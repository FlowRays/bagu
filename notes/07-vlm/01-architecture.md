# VLM 架构：一张图片怎么进 LLM

## 1. Projector 不是"把 1024 维硬变成 4096 维"

最经典的 LLaVA-style：$336\times336$ 图、patch=14

$$\frac{336}{14}=24\ \Rightarrow\ 24\times24=576\ \text{visual tokens}$$

每个 patch（$14\times14\times3$）先线性投影成 embedding，过 ViT 的 self-attention，得到

$$Z_v\in\mathbb R^{576\times d_v},\qquad d_v=1024$$

但 LLM 的 $d=4096$，所以需要 projector（实际常是两层 MLP）：

$$h_i=W_2\,\sigma(W_1z_i)\ \Rightarrow\ H_v\in\mathbb R^{576\times4096}$$

然后直接拼：

$$[\text{BOS},h_1^{img},\dots,h_{576}^{img},h_1^{text},\dots,h_n^{text}]$$

$$\boxed{\text{对 LLM 来说，图片不是特殊对象，就是一串连续 embedding token}}$$

**但如果 projector 是随机初始化的，维度对上也没用**：它输出的 4096 维向量和 LLM 已经学会的语义空间毫无对应关系。所以 projector 真正做的是

$$\boxed{\text{representation alignment，而不只是 dimension matching}}$$

## 2. 为什么一个简单 MLP 就够

**因为 Vision Encoder 本身已经很强了。** CLIP ViT 经过几十层 Transformer，输出的 visual feature 已经是高度语义化的（这是一个人 / 他拿着杯子 / 这个区域是红色 / 这里像一只狗）。

projector 不需要重新"学会看图"，它只需要学

$$\text{Vision representation}\longrightarrow\text{LLM 能消费的 representation}$$

这比 `raw pixel → language understanding` 简单太多。

### 翻译器类比

Vision Encoder 说的是"视觉语"，$z_v=[0.2,-1.7,0.8,\dots]$ 内部其实已经表达"一只狗坐在草地上"，但 LLM 不懂这个方言。projector 就是那个翻译。

**注意**：目标**不是** $P(z_{\text{dog}})=E(\text{"dog"})$。只要求 $P(z)$ 进入 LLM 后，经过若干层 self-attention，能让 LLM 正确预测 "There is a dog in the image."。它只需要成为 **LLM 可解释的 conditioning representation**。

## 3. CLIP 为什么在这里特别关键

普通 ImageNet supervised ViT 学的是 $I\to\text{class}$；CLIP 学的是让 $\text{sim}(z_I,z_T)$ 高，**它的视觉空间本来就被语言监督塑形过**。

$$\boxed{\text{CLIP 已经负责了第一层 visual-language alignment}}$$
$$\boxed{\text{projector 只负责 CLIP representation}\to\text{LLM representation}}$$

这就是 LLaVA 当时最重要的 insight：`CLIP + 简单 MLP + LLM` 居然就很好用。

## 4. Stage 1：怎么训 projector

$$\boxed{\text{Frozen Vision Encoder}+\text{Trainable Projector}+\text{Frozen LLM}}$$

数据是 image-caption pair，loss 就是普通 next-token prediction：

$$\mathcal L=-\sum_t\log p_\theta(y_t\mid y_{<t},I)$$

能动的只有 projector，所以梯度 $\partial\mathcal L/\partial\theta_{\text{proj}}$ 逼着它学：

> 我要把视觉 feature 转换成什么样的 embedding，才能让这个 **frozen** LLM 正确生成 caption？

## 5. 卡点：LLM 冻结了，梯度还能穿过它吗

**能。** 这是最容易搞混的一点。

图片是猫，projector 随机初始化，LLM 输出 "A car …"，$\mathcal L=-\log p(\text{cat})$ 很大。反向：

$$\mathcal L\to\text{LLM}\to h_{\text{vision}}\to\text{Projector}$$

冻结 $\theta_{LLM}$ 只意味着 **不执行** $\theta_{LLM}\leftarrow\theta_{LLM}-\eta\nabla_{\theta_{LLM}}\mathcal L$ 这一步，但

$$\boxed{\frac{\partial\mathcal L}{\partial h_{\text{vision}}}\ \text{照样能算出来}}$$

大量 image-caption pair 之后，视觉空间就接上去了。

## 6. 为什么不用一个巨大的 cross-attention 模块

因为如果 Vision Encoder 已经产生高度语义化、且和语言相关的特征，那两个 representation manifold 之间可能**只需要一个相对简单的映射**。这也是 LLaVA 相比 Flamingo/BLIP-2 那种重 cross-attention / Q-Former 设计的简洁之处。

## 7. 统一 self-attention：四个象限

拼起来 $H=[H_v;H_t]$ 之后就是普通 Transformer：$Q=HW_Q$、$K=HW_K$、$V=HW_V$。于是出现四种 attention：

| | $K_{\text{vision}}$ | $K_{\text{text}}$ |
|---|---|---|
| $Q_{\text{vision}}$ | V→V | V→T |
| $Q_{\text{text}}$ | **T→V** | T→T |

其中 **T→V** 是关键：文本 token "dog" 的 query 直接 attend 到相关的 visual patch，$q_{\text{dog}}^\top k_{\text{image patch}}$ 大 → attention weight 高 → 模型从那块图像区域取信息。

这就是 **single-stream / unified self-attention** 的核心直觉。

## 8. 高频八股答法："VLM 怎么训练"

分三层答：

1. **视觉表征**：$I\xrightarrow{ViT}Z_v$，通常初始化自 CLIP / SigLIP 等预训练 encoder
2. **模态对齐**：$Z_v\xrightarrow{Projector}H_v$，冻结 VE 和 LLM，用 image-caption 训 projector
3. **多模态 instruction tuning**：$[I,\text{instruction},\text{response}]$，autoregressive CE 联合训练 projector + LLM，必要时解冻 VE

$$\boxed{\text{很多经典 VLM 的核心训练就是普通 next-token cross entropy，只是 input 前面多了一堆 visual embedding}}$$

## 自测（口述版）

1. $336\times336$、patch=14 得到多少 visual token？为什么？
2. projector 如果随机初始化为什么没用？它真正在做什么？
3. 为什么一个两层 MLP 就够？projector 的目标是不是 $P(z_{\text{dog}})=E(\text{"dog"})$？
4. Stage 1 冻结哪些、训哪些？loss 是什么？
5. LLM 冻结了，梯度还能穿过它吗？"冻结"到底冻的是哪一步？
6. 画出 visual/text token 的四象限 attention，指出哪一格最关键。
