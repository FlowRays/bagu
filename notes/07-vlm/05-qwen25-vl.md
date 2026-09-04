# Qwen2.5-VL：一张图从 pixel 走到 LLM 的完整路径

> 前置：[01b Patch Embedding](01b-patch-embedding.md)（图片怎么变成 $N\times1280$）。
> 这篇把 Qwen2.5-VL 的 **ViT 本体 → PatchMerger → 塞进 LLM sequence → 三阶段训练** 一次走完，每一步都追踪 shape。

$$\boxed{\text{Image/Video}\rightarrow\text{3D Patch Embed}\rightarrow32\times\text{ViT Block}\rightarrow\text{PatchMerger}\rightarrow\text{LLM}}$$

## 1. 先背下这七个数字

Qwen2.5-VL-7B 的 vision config：

$$\boxed{\begin{aligned}&\text{patch}: &&14\times14\\&\text{temporal patch}: &&2\\&\text{hidden}: &&1280\\&\text{layers}: &&32\\&\text{heads}: &&16\ (d_h=80)\\&\text{MLP intermediate}: &&3420\\&\text{window}: &&112\times112\\&\text{full-attn layers}: &&7,\ 15,\ 23,\ 31\end{aligned}}$$

ViT backbone 约 **630M** 参数 —— 相比 7B LLM 是小，但单看视觉 encoder 已经不算小了。

⚠️ 注意它**不是拿现成的 SigLIP/CLIP**，是 Qwen 自己设计并从零训练的 ViT（对比 [Qwen3-VL 用 SigLIP-2 初始化](02-vision-encoder.md#4-主流模型的实际配置)）。

## 2. ViT block：标准 Pre-Norm

$$X'=X+\operatorname{Attention}(\operatorname{RMSNorm}(X))$$
$$X''=X'+\operatorname{MLP}(\operatorname{RMSNorm}(X'))$$

```text
X ── RMSNorm ── Attention ──+──  RMSNorm ── SwiGLU ──+── out
 └──────────────residual────┘  └─────────residual────┘
```

两处和经典 ViT 不一样：

- **norm 用 RMSNorm**，不是 LayerNorm
- **FFN 是 SwiGLU 风格**，不是传统的单层 GELU：

$$\operatorname{MLP}(x)=W_{down}\big[\operatorname{SiLU}(W_{gate}x)\odot W_{up}x\big],\qquad 1280\to3420\to1280$$

也就是说这个 ViT 长得比"经典 ViT"更像现代 LLM 的 block。

### Attention 是双向的

$$Q=XW_Q,\ K=XW_K,\ V=XW_V,\qquad A=\operatorname{softmax}\Big(\frac{QK^\top}{\sqrt{80}}\Big)V$$

和 LLM 最大的区别：**ViT 没有 causal mask**，一个 patch 可以看到窗口/图像里的其他 patch。官方实现里就是 `is_causal=False`。

## 3. Window Attention：28 层局部 + 4 层全局

如果 32 层全做 full attention，高分辨率图片的成本很恐怖：$N=\frac H{14}\frac W{14}$，复杂度 $O(N^2)$，$1024\times1024$ 的图就是几千个 patch。

所以 Qwen2.5-VL：

$$\boxed{28\text{ 层 Window Attention}+4\text{ 层 Full Attention}}$$

只有第 $\boxed{7,15,23,31}$ 层做 full attention：

```text
Layer 0-6    Window
Layer 7      FULL      ← 全局信息交换
Layer 8-14   Window
Layer 15     FULL
Layer 16-22  Window
Layer 23     FULL
Layer 24-30  Window
Layer 31     FULL
```

$$\boxed{\text{大量 local perception}+\text{周期性的 global information exchange}}$$

### window 多大

`window_size=112`，patch 是 $14\times14$，所以一个 window 覆盖 $112\times112$ 像素 = 空间上 $8\times8$ 个 ViT patch。

```text
┌──────┬──────┬──────┬──────┐
│window│window│window│window│   普通层：每个 window 内部各自 attention
├──────┼──────┼──────┼──────┤
│window│window│window│window│   第 7/15/23/31 层：所有 patch 一起 attention
└──────┴──────┴──────┴──────┘
```

⚠️ **Window attention 不减少 token 数 $N$**，它只减少每个 token 能看到的范围。压 token 数是后面 PatchMerger 的事。

这套设计正是它能支持 **native dynamic resolution**（不强行 resize 到固定尺寸）的原因。

## 4. ViT 内部是 2D RoPE，不是 MRoPE

这里最容易和 LLM 里的 MRoPE 搞混。

ViT 内部不用 learned absolute PE，而是在 $Q/K$ 上加 **二维 rotary**，按 patch 的 $(h,w)$ 网格坐标构造 position id：

$$(h,w)\ \longrightarrow\ \operatorname{RoPE}(Q,K)$$

也就是告诉 attention"**这个 patch 在图像的第几行、第几列**"。

$$\boxed{\text{ViT 内部：2D RoPE}}\qquad\ne\qquad\boxed{\text{LLM 内部：MRoPE}}$$

**两个不同层级的位置编码模块，面试千万别混。**

## 5. PatchMerger：既是压缩，也是 projector

ViT 出来仍然是 $[1024,1280]$（32 层不改变 token 数和维度）。PatchMerger 干两件事：

**① 空间上 $2\times2$ 合并 —— concat，不是 pooling**

```text
x₁  x₂        每个都是 1280 维
x₃  x₄   →   concat [x₁;x₂;x₃;x₄] → 4×1280 = 5120
```

$$\text{hidden}=\text{context\_dim}\times\text{spatial\_merge\_size}^2=1280\times2^2=5120$$

grid $32\times32\to16\times16$，所以 $\boxed{N\to N/4}$，$1024\to256$。

**先 concat 保住信息，再让 MLP 学怎么压** —— 不是 average pooling，也不是求和。

**② 一个 MLP 把维度打到 LLM 的 hidden size**

$$\boxed{\text{RMSNorm}\rightarrow\text{Linear}(5120,5120)\rightarrow\text{GELU}\rightarrow\text{Linear}(5120,3584)}$$

所以完整 shape 链：

| 阶段 | shape |
|---|---|
| ViT 前 | $[1024,\ 1280]$ |
| ViT 32 层后 | $[1024,\ 1280]$ |
| $2\times2$ concat | $[256,\ 5120]$ |
| MLP | $\boxed{[256,\ 3584]}$ |

### ⭐ 一个 LLM visual token 对应原图多大

ViT patch 是 $14\times14$ 像素，PatchMerger 再合 $2\times2$ 个 patch，所以

$$\boxed{1\text{ 个 LLM visual token}\ \leftrightarrow\ 28\times28\text{ 像素}}$$

$$\boxed{N_{\text{LLM-vis}}=\frac H{28}\cdot\frac W{28}}$$

$448\times448\Rightarrow16\times16=256$。这也解释了为什么 preprocessing 要求边长能被 28 整除。

**这个 PatchMerger 就是 Qwen2.5-VL 的 projector**，只不过它不是简单的 $\text{Linear}(1280,3584)$，而是"先 $2\times2$ concat 再两层 MLP"。

## 6. ⭐⭐ visual token 怎么真的进到 LLM 的序列里

这一步是很多人从没搞清楚的。**没有 cross-attention**，是直接替换 embedding。

**① chat template 里图片只是一个占位符**

```text
<|im_start|>user
描述这张图片：
<|vision_start|><|image_pad|><|vision_end|>
<|im_end|>
```

$$\boxed{\texttt{<|image\_pad|>}\ \text{最开始只是占位符}}$$

它**不是**"一张图只占一个 token"的意思。

**② processor 把它展开成 N 个**

$$N_{\text{image}}=\frac{T_{grid}H_{grid}W_{grid}}{\text{merge\_size}^2}=\frac{1\times32\times32}{4}=256$$

于是序列里真的出现 256 个连续的 `<|image_pad|>`。

**③ 文字走正常 embedding lookup**

$e_i=E[id_i]$，$E\in\mathbb R^{V\times3584}$。注意 `<|vision_start|>`、`<|vision_end|>` 这些 special token **也有自己的 learned embedding**。

**④ 关键：把 image_pad 的 embedding 整个替换掉**

vision 侧同时算出 $V=[v_1,\dots,v_{256}]$，$v_i\in\mathbb R^{3584}$。模型找到所有 `image_token_id` 的位置，用 `masked_scatter` 把原来的 embedding **换掉**：

```text
<image_pad> embedding          visual embedding v1
<image_pad> embedding    →     visual embedding v2
    ...                            ...
<image_pad> embedding          visual embedding v256
```

$$\boxed{\text{image\_pad token 的位置}\ \longrightarrow\ \text{Vision Encoder 的输出}}$$

**⑤ 最终送进 LLM 的就是一条普通序列**

```text
<im_start> user 描述 这 张 图片 <vision_start> v1 v2 ... v256 <vision_end> <im_end>
```

$$X_{LLM}\in\mathbb R^{L\times3584},\qquad L=N_{text}+N_{visual}+N_{special}$$

**这就是 projector 必须输出 3584 的原因** —— 它得和 text embedding 落在同一个空间、同一个维度，才能拼在一条序列里被同一套 self-attention 处理。进入 LLM 之后就不再区分"视觉 Transformer"和"文本 Transformer"了。

## 7. 三阶段训练

| 阶段 | 数据量 | context | 更新 | 数据 |
|---|---:|---:|---|---|
| Visual Pre-training | 1.5T | 8K | **ViT** | caption / 视觉知识 / OCR |
| Multimodal Pre-training | 2.0T | 8K | **全部** | 图文交错、VQA、Math、Video、Agent、纯文本 |
| Long-context Pre-training | 0.6T | 32K | **全部** | 长视频、长文档、长 Agent |

合计 $1.5+2.0+0.6=\boxed{4.1\text{T tokens}}$（Qwen2-VL 约 1.2T，扩大了不少）。

### 初始化

| 模块 | 初始化 |
|---|---|
| ViT | random → DataComp 等做 CLIP 式视觉预训练（**不是拿现成 SigLIP 权重**） |
| PatchMerger | 全新模块，无预训练权重 |
| LLM | 直接用 pretrained **Qwen2.5** |

### Stage 1：LLM 冻结，靠梯度穿过去训 ViT

论文明确写 Training = ViT，LLM frozen。loss 就是普通的文本 CE：

$$L_{CE}=-\sum_t\log p(y_t\mid y_{<t},I)$$

**visual token 本身不做监督，只监督 text token。**

LLM 冻结不代表梯度过不去：

$$\frac{\partial L}{\partial\theta_V}=\frac{\partial L}{\partial V}\cdot\frac{\partial V}{\partial\theta_V}$$

$\frac{\partial L}{\partial\theta_{LLM}}$ 算出来只是不拿去更新，但 $\frac{\partial L}{\partial V}$ 照样往前传。所以效果是：

> Qwen2.5 LLM 当一个**固定的语言解释器**，逼 ViT 学会输出"能让这个 LLM 正确预测 caption/OCR 文本"的视觉表示。

这就是 $\boxed{\text{Vision-Language Alignment}}$。

### Stage 1 的 projector 到底训不训（报告没说）

论文只写了 "only the Vision Transformer is trained"，**没有单独交代 merger 的 freeze 状态**。两种解释都讲得通：

- **解释 A**：字面意思，ViT ✅ / Projector ❌ / LLM ❌
- **解释 B**：表格把 ViT+Merger 统称为 vision part，实际 ViT ✅ / Projector ✅ / LLM ❌

从架构上看 A 有点奇怪（随机 projector 全程冻结，只逼 ViT 去适配一个随机映射，理论上能训但不是自然的 alignment recipe），但**不能替论文补全**。

被问到就这么答：

> Qwen2.5-VL 报告明确写 Stage 1 只训练 ViT、冻结 LLM，但没有单独交代 VL merger 的 freeze 状态，因此 projector 是否被包含在其 "ViT training" 的表述里并不清楚。

### Stage 2 / 3：全部解冻

Stage 2 官方明确 **all model parameters are unfrozen**，Stage 3 继续全参数训练，只是 context $8K\to32K$、数据换成长视频/长文档/长 agent。

$$\boxed{\begin{array}{c|ccc} & ViT & Projector & LLM\\\hline \text{初始化} & \text{视觉预训练} & \text{新模块} & \text{Qwen2.5}\\ \text{Stage 1} & \checkmark & ? & \times\\ \text{Stage 2} & \checkmark & \checkmark & \checkmark\\ \text{Stage 3} & \checkmark & \checkmark & \checkmark\end{array}}$$

### 不是三个模块三个 loss

$$L_{ViT}+L_{projector}+L_{LLM}\quad\text{✗}$$

只有一个 autoregressive language modeling objective：

$$\boxed{L_{NTP}=-\sum_{\text{target text tokens}}\log p_\theta(y_t\mid y_{<t},I)}$$

因为计算图是连着的（$ViT\to Projector\to LLM\to L$），**一个 loss 就能更新三个模块**，和普通神经网络 $Layer_1\to Layer_2\to Layer_3\to CE$ 完全一样，不需要每层各配一个 loss。详见 [04 训练阶段](04-training-stages.md#6-卡点三个模块并不各有一个-loss)。

## 面试版

> Qwen2.5-VL 的 vision 塔是 Qwen 自研、从零训练的 32 层 ViT（hidden 1280、patch 14、temporal patch 2、约 630M）。为了控制高分辨率成本，32 层里只有第 7/15/23/31 层做 full attention，其余是 $112\times112$ 的 window attention，因此支持 native dynamic resolution；位置信息在 ViT 内部用 2D RoPE 表达（和 LLM 里的 MRoPE 不是一回事）。ViT 输出后由 PatchMerger 把空间上 $2\times2$ 的四个 token concat 成 5120 维再过 MLP 打到 3584，token 数变成 $N/4$，所以一个 LLM visual token 对应原图 $28\times28$ 像素。这些 visual embedding 不走 cross-attention，而是把 chat template 里展开出来的 N 个 `<|image_pad|>` 占位符的 embedding 直接替换掉，之后就是一条普通序列走 LLM。训练是三阶段共 4.1T token：先冻 LLM 只训 ViT 做对齐，再全参数多模态预训练，最后长上下文；全程只有文本 NTP 一个 loss。

## 自测

**1.** ⭐ 背出 Qwen2.5-VL ViT 的关键配置。

> **答：** patch $14\times14$、temporal patch 2、hidden 1280、32 层、16 heads（$d_h=80$）、MLP 3420、window $112\times112$、full-attn 在第 7/15/23/31 层。backbone 约 630M。

**2.** ⭐⭐ 32 层里 attention 怎么排？为什么这么排？window 多大？

> **答：** **28 层 window + 4 层 full**，full 在第 7/15/23/31 层。因为全用 full attention 时复杂度 $O(N^2)$，高分辨率下 $N$ 有几千，成本爆炸。所以做成"大量 local perception + 周期性 global information exchange"。
> window_size=112 像素 = $8\times8$ 个 ViT patch。
> ⚠️ window attention **不减少 token 数**，只限制每个 token 能看到谁。

**3.** ⭐ Qwen2.5-VL 的 ViT block 和经典 ViT block 有什么不同？

> **答：** ① norm 用 **RMSNorm** 不是 LayerNorm；② FFN 是 **SwiGLU** 风格 $W_{down}[\text{SiLU}(W_{gate}x)\odot W_{up}x]$，$1280\to3420\to1280$，不是单层 GELU。整体更像现代 LLM 的 block。attention 是 **双向的**（`is_causal=False`）。

**4.** ⭐⭐ ViT 里的 2D RoPE 和 LLM 里的 MRoPE 是一回事吗？

> **答：** **不是。** ViT 内部按 patch 的 $(h,w)$ 网格坐标构造 position id，在 $Q/K$ 上加二维 rotary，告诉 attention"这个 patch 在第几行第几列"。MRoPE 是 LLM 内部的位置编码。**两个不同层级的模块。**

**5.** ⭐ PatchMerger 做了哪两件事？为什么是 concat 不是 pooling？

> **答：** ① 空间上 $2\times2$ 合并：四个 1280 维 token **concat** 成 $4\times1280=5120$，token 数 $N\to N/4$；② $\text{RMSNorm}\to\text{Linear}(5120,5120)\to\text{GELU}\to\text{Linear}(5120,3584)$ 打到 LLM hidden size。
> 用 concat 是为了**先尽量保住信息，再让 MLP 学怎么压**；average pooling 会直接丢掉四个 token 之间的差异。

**6.** ⭐ 一个 LLM visual token 对应原图多大区域？$448\times448$ 有多少 visual token？

> **答：** ViT patch $14\times14$，再 $2\times2$ merge，所以一个 LLM visual token 对应 $\boxed{28\times28}$ 像素，$N_{\text{LLM-vis}}=\frac H{28}\frac W{28}$。
> $448\times448\Rightarrow16\times16=\boxed{256}$。（ViT 内部是 1024，别混。）

**7.** ⭐⭐ visual embedding 到底怎么进 LLM 序列的？有 cross-attention 吗？

> **答：** **没有 cross-attention。** chat template 里先放一个占位符 `<|image_pad|>`，processor 按 $\frac{T_gH_gW_g}{\text{merge}^2}$ 把它**展开成 256 个**；文字走正常 embedding lookup；然后模型找到所有 `image_token_id` 的位置，用 `masked_scatter` 把这些占位符的 embedding **整个替换成** Vision Encoder 的输出 $v_1,\dots,v_{256}$。之后就是一条普通的 $L\times3584$ 序列走 LLM。
> 这就是 projector 必须输出 3584 的原因 —— 要和 text embedding 同空间同维度。

**8.** ⭐ Stage 1 里 LLM 冻结了，ViT 怎么学？

> **答：** 冻结只是**不更新** $\theta_{LLM}$，梯度照样穿过去：$\frac{\partial L}{\partial\theta_V}=\frac{\partial L}{\partial V}\frac{\partial V}{\partial\theta_V}$。效果是把 Qwen2.5 LLM 当成一个固定的语言解释器，逼 ViT 学出"能让这个 LLM 正确预测 caption/OCR 文本"的视觉表示，即 vision-language alignment。

**9.** ⭐⭐ Qwen2.5-VL 的 Stage 1 训不训 projector？

> **答：** **报告没说。** 只写了 "only the Vision Transformer is trained" + LLM frozen，没单独交代 merger。
> 可能是字面意思（projector 也冻结），也可能是表格把 ViT+Merger 统称 vision part。**不要编**，就答"报告明确 ViT train / LLM frozen，但没有单独交代 merger 的 freeze 状态"。

**10.** 三阶段的数据量、context 和更新对象分别是什么？

> **答：** Visual PT（1.5T / 8K / 只 ViT，caption+视觉知识+OCR）→ Multimodal PT（2.0T / 8K / **全部解冻**，图文交错+VQA+Math+Video+Agent+纯文本）→ Long-context PT（0.6T / 32K / 全部，长视频+长文档+长 agent）。合计 4.1T。

**11.** Qwen2.5-VL 的 ViT 是从 SigLIP 初始化的吗？

> **答：** **不是。** 它是 Qwen 自己设计并从零训练的 ViT，先用 DataComp 等数据做 CLIP 式视觉预训练，再进入三阶段 VLM 训练。"from scratch" 指的是**没有拿现成 SigLIP/CLIP 权重当视觉塔**，不是说随机 ViT 直接和 LLM 一起裸训（那是 [Kimi K3 MoonViT-V2](02-vision-encoder.md#5-kimi-k3从零训练的-moonvit-v2) 更激进的做法）。对比 Qwen3-VL 则是明确从 SigLIP-2 checkpoint 初始化的。
