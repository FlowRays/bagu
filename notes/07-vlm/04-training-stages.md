# VLM 多阶段训练：每阶段的 loss 和更新谁

> 先分清楚两套"阶段"：**A. Vision Encoder 自己的预训练**（CLIP/SigLIP/DINO）；
> **B. 组装成 VLM 后的训练**（alignment → multimodal pretrain → SFT → distill → RL）。

## 1. Stage −1：Vision Encoder pretraining

| 方法 | loss | 更新谁 |
|---|---|---|
| CLIP | $L=\frac12(L_{I\to T}+L_{T\to I})$，softmax contrastive | Vision Encoder + Text Encoder |
| SigLIP | sigmoid pairwise 二分类 | Vision Encoder + Text Encoder |
| DINO | $\text{CE}(p_{\text{teacher}},p_{\text{student}})$ | student 走 SGD，**teacher 走 EMA**，无 text encoder |

## 2. Stage 0：Vision-Language Alignment

$$\boxed{VE:\text{frozen}\quad Projector:\text{train}\quad LLM:\text{frozen}}$$

loss 就是普通 causal LM：

$$L_{\text{NTP}}=-\sum_t\log p(y_t\mid I,y_{<t})$$

梯度 $L\to LLM\to Projector$，但只有 $\theta_P\leftarrow\theta_P-\eta\nabla_{\theta_P}L$ 真正执行。

Qwen3-VL 明确有这个阶段（S0），约 67B tokens，只更新 merger。

## 3. Stage 1：Multimodal Pretraining

$$\boxed{VE+Projector+LLM\ \text{全部 train}}$$

数据是 caption / OCR / VQA / interleaved image-text / grounding / document / video / code / pure text。loss 本质仍是 next-token cross entropy，只是梯度这次一路更新到 ViT。

### Qwen3-VL 的四个阶段

| Stage | Context | 更新 |
|---|---:|---|
| S0 Alignment | 8K | projector |
| S1 MM pretrain | 8K | ALL |
| S2 long context | 32K | ALL |
| S3 ultra-long | 256K | ALL |

S1/S2/S3 不是换了 loss，主要是**改 data mixture + context length**。

### Kimi K3 更激进

连 Stage −1 都不要，MoonViT from scratch，从训练一开始就 $VE+\text{projector}+LLM$ 一起在统一的 $L_{\text{NTP}}$ 下更新，官方称为 **native multimodal training**。

$$\text{早期 VLM}=\underbrace{\text{Vision}}_{\text{独立训练}}+\underbrace{\text{LLM}}_{\text{独立训练}}+\underbrace{\text{Projector}}_{\text{把两个焊起来}}$$
$$\text{现在}\rightarrow\boxed{\text{Vision + Language 从 foundation pretraining 就共同优化}}$$

## 4. Instruction SFT

$$L_{\text{SFT}}=-\sum_{t\in\text{Assistant}}\log p(y_t\mid I,x,y_{<t})$$

## 5. Pretrain 的 CE 和 SFT 的 CE 差在哪

数学形式**几乎一样**，区别是 **loss mask**：

```text
Pretrain :  The cat sits on the table.     ← 几乎全部 token 都算 loss

SFT      :  System   → mask
            User     → mask
            Image    → 没有离散 CE target
            Assistant→ loss
```

## 6. 卡点：三个模块并不各有一个 loss

$$\boxed{\text{通常不是 ViT / Projector / LLM 各有一个 loss}}$$

三个模块共享**同一个**最终任务 loss，梯度一路 $\mathcal L\to LLM\to Projector\to ViT$，**谁 `requires_grad=True` 谁就更新**。

设 $I\xrightarrow[\theta_V]{ViT}Z\xrightarrow[\theta_P]{Proj}H\xrightarrow[\theta_L]{LLM}y$：

| 情况 | ViT | Projector | LLM | loss |
|---|---|---|---|---|
| A | frozen | frozen | **train** | $L_{\text{SFT}}$ |
| B | frozen | **train** | **train** | $L_{\text{SFT}}$ |
| C | **train** | **train** | **train** | $L_{\text{SFT}}$ |

情况 C 下三个梯度同时存在：

$$\theta_V\leftarrow\theta_V-\eta_V\nabla_{\theta_V}L,\quad \theta_P\leftarrow\theta_P-\eta_P\nabla_{\theta_P}L,\quad \theta_L\leftarrow\theta_L-\eta_L\nabla_{\theta_L}L$$

pretraining 有时会额外加 auxiliary loss（image-text contrastive、grounding/bbox、masked image modeling、DINO self-distillation、detection），但那是**帮助 representation learning**，不是"ViT 必须有专门 loss 才能更新"。

## 7. 为什么语言 CE 能更新 ViT

```text
<image>  Which button should I click?      GT: the blue button
```

模型没看清颜色：$p(\text{red})=0.6$、$p(\text{blue})=0.1$，$L=-\log0.1$。反向：

$$L\to\text{logits}\to\text{LLM hidden}\to\text{visual embedding}\to\text{Projector}\to\text{ViT feature}$$

等于告诉 ViT：**你现在产生的视觉表示没让后面区分出 blue。** 这就是 end-to-end multimodal training。

类比：$\text{image}\to\text{CNN}\to\text{classifier}\to\text{CE}$，输出也不是图片，CE 一样能训 CNN。

## 8. 卡点：image token 是 condition，不是 prediction target

因为 VLM 只有文本输出，所以确实不会对 image token 计算 next-token CE，loss mask 类似

$$[\underbrace{0,\dots,0}_{\text{image}},\underbrace{0,\dots,0}_{\text{user}},\underbrace{1,\dots,1}_{\text{assistant}}]$$

**但这不代表图片这一支没有梯度。** assistant 的预测依赖图片，所以链式法则给出

$$\frac{\partial L}{\partial H_v}\neq0,\qquad \frac{\partial L}{\partial\theta_P}=\frac{\partial L}{\partial H_v}\frac{\partial H_v}{\partial\theta_P},\qquad \frac{\partial L}{\partial\theta_V}=\frac{\partial L}{\partial H_v}\frac{\partial H_v}{\partial Z_v}\frac{\partial Z_v}{\partial\theta_V}$$

$$\boxed{\text{Image token 是 condition，不是 prediction target；但 condition 仍然在 computation graph 里}}$$

（和 PPO/GRPO 很像：loss 定义在最终 action/token 上，但梯度可以更新整个产生中间 representation 的前向网络。）

## 9. SFT 到底要不要训 ViT

判断标准很简单 —— 你的数据在改变什么：

| 改变 | 例子 | ViT |
|---|---|---|
| **怎么思考 / 怎么回答** | math CoT、instruction following、agent planning、输出格式 | 大概率 **freeze** 就够（perception 没有 domain shift） |
| **怎么看** | 医学影像、卫星图、GUI 小图标、游戏画面、小目标检测、OCR、spatial grounding、特殊传感器 | **unfreeze 可能明显更重要**（瓶颈就在 $I\to Z_v$） |

## 10. RL 为什么更倾向 freeze ViT

GRPO/PPO 的 policy loss 定义在输出 token 的 log probability 上：

$$L_{\text{GRPO}}=-\sum_t\min\big(r_tA_t,\ \text{clip}(r_t)A_t\big)+\beta\,\text{KL},\qquad r_t=\frac{\pi_\theta(y_t|I,x,y_{<t})}{\pi_{\text{old}}(y_t|I,x,y_{<t})}$$

如果 ViT 解冻，policy gradient 同样能一路传回 ViT，reward 可以真的改变视觉表示（Game Agent 里 `frame → VLM → action=LEFT → reward=-1` 会同时压低相关视觉 feature）。

**但实践中 RL 往往冻结 vision encoder**，因为 reward 对 perception 来说太间接：$R=-1$ 可能因为图没看清、reasoning 错、planning 错、action sampling 错、长程 credit assignment 错。全参更新会让 ViT 为这个 reward 承担部分责任，容易造成

$$\boxed{\text{representation drift}}$$

甚至破坏已有的 OCR / detection / general vision 能力。

$$\boxed{\text{SFT：可以较积极地 unfreeze vision}}\qquad\boxed{\text{RL：更保守地 freeze vision}}$$

## 11. 实用组合表

| 目标 | ViT | Projector | LLM |
|---|---|---|---|
| Alignment | freeze | **train** | freeze |
| MM Pretrain | **train** | **train** | **train** |
| 普通 VLM SFT | freeze / train | **train** | **train** |
| LoRA SFT | freeze | optional | **LoRA** |
| reasoning RL | 通常 freeze | freeze/train | **train** |
| perception-heavy RL | 可 train | **train** | **train** |

训练框架一般允许三个模块独立 freeze（如 NVIDIA 的 Qwen2.5-VL finetuning recipe 分别提供 language model / vision model / vision projection 的冻结开关）。

## 12. 整条 pipeline

$$\underbrace{\text{Vision Pretraining}}_{L_{\text{contrastive / SSL}}}\rightarrow\underbrace{\text{VL Alignment}}_{L_{\text{NTP}},\ \text{更新 Projector}}\rightarrow\underbrace{\text{MM Pretraining}}_{L_{\text{NTP}},\ \text{更新 VE+P+LLM}}\rightarrow\underbrace{\text{Instruction SFT}}_{\text{CE on assistant tokens}}\rightarrow\underbrace{\text{Distillation}}_{\text{CE / KL}}\rightarrow\underbrace{\text{RL}}_{\text{PPO/GRPO/DAPO}}$$

Distillation 阶段：有 teacher logits 就做 $L_{KD}=D_{KL}(p_T\|p_S)$（见 [蒸馏目录](../06-post-training/distill/00-map.md)）；只有 teacher rollout 就退化成 $L_{CE}=-\log p_S(y_T)$。Qwen3-VL post-training 明确是 `SFT → Strong-to-Weak Distillation → RL`，distillation 阶段用 text-only 数据增强 LLM reasoning。

## 13. 面试标准答法

> **VLM 的 ViT、projector 和 LLM 并不需要分别定义三个 loss。** 普通 multimodal SFT 就是 assistant token 上的 autoregressive cross-entropy。只训练 LLM 时梯度只更新 LLM；同时训练 projector 时 language loss 会通过 LLM 反传到 projector；ViT 也解冻时，同一个 language loss 会继续反传到 ViT，实现 end-to-end visual-language adaptation。
>
> **RL 同理**，GRPO/PPO 的 policy loss 定义在输出 token 的 log probability 上，vision encoder 解冻时 policy gradient 同样能传回 ViT。但实践中 RL 更倾向冻结 vision encoder，因为 reward noisy、credit assignment 间接，直接更新 ViT 容易造成视觉 representation drift。

## 自测（口述版）

**1.** CLIP / SigLIP / DINO 三者的 loss 和更新对象分别是什么？

> **答：** **CLIP**：$L=\frac12(L_{I\to T}+L_{T\to I})$ softmax contrastive，更新 Vision Encoder + Text Encoder；
> **SigLIP**：sigmoid pairwise 二分类，同样更新 VE + Text Encoder；
> **DINO**：$\text{CE}(p_{\text{teacher}},p_{\text{student}})$，student 走 SGD、**teacher 走 EMA**，**没有 text encoder**。

**2.** Stage 0 冻结谁、训谁、loss 是什么？Qwen3-VL 的 S0–S3 分别改了什么？

> **答：** Stage 0（VL Alignment）：**VE frozen、Projector train、LLM frozen**，loss 就是普通 $L_{\text{NTP}}=-\sum_t\log p(y_t|I,y_{<t})$。Qwen3-VL 的 S0 约 67B tokens，只更新 merger。
> S0 (8K, projector) → S1 (8K, ALL) → S2 (32K, ALL) → S3 (256K, ALL)。**S1/S2/S3 不是换了 loss，主要是改 data mixture + context length。**

**3.** Pretrain 的 CE 和 SFT 的 CE 差多少？真正的区别在哪？

> **答：** **数学形式几乎一样**，都是 $-\log p(y_t|\cdots)$。真正的区别是 **loss mask**：
> pretrain 几乎全部 token 都算 loss；SFT 是 System→mask、User→mask、Image→没有离散 CE target、**只有 Assistant 段算 loss**。

**4.** 三个模块是不是各有一个 loss？如果三个都训，写出三个更新式。

> **答：** **通常不是。** 三个模块共享**同一个**最终任务 loss，梯度一路 $\mathcal L\to LLM\to Projector\to ViT$，谁 `requires_grad=True` 谁就更新。
> $$\theta_V\leftarrow\theta_V-\eta_V\nabla_{\theta_V}L,\quad\theta_P\leftarrow\theta_P-\eta_P\nabla_{\theta_P}L,\quad\theta_L\leftarrow\theta_L-\eta_L\nabla_{\theta_L}L$$
> pretraining 有时会额外加 auxiliary loss（image-text contrastive、grounding/bbox、masked image modeling、DINO self-distillation、detection），但那是**帮助 representation learning**，不是「ViT 必须有专门 loss 才能更新」。

**5.** 用一个具体例子说明语言 CE 怎么把梯度传到 ViT。

> **答：** `<image> Which button should I click?`，GT 是 “the blue button”。模型没看清颜色：$p(\text{red})=0.6$、$p(\text{blue})=0.1$，$L=-\log0.1$ 很大。
> 反向路径：$L\to\text{logits}\to\text{LLM hidden}\to\text{visual embedding}\to\text{Projector}\to\text{ViT feature}$，等于告诉 ViT「**你现在产生的视觉表示没让后面区分出 blue**」。
> 类比：$\text{image}\to\text{CNN}\to\text{classifier}\to\text{CE}$，输出也不是图片，CE 一样能训 CNN。

**6.** 图片不是输出，那它有梯度吗？写出链式法则。真正"不需要考虑"的是什么？

> **答：** **有梯度。** loss mask 上 image token 确实是 0（不作为 prediction target），但 assistant 的预测**依赖**图片，所以 $\frac{\partial L}{\partial H_v}\ne0$，进而
> $$\frac{\partial L}{\partial\theta_P}=\frac{\partial L}{\partial H_v}\frac{\partial H_v}{\partial\theta_P},\qquad \frac{\partial L}{\partial\theta_V}=\frac{\partial L}{\partial H_v}\frac{\partial H_v}{\partial Z_v}\frac{\partial Z_v}{\partial\theta_V}$$
> 真正「不需要考虑」的只是：**给 image token 定义一个 token-level 的 CE label**。
> 一句话：**image token 是 condition，不是 prediction target；但 condition 仍然在 computation graph 里。**

**7.** 什么情况下 SFT 要解冻 ViT？什么情况不用？

> **答：** 看数据在改变什么：
> 改变「**怎么思考 / 怎么回答**」（math CoT、instruction following、agent planning、输出格式）→ **freeze 就够**，perception 没有 domain shift；
> 改变「**怎么看**」（医学影像、卫星图、GUI 小图标、游戏画面、小目标检测、OCR、spatial grounding、特殊传感器）→ **unfreeze 可能明显更重要**，因为瓶颈就在 $I\to Z_v$。

**8.** RL 为什么倾向 freeze ViT？如果解冻会发生什么？

> **答：** 数学上完全可以解冻，policy gradient 一样能传回 ViT，reward 可以真的改变视觉表示。
> 但**实践中倾向 freeze**，因为 reward 对 perception 来说太**间接**：$R=-1$ 可能因为图没看清、reasoning 错、planning 错、action sampling 错、长程 credit assignment 错。全参更新会让 ViT 为这个 reward 承担部分责任，容易造成 **representation drift**，甚至破坏已有的 OCR / detection / general vision 能力。
> 所以：**SFT 可以较积极地 unfreeze vision，RL 更保守地 freeze vision。**

