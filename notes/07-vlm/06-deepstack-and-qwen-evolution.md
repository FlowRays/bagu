# DeepStack，以及 Qwen2.5-VL → Qwen3-VL → Qwen3.5 的 vision 演进

> 前置：[05 Qwen2.5-VL](05-qwen25-vl.md)。
> 这篇回答两件事：**DeepStack 到底在哪一层取什么、怎么变维、加到哪里**；以及 **Qwen3.5 为什么反而把它删了**。

## 1. 三代放一起看

用 Qwen2.5-VL-7B / Qwen3-VL-8B / Qwen3.5-9B 做代表：

| | Qwen2.5-VL-7B | Qwen3-VL-8B | Qwen3.5-9B |
|---|---:|---:|---:|
| patch size | 14×14 | **16×16** | **16×16** |
| temporal patch | 2 | 2 | 2 |
| ViT hidden | 1280 | **1152** | **1152** |
| ViT layers | 32 | **27** | **27** |
| heads | 16 | 16 | 16 |
| MLP intermediate | 3420 | **4304** | **4304** |
| ViT attention | window 为主 + 4 层 global | **全局** | **全局** |
| learned absolute PE | ❌ | **✅ 48×48 + 插值** | **✅** |
| ViT 2D RoPE | ✅ | ✅ | ✅ |
| spatial merge | 2×2 | 2×2 | 2×2 |
| **DeepStack** | ❌ | **✅ [8,16,24]** | **❌ []** |
| 输出到 LLM hidden | 3584 | 4096 | 4096 |

三个值得注意的点：

**① vision encoder 并没有被 scale 大。** Qwen3-VL 的 hidden 和 depth 反而都比 Qwen2.5-VL **小**（$1280\to1152$、$32\to27$）。这条和 [为什么大家不 scale vision encoder](02-vision-encoder.md#6-卡点为什么大家不-scale-vision-encoder) 是一致的。

**② window attention 没了。** Qwen3-VL 代码里已经没有 Qwen2.5 那套 window index，27 层全部对完整 patch sequence 做 attention。

**③ Qwen3.5 的 vision 塔几乎就是 Qwen3-VL 那一套** —— HF 文档直接说 "vision tower reuses the Qwen3-VL encoder"。

### Qwen3-VL 多了一套 learned absolute PE

Qwen2.5-VL 的 ViT 只靠 2D RoPE 表达位置。Qwen3-VL 额外加了：

```python
self.pos_embed = nn.Embedding(2304, hidden_size)   # 2304 = 48 × 48
```

学一张标准的 $48\times48$ 位置 embedding 网格；实际图片 patch grid 不是 $48\times48$ 时做 **bilinear interpolation** 插值到 $H_p\times W_p$，然后 $x_i\leftarrow x_i+p_i$ 再进 ViT。

$$\boxed{\text{Qwen3-VL ViT}=\text{Learned Absolute PE}+\text{2D RoPE}}$$

分工可以这么理解：

- **Absolute PE**：直接告诉 token"我大概在哪"
- **2D RoPE**：在 attention 里告诉 $Q,K$"你俩的相对空间关系是什么"

## 2. DeepStack 要解决什么

以前是：

```text
Image → ViT × 32 → 最后一层 feature → Patch Merger → LLM
```

$$\boxed{\text{LLM 只吃 ViT 最后一层}}$$

但 ViT 不同深度的信息不一样：

```text
浅层   →  edge / texture / small text / OCR 字符形态 / 局部细节
中层   →  parts / object / region / 空间结构
深层   →  semantic concepts / 高层语义
```

只拿 $X_{27}$ 的问题是：**语义很好，但低层视觉细节已经过了 20 多层 attention/MLP，被重新编码甚至弱化了。** 而 VLM 恰恰很关心 OCR、GUI、chart、document、fine-grained grounding、小目标 —— 这些对局部空间细节特别敏感。

$$\boxed{\text{不要要求 ViT final feature 一个人承担所有视觉信息}}$$

## 3. 具体怎么做

Qwen3-VL-8B 的配置：

```json
deepstack_visual_indexes: [8, 16, 24]
```

除了最终的 $X_{27}$，还额外抽 $X_8,X_{16},X_{24}$，**每一路配自己的 Patch Merger**：

$$X_8\to V_1,\qquad X_{16}\to V_2,\qquad X_{24}\to V_3,\qquad X_{27}\to V_{final}$$

为什么每路都要一个自己的 merger？因为 ViT hidden 是 1152、LLM hidden 是 4096，**token 数和维度两边都对不上**，必须各自过一遍 $2\times2$ merge + MLP：

$$4\times1152=4608\ \longrightarrow\ 4608\ \longrightarrow\ 4096$$

## 4. ⭐⭐ 最关键的一点：不是 concat，是逐元素相加

$$[V,D_8,D_{16},D_{24}]\quad\text{✗ 绝对不是这样}$$

那样 sequence length 会直接变 4 倍。实际做的是 **element-wise addition**，源码非常直白：

```python
hidden_states[visual_pos_masks, :] + visual_embeds
```

$$h_{visual}\ \leftarrow\ h_{visual}+D_k$$

$$\boxed{\text{DeepStack 不增加 sequence length}}$$

visual token 数仍然只有 $N/4$，只是**反复给 LLM 的视觉位置补充不同深度的 vision feature**。

## 5. 加在哪里：反直觉的一点

你可能会问：LLM 的输入已经是 ViT 最终层了，为什么 layer 0 之后又加一个**更浅的** $X_8$？

**不是**这样：

```text
✗  ViT 8 → LLM 0,  ViT 16 → LLM 1,  ViT 24 → LLM 2,  ViT final → LLM 3
```

**而是**：

```text
✓  ViT final → LLM input（正常的 visual token）

   然后额外：
   ViT 8  → merger → LLM layer 0 之后 residual 加上去
   ViT 16 → merger → LLM layer 1 之后
   ViT 24 → merger → LLM layer 2 之后
```

```text
ViT layer 8  ─→ merger ─────────────┐
                                    ↓ +
ViT layer 16 ─→ merger ────────┐  LLM layer 1
                               ↓ +
ViT layer 24 ─→ merger ────┐ LLM layer 2
                           ↓ +
ViT final ───→ merger → LLM input → layer0 → layer1 → layer2 → ...
```

所以本质是：

$$\boxed{\text{final semantic vision feature}+\text{multi-level residual visual features}}$$

## 6. 从 residual 的角度理解最简单

普通 Transformer residual：

$$h_{l+1}=F_l(h_l)+h_l$$

DeepStack 相当于多接一条来自 ViT 中间层的旁路：

$$h_{l+1}^{vision}=F_l(h_l)^{vision}+\underbrace{D_l}_{\text{来自 ViT 中间层}}$$

```text
Vision Encoder
      │
      └──────────────────────────┐
                                 ↓
LLM hidden ─→ LLM Block ───────→ +
```

$$\boxed{\text{DeepStack}=\text{ViT}\rightarrow\text{LLM 的 skip connection}}$$

### 名字为什么叫 DeepStack

传统 VLM 只有**一个**连接点（vision final → LLM input）。DeepStack 变成沿网络深度的多个连接点：

$$\boxed{\text{不是在 sequence 维度上 stack，是在 network depth 上 stack}}$$

## 7. 一个 visual token 对应多大区域（三代对比）

| | patch | merge | 一个 LLM visual token |
|---|---|---|---|
| Qwen2.5-VL | 14×14 | 2×2 | $\boxed{28\times28}$ 像素 |
| Qwen3-VL / Qwen3.5 | 16×16 | 2×2 | $\boxed{32\times32}$ 像素 |

所以同分辨率下 Qwen3-VL 的 visual token 更少，粗略比例：

$$\Big(\frac{28}{32}\Big)^2\approx0.766$$

**大约少 23%**（实际还受 dynamic resolution preprocessing 影响）。

## 8. 那 Qwen3.5 为什么把 DeepStack 删了

```json
Qwen3-VL-8B :  deepstack_visual_indexes = [8, 16, 24]
Qwen3.5-9B  :  deepstack_visual_indexes = []
```

$$\boxed{\text{Qwen3.5 Vision}\approx\text{Qwen3-VL Vision}-\text{DeepStack}}$$

### 主要原因：训练范式变了

**Qwen3-VL** 是"两个各自练好的模块拼起来"：

```text
SigLIP2 已经单独训练好  +  Qwen3 已经单独训练好
              ↓ 拼起来
        S0：先只训 projector 做对齐
```

$$\text{Vision representation}\ \ne\ \text{LLM-native representation}$$

正因为二者不是一起长大的，才需要显式的结构手段告诉模型"别只依赖 ViT 最后一层，我把浅中深层都送给你" —— DeepStack 是一种 **architecture-level inductive bias**，技术报告也把它描述为**加强 vision-language alignment**。

**Qwen3.5** 走的是 **early text-vision fusion / native multimodal pretraining**：从 foundation pretraining 一开始，text / image / video / 交错数据就一起训整个模型，$ViT\leftrightarrow Merger\leftrightarrow LLM$ 长期联合优化。

于是训练本身就能把 ViT final representation 调成"最适合后面这个 backbone 使用的表示"，没必要再规定 $X_8,X_{16},X_{24}$ 必须分别绕路送进去。

⚠️ **这是基于架构和官方 early-fusion 描述的推断，不是 Qwen 公开的 DeepStack 删除 ablation 结论。**

### 次要原因：DeepStack 不免费

虽然不增加 token 数，但有三个成本：

| 成本 | 说明 |
|---|---|
| **参数** | 3 个额外 merger $\approx120M$。对 8B 整体不多，但对 vision connector 来说已经很大（主 merger 才 $\approx40M$） |
| **Vision prefill 计算** | 四套大 MLP projector 都要跑，一个 merger 里就有 $4608\times4608$ 和 $4608\times4096$ 两次大 GEMM |
| **工程复杂度** | ViT forward 要保留 $X_8,X_{16},X_{24}$ 的中间 activation；LLM 还得知道哪些位置是 visual token、第几个 decoder layer 该加哪一级 feature |

删掉之后链路干净很多，特别适合 Qwen3.5 后面复杂得多的 **Gated DeltaNet + Full Attention** 混合 backbone。

## 9. ⚠️ "Native multimodal" 不等于没有 ViT

这个特别容易误会。Qwen3.5 **仍然有 vision encoder**：

```text
Image → Conv3D patch embed → Learned PE + 2D RoPE → 27 层 ViT
      → 2×2 Patch Merger → 4096-d visual embeddings → Qwen3.5 backbone
```

"Native multimodal" 讲的是**训练方式，不是去掉 vision encoder**：

$$\text{以前}:\ \text{先有强 LLM}\rightarrow\text{再接 ViT}\rightarrow\text{大量 VL 数据对齐}$$
$$\text{Qwen3.5}:\ \text{从 foundation pretraining 开始，text/image/video 交错数据一起训}$$

也**不等于把图片也 autoregressive 生成** —— 它照样是"图片走 encoder 进来，只在 text token 上算 loss"。所以 Qwen3.5 训练 ViT 仍然靠文本 loss 反传：

$$L\rightarrow\text{Qwen3.5}\rightarrow\text{Merger}\rightarrow\text{ViT},\qquad\boxed{\nabla_{\theta_{ViT}}L\ne0}$$

这一点和 Qwen3-VL 的 S1 之后其实一样。**真正不同的是：Qwen3.5 把这种 joint optimization 提前到了 foundation pretraining 的起点**，而不是在两个独立 pretrained model 拼起来之后才做。

这也是为什么它不再叫 Qwen3.5-VL 而直接叫 Qwen3.5 —— 视觉是 foundation model 的 native capability。

### 真正的大架构变化其实在 LLM 侧

vision 还是 Qwen3-VL 那套，但语言 backbone 换成了 3:1 的混合：

$$8\times\big(3\times\text{Gated DeltaNet}+1\times\text{Full Attention}\big)=32\ \text{层}$$

这才是它能在大量 visual token、长视频、长 context 下控制成本的原因。

## 10. 三张脑图

```text
Qwen2.5-VL                Qwen3-VL                      Qwen3.5
──────────                ────────                      ───────
Conv3D (2,14,14)          Conv3D (2,16,16)              Conv3D (2,16,16)
d=1280                    d=1152                        d=1152
     ↓                    Learned Abs PE + 2D RoPE      Learned Abs PE + 2D RoPE
2D RoPE                        ↓                             ↓
     ↓                    27 层 global ViT               27 层 global ViT
32 层 ViT                      ├─ layer 8  → merger ─┐        ↓
window 为主                    ├─ layer 16 → merger ─┼ DeepStack   2×2 Merger
+ 4 层 global                  ├─ layer 24 → merger ─┘        ↓
     ↓                         └─ final → merger              LLM input
2×2 Merger                          ↓                         ↓
     ↓                         LLM input                 Native Multimodal
Qwen2.5 LLM + MRoPE            ↓  ↑ DeepStack 注入前几层   Qwen3.5
                          Qwen3 LLM                      3 DeltaNet : 1 Full Attn
```

$$\boxed{\begin{aligned}\text{Qwen2.5-VL}:&\ \text{dynamic ViT}+\text{window attention}+\text{merger}\\\text{Qwen3-VL}:&\ \text{global ViT}+\text{abs PE}+\text{DeepStack}\\\text{Qwen3.5}:&\ \text{Qwen3-VL 式 ViT}+\text{native multimodal early-fusion backbone}\end{aligned}}$$

## 面试版

> DeepStack 是 Qwen3-VL 的核心结构升级：除了 ViT 最后一层，还在第 8/16/24 层各抽一次 feature，每路配一个自己的 Patch Merger（因为 token 数和维度两边都对不上），然后在 LLM 的前几个 decoder layer 之后**逐元素加到 visual token 的 hidden state 上**。注意它不是 concat，**不增加 sequence length**，本质是 ViT 到 LLM 的 skip connection —— 在 network depth 上 stack 而不是在 sequence 上。动机是浅层保留 OCR/GUI/细粒度空间信息，只用高度 semantic 的最后一层会丢掉这些。Qwen3.5 把它删了，我的理解是训练范式变了：Qwen3-VL 是 SigLIP2 和 Qwen3 两个各自练好的模块拼起来，需要结构上的 inductive bias 来强制保留 multi-level 特征；Qwen3.5 走 early-fusion native multimodal pretraining，ViT 和 backbone 从一开始就联合优化，final representation 本身就是为这个 backbone 长出来的。不过这是推断，官方没有给删除 DeepStack 的 ablation。

## 自测

**1.** ⭐ 三代的 patch size / ViT depth / hidden / attention 分别是什么？

> **答：** Qwen2.5-VL：patch 14、32 层、1280、window 为主 + 4 层 global。Qwen3-VL 和 Qwen3.5：patch 16、27 层、1152、**全局 attention**。
> 注意 vision encoder **并没有被 scale 大**，hidden 和 depth 反而都变小了。

**2.** Qwen3-VL 的 ViT 位置编码和 Qwen2.5-VL 差在哪？

> **答：** Qwen2.5-VL 只有 **2D RoPE**。Qwen3-VL 额外加了一套 **learned absolute PE**：`nn.Embedding(2304, hidden)`，$2304=48\times48$，实际 grid 不是 $48\times48$ 时做 bilinear 插值，然后 $x_i\leftarrow x_i+p_i$。
> 分工：absolute PE 说"我大概在哪"，2D RoPE 在 attention 里说"你俩的相对空间关系"。

**3.** ⭐⭐ DeepStack 取哪几层？每层怎么变维？

> **答：** `deepstack_visual_indexes=[8,16,24]`，加上最终的 $X_{27}$。**每一路都有自己的 Patch Merger**，因为 ViT hidden 1152 和 LLM hidden 4096 的**token 数和维度两边都对不上**：$4\times1152=4608\to4608\to4096$。

**4.** ⭐⭐ DeepStack 是把四路 feature concat 进 sequence 吗？

> **答：** **绝对不是。** 那样 token 数会变 4 倍。它是 **element-wise addition**：`hidden_states[visual_pos_masks,:] + visual_embeds`，即 $h_{visual}\leftarrow h_{visual}+D_k$。
> $\boxed{\text{DeepStack 不增加 sequence length}}$，visual token 仍然只有 $N/4$。

**5.** ⭐⭐ LLM 输入已经是 ViT 最后一层了，为什么 layer 0 之后又加更浅的 $X_8$？

> **答：** 因为不是"ViT 8→LLM 0、ViT 16→LLM 1、ViT final→LLM 3"那种一一对应。实际是 **ViT final 作为正常的 LLM input**，然后 $X_8/X_{16}/X_{24}$ 经各自 merger 后，在 LLM layer 0/1/2 **之后**作为额外 residual 加到 visual 位置上。
> 本质：**final semantic feature + multi-level residual visual features**，等于 ViT → LLM 的 skip connection。

**6.** 名字为什么叫 DeepStack？

> **答：** 传统 VLM 只有一个连接点（vision final → LLM input），DeepStack 沿**网络深度**开了多个连接点。$\boxed{\text{不是在 sequence 维度 stack，是在 network depth 上 stack}}$。

**7.** ⭐ DeepStack 的动机是什么？为什么 VLM 特别需要？

> **答：** ViT 浅层保留 edge/texture/小字/OCR 字符形态，中层是 parts/object/region，深层才是 semantic。只用最后一层，**语义很好但低层细节已经被 20 多层重新编码甚至弱化了**。而 VLM 恰恰很吃 OCR / GUI / chart / document / fine-grained grounding / 小目标这类对局部空间细节敏感的任务。所以"不要要求 ViT final feature 一个人承担所有视觉信息"。

**8.** ⭐⭐ Qwen3.5 为什么删掉 DeepStack？

> **答：** 主因是**训练范式变了**。Qwen3-VL 是 SigLIP2 + Qwen3 两个各自练好的模块拼起来，vision representation 不是 LLM-native 的，所以需要 S0 projector alignment，也需要 DeepStack 这种 **architecture-level inductive bias** 强制保留 multi-level 特征。Qwen3.5 走 early-fusion native multimodal pretraining，$ViT\leftrightarrow Merger\leftrightarrow LLM$ 从 foundation pretraining 起就联合优化，final representation 本身就是为这个 backbone 长出来的，不必再绕路。
> 次因是成本：3 个额外 merger $\approx120M$ 参数、四套大 MLP 的 prefill 计算、以及要保留中间 activation + LLM 得知道哪层加哪一级的工程复杂度。
> ⚠️ **这是推断，官方没有公开删除 DeepStack 的 ablation。**

**9.** ⭐⭐ "Native multimodal" 是不是意味着没有 ViT / 图片也 autoregressive 生成？

> **答：** **都不是。** Qwen3.5 仍然有完整的 vision encoder（Conv3D → 27 层 ViT → $2\times2$ merger → 4096-d），HF 文档说 "vision tower reuses the Qwen3-VL encoder"。也仍然只在 **text token** 上算 loss，ViT 靠文本 CE 反传更新（$\nabla_{\theta_{ViT}}L\ne0$）。
> "Native" 讲的是**训练方式**：不是先训好 text-only backbone 再外挂视觉，而是从 foundation pretraining 起就用 text/image/video 交错数据联合优化。所以它不叫 Qwen3.5-VL，直接叫 Qwen3.5。

**10.** 同分辨率下 Qwen3-VL 的 visual token 比 Qwen2.5-VL 多还是少？

> **答：** **少约 23%**。Qwen2.5-VL 一个 LLM visual token 对应 $28\times28$ 像素（patch 14 × merge 2），Qwen3-VL 是 $32\times32$（patch 16 × merge 2），比例 $(28/32)^2\approx0.766$。

**11.** Qwen3.5 真正的大架构变化在哪？

> **答：** **在 LLM 侧，不在 vision 侧。** vision 还是 Qwen3-VL 那套，但语言 backbone 换成 3:1 混合：$8\times(3\times\text{Gated DeltaNet}+1\times\text{Full Attention})=32$ 层。这才是它能在大量 visual token、长视频、长 context 下控制成本的原因。
