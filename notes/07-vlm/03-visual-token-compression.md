# Visual token 太多怎么办

## 1. 问题有多严重

$H=W=1024$、patch=14：

$$\frac{1024}{14}\approx73\ \Rightarrow\ 73\times73\approx5329\ \text{个 ViT patch}$$

全送给 LLM 就是 context 多 5300 token，而 attention 是

$$O\big((N_t+N_v)^2\big)$$

视频再乘帧数，直接爆炸。

$$\boxed{\text{ViT 内部可以保留较密的 token，但送进 LLM 前必须压缩}}$$

## 2. Spatial merge / pixel shuffle

最常见。Qwen 的 $2\times2$：四个相邻 patch token concat 后过 MLP 合成一个

$$[z_1;z_2;z_3;z_4]\in\mathbb R^{4608}\ \xrightarrow{\ 4608\to4608\to4096\ }\ z'$$

$$N\rightarrow N/4$$

### 卡点：为什么 concat + MLP 而不是 average pooling

平均池化把 4 个 patch 的空间信息**抹平**了（谁在左上、谁在右下都一样）；concat 保留了 $2\times2$ 内部的相对位置，MLP 可以学怎么融合。代价就是 projector 参数从"很小"涨到 40–50M（见 [02](02-vision-encoder.md#卡点projector-居然有-40m)）。大家宁愿付这个参数，也说明空间信息值这个钱。

## 3. 更激进的 spatial pooling：DeepSeek 的 3×3

DeepSeek-V4-Flash-Vision-Exp（2026-08-21 发布）：

$$\boxed{9\ \text{ViT tokens}\rightarrow1\ \text{LLM token}},\qquad N\rightarrow N/9$$

## 4. Learned resampler

Q-Former / Perceiver Resampler / latent queries：准备固定的 $M=64/128/256$ 个 learnable query

$$Q'=\text{CrossAttn}(Q,Z_v)$$

$3000\to256$。好处是**固定预算**，缺点是容易丢 dense spatial information。

## 5. Token selection / pruning

只保留最重要的 $K$ 个 visual token，依据 attention score / saliency / token similarity / text-query relevance。对视频尤其重要。

## 6. 案例：DeepSeek-V4-Flash-Vision-Exp

在 DeepSeek-V4-Flash 上加视觉模块并继续训练。

| 参数 | 值 |
|---|---:|
| ViT layers | 32 |
| hidden | 1024 |
| heads | 16 |
| FFN intermediate | 2816 |
| patch size | 14 |
| 位置编码 | **2D RoPE** |
| spatial downsample | **3×3** |
| max image tokens | **384** |

ViT 内部是 **full bidirectional attention over one image**（不是 causal），每个 patch 能看整张图。

### Aligner

$$h=W_2\,\text{GELU}\big(W_1[z_1;\dots;z_9]\big),\qquad 9\times1024=9216\to4096\to4096$$

参数：$37.75\text{M}+16.78\text{M}=54.53$M；加 ViT 的 411.84M 共约 **466M**。

### 两层控制，不是先生成几十万 token 再压

`vision_max_n_token = 384`，API 也明确一张图最多 384 token。**预处理阶段**就根据 patch=14、downsample=3、max=384 反推允许的 resize 分辨率：

```text
raw image → 按长宽比动态 resize → patch=14 → ViT patches
         → 32-layer ViT → 3×3 spatial merge → ≤384-token image block → LLM
```

$$\boxed{\text{限制输入 resolution}\ +\ 3\times3\text{ token merge}}$$

### 一个很聪明的细节：sliding window 里的 image span

DeepSeek-V4-Flash 的 LLM attention 有 **128-token sliding window**，但一张图可以有 384 token —— 按普通 sliding window，第 384 个 image token 根本看不到第 1 个，**图片会被切碎**。

所以它在 attention mask 里专门识别 `[IMAGE_START, IMAGE_END]` 区间，对同一张 image span 扩展可见范围，让图像内部 token 能跨过 128-token window 互相访问（实现里的 `get_image_visible()` 就是在算当前 token 距 image span 左右边界多远）。

## 7. 汇总

| 方法 | 压缩比 | 代表 | 特点 |
|---|---|---|---|
| Spatial merge | 4→1 | Qwen3-VL | 简单、保空间信息 |
| 激进 pooling | 9→1 | DeepSeek-V4-Flash-Vision | 再配死 token budget |
| Learned resampler | $N\to M$ 固定 | Q-Former / Perceiver | 预算固定，易丢 dense 信息 |
| Token pruning | $N\to K$ | 视频场景 | 依赖打分，动态 |

## 自测（口述版）

**1.** $1024\times1024$、patch=14 会产生多少 patch？为什么不能直接送 LLM？

> **答：** $1024/14\approx73$，$73\times73\approx\mathbf{5329}$ 个 patch。
> 直接送 LLM 会让 context 多 5300 token，而 attention 是 $O((N_t+N_v)^2)$，视频再乘帧数直接爆炸。所以 **ViT 内部可以保留较密的 token，但送进 LLM 前必须压缩**。

**2.** 四类压缩方法各是什么？各自的优缺点？

> **答：** ① **spatial merge / pixel shuffle**（Qwen $2\times2$，$N\to N/4$）：简单、保空间信息；
> ② **更激进的 spatial pooling**（DeepSeek $3\times3$，$N\to N/9$）：压缩比高，通常还配死 token budget；
> ③ **learned resampler**（Q-Former / Perceiver，固定 $M$ 个 learnable query 做 cross-attention）：预算固定，缺点是容易丢 dense spatial information；
> ④ **token selection / pruning**（按 attention score / saliency / similarity / text-query relevance 留 $K$ 个）：动态，视频场景尤其重要。

**3.** 为什么是 concat + MLP 而不是 average pooling？代价是什么？

> **答：** 平均池化把 4 个 patch 的**空间信息抹平**了（谁在左上、谁在右下都一样）；concat 保留了 $2\times2$ 内部的相对位置，MLP 可以学怎么融合。
> 代价是 projector 参数从「很小」涨到 40–50M。大家宁愿付这个参数，正说明空间信息值这个钱。

**4.** DeepSeek-V4-Flash-Vision 的 aligner 怎么算？参数量多少？

> **答：** $9\times1024=9216\to4096\to4096$，$h=W_2\,\text{GELU}(W_1[z_1;\dots;z_9])$，即 **9 个 ViT token → 1 个 LLM token**。
> 参数：$37.75+16.78=\mathbf{54.53M}$，加 ViT 的 411.84M 共约 **466M**。
> ViT 本身是 32 层、hidden 1024、16 heads、FFN 2816、patch 14、2D RoPE，内部是 **full bidirectional attention**（不是 causal）。

**5.** 它是先生成海量 ViT token 再压到 384 吗？两层控制分别是什么？

> **答：** **不是。** 有**两层控制**：① `vision_max_n_token = 384` 卡死每张图的 LLM token 预算；② **预处理阶段**就根据 patch=14、downsample=3、max=384 **反推允许的 resize 分辨率**。
> 流程：`raw image → 按长宽比动态 resize → patch=14 → 32-layer ViT → 3×3 merge → ≤384-token block → LLM`。

**6.** 384 个 image token 遇上 128-token sliding window 会怎样？它怎么解决？

> **答：** 按普通 sliding window，第 384 个 image token **根本看不到第 1 个**，一张图会被切碎。
> DeepSeek 在 attention mask 里专门识别 `[IMAGE_START, IMAGE_END]` 区间，对同一张 image span **扩展可见范围**，让图像内部 token 能跨过 128-token window 互相访问（实现里的 `get_image_visible()` 就是在算当前 token 距 image span 左右边界多远）。

