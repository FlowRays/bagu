# 现代大模型的参数都长在哪（MoE / attention / embedding / vision）

> 一个非常有用的直觉：**大 MoE 的 97–99% 参数都在 expert FFN 里。**
> 这直接决定了为什么 EP 那么重要，也解释了 total vs activated 的巨大差距。

## 1. 结论先行

$$\boxed{P_{\text{MoE}}\gg P_{\text{attn}}\gg P_{\text{emb/head}}\gtrsim P_{\text{vision}}}$$

| Model | 总参数 | Vision | Attention | MoE / FFN | Emb + LM Head |
|---|---:|---:|---:|---:|---:|
| DeepSeek-V4-Flash | 284B | 0 | ~5.1B / **1.8%** | ~278.1B / **97.8%** | ~1.06B / 0.37% |
| DeepSeek-V4-Pro | 1.6T | 0 | ~19.5B / ~1.2% | ~1.551T / ~98% | ~1.85B / 0.12% |
| GLM-5.3 | 753B | 0 | ~13.2B / 1.76% | ~738.0B / **98.0%** | ~1.90B / 0.25% |
| Kimi-K3 | 2.8T | **0.401B / 0.014%** | ~36.2B / 1.29% | ~2.741T / 97.9% | ~2.35B / 0.084% |

（这些是按公开 config / 实现逐项算的估计；1.6T、2.8T 本身是官方 rounded number，不要把小数位当审计值。）

## 2. 为什么 MoE 能占到 98%

现在的 expert 基本是 GLU 类 FFN，所以一个 expert：

$$P_{\text{expert}}\approx 3\,d_{\text{model}}\,d_{\text{ffn}}$$

真正恐怖的是再乘 expert 数和层数：

$$\boxed{P_{\text{MoE}}\approx3\,d\,d_{\text{ffn}}\times N_{\text{expert}}\times N_{\text{MoE layer}}}$$

### DeepSeek-V4-Flash 手算一遍

$d=4096$、43 层、每层 256 routed + 1 shared expert、$d_{\text{ffn}}=2048$、top-6：

$$3\times4096\times2048=25.17\text{M（一个 expert）}$$
$$25.17\text{M}\times257\times43\approx\boxed{278.1\text{B}}$$

Embedding + LM head（$V=129280$，`tie_word_embeddings=false` 所以是两份）：

$$2Vd=2\times129280\times4096\approx1.06\text{B}\quad(0.37\%)$$

$$\boxed{284\text{B}\approx278\text{B MoE}+5\text{B Attention}+1\text{B Emb/Head}}$$

### GLM-5.3 是最整齐的例子

$d=6144$、78 层（3 dense FFN + 75 MoE）、256 routed + 1 shared、top-8、$d_{\text{MoE}}=2048$，另有 1 个 MTP layer：

$$738.0+13.2+1.90\approx\boxed{753.1\text{B}}$$

和 HF 标的 753B 几乎正好对上。attention 是 MLA + DSA sparse indexer。

### Kimi-K3 的 latent MoE

$d_{\text{model}}=7168$，但 routed expert 不直接吃 7168：

$$7168\rightarrow\boxed{3584\text{ latent}}\rightarrow\text{expert}\rightarrow7168$$

所以一个 routed expert $3\times3584\times3072\approx33.03$M，$896\times92$ 份 $\approx2.723$T，加 shared experts / latent projection / router / dense FFN 后约 **2.741T**。

## 3. 为什么 activated 参数小这么多

Kimi-K3：2.8T total，但每 token 只选 $16/896$ 个 routed expert：

$$2.723\text{T}\times\frac{16}{896}\approx48.6\text{B}$$

再加 shared experts ~12.2B、latent projections ~4.7B、attention ~36.2B、dense FFN ~0.7B：

$$\approx103\text{B}\quad\to\quad\text{官方 }\boxed{104\text{B activated}}$$

同理 V4-Pro 1.6T→49B、V4-Flash 284B→13B。

## 4. 卡点：这对训练显存意味着什么

Dense 模型可以认为 $P_{\text{total}}\approx P_{\text{active}}$，但大 MoE：

$$\boxed{P_{\text{total}}\gg P_{\text{active}}}$$

于是出现一个关键分离：

$$\boxed{\text{权重 / optimizer 显存由 total params 决定}}$$
$$\boxed{\text{每 token 的 FFN FLOPs 由 activated params 决定}}$$

**2.8T MoE 的训练不是"104B 模型的显存"** —— 那 2.8T 参数和对应的 optimizer state 你还是得存下。这正是 [EP](02-parallelism-map.md#5-expert-parallelmoe-专用) 对这类模型如此重要的原因：约 98% 参数是 experts，EP 就是专门切这 98%。而 TP 主要在处理那 1–2% 的 dense attention / shared path 及其计算。

## 5. 顺带一个 VLM 直觉

Kimi-K3 的 MoonViT-V2 是 401M，相对 2.8T 只有

$$\frac{0.401}{2800}\approx0.014\%$$

$$\boxed{\text{"多模态模型"不意味着 vision 参数占很多}}$$

超大 MoE VLM 的视觉塔可能只占万分之几，绝大多数 capacity 还是语言侧的 MoE experts。详见 [VLM/02 vision encoder](../07-vlm/02-vision-encoder.md)。

## 自测（口述版）

1. 写出一个 GLU expert 的参数量公式和整个 MoE 的参数量公式。
2. 用 DeepSeek-V4-Flash 的 config 手算 MoE 参数量。
3. Embedding + LM head 为什么要乘 2？什么时候不用乘 2？
4. Kimi-K3 的 latent MoE 和普通 MoE 的区别？
5. 从 2.8T total 推出 104B activated。
6. 大 MoE 的显存由 total 还是 activated 决定？FLOPs 呢？这对并行选择意味着什么？

> 带答案的题库在 [显存 / 训练工程 / 分布式 自测](../03-training-fundamentals/self-test.md)。
