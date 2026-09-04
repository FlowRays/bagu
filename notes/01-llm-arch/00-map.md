# LLM 架构总图

> 对应 [bagu.md](../../bagu.md) 的 (1) llm arch。这是面试问得最多的一块，尤其 attention 家族。

## 一条前向路径

```text
文本 → tokenizer → token id → embedding  (B,S,d)
   → [ pre-norm → attention → residual → pre-norm → FFN/MoE → residual ] × N
   → final norm → lm_head → logits (B,S,V) → softmax → sampling → next token
```

每一段对应一篇：

| 篇 | 内容 | 高频度 |
|---|---|---|
| [01 Tokenizer 与词表](01-tokenizer.md) | BPE / BBPE / WordPiece / Unigram、encode 优先级、chat template、词表大小影响 | 中 |
| [02 位置编码](02-position-encoding.md) | 绝对 PE、ALiBi、**RoPE**、partial RoPE、NTK / YaRN 外推、M-RoPE | **高** |
| [02b RoPE 从零](02b-rope-from-zero.md) | **数学前置**：旋转矩阵、两 token 手算、多频率的钟表类比、2D RoPE、MRoPE | — |
| [03 Normalization](03-normalization.md) | LayerNorm / RMSNorm / BatchNorm、**pre vs post norm**、residual stream、QK-norm | **高** |
| [04 Attention 家族](04-attention.md) | MHA / MQA / **GQA** / **MLA**、KV cache 账、attention sink、FlashAttention、线性与稀疏注意力 | **最高** |
| [05 FFN 与 MoE](05-ffn-moe.md) | SwiGLU 与 $\frac83d$、router / top-k、shared expert、**负载均衡**、EP | **高** |
| [05b 激活函数](05b-activations.md) | **数学前置**：正态分布 / CDF / sigmoid 的图像、"输入×门"模板、GLU 家族 | — |
| [06 LM head 与 MTP](06-lm-head-and-mtp.md) | weight tying、logits 显存、MTP 的双重作用 | 中 |
| [自测题](self-test.md) | 44 题带答案 | — |

## 必须能当场答出来的十条

1. **为什么除 $\sqrt{d_h}$**：$q^\top k$ 的方差是 $d_h$，不缩放会让 softmax 饱和、梯度消失。除标准差不是除方差。
2. **causal mask 必须在 softmax 之前填 $-\infty$**：之后乘 0 的话分母里还算了被屏蔽的位置。
3. **RoPE 的核心性质**：$\langle R_mq,R_nk\rangle=q^\top R_{n-m}k$，用绝对位置的旋转得到相对位置的效果。
4. **NTK vs YaRN**：NTK 调大 base 让低频多拉伸；YaRN 按频段分别处理并加 attention scaling。
5. **LLM 不用 BatchNorm**：推理时 batch/seq 一直变，且语义上应该是样本内比较而非跨样本。
6. **pre-norm 更好训**：展开后存在一条不被 norm 阻断的恒等路径，梯度能直接回流。
7. **GQA 省的是 KV cache**，$W_K/W_V$ 比 $W_Q$ 窄；广播用 `repeat_interleave`。
8. **MLA 压的是 KV 的维度**（低秩 latent），和 RoPE 冲突所以要拆一个独立的 RoPE 分支。
9. **SwiGLU 有三个矩阵**，所以 $d_{ff}$ 要缩到 $\frac83d$ 才和普通 FFN 参数量持平。
10. **MoE 的显存按 total 参数算，FLOPs 按 activated 算**，它换的是容量不是显存。

## 一个贯穿全篇的数字感

| 量 | 典型值 | 出现在 |
|---|---|---|
| 32K 上下文的 KV cache | **约 17 GB**（32 层 / 32 头 / $d_h$=128 / BF16） | [04](04-attention.md#3-kv-cache-是这一切的动机) |
| logits 激活 | **约 9.8 GB**（$B$=4, $L$=8K, $V$=150K, BF16） | [06](06-lm-head-and-mtp.md#1-从-hidden-到-token) |
| 大 MoE 的参数分布 | MoE 97–99%，attention 1–2%，emb+head 0.1–0.4% | [参数构成](../04-distributed-infra/03-model-param-breakdown.md) |

## 相关

- [手撕：大模型](../code/07-handwrite/03-llm.md) — Self-Attention / MHA / GQA / MLA / KV Cache / RMSNorm / SwiGLU 的可默写实现，NumPy 和 PyTorch 两版
- [推理与部署](../02-inference-serving/00-map.md) — KV cache 在推理侧怎么管
- [显存账本](../03-training-fundamentals/01-memory-accounting.md) — 这些结构在训练时各占多少显存
