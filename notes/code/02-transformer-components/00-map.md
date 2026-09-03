# Transformer 组件（代码）

> 对应 [bagu.md](../../../bagu.md) 的 code (2) transformer components。
> **实现都在 [手撕代码合集](../07-handwrite/00-map.md) 里**，这一页只做索引，避免重复。

| 组件 | 可默写实现（NumPy + PyTorch 两版） | 理论 |
|---|---|---|
| Self-Attention / MHA / Cross / GQA / MLA | [手撕：大模型](../07-handwrite/03-llm.md) | [Attention 家族](../../01-llm-arch/04-attention.md) |
| KV Cache 增量解码 | [手撕：大模型](../07-handwrite/03-llm.md#kv-cache-增量注意力) | [推理基础](../../02-inference-serving/01-inference-basics.md#2-kv-cache) |
| LayerNorm / BatchNorm | [手撕：深度学习](../07-handwrite/02-dl.md) | [Normalization](../../01-llm-arch/03-normalization.md) |
| RMSNorm | [手撕：大模型](../07-handwrite/03-llm.md#rms-normalization) | 同上 |
| FFN / SwiGLU | [手撕](../07-handwrite/03-llm.md#swiglullama-系激活前向传播) | [FFN 与 MoE](../../01-llm-arch/05-ffn-moe.md) |
| Transformer Encoder Block | [手撕：大模型](../07-handwrite/03-llm.md#transformer-encoder-block) | [LLM 架构总图](../../01-llm-arch/00-map.md) |
| 卷积 / 池化 / im2col | [手撕：深度学习](../07-handwrite/02-dl.md) | — |

面试写这些时的通用要点见 [手撕总表的「几个反复出现的套路」](../07-handwrite/00-map.md#几个反复出现的套路)。
