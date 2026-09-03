# 推理与部署总图

> 对应 [bagu.md](../../bagu.md) 的 (2) inference and serving。
> **整块只需要记住一条主线**：prefill 是 compute-bound，decode 是 memory-bound，所有优化都在打这两个中的一个。

| 篇 | 内容 |
|---|---|
| [01 推理基础](01-inference-basics.md) | prefill / decode、算术强度、KV cache、TTFT / TPOT / 吞吐 / 延迟、采样 |
| [02 服务化优化](02-serving-optimization.md) | PagedAttention、continuous batching、prefix caching、chunked prefill、KV 量化与 offload、FlashInfer |
| [03 投机解码与量化](03-speculative-and-quant.md) | speculative sampling 的无损性、MTP 自投机、GPTQ / AWQ / FP8、激活异常值 |
| [自测题](self-test.md) | 30 题带答案 |

## 一张总表

| 优化 | 治的瓶颈 | 直接改善 |
|---|---|---|
| PagedAttention | KV 显存碎片 | 并发数 → 吞吐 |
| Continuous batching | GPU 空转等待 | 吞吐 |
| Prefix caching | 重复 prefill | **TTFT** |
| Chunked prefill | prefill 阻塞 decode | **TPOT 稳定性** |
| KV 量化 / offload | KV 显存 | 并发数 |
| GQA / MLA | KV cache 大小（结构层面） | 并发数 |
| 投机解码 | decode 的串行性 | **TPOT**（小 batch 才划算） |
| 权重量化 | 权重读取字节数 | **TPOT** |
| FlashAttention / FlashInfer | attention 的 IO | 两个阶段都有 |

## 面试怎么答「怎么优化推理」

先反问场景，再对号入座：

> **先定位瓶颈**。prefill 是 compute-bound，决定 TTFT；decode 是 memory-bound，决定 TPOT，因为每步都要把整个模型权重从 HBM 读一遍才产出一个 token。
>
> **如果要压 TTFT**：prefix caching 复用相同 system prompt / 历史的 KV，chunked prefill 避免长 prompt 阻塞，必要时上 TP 摊薄单卡算力。
>
> **如果要压 TPOT**：权重量化直接减少要读的字节数；投机解码把串行的多步合并成一次前向验证，而且是无损的；结构上换 GQA/MLA 减小 KV cache。
>
> **如果要提吞吐**：continuous batching 保持 batch 满，PagedAttention 提高显存利用率从而提高并发；这两个是 vLLM 的核心。
>
> **注意延迟和吞吐是矛盾的**，batch 增大吞吐升但单条 TPOT 降，所以对话产品和离线批处理的取舍完全不同。

## 相关

- [Attention 家族](../01-llm-arch/04-attention.md) — GQA / MLA 从结构上减小 KV cache
- [MTP](../01-llm-arch/06-lm-head-and-mtp.md#3-mtpmulti-token-prediction) — 自带的 draft model
- [显存账本](../03-training-fundamentals/01-memory-accounting.md) — 训练侧的显存构成，和推理侧对照着看
- [RL 显存](../06-post-training/memory-sft-opd-rl.md) — RL 里 rollout engine 就是一个推理服务
