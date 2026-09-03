# 预训练总图

> 对应 [bagu.md](../../bagu.md) 的 (5) pre-training。面试问得不如 post-training 多，
> 但**数据和 scaling 的判断力**是区分「调过参」和「训过模型」的地方。

| 篇 | 内容 |
|---|---|
| [01 数据](01-data.md) | 来源配比、清洗流水线、去重（MinHash+LSH）、质量过滤、退火 / mid-training、长上下文分阶段 |
| [02 Scaling law](02-scaling.md) | 幂律形式、**Chinchilla 与它为什么被超过**、MoE 的 scaling、涌现争议、$C\approx6ND$ 与 MFU |
| [03 训练稳定性](03-stability.md) | loss spike、BF16 vs FP16、初始化、warmup 与 WSD、muP、该监控什么 |

## 三条最该记住的

1. **数据决定上限。** 去重是收益最确定的一步；退火期的数据质量是性价比最高的干预点。
2. **Chinchilla 说 $D\approx20N$，但实践远超它**，因为要优化的是「训练 + 全生命周期推理」的总算力，不只是训练算力。
3. **现代 LLM 的很多结构改动治的是稳定性不是效果**：pre-norm、RMSNorm、QK-norm、去 bias、gradient clipping、BF16。

## 一个能当场算的例子

7B 训 2T token：$C=6ND=8.4\times10^{22}$ FLOPs。1024×H100、MFU 45% → **约 2.1 天**。

## 相关

- [LLM 架构](../01-llm-arch/00-map.md) — 那些为稳定性做的结构设计
- [显存账本](../03-training-fundamentals/01-memory-accounting.md) 与 [并行全图](../04-distributed-infra/02-parallelism-map.md) — 怎么把这些算力真正用上
- [位置编码](../01-llm-arch/02-position-encoding.md) — 长上下文扩展阶段配套的 NTK / YaRN
