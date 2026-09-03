# 评测与 harness 总图

> 对应 [bagu.md](../../bagu.md) 的 (8) eval and harness。

| 篇 | 内容 |
|---|---|
| [01 评测](01-eval.md) | LLM / VLM / Agent benchmark 清单、**setting 的影响**、pass@k 与 maj@k、污染检测、LLM-as-judge 的偏置 |
| [02 Agent harness](02-harness.md) | ReAct、harness 要解决的问题、编码智能体的设计哲学、agent 评测的难点、agent RL |

## 三条最该记住的

1. **报分数必须报 setting**（shot 数 / CoT / 答案提取 / 打分方式 / 采样），否则不可比。
2. **pass@1 衡量稳定做对，pass@k 衡量有没有能力**。RL 的典型效果是把 pass@k 压进 pass@1，pass@k 本身可能不涨。
3. **agent 只报成功率是不完整的**，要连带报平均步数和 token 消耗。

## 相关

- [RM 的问题](../06-post-training/preference/01-rm-and-rlhf.md#3-rm-的问题) — LLM-as-judge 的偏置和 RM 是同一类问题
- [agentic SFT](../06-post-training/sft/02-data-and-cot.md#5-agentic-sft) — harness 的轨迹怎么变成训练数据
- [prefix caching](../02-inference-serving/02-serving-optimization.md#3-prefix-caching复用相同前缀的-kv) — agent 场景的关键推理优化
