# 世界模型总图

| 篇 | 内容 |
|---|---|
| [01 三条路线](01-world-model-map.md) | pixel / latent dynamics / JEPA 的分水岭、Dreamer 与 RSSM、Genie 的 latent action model、视频生成能否当世界模型、三种决策方式 |
| （JEPA 单独成册） | [04 JEPA](../04-jepa/01-jepa.md) |

## 一句话

$$\boxed{\text{动力学任务无关可以用便宜数据学；策略任务相关只能用贵数据学}}$$

这就是世界模型这条路线存在的全部理由。三条子路线的分歧只在一件事上：**要不要重建像素**。

## 相关

- [JEPA](../04-jepa/01-jepa.md) — 纯表示空间的那一条
- [视觉生成](../02-visual-generation/00-map.md) — pixel 路线的技术基础
- [RL 目录](../../notes/06-post-training/rl/00-map.md) — imagination training 用的还是 actor-critic
