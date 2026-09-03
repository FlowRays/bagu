# VLA 模型总图

| 篇 | 内容 |
|---|---|
| [01 架构与训练](01-vla-models.md) | RT-1→RT-2→OpenVLA→π0 演进、典型架构的三个设计位、co-training 防遗忘、两阶段训练、数据金字塔、推理效率 |

## 一句话

$$\boxed{\text{VLA} = \text{VLM backbone} + \text{action expert} + \text{action chunk}}$$

三个核心约束导出全部设计：**动作多模态**（生成式 head）、**实时性**（chunk + 小 expert + flow）、**数据极贵**（跨本体预训练 + 无标注视频）。

## 相关

- [Action](../05-action/00-map.md) — head 怎么设计
- [视觉生成](../02-visual-generation/00-map.md) — flow matching 的推导
- [VLM 架构](../../notes/07-vlm/00-map.md) — backbone 那一半
