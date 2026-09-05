# VLA 模型总图

| 篇 | 内容 |
|---|---|
| [01 架构与训练](01-vla-models.md) | RT-1→RT-2→OpenVLA→π0 演进、典型架构的三个设计位、co-training 防遗忘、两阶段训练、数据金字塔、推理效率 |
| [02 第一代 AR VLA](02-gen1-ar-vla.md) | **RT-1 / RT-2 / RT-X / OpenVLA 逐 tensor 拆开**：FiLM、TokenLearner、动作伪装成语言、co-fine-tuning、粗对齐的 cross-embodiment、DINOv2+SigLIP 通道拼接 |

## 一句话

$$\boxed{\text{VLA} = \text{VLM backbone} + \text{action expert} + \text{action chunk}}$$

三个核心约束导出全部设计：**动作多模态**（生成式 head）、**实时性**（chunk + 小 expert + flow）、**数据极贵**（跨本体预训练 + 无标注视频）。

## 三个必须记住的"不等于"

$$\boxed{\text{discrete action}\neq\text{LLM-style autoregressive action}}\quad\text{(RT-1 是并行分类)}$$

$$\boxed{\text{coarse alignment}\neq\text{true universal action representation}}\quad\text{(RT-X)}$$

$$\boxed{\text{shared token}\neq\text{shared physical magnitude}}\quad\text{(per-dataset 归一化)}$$

展开都在 [02](02-gen1-ar-vla.md)。

## 相关

- [Action](../05-action/00-map.md) — head 怎么设计
- [视觉生成](../02-visual-generation/00-map.md) — flow matching 的推导
- [VLM 架构](../../notes/07-vlm/00-map.md) — backbone 那一半
