# VLA 模型总图

| 篇 | 内容 |
|---|---|
| [01 架构与训练](01-vla-models.md) | RT-1→RT-2→OpenVLA→π0 演进、典型架构的三个设计位、co-training 防遗忘、两阶段训练、数据金字塔、推理效率 |
| [02 第一代 AR VLA](02-gen1-ar-vla.md) | **RT-1 / RT-2 / RT-X / OpenVLA 逐 tensor 拆开**：FiLM、TokenLearner、动作伪装成语言、co-fine-tuning、粗对齐的 cross-embodiment、DINOv2+SigLIP 通道拼接 |
| [03 第二代：π0 → π0.5 → G0](03-pi0-pi05-g0.md) | **π0 是一个 Transformer 两套权重**、块状 attention mask、π0.5 的 FAST 预训练 + 子任务 + flow 执行、G0 双系统与三阶段训练、跨本体负迁移 |
| [04 G0.5 架构与源码](04-g05-architecture.md) | Qwen3.5 多模态输入、ActionCodec（32 步→码的形状流、RVQ 轮数、一致性正则）、统一推理与动作 token、视觉记忆、预训练 recipe、AR / FM 配置与源码阅读顺序 |
| [05 前沿路线](05-frontier-routes.md) | DM0.5 vs 小米（Robotics-1 / U0）、RoboDojo 前 10 各自路线、谁用了 Ego / UMI、三种第一视角、四个派系 |

## 一句话

VLA 将视觉、语言和机器人状态映射为动作。常见实现包括 **VLM + 连续动作 expert**，以及 **VLM 自回归生成动作 token + codec 解码**（见 [G0.5](04-g05-architecture.md)）。

三个核心约束帮助理解设计：**动作多模态**（生成式建模）、**实时性**（chunk、动作压缩或小 expert）、**数据极贵**（跨本体预训练与视觉语言数据共训）。

## 三个必须记住的"不等于"

$$\boxed{\text{discrete action}\neq\text{LLM-style autoregressive action}}\quad\text{(RT-1 是并行分类)}$$

$$\boxed{\text{coarse alignment}\neq\text{true universal action representation}}\quad\text{(RT-X)}$$

$$\boxed{\text{shared token}\neq\text{shared physical magnitude}}\quad\text{(per-dataset 归一化)}$$

展开都在 [02](02-gen1-ar-vla.md)。

## 相关

- [Action](../02-action/00-map.md) — head 怎么设计
- [视觉生成](../09-visual-generation/00-map.md) — flow matching 的推导
- [VLM 架构](../../notes/07-vlm/00-map.md) — backbone 那一半
