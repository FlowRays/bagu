# VLA / WAM 八股总图

> 大纲见 [embodied.md](../embodied.md)。这一页是各册的入口和主线。
> 前置是 [LLM/VLM 八股](../bagu.md)，尤其 [VLM 架构](../notes/07-vlm/00-map.md) 和 [RL 目录](../notes/06-post-training/rl/00-map.md)。

## 各册

目录按「主线在上、支撑原理在下」排：先 VLA 这条线，再机器人基础、数据仿真评测，生成模型与 JEPA 放最后。四块内容与演变时间线见 [embodied.md](../embodied.md)。

| 册 | 属于 | 内容 | 状态 |
|---|---|---|---|
| [01 VLA 模型](01-vla/00-map.md) | VLA 主线 / 最新进展 | 演进一页纸；**02 第一代逐 tensor 走查**、**03 π0→π0.5→G0**、**04 G0.5 架构/ActionCodec/数据**、**05 前沿路线（DM0.5、小米、RoboDojo 前排）** | 第一版 |
| [02 Action](02-action/00-map.md) | VLA 主线 / 机器人基础 | **01b 从零补机器人前置**（DoF/FK/IK/旋转表示/正逆动力学）、动作空间与 6D 旋转、**四种 action head**、chunking 与实时性、**04 ACT 与 Diffusion Policy** | 第一版 |
| [03 总览](03-overview/00-map.md) | 机器人基础 | 和 LLM 的五个本质差异、三种范式各赌什么、概念对照表 | 第一版 |
| [04 数据与仿真](04-data-sim/00-map.md) | 数据集、仿真、评测 | 四类数据源、遥操作质量、sim2real；**02 UMI / Ego / Teleop 等数据类型与采集**、**03 MuJoCo / LIBERO / Isaac 环境栈** | 第一版 |
| [05 评测](05-eval/00-map.md) | 数据集、仿真、评测 | 仿真 benchmark、真机评测难点、五个陷阱 | 第一版 |
| [06 具身 RL](06-embodied-rl/00-map.md) | 训练 | 和 LLM RL 的五个差异、reward hacking、GRPO 在 VLA 上的三个问题 | 第一版 |
| [07 世界模型](07-world-model/00-map.md) | 支撑原理 | 三条路线、Dreamer/RSSM、Genie 的 latent action、视频生成能否当世界模型 | 第一版 |
| [08 手撕实现](08-handwrite/00-map.md) | 支撑原理 | Diffusion / DDIM / CFG / Flow matching / Action chunking 的最小实现，含**证明「不能用 MSE 回归动作」的可运行实验** | 第一版 |
| [09 视觉生成](09-visual-generation/00-map.md) | 支撑原理 | 生成模型谱系、**01b AE→VAE→VQ-VAE→Latent Diffusion 从零**、**Diffusion**、**Flow Matching** | 第一版 |
| [10 JEPA](10-jepa/01-jepa.md) | 支撑原理 | 表示空间预测、防坍缩、I-JEPA、**V-JEPA 2-AC 与规划** | 第一版 |

## 两个约束推出全部设计

$$\boxed{\text{约束一：数据极贵}\qquad\boxed{\text{约束二：必须实时闭环}}}$$

**数据极贵** 推出：
- 用无标注视频（JEPA 学表示、Genie 的 latent action 反推动作）
- 用仿真（+ domain randomization 跨 gap）
- 跨本体预训练（Open X-Embodiment）
- 世界模型（动力学任务无关，可以用便宜数据学）

**必须实时闭环** 推出：
- action chunking（一次出 H 步，解耦推理频率和控制频率）
- 小 action expert（大 backbone 跑一次，小 expert 跑 K 次）
- flow matching 而不是 diffusion（K 从 50 降到 4–10）
- 异步推理（并注意对齐延迟偏移）

## 一条必须记住的推理链

> **为什么 VLA 的 action head 长成现在这样？**
>
> 动作分布是**多模态**的（绕障碍可左可右）→ MSE 回归输出条件均值会直接撞上去 → 必须**生成式建模** → diffusion 能建模多模态但要 50 步 → 机器人 10–50 Hz 闭环跑不起 → 换 **flow matching**（直线路径，1–10 步）→ 还是贵 → 大 backbone 跑一次、**小 action expert** 跑那 K 步 → 一次只出一个动作太浪费 → **chunk** 一次出 H 步。

## 和 LLM 八股的对照

| 具身 | LLM 里的对应 |
|---|---|
| BC / 模仿学习 | [SFT](../notes/06-post-training/sft/01-sft-basics.md) |
| compounding error | [exposure bias](../notes/06-post-training/distill/02-sft-and-kd.md#3-exposure-biassft-真正的病) |
| action chunking | 某种意义上的"多 token 预测" |
| 世界模型 | next-token prediction 是一维世界模型 |
| co-training 防遗忘 | [灾难性遗忘的 replay](../notes/06-post-training/sft/01-sft-basics.md#6-灾难性遗忘) |
| 仿真拒绝采样 | [RFT](../notes/06-post-training/sft/02-data-and-cot.md#2-数据合成) |
| VLA backbone | [VLM](../notes/07-vlm/00-map.md) |
| 具身 RL | [PPO / GRPO / DAPO](../notes/06-post-training/rl/00-map.md) |

## 还缺什么（下一版补）

| 块 | 待补 |
|---|---|
| VLA 和知名架构 | Octo、RDT、OpenVLA-OFT；GR00T N1 系列；Helix / Gemini Robotics 等双系统；实时 chunking 与推理加速 |
| 机器人基础 | 坐标系与相机标定、PD / 阻抗控制、灵巧手与触觉力控、双臂协同、移动操作、人形全身控制 |
| 数据集、仿真、评测 | 主流数据集逐个拆（OXE、DROID、AgiBot World、Galaxea Open-World）；benchmark 逐个拆（LIBERO、SimplerEnv、CALVIN、RoboTwin、BEHAVIOR-1K、RoboDojo）的任务、指标和坑 |
| 最新进展 | 世界模型当数据引擎（Cosmos 等）、WAM 架构细节、VLA 的 RL 后训练新做法 |
| 支撑原理 | mRoPE / 多视角几何、点云与 3D 表示 |

论文精读放进 [论文笔记](../papers.md) 那一本。
