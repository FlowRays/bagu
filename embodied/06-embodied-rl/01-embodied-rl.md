# 具身 RL

> 前置：[LLM RL 目录](../../notes/06-post-training/rl/00-map.md)。这一篇只讲**具身特有的部分**。

## 1. 和 LLM RL 的五个差异

| | LLM RL | 具身 RL |
|---|---|---|
| rollout 成本 | 一次前向，可大批并行 | **真机上是物理时间**，无法并行；仿真可并行但有 gap |
| episode 长度 | 几百到几千 token | 几百到几千步，且**每步都要和环境交互** |
| reward | rule-based 很干净（答案对错） | **稀疏**（只有成功/失败），或需要手工设计 |
| 安全 | 说错话 | **可能撞坏硬件或伤人**，探索受限 |
| 动作空间 | 离散，词表有限 | **连续高维** |

$$\boxed{\text{LLM RL 的瓶颈是 reward 质量，具身 RL 的瓶颈是采样成本和安全}}$$

## 2. reward 设计

| 类型 | 说明 |
|---|---|
| **稀疏 reward** | 只在任务成功时给 1。最诚实但最难学 |
| **shaped reward** | 手工加中间奖励（靠近物体、抓住了…）。学得快但**容易被 hack**（模型找到刷分捷径而不完成任务） |
| **学出来的 reward** | 用 VLM 判断"是否完成"，或训一个成功分类器。灵活但引入 judge 的偏置 |
| **过程奖励** | 每个子阶段给分，和 LLM 的 PRM 同构 |

reward hacking 在具身里更直观：让机械臂"把方块推进洞里"，它可能学会把整个桌子掀起来。这和 [RM 的 reward hacking](../../notes/06-post-training/preference/01-rm-and-rlhf.md#3-rm-的问题) 是同一类问题。

## 3. 几条实用路线

| 路线 | 做法 |
|---|---|
| **仿真里 RL + sim2real** | 最主流。仿真可以并行几千个环境，配 DR 迁移到真机 |
| **IL 初始化 + RL 微调** | 先 BC 学个能用的策略，再用 RL 提升。和 LLM 的 `SFT → RL` 完全同构 |
| **offline RL** | 只用已有数据不交互，回避采样成本，但受数据覆盖限制 |
| **residual policy** | 学一个在传统控制器输出上的**增量**，安全性有保障、学习更容易 |
| **世界模型里 RL** | Dreamer 路线，在想象中训练，真机采样极少 |

## 4. GRPO / PPO 在 VLA 上怎么用

思路可以直接搬（见 [GRPO](../../notes/06-post-training/rl/06-grpo.md)），但要处理三件事：

1. **连续动作的 log prob**：动作 head 是 diffusion / flow 时，log prob 不好算。常见做法是改用高斯 head 做 RL，或者只对离散化的动作 token 做 RL
2. **rollout 极慢**：GRPO 要一个 prompt 采 $G$ 条，在真机上不现实 → 只能在仿真里做
3. **advantage 的粒度**：一条轨迹几百步只有一个成功/失败信号，credit assignment 比长 CoT 还难

$$\boxed{\text{LLM RL 的 credit assignment 已经难，具身还要再乘上「rollout 贵几个数量级」}}$$

所以具身里**dense 监督**（过程奖励、世界模型、[OPD 式的 teacher 逐步指导](../../notes/06-post-training/distill/03-opd.md)）比在 LLM 里更有价值。

## 5. 安全约束

- **动作限幅**：软件层面卡死速度、力、工作空间
- **仿真中先验证**，真机上先低速跑
- **人在回路**：随时能急停
- **constrained RL / safe RL**：把安全当约束而不是 reward 的一项

## 自测

**1.** 具身 RL 和 LLM RL 的五个差异？各自的瓶颈是什么？

> **答：** ① **rollout 成本**（真机是物理时间、无法并行）；② episode 长且每步都要交互；③ **reward 稀疏**或要手工设计；④ **安全**（可能撞坏硬件伤人），探索受限；⑤ **动作空间连续高维**。
> **LLM RL 的瓶颈是 reward 质量，具身 RL 的瓶颈是采样成本和安全。**

**2.** 四类 reward 各有什么问题？举一个具身 reward hacking 的例子。

> **答：** **稀疏**最诚实但最难学；**shaped** 学得快但容易被 hack；**学出来的**（VLM 判断/成功分类器）灵活但引入 judge 偏置；**过程奖励**和 LLM 的 PRM 同构。
> hacking 例子：让机械臂"把方块推进洞里"，它可能学会**把整个桌子掀起来**让方块掉进去。

**3.** 五条实用路线？哪条和 LLM 的 SFT→RL 同构？

> **答：** 仿真 RL + sim2real（最主流）、**IL 初始化 + RL 微调**（和 `SFT → RL` 完全同构）、offline RL、residual policy（学传统控制器输出的增量，安全有保障）、世界模型里 RL（Dreamer）。

**4.** GRPO 用在 VLA 上要处理哪三件事？

> **答：** ① **连续动作的 log prob 不好算**（diffusion/flow head 没有显式密度）→ 改用高斯 head 或只对离散动作 token 做 RL；② **rollout 极慢**，一个 prompt 采 $G$ 条在真机上不现实 → 只能在仿真里做；③ **credit assignment 更难**，几百步只有一个成功/失败信号。

**5.** 为什么 dense 监督在具身里比在 LLM 里更有价值？

> **答：** 因为具身的 credit assignment 已经很难（长 episode + 稀疏 reward），而**rollout 还贵几个数量级** —— 每一条轨迹的信息量必须被榨干。所以过程奖励、世界模型、OPD 式的 teacher 逐步指导都比稀疏 reward 的 RL 更划算。
