# 具身智能：技术演变笔记（VLA / WAM）

**目的：把具身的技术演变整理成笔记，能理解并说出历史演变、最新进展，以及每一部分的细节原理。**

和 [LLM/VLM 八股](bagu.md) 的关系：LLM/VLM 是底座，这里只整理"从看懂世界到产生动作"这一段特有的东西。

学习思路：跟 GPT 交互理解 → 整理结构化笔记 → 关掉笔记口述 / 纸上推导 → 暴露问题 → 回填。每一代都按 **上一代的问题 → 这一代改了什么 → 逐 tensor 的细节** 来写。

---

## 四块内容

| 块 | 要能说出什么 | 笔记 |
|---|---|---|
| **一、VLA 和知名架构**（主线） | 每一代为什么这样设计、修了上一代什么；动作怎么表示和解码 | [VLA 模型](embodied/01-vla/00-map.md)：第一代 RT-1→RT-2→RT-X→OpenVLA、第二代 π0→π0.5→G0、G0.5；[Action](embodied/02-action/00-map.md)：action head、chunking、ACT 与 Diffusion Policy |
| **二、机器人基础** | 一条 action 里每个数字是什么；控制链从目标位姿到电机力矩 | [动作空间从零](embodied/02-action/01b-action-space-from-zero.md)（DoF / FK / IK / 旋转 / 正逆动力学）、[动作空间与表示](embodied/02-action/01-action-space.md)、[三种范式](embodied/03-overview/01-paradigms.md) |
| **三、数据集、仿真、评测** | 数据从哪来、离真实 robot action 多远；环境栈谁搭的；分数怎么读 | [数据与仿真](embodied/04-data-sim/00-map.md)：数据类型与采集、MuJoCo / LIBERO / Isaac、sim2real；[评测](embodied/05-eval/00-map.md)；[具身 RL](embodied/06-embodied-rl/00-map.md) |
| **四、最新进展** | 前排模型各押注什么，谁在用 Ego / UMI / 世界模型 | [前沿路线](embodied/01-vla/05-frontier-routes.md)：DM0.5、小米 Robotics-1 / U0、RoboDojo 前 10 |
| 支撑原理（目录最下） | 生成式动作头和世界模型背后的数学 | [世界模型](embodied/07-world-model/00-map.md)、[手撕实现](embodied/08-handwrite/00-map.md)、[视觉生成](embodied/09-visual-generation/00-map.md)（AE/VAE、Diffusion、Flow Matching）、[JEPA](embodied/10-jepa/01-jepa.md) |

## 技术演变时间线

### 主线：policy / VLA

| 时间 | 代表 | 修了上一代什么 | 笔记 |
|---|---|---|---|
| 2022 前 | 模块化（感知→规划→控制）、单任务 BC | — | [范式](embodied/03-overview/01-paradigms.md) |
| 2022 | **RT-1** | 一个模型学很多任务（130k episodes、700+ 任务），还不是 VLA | [第一代](embodied/01-vla/02-gen1-ar-vla.md) |
| 2023 | **ACT、Diffusion Policy** | 单步 BC 误差累积、MSE 平均多模态 → 连续 chunk + 生成式 | [ACT 与 DP](embodied/02-action/04-act-and-diffusion-policy.md) |
| 2023 | **RT-2** | 机器人数据装不下世界知识 → 用 VLM，动作伪装成语言 token | [第一代](embodied/01-vla/02-gen1-ar-vla.md) |
| 2023 | **Open X-Embodiment / RT-X** | 单一本体 → 多本体混训（粗对齐） | [第一代](embodied/01-vla/02-gen1-ar-vla.md) |
| 2024 | **OpenVLA** | 不开源 → 开源 7B，现代 VLM 架构 | [第一代](embodied/01-vla/02-gen1-ar-vla.md) |
| 2024 | **π0** | 逐标量 token 粗、慢、单步 → VLM + flow matching action expert，50 步连续 chunk | [第二代](embodied/01-vla/03-pi0-pi05-g0.md) |
| 2025 | **FAST、π0.5** | 动作 token 太多；长程任务不会拆 → 压缩 token 预训练 + 子任务 + flow 执行 | [第二代](embodied/01-vla/03-pi0-pi05-g0.md) |
| 2025 | **G0** | 跨本体负迁移 → 独立 planner + 单本体预训练阶段 | [第二代](embodied/01-vla/03-pi0-pi05-g0.md) |
| 2026 | **G0.5** | 三套参数、动作表示差 → 一个 VLM 生成推理 + ActionCodec token | [G0.5](embodied/01-vla/04-g05-architecture.md) |
| 2026 | **DM0.5、Xiaomi-Robotics-1 / U0、OpenWAM-α 等** | action head 收敛后比数据 scale、记忆推理、世界模型数据引擎 | [前沿路线](embodied/01-vla/05-frontier-routes.md) |

### 数据线

| 阶段 | 代表 | 笔记 |
|---|---|---|
| 单机构真机遥操作 | RT-1 数据 | [第一代](embodied/01-vla/02-gen1-ar-vla.md) |
| 多机构汇总 / 大规模多场景 | Open X-Embodiment、DROID | [数据与 sim2real](embodied/04-data-sim/01-data-and-sim2real.md) |
| 便宜的人类数据 | UMI、人类 Ego、MoCap retarget | [数据类型](embodied/04-data-sim/02-data-types.md) |
| 合成与世界模型当数据引擎 | 仿真 + DR、GR00T 数据金字塔、Xiaomi-Robotics-U0 | [数据类型](embodied/04-data-sim/02-data-types.md)、[前沿路线](embodied/01-vla/05-frontier-routes.md) |

### 支线：世界模型与表示

Dreamer（latent dynamics）→ Genie（latent action）→ 视频生成当世界模型 → V-JEPA 2-AC（表示空间预测 + 规划）→ WAM（视频 DiT 与动作 DiT 联合去噪）。见 [世界模型](embodied/07-world-model/00-map.md)、[JEPA](embodied/10-jepa/01-jepa.md)。

---

## 和 LLM 最本质的五个差异

先建立这个对照，后面所有设计取舍都从这里长出来：

| | LLM / VLM | 具身（VLA / WAM） |
|---|---|---|
| 输出 | 离散 token，词表有限 | **连续动作**，高维、有物理量纲 |
| 时间 | 用户等得起几百 ms | **实时闭环**，控制频率 10–1000 Hz |
| 错误代价 | 说错话 | **撞坏硬件 / 伤人**，不可回退 |
| 数据 | 互联网规模文本 | **遥操作数据极贵**，几万条已经算大 |
| 分布 | 训练=测试分布 | **sim2real、跨本体、跨场景**都是分布迁移 |

$$\boxed{\text{一句话：LLM 缺的是推理，具身缺的是数据和闭环}}$$

---

## 详细大纲

### (1) VLA 模型

- 演进线
    - RT-1 → RT-2（把动作当 token 接到 VLM 上）
    - RT-X / Open X-Embodiment（多本体混合训练，粗对齐）
    - OpenVLA（开源基线）
    - **π0 / π0.5**（flow matching action expert；π0.5 FAST 预训练 + 子任务 + flow 执行）
    - **G0 → G0.5**（双系统 + 单本体预训练 → 一个 VLM 生成推理与 ActionCodec token）
    - GR00T、Helix、以及最新的一批（DM0.5、Xiaomi-Robotics-1 / U0、OpenWAM 等）
- 架构要点
    - VLM backbone（通常冻结或小 lr）+ **action expert**
    - 视觉输入：多相机、腕部相机、历史帧
    - 本体感知（proprioception）怎么注入
    - 语言指令怎么条件化
- 训练范式
    - 大规模跨本体预训练 → 特定本体后训练
    - co-training：机器人数据 + 网络图文数据一起训，防遗忘
    - 数据配比
- 推理效率：action chunk、并行解码、蒸馏

### (2) Action 表示与解码

VLA 和 LLM 差别最大的一段，面试必问。

- 动作空间
    - joint position / velocity / torque
    - end-effector pose（位置 + 姿态）、6D vs 7D、四元数 vs 6D 旋转表示
    - **delta（增量）vs absolute（绝对）**
    - gripper 开合、力控
    - 归一化、坐标系（base / camera / tool）
- 表示与解码方式
    - **连续回归**：MSE / L1，简单但**无法建模多模态**
    - **离散 tokenization**：分 bin → 当 token 预测（RT-2、OpenVLA）；bin 数与精度的权衡
    - **FAST**：DCT + BPE 的动作压缩 tokenizer
    - **Diffusion policy**：把动作序列当作要去噪的样本
    - **Flow matching head**：π0 系列，少步、快
- **Action chunking**
    - ACT：一次预测未来 $H$ 步动作；CVAE 结构，推理取 $z=0$
    - Diffusion Policy：$T_o/T_p/T_a$、receding horizon
    - 为什么能缓解 compounding error 和多模态
    - **temporal ensembling**：重叠 chunk 加权平均
    - chunk 长度与延迟的权衡
- 实时性
    - 控制频率 vs 推理频率
    - 异步推理、动作缓冲
    - 大模型怎么跑到 50 Hz：小 action expert、量化、蒸馏
- 损失设计
    - 回归 loss vs 分类 loss vs 扩散 loss
    - 多模态动作分布为什么必须用生成式建模

### (3) 具身总览与范式

- 任务形态
    - manipulation（抓取、装配、柔性物体）、navigation、locomotion、mobile manipulation
    - 长程任务、多阶段任务
- 三种范式，以及各自赌的是什么
    - **模块化**：感知 → 规划 → 控制，各段独立。可解释、好调试，但误差累积、接口僵硬
    - **端到端 VLA**：观测直接映射到动作。赌的是"数据够多就能学到"
    - **世界模型 + 规划**：先学环境动力学，再在想象中搜索。赌的是"学 dynamics 比学 policy 更省数据"
    - **分层**：高层 VLM 出子目标，低层 policy 出动作。现在的实用主流
- 关键概念
    - embodiment / morphology、observation space、action space
    - open-loop vs closed-loop
    - imitation learning（BC）vs RL
    - cross-embodiment 泛化
- 和 LLM 的关系：VLM 提供语义理解，具身要补的是**动作**和**物理**

### (4) 数据

- 来源
    - **遥操作**（VR、主从臂、外骨骼）：质量高、极贵
    - **仿真**：便宜、可无限扩展、有 gap
    - **人类视频**：量大、缺动作标注 → latent action / 手部姿态估计
    - play data、autonomous collection
    - UMI（手持夹爪 + 腕部相机 + SLAM/IMU 恢复位姿）、人类 Ego、MoCap retarget、神经合成数据
- 数据集：Open X-Embodiment、DROID、RT-X、AgiBot 等
- 数据金字塔：网络数据（语义）→ 人类视频（动作先验）→ 仿真 → 真机
- 质量问题：遥操作抖动、多模态标注、失败轨迹要不要留

### (5) 仿真与 sim2real

- 仿真器：Isaac Sim / Isaac Lab、MuJoCo、SAPIEN、Genesis
    - 物理引擎 ≠ 任务环境：MuJoCo → robosuite → LIBERO；Isaac Sim → Isaac Lab
- 物理：刚体、接触求解、软体与流体的难点
- sim2real 手段
    - **domain randomization**（视觉、动力学）
    - system identification、real2sim
    - digital twin
- 渲染与真实感、传感器建模

### (6) 评测

- 仿真 benchmark：LIBERO、SimplerEnv、CALVIN、Meta-World、RoboCasa
- 真机评测：成功率、试验次数、方差、可复现性
- 泛化维度：新物体 / 新场景 / 新指令 / 新本体
- 常见陷阱：仿真涨点不代表真机涨点；成功率之外还要看效率和安全

### (7) 具身 RL

- 和 LLM RL 的区别
    - reward 更稀疏、episode 更长、rollout 极慢且不可并行采样
    - 安全约束、真机采样的代价
- 手段
    - sim 里 RL 再迁移
    - offline RL / IL + RL 混合
    - residual policy
    - 世界模型里 RL（Dreamer 路线）
- GRPO / PPO 在 VLA 上怎么用，reward 怎么设计

### (8) 世界模型 WAM

- 定义：学 $p(s_{t+1}\mid s_t,a_t)$ 或它的表示空间版本
- 三条技术路线
    - **Latent dynamics**（Dreamer 系）：RSSM、在 latent 里 imagine、actor-critic 在想象中训练
    - **Pixel / video 生成**（Genie、Sora-like）：直接生成未来帧
    - **表示空间预测**（JEPA 系）：见 (4)
- Dreamer 系列
    - RSSM：deterministic + stochastic 双路
    - 三个 loss：重建、动力学、reward
    - imagination rollout、$\lambda$-return
    - DreamerV3 的 symlog、two-hot 等稳定化技巧
- 可交互世界模型
    - Genie / Genie-2 / Genie-3：**latent action model**（从无标注视频里反推动作）
    - 可控性、长时一致性、实时性
- 视频生成模型能当世界模型吗
    - "Sora is a world simulator" 的争议
    - 缺什么：动作条件、因果、物理一致性、可交互
- 用世界模型做决策
    - MPC / CEM / MPPI 在想象中搜索
    - policy in imagination（Dreamer）
    - 世界模型当数据增强器
- 和 LLM 的联系：next-token prediction 就是一维世界模型

### (9) 视觉生成基础

这一块是世界模型和 diffusion action head 的共同地基。

- 生成模型谱系
    - VAE：ELBO、重参数化、posterior collapse
    - GAN：对抗训练、mode collapse
    - Autoregressive：VQ-VAE / VQ-GAN → token 化后当语言模型做
    - **Diffusion**：现在的主流
    - **Flow Matching / Rectified Flow**：更直、更快，VLA 的 action head 正在大规模转向它
- Diffusion 核心
    - 前向加噪 $q(x_t|x_0)$ 的闭式解、方差 schedule
    - 反向去噪、噪声预测 $\epsilon_\theta$、与 score matching 的等价
    - 训练目标为什么可以简化成 MSE
    - DDPM vs **DDIM**（确定性、少步采样）
    - **Classifier-free guidance**：条件与无条件的线性外推
    - v-prediction、SNR 加权
- Flow Matching
    - ODE 视角、连续归一化流
    - **Rectified Flow**：直线路径、$v_\theta$ 预测、少步甚至一步
    - 和 diffusion 的关系与取舍
- 架构
    - U-Net → **DiT**（Transformer 做 backbone）
    - **Latent diffusion**：先 VAE 压到 latent 再扩散
    - 条件注入：cross-attention / AdaLN / in-context
- 视频生成
    - 时空注意力（full 3D / factorized）
    - **causal video VAE**、时间压缩
    - 长视频：autoregressive、滑窗、chunk 生成
    - 一致性问题：时序抖动、物体永存性
- 评价：FID / FVD 及其局限

### (10) JEPA 系列

- 核心主张：**在表示空间预测，而不是在像素空间重建**
    - LeCun 的论据：像素级重建把大量容量浪费在不可预测的细节上
    - 和 MAE 的区别（MAE 重建像素，JEPA 预测表示）
    - 和对比学习的区别（不需要负样本 / 大 batch）
- 防坍缩机制：EMA target encoder、stop-gradient、asymmetric 结构
- **I-JEPA**：context block → 预测多个 target block 的表示
- **V-JEPA / V-JEPA 2**：视频、mask 时空块
    - V-JEPA 2-AC：动作条件的后训练，直接用于规划
- 能量模型视角（EBM）、JEPA 与规划的关系
- 争议与局限：表示评估困难、下游需要探针、生成能力缺失

---

## 面向代码的维度

- (a) diffusion / flow matching 的最小实现
- (b) action head 与 chunking 的实现
- (c) 仿真环境接口与数据管线
- (d) VLA 训练框架（LeRobot、Isaac Lab 等）
