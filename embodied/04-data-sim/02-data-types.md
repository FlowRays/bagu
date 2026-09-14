# 具身数据类型与采集方式

> 前置：[数据与 sim2real](01-data-and-sim2real.md)、[动作空间从零](../02-action/01b-action-space-from-zero.md)。
> 相关：[仿真器与环境栈](03-simulators.md)、[第一代 VLA](../01-vla/02-gen1-ar-vla.md)、[评测](../05-eval/01-eval.md)。

**UMI、Ego、Teleop、Simulation 经常被并列说，但它们不是同一个维度的概念。** 先把维度拆开，再看每一类。

## 1. 四个维度

| 维度 | 问的是什么 | 例子 |
|---|---|---|
| **谁在动** | 人、人控制的机器人、机器人自己、生成模型 | Ego 是人在动；Teleop 是人控制机器人动 |
| **有没有 robot action** | 数据里有没有能直接喂给机器人的动作标签 | Ego 没有；UMI 有代理动作；Teleop 有真动作 |
| **视角** | 头部 / 腕部 / 第三人称 | Ego 是头部视角；UMI 是夹爪上的腕部视角 |
| **采集方式** | 用什么设备、什么流程拿到数据 | UMI 是一种采集方案；leader-follower 是一种遥操作方式 |

所以「UMI 是采集方案、Ego 是视角 / 数据来源、Teleop 是数据产生方式」。把它们统一放到一条轴上看最清楚：

$$\boxed{\text{离真实 robot action 有多远}}$$

## 2. 九类数据总表

| 类型 | 谁在动 | 主要记录什么 | 有真实 robot action？ | 规模潜力 | 典型用途 |
|---|---|---|---|---|---|
| **Web Video** | 人 | 第三人称公开视频、文本 | ❌ | ⭐⭐⭐⭐⭐ | 世界知识、物体 / 动作语义 |
| **Ego Video** | 人 | 第一人称视频，手-物交互 | ❌ | ⭐⭐⭐⭐⭐ | manipulation 先验、视觉预训练 |
| **MoCap / Hand Tracking** | 人 | 手、腕、身体 3D pose | ❌，但可 retarget | ⭐⭐⭐⭐ | 人形 / 灵巧手动作 |
| **UMI 类** | 人 | 腕部 RGB + 6DoF 夹爪位姿 + 开合 | ⚠️ 代理动作（proxy） | ⭐⭐⭐⭐ | 从人类操作直接学 manipulation |
| **Robot Teleop** | 人控制机器人 | RGB + 本体状态 + robot action | ✅ | ⭐⭐⭐ | VLA 最核心的监督数据 |
| **Robot Play / Autonomous** | 机器人 | observation-action 轨迹 | ✅ | ⭐⭐⭐ | 自主采集、offline RL、持续学习 |
| **Simulation** | 人 / 策略 / 规划器 | 完整 state + 图像 + action + reward | ✅（合成） | ⭐⭐⭐⭐⭐ | scaling、RL、技能生成 |
| **Neural Synthetic** | 生成模型 | 生成的视频 / 轨迹 / 动作 | 合成 | ⭐⭐⭐⭐⭐ | 大规模扩数据 |
| **Real-world RL / Failure** | 机器人 | 成功 + 失败 + reward / 人工干预 | ✅ | ⭐⭐ | 在线 RL、纠错、鲁棒性 |

## 3. Ego：人类第一视角视频

人戴头戴相机正常生活：看到桌子 → 手伸过去 → 拿起杯子 → 放到架子上。

- **Ego4D**：数千小时第一人称日常活动视频，部分带 audio、IMU、gaze。
- **Ego-Exo4D**：人戴 **Aria 眼镜**记录第一视角，周围再摆 4–5 个 GoPro 同步拍第三视角；Aria 自身还有 RGB、IMU、audio、eye gaze、SLAM。

**优势是极易 scale**：把「机器人采数据」变成「人正常生活顺便产生数据」。

**问题是没有 robot action**。只知道「人把杯子拿起来了」，不知道机器人该输出

$$a_t=(\Delta x,\Delta y,\Delta z,\Delta r_x,\Delta r_y,\Delta r_z,g)$$

所以 Ego 天然适合学**视觉、dynamics、affordance、任务语义**，而不是最直接的 policy 监督。「human Ego → VLA」是 2026 年很活跃的方向（例如把 egocentric 人类数据和机器人数据一起用于 π0.5 的工作）。

## 4. UMI：介于 Ego 和 Teleop 之间

UMI = **Universal Manipulation Interface**。它要解决的正是 Ego 的问题：

> 能不能保持「人采数据很便宜」，同时拿到接近 robot action 的信息？

做法是人**不直接拿物体**，而是拿一个**长得像机器人末端执行器的手持夹爪**去完成任务，夹爪上装 GoPro：

$$\text{Human}\rightarrow\boxed{\text{UMI gripper}}\rightarrow\text{object}$$

同时得到 $RGB_t$、$SE(3)$ 位姿 $T_t$、夹爪开合宽度 $g_t$，于是可以构造代理动作：

$$a_t\approx \text{pose}_{t+1}-\text{pose}_t$$

$$\boxed{\text{UMI}\approx\text{Ego video}+\text{action proxy}}$$

几个关键设计：

| 设计 | 为什么 |
|---|---|
| **wrist-only 相机**，没有头部 / 第三人称相机 | 部署时在机器人末端装同样视角的相机，**人→机器人的观测分布几乎一致**，省掉视角 gap |
| GoPro 超广角 fisheye（约 155°）+ 夹爪侧面**镜子** | 腕部视角全局信息弱，夹爪低头时可能看不到远处目标；广角和镜子补视野与几何信息 |
| **视觉 SLAM + GoPro 内置 IMU** 离线恢复轨迹 | 手快速运动时画面有 motion blur，纯视觉跟踪会丢，IMU 补上（见第 11 节） |
| 不需要机器人在现场 | 可以拿去咖啡馆、厨房等真实环境采；原论文示例有抛物、摆杯子、叠衣服、洗盘子 |

⚠️ **GoPro 同时承担两个角色**：拍 policy 的视觉观测，又帮助恢复 action。

后续的 **FastUMI** 等工作进一步降低硬件依赖和部署成本，目标都是让这类数据继续 scale。

## 5. Robot Teleop：目前最「黄金」的 VLA 数据

人通过某种接口控制**真实机器人**，得到 $(o_t,a_t)$：

$$o_t=(RGB_{\text{head}},\ RGB_{\text{wrist}},\ q_t,\ \dot q_t,\ g_t,\ \text{language})$$

action 是真正执行到机器人上的量：$\Delta$EE pose、joint position 或 joint velocity。

| 接口 | 怎么映射 | 例子 |
|---|---|---|
| **VR 手柄** | 手柄的**位姿变化** → 机器人末端位姿变化，扳机控制夹爪 | DROID（Oculus → Franka） |
| **SpaceMouse** | 六轴鼠标 → 末端增量 | 仿真遥操作常用 |
| **Leader-Follower** | 人直接掰 leader 臂，关节角 1:1 发给 follower：$q^{\text{follower}}_t\leftarrow q^{\text{leader}}_t$ | SO-101、ALOHA 类平台 |
| Apple Vision Pro / 手柄 | 手部 / 头部追踪 → 末端 | 各类新平台 |

DROID 一次采集持续记录的字段：

```text
camera_1 RGB / camera_2 RGB
robot joint position / joint velocity
end-effector pose
action
timestamp
```

**优势**：分布完全对齐机器人。Open X-Embodiment 就是这类轨迹的大规模汇总。

**问题**：贵。$1\text{ 人}+1\text{ 台机器人}+1\text{ 个环境}$ 才能持续采，是今天 robot foundation model 最大的数据瓶颈之一。**leader-follower 不需要 VR / 动捕，是很多廉价采集平台的本质**。

## 6. MoCap / Hand Tracking 与 retargeting

戴动捕服、数据手套（Manus）、Vision Pro、Leap Motion、VR tracker，得到人体骨架，再映射到机器人：

$$\text{Human motion}\xrightarrow{\text{retargeting}}\text{Robot motion},\qquad \text{human wrist pose}\rightarrow\text{robot EE pose}$$

和 UMI 的区别：**UMI 围绕 manipulation 接口；MoCap 关注整个人体 / 手的运动学**。在人形 locomotion、全身操作、灵巧手里特别重要。

OmniH2O 的链路：

```text
Human → VR / RGB camera → human body pose → retargeting
      → humanoid target pose → low-level controller → Unitree H1
```

不是「记录人的视频」，而是 $\text{Human Pose}\to\text{Robot-compatible Pose}\to\text{Robot Action}$；遥操作轨迹之后再用来训练自主策略。

## 7. Simulation

仿真里几乎什么监督都有：RGB、depth、segmentation、state、action、reward、contact、force、pose，甚至真实世界不可观测的**物体精确位姿**。

所以可以 $1\text{k demos}\to100\text{k}/1\text{M trajectories}$，再用 domain randomization（纹理、光照、相机、物体、动力学）扩展。BEHAVIOR 2026 的数据就是典型的**仿真遥操作**数据：带本体状态、action、相机位姿和语言标注。

最大问题仍然是 $\boxed{\text{Sim2Real gap}}$，见 [01](01-data-and-sim2real.md)。仿真器本身见 [03](03-simulators.md)。

## 8. Neural Synthetic

不一定真的跑物理引擎，而是

$$\text{real robot video}\rightarrow\text{video / world model}\rightarrow\text{new trajectory}$$

改变环境、物体、背景、动作、轨迹。GR00T N1 / N1.5 明确把数据分成 **Human Video + Synthetic + Real Robot**，synthetic 里又包括仿真轨迹和神经网络生成的轨迹。

$$\boxed{\text{世界模型不只是 policy 的一部分，也可能直接成为 robot data engine}}$$

## 9. Robot Play / Autonomous

不是人规定「把红杯子放进箱子」，而是机器人自由交互：摸、推、抓、撞、失败、重抓。

- **优势**：机器人可以 24/7 自己产生数据。
- **缺点**：大量数据并不 interesting。

所以必须配套：**goal relabeling**、reward model、**success detector**、novelty selection、automatic curriculum。

## 10. 数据金字塔

```text
                   Real-world RL
                  / Autonomous
                       ↑
                Robot Teleoperation
                       ↑
            Simulation / Synthetic
                       ↑
              UMI / MoCap / VR
                       ↑
                 Ego Human Video
                       ↑
              Internet / Web Data

数据量        大  ←────────────→  小
action 质量   弱  ────────────→  强
采集成本      低  ────────────→  高
robot 对齐    弱  ────────────→  强
```

- 底层 **Web + Ego**：数据极多，离 action 很远，负责规模和语义。
- **MoCap / UMI**：人类数据，但开始有 action / 几何。
- **Sim + Synthetic**：有 action，可大量生成。
- **Real Robot Teleop**：非常准，但昂贵。
- 顶层 **Autonomous / RL**：进入「自己行动 → 观察后果 → 再学习」的闭环。

这和 NVIDIA 描述 GR00T 的 data pyramid 基本一致。

$$\boxed{\text{真正缺的不是 video，而是便宜、规模化、带 action、且与物理交互闭环的数据}}$$

## 11. IMU 是什么

IMU = **Inertial Measurement Unit，惯性测量单元**：感知「设备自己怎么动」的小传感器，手机、无人机、VR 头显、相机、机器人里都有。

| 传感器 | 测什么 | 例子 |
|---|---|---|
| **加速度计** accelerometer | 三轴线加速度 $a_x,a_y,a_z$；静止时也感受到**重力**，能判断倾斜 | 夹爪突然往右甩 → $a_x$ 很大 |
| **陀螺仪** gyroscope | 三轴角速度 $\omega_x,\omega_y,\omega_z$ | 夹爪绕 z 轴转 → $\omega_z$ 变大 |

直觉：闭眼转一下手机，相机什么都没看到，但手机依然知道「刚才被快速旋转了」—— 这就是 IMU。

### UMI 为什么需要 IMU

UMI 最终要的是夹爪每一时刻的位姿 $T_t=(x,y,z,R)$。只靠视频做视觉 SLAM（看环境特征点怎么移动 → 估相机运动）在慢速时够用，但**人手运动很快**，快速甩动时帧间有 motion blur，特征点很难匹配。于是融合：

```text
RGB ──→ visual tracking ──┐
                          ├──→ camera pose ──→ gripper trajectory
IMU (accel + gyro) ───────┘
```

也就是 **visual-inertial** 位姿估计。采集到的原始数据是 $RGB+IMU+\text{gripper state}$，离线处理后得到

$$RGB_t\ +\ 6\text{DoF gripper pose}_t\ +\ \text{gripper width}_t$$

$$\boxed{\text{IMU 主要用在数据处理阶段恢复可靠的轨迹 / action，通常不是 policy 的主要输入}}$$

## 12. 采集过程怎么看：演示视频

| 采集方式 | 演示 | 看什么 |
|---|---|---|
| **UMI** | [项目页](https://umi-gripper.github.io/)、[官方 YouTube Demo](https://www.youtube.com/watch?v=EJmAg1Bnp-k) | 人手里拿着**带 GoPro 的夹爪**操作物体，没有机器人在场 |
| **Leader-Follower** | [SO-101 数据采集](https://www.youtube.com/watch?v=2mo1Rzy8sCw) | 人掰一台小机械臂，另一台 1:1 跟动，同时录相机、关节、action（示例采了 50 条 pick-and-place） |
| **VR Teleop** | [DROID 遥操作实拍](https://drive.google.com/file/d/1Rg10T5rVaK9m_0BYXq2EkVdNgLpZnj3P/preview) | Oculus 手柄位姿变化映射到 Franka 末端，扳机控夹爪 |
| **Ego / Ego-Exo** | [Ego4D 介绍](https://www.youtube.com/watch?v=9V_QfuXhgas)、[Ego-Exo4D](https://www.youtube.com/watch?v=GdooXEBAnI8) | 人戴 Aria 正常做饭、修东西；Ego-Exo 周围多台 GoPro 同步 |
| **Humanoid 全身** | [OmniH2O](https://omni.human2humanoid.com/) | VR / RGB pose → retarget → H1 全身跟随 |
| **仿真遥操作** | [Isaac Lab Teleop + Data Collection](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/overview/teleop_imitation.html) | SpaceMouse / Vision Pro 控制**仿真机器人**，动作和 state 写入 HDF5 |

只有 20 分钟的话，按 **UMI → SO-101 → DROID → Ego-Exo4D → OmniH2O** 看。

### 五张采集链路图

```text
① Ego
Human ──→ head camera
只有 observation
```

```text
② UMI
Human → Instrumented Gripper → Object
                ↓
          RGB + Pose
human action proxy
```

```text
③ Robot Teleop
Human → VR / SpaceMouse → Robot → Object
                           ↓
                  RGB + state + action
真正 robot data
```

```text
④ Leader-Follower
Human → Leader Arm → Follower Arm → Object
                          ↓
                 RGB + state + action
廉价 robot data
```

```text
⑤ MoCap / Humanoid
Human motion → Pose estimation → Retargeting → Humanoid
```

## 自测

**1.** 为什么说「UMI、Ego、Teleop、Sim」不是同一维度？该用哪条轴统一？

> **答：** UMI 是**采集方案**，Ego 是**视角 / 数据来源**，Teleop 是**数据产生方式**，Sim 是**环境**。拆开的四个维度是：谁在动、有没有 robot action、视角、采集方式。
> 统一看时用一条轴：**离真实 robot action 有多远**。

**2.** Ego 数据的最大优势和最大问题？它适合学什么？

> **答：** 优势是**极易 scale**（人正常生活顺便产生）。问题是**没有 robot action**，只知道「人拿起了杯子」，不知道机器人该输出的 $\Delta$pose 和夹爪指令。
> 所以适合学视觉表示、dynamics、affordance、任务语义，不是最直接的 policy 监督。

**3.** UMI 怎么同时做到「便宜」和「接近 robot action」？

> **答：** 人不直接拿物体，而是拿一个**像机器人末端执行器的手持夹爪**（带 GoPro），不需要机器人在场所以便宜；GoPro 视频 + IMU 离线恢复夹爪 $SE(3)$ 轨迹，加上夹爪开合宽度，就能构造 $a_t\approx\text{pose}_{t+1}-\text{pose}_t$ 这样的**代理动作**。

**4.** ⭐ 原版 UMI 为什么只用腕部相机？代价是什么，怎么补？

> **答：** 部署时在机器人末端装同视角相机，**训练和部署的观测分布几乎一致**，避免人头部视角和机器人视角之间的 domain gap。
> 代价是**全局信息弱**，夹爪低头时可能看不到远处目标。补法：约 155° 的超广角 fisheye + 夹爪侧面镜子。

**5.** IMU 测什么？UMI 为什么需要它？

> **答：** 加速度计测三轴线加速度（静止时含重力，可判断倾斜），陀螺仪测三轴角速度。
> 人手运动快，视频帧间 motion blur 导致视觉 SLAM 特征点匹配失败；融合 IMU 做 visual-inertial 估计，快速运动时位姿轨迹仍然稳定。

**6.** IMU 数据一般是 VLA policy 的输入吗？

> **答：** **通常不是主要输入。** 它主要用在数据处理阶段，帮助恢复可靠的夹爪轨迹 / action。最终喂给 policy 的是 $RGB_t$、6DoF 夹爪位姿、夹爪宽度。

**7.** VR 遥操作和 leader-follower 分别怎么把人的动作映射到机器人？后者为什么流行？

> **答：** VR：手柄的**位姿变化**映射成机器人末端位姿变化，扳机控夹爪（DROID）。Leader-follower：人掰 leader 臂，读出每个关节角直接发给 follower，$q^{\text{follower}}_t\leftarrow q^{\text{leader}}_t$（关节空间 1:1）。
> 后者不需要 VR / 动捕设备，硬件便宜直观，是很多廉价采集平台的本质。

**8.** MoCap 和 UMI 的区别？retargeting 在做什么？

> **答：** UMI 围绕 **manipulation 接口**（末端夹爪），MoCap 记录**整个人体 / 手的运动学**，在人形和灵巧手里更重要。
> retargeting 把人体 pose 映射成机器人兼容的 pose（如 human wrist pose → robot EE pose），再由低层控制器执行，例如 OmniH2O 的 `human pose → retarget → humanoid target pose → controller → H1`。

**9.** Play / autonomous 数据的优缺点？需要配什么？

> **答：** 优点是机器人 24/7 自己产生数据；缺点是大部分数据不 interesting。需要 goal relabeling、reward model、success detector、novelty selection、automatic curriculum 来挑出有用的部分。

**10.** 画出数据金字塔，并说出四条轴的变化方向。

> **答：** 自底向上：Web → Ego → UMI / MoCap / VR → Sim / Synthetic → Robot Teleop → Real-world RL / Autonomous。
> 越往上：**数据量越小、action 质量越强、采集成本越高、与机器人对齐越好**。核心矛盾是真正缺的不是视频，而是便宜、规模化、带 action 且和物理交互闭环的数据。
