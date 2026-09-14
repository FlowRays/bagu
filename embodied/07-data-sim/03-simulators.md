# 仿真器与环境栈：MuJoCo、robosuite、LIBERO、Isaac

> 前置：[数据与 sim2real](01-data-and-sim2real.md)、[动作空间从零](../05-action/01b-action-space-from-zero.md)。
> 相关：[具身数据类型与采集方式](02-data-types.md)、[评测](../09-eval/01-eval.md)。

**最容易混的一件事：物理引擎、任务环境、训练框架、算法是四层不同的东西。** 先把 MuJoCo 摆到正确的层上，再看谁在它上面搭了什么，最后看 Isaac 那条栈。

## 1. MuJoCo 是什么

MuJoCo = **Multi-Joint dynamics with Contact**，一个**物理仿真器**。它做的事情是：

$$s_t+a_t\ \xrightarrow{\ \text{MuJoCo}\ }\ s_{t+1}$$

```text
机器人当前关节角度
        ↓
action = 给关节施加 torque
        ↓
MuJoCo：动力学 + 碰撞 + 摩擦
        ↓
下一帧机器人姿态
```

内部核心就是 [正动力学](../05-action/01b-action-space-from-zero.md)：$(q,\dot q,\tau)\to\ddot q$，再积分得到 $\dot q,q$。从 RL 的角度，它就是 **environment**，和游戏模拟器在 agent 循环里的位置完全一样：

```text
Agent / VLA / Policy
  ↓ action
MuJoCo
  ↓ 物理世界变化
  ↓ camera / proprioception
Agent
```

### 模型文件：MJCF

机器人和场景用 XML（**MJCF**）描述：

```xml
<body name="arm">
    <joint name="joint1" type="hinge"/>
    <geom type="capsule"/>
</body>
```

| 元素 | 含义 |
|---|---|
| `body` | 刚体 |
| `joint` | 关节 |
| `geom` | 几何形状 / 碰撞体 |
| `actuator` | 电机 |
| `sensor` | 传感器 |

### 为什么机器人领域常用 MuJoCo

机器人仿真的核心不是画面好看，而是**动力学够准、接触够稳、仿真够快**。MuJoCo 擅长刚体动力学、多关节 articulated body、接触与摩擦、执行器，几十个关节互相耦合的 humanoid 也能快速算。

## 2. 四层栈：引擎 ≠ 环境 ≠ 算法

```text
你的 VLA / RL Policy          ← 算法：PPO、SAC、diffusion policy、VLA
        ↓
任务环境 / Benchmark           ← LIBERO、Meta-World、Gymnasium 任务
        ↓
环境框架                       ← robosuite、dm_control
        ↓
MuJoCo                        ← 物理引擎
```

$$\boxed{\text{MuJoCo 几乎不负责「给你造任务」，它主要负责物理}}$$

类比：MuJoCo 之于 robosuite，就像 PyTorch 之于某个基于 PyTorch 写的训练框架。

## 3. 谁在 MuJoCo 上搭了什么

| 谁 | 做什么 | 典型东西 |
|---|---|---|
| **MuJoCo**（DeepMind / Google） | 物理引擎 | collision / contact / dynamics |
| **Gymnasium**（Farama Foundation） | 经典 RL 环境 | Ant、Hopper、Humanoid |
| **dm_control**（Google DeepMind） | RL 环境框架 | cartpole、walker、finger |
| **robosuite**（ARISE Initiative） | manipulation 框架 | Lift、Door、PickPlace |
| **Meta-World** | manipulation benchmark | 50 个任务 |
| **LIBERO** | 语言条件机器人 benchmark | LIBERO-90 等 |

### Gymnasium MuJoCo：传统 RL

`HalfCheetah / Hopper / Walker2d / Ant / Humanoid / Reacher / Pusher / Swimmer`。

```python
env = gymnasium.make("Ant-v5")   # 四足机器人, action 是 8 个关节的 torque
```

最早属于 OpenAI Gym 生态，现在由 **Farama Foundation 的 Gymnasium** 维护。分工是：**MuJoCo 算物理，Gymnasium 定义 Ant 要干什么、obs / action / reward 是什么**。适合 PPO / SAC 基础研究，但不像 VLA 做的 manipulation。

### dm_control：DeepMind 的环境栈

| 模块 | 作用 |
|---|---|
| `dm_control.mujoco` | MuJoCo 的 Python 封装 |
| `dm_control.suite` | 具体 RL 环境（cartpole、walker、cheetah、humanoid、finger、reacher、fish、hopper…） |
| `dm_control.composer` | 自己组合机器人 + 场景 + task |

它不只是跑仿真，而是一套「**怎么用 MuJoCo 构造 RL environment**」的框架。

### robosuite：manipulation 框架

ARISE Initiative 开发维护，底层 MuJoCo。任务是机械臂的：`Lift / PickPlace / NutAssembly / Door / Wipe / ToolHang / TwoArmLift / TwoArmHandover / TwoArmPegInHole`。

以 `Lift`（Franka 抓起桌上 cube）为例，**robosuite 作者构造的**是：机器人模型、桌子、cube、相机、reset 分布、observation、action、reward、success 条件；**MuJoCo 算的**只是 cube 被夹住后怎么动、摩擦、碰撞。

### Meta-World：50 个任务的 benchmark

reach、push、pick-place、开关门、开关抽屉、拧水龙头、按按钮、用棍子推、sweep……共 **50 个**机械臂操作任务，组织成：

```text
MT1 / MT10 / MT50     multi-task RL
ML1 / ML10 / ML45     meta-RL
```

### LIBERO：进入 VLA 主线

```text
LIBERO → robosuite → MuJoCo
```

LIBERO 代码直接依赖 robosuite。它研究的不是「PPO 能不能学会 lift cube」，而是**给一个语言指令，policy / VLA 能不能完成操作**：`"put the red bowl on the plate"`、`"open the drawer"`。组织成 5 个 suite：

```text
LIBERO-Spatial / LIBERO-Object / LIBERO-Goal / LIBERO-10 / LIBERO-90
```

作者包括 Bo Liu、Yifeng Zhu、Chongkai Gao、Yihao Feng、Qiang Liu、Yuke Zhu、Peter Stone 等。π0、π0.5、OpenVLA、Octo、Diffusion Policy、G0.5 这类工作经常报 LIBERO 结果。

## 4. 一个「环境」到底是谁构造的

假设研究「Franka 把杯子放进抽屉」。MuJoCo 不会凭空生成 Franka、桌子、杯子、抽屉，这些是**环境作者写的**：

$$\text{Environment}=\text{Scene}+\text{Robot}+\text{Objects}+\text{Task}+\text{Interface}$$

| # | 项 | 例子 |
|---|---|---|
| ① | robot | Franka Panda |
| ② | scene | 桌子 / 厨房 / 地面 |
| ③ | objects | 杯子 / 碗 / 抽屉 |
| ④ | initial state | 杯子初始位置随机在哪 |
| ⑤ | observation | RGB、depth、joint position、gripper state |
| ⑥ | action | joint torque / joint position / 末端 delta pose |
| ⑦ | task | "put cup in drawer" |
| ⑧ | success condition | 杯子是否真的进了抽屉 |
| ⑨ | reward | 成功 +1，或 dense reward |

场景写成 `scene.xml`（world → table / Franka / cup / drawer / camera / light），任务逻辑写成 Python：

```python
class PutCupInDrawerEnv:
    def reset(self):
        ...                          # 随机摆放杯子
    def get_obs(self):
        return {"image": ..., "joint_position": ...}
    def step(self, action):
        mujoco.step(...)             # ← 只有这一行是物理引擎的事
        return obs, reward, done
    def success(self):
        return cup_inside_drawer
```

$$\boxed{\text{九项里只有最底下的 physics transition }(s_t,a_t)\to s_{t+1}\text{ 归 MuJoCo}}$$

## 5. Isaac Sim：NVIDIA 的完整机器人仿真平台

如果 MuJoCo 更像「机器人版物理引擎」，**Isaac Sim 更像「完整的虚拟机器人世界 + 物理 + 渲染 + 传感器 + 工具链」**。从 VLA 看它的角色还是 $(o_t,a_t)\to o_{t+1}$，只是世界复杂得多。

```text
Isaac Sim
├── Physics        PhysX / Newton
├── Rendering      RTX
├── Scene          OpenUSD
├── Sensors        RGB / Depth / LiDAR / Radar / IMU / Contact / Force
├── Robot assets   可导入 URDF、MJCF、Onshape CAD、USD
├── Synthetic Data Replicator
├── ROS 2          bridge
└── Python / GUI / OmniGraph
```

为什么比 MuJoCo 重：MuJoCo 的目标是**动力学算得快**；Isaac Sim 还要回答**真实相机看到什么、LiDAR 打到哪、光照怎么反射、材质和阴影什么样**，所以渲染本身是核心部分。

| 能力 | 说明 |
|---|---|
| **RTX 渲染** | 光照、材质、阴影、反射、depth、相机噪声；对**以 RGB 为输入的 VLA** 很重要（传统 RL 只看 joint_pos / object_pos，不需要漂亮画面） |
| **OpenUSD 场景** | Isaac Sim 是 USD-native（`.usd / .usda / .usdc`），比 MJCF 更强的 3D 世界描述格式；也能导入 MJCF、URDF |
| **Replicator** | 自动随机化相机位姿、光照、纹理、物体位置、背景、颜色，批量产出 RGB、depth、segmentation、bbox、pose |
| **ROS 2 bridge** | 真实机器人的 ROS 软件栈（camera → perception → planning → controller）可以先接到 Isaac Sim 上全部测一遍 |

### Isaac 是生态名，不是单个软件

```text
NVIDIA Isaac
├── Isaac Sim     机器人 simulator
├── Isaac Lab     robot learning framework
├── Isaac ROS     ROS / perception / deployment
└── 硬件与机器人栈
```

$$\boxed{\text{Isaac Sim}\neq\text{Isaac Lab}}$$

## 6. Isaac Lab 与 Isaac Gym 的关系

和「MuJoCo → robosuite」完全同构：

| | 负责 |
|---|---|
| **Isaac Sim** | 机器人、场景、物体、physics、camera、sensor、rendering |
| **Isaac Lab** | environment、task、observation、action、reward、reset、**parallel env**、RL / IL 训练 |

```text
Policy → Isaac Lab → Isaac Sim → PhysX / RTX
```

Isaac Lab 是 NVIDIA 官方定义的 Isaac Sim 上的 robot-learning 框架。以「抓起红色方块」为例：Isaac Sim 负责 Franka、桌子、cube、相机、光照，以及接触、摩擦、抓起的动力学；Isaac Lab 定义

```python
observation = {"joint_pos": ..., "joint_vel": ..., "cube_pos": ...}
action = joint_target
reward = cube_height
done = cube_height > threshold
```

然后 PPO 在**几千个并行环境**里训练。

### GPU 大规模并行

Isaac 生态的一个核心目标是**在 GPU 上同时跑大量环境**：

```text
GPU:  env 1 | env 2 | env 3 | ... | env 4096     (每个都是 Franka + cube)
```

传统 MuJoCo 用法多是几十 / 几百个 env，Isaac Lab 这类 GPU 仿真是几千个，思路差异很大。这也是当年 Isaac Gym 出名的原因。

### 历史：谁被谁取代

```text
Isaac Gym → OmniIsaacGymEnvs → Orbit → Isaac Lab
```

NVIDIA 当前文档已把 IsaacGymEnvs、OmniIsaacGymEnvs、Orbit 列为 deprecated，建议迁移到 Isaac Lab。新项目的认知应该是：**旧 = Isaac Gym；新 = Isaac Sim + Isaac Lab**。

## 7. MuJoCo vs Isaac Sim

| | MuJoCo | Isaac Sim |
|---|---|---|
| 谁做的 | DeepMind / Google | NVIDIA |
| 定位 | 物理引擎 | 完整机器人仿真平台 |
| Physics | 强 | 强 |
| GPU 并行 | 有现代 GPU 路线 | **核心卖点之一** |
| Rendering | 相对简单 | **RTX 高保真** |
| Sensors | 基础 | **非常丰富** |
| 复杂视觉 VLA | 能做 | **更适合** |
| LiDAR / Radar | 不主打 | **强** |
| ROS | 可以接 | **官方重点支持** |
| 场景格式 | MJCF / XML | **USD** |
| 学习框架 | robosuite 等 | **Isaac Lab** |
| 系统复杂度 | 低 | **高很多** |

**不是「Isaac 更高级所以都用 Isaac」**：

- Franka 抓 cube、只研究 RL / 控制、不关心真实视觉 → **MuJoCo 更简单**。
- 复杂机器人 + 真实 RGB + LiDAR + 复杂场景 + sim2real + ROS → **Isaac Sim 优势明显**。

## 8. 两条栈对齐

```text
                    Robot Policy / VLA
                           │
              ┌────────────┴────────────┐
              ↓                         ↓
           LIBERO                   Isaac Lab
              ↓                         ↓
          robosuite                  Isaac Sim
              ↓                         ↓
           MuJoCo                 PhysX + RTX
```

**RoboDojo 为什么需要带 RT Core 的 GPU**（例如 RTX 6000 Ada 能跑，一些没有 RT Core 的计算卡不行）：

```text
VLA → RoboDojo → Isaac Sim → RTX Camera Rendering → RT Core
```

$$\boxed{\text{需要 RT Core 的是仿真器的渲染层，不是 VLA 本身}}$$

建议的学习顺序：MuJoCo 基本概念 → robosuite → LIBERO → 跑一个 VLA rollout → 再对齐 Isaac Sim / Isaac Lab / RoboDojo。Gymnasium 的 Ant / Hopper 知道是传统 RL benchmark 即可。

## 自测

**1.** MuJoCo 是什么？它在 agent 循环里处在哪个位置？

> **答：** Multi-Joint dynamics with Contact，一个**物理仿真器**，计算 $(s_t,a_t)\to s_{t+1}$（动力学、碰撞、摩擦、接触）。在 RL 循环里它就是 environment：policy 出 action → MuJoCo 算下一状态 → 返回 camera / 本体状态。

**2.** MJCF 里 body / joint / geom / actuator / sensor 分别是什么？

> **答：** body 是刚体；joint 是关节；geom 是几何形状 / 碰撞体；actuator 是电机；sensor 是传感器。

**3.** ⭐ 「物理引擎、环境框架、任务 benchmark、算法」四层各举一例，说明 LIBERO 的依赖链。

> **答：** 引擎：MuJoCo；框架：robosuite / dm_control；benchmark：LIBERO / Meta-World / Gymnasium 任务；算法：PPO、SAC、diffusion policy、VLA。
> LIBERO → robosuite → MuJoCo。MuJoCo 不造任务，只负责物理。

**4.** 一个任务环境包含哪九项？哪一项归物理引擎？

> **答：** robot、scene、objects、initial state、observation、action、task、success condition、reward。
> 只有最底下的 physics transition $(s_t,a_t)\to s_{t+1}$ 归引擎，其余全是环境作者写的（scene.xml + reset / get_obs / step / success）。

**5.** Gymnasium MuJoCo、dm_control、robosuite、Meta-World、LIBERO 各自是谁、面向什么？

> **答：** Gymnasium（Farama）：Ant / Hopper 等经典 RL 环境。dm_control（DeepMind）：suite 提供 RL 环境、composer 自己组合场景和任务。robosuite（ARISE）：机械臂 manipulation 框架（Lift、Door、PickPlace）。Meta-World：50 个操作任务，MT1/10/50 与 ML1/10/45 划分，研究多任务和 meta-RL。LIBERO：语言条件操作 benchmark，Spatial / Object / Goal / 10 / 90 五个 suite，VLA 常用。

**6.** Isaac Sim 比 MuJoCo 多了哪些东西？为什么这对 VLA 重要？

> **答：** 除物理（PhysX / Newton）外还有 **RTX 渲染**、OpenUSD 场景、丰富传感器（RGB、depth、LiDAR、Radar、IMU、力）、Replicator 合成数据、ROS 2 bridge。
> VLA 以 RGB 图像为输入，仿真器的视觉质量直接影响训练和 sim2real；传统 RL 只看关节和物体坐标，不需要真实感画面。

**7.** Isaac Sim 和 Isaac Lab 的分工？Isaac Gym 现在是什么状态？

> **答：** Isaac Sim 负责世界（机器人、场景、物理、传感器、渲染）；Isaac Lab 负责学习环境（task、obs、action、reward、reset、并行 env、RL / IL 训练），关系同 MuJoCo 与 robosuite。
> 历史线 Isaac Gym → OmniIsaacGymEnvs → Orbit → Isaac Lab，前三者已被 NVIDIA 列为 deprecated。

**8.** Isaac 生态的 GPU 并行和传统 MuJoCo 用法差在哪？

> **答：** Isaac Lab 在 GPU 上同时跑几千个环境（如 4096 个 Franka + cube），传统 MuJoCo 用法通常是几十到几百个 env。大规模并行让 on-policy RL 的采样吞吐高得多。

**9.** 什么时候选 MuJoCo，什么时候选 Isaac Sim？

> **答：** 只研究 RL / 控制、不关心真实视觉（如 Franka 抓 cube）→ MuJoCo，轻量简单。需要真实 RGB、LiDAR、复杂场景、sim2real、ROS 集成 → Isaac Sim。不存在谁更高级，看任务需要什么。

**10.** ⭐ 跑 RoboDojo 为什么要求带 RT Core 的 GPU？

> **答：** RoboDojo 建在 Isaac 生态上，相机观测由 Isaac Sim 的 **RTX 渲染**生成，光追渲染需要 RT Core。**是仿真器的渲染层需要，不是 VLA 推理需要**。
