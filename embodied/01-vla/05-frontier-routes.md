# 前沿 VLA 路线：DM0.5、小米、RoboDojo 前排

> 2026-09 按公开资料整理，榜单为 RoboDojo Simulation Leaderboard 2026-09-10 快照。模型更新很快，具体数字以各家原始报告为准。
> 前置：[π0 / π0.5 / G0](03-pi0-pi05-g0.md)（连续 flow expert 这一系）、[G0.5 架构与源码](04-g05-architecture.md)（离散 action token 这一系）、[具身数据类型](../04-data-sim/02-data-types.md)（UMI / Ego 到底是什么）。

**一句话：action head 已经基本收敛，大多数前排模型都是「VLM + flow matching 生成连续 action chunk」。真正拉开差距的是三件事：数据怎么 scale、要不要显式做长历史和具身推理、有没有把世界模型当成独立的数据引擎。**

另外还有一条贯穿全篇的分叉：action 到底怎么表示。

$$\text{VLM latent}\rightarrow\text{Flow Matching Action Expert}\qquad\text{vs}\qquad\text{VLM}\rightarrow\text{ActionCodec}\rightarrow\text{离散 action token}$$

前者是 π0 / DM0.5 / 小米 / 腾讯 Hy 这一系，后者是 G0.5。问题的本质是：action 应该一直作为挂在 VLM 上的特殊连续 head，还是最终像文本、图像一样成为 foundation model 的原生模态。

## 1. DM0.5：让 policy 本身更像智能体

DM0.5 来自 Dexmal（原力灵机）。

### 架构：标准的 π0 系

| 模块 | 配置 |
|---|---|
| VLM | Gemma3，约 4B |
| Action module | 约 680M Action Expert |
| 动作 | 连续 action chunk |
| 生成方式 | Flow Matching |
| 组织方式 | VLM prefix → Action Expert suffix |

公开代码的结构就是 `Gemma3ForConditionalGeneration + DM05ActionExpert`：动作投影进 Action Expert，加时间条件，用 flow matching 输出连续动作。**action tokenizer 不是它的核心路线。**

```text
当前 observation + 历史 observation + instruction + robot state
        ↓
   Gemma3 VLM  → 语义 / 历史表示
        ↓
   Action Expert (flow matching)
        ↓
   action chunk
```

### 真正的创新在三处

它围绕「VLA 在真实世界为什么会挂」做了三个增强。

**① 长历史 / 记忆。** 不再只看当前帧。**Context Abstraction Layer** 把历史视觉压缩成 token，据公开介绍最长覆盖约 **60 秒**。例子是「拿起杯子 → 擦桌子 → 把杯子放回原位」：擦完桌子时原位置可能已经不在当前画面里，只能靠过去的信息。

**② 具身推理（Embodied Reasoning）。** 不只训练 $(o_t,\text{instruction})\rightarrow a_t$，还在机器人数据上加入多类**自回归**具身任务：任务阶段、环境变化、未来事件、动作意图。VLM 承担理解、规划、预测，Action Expert 负责低层连续控制。

**③ 动态轨迹匹配。** 不同遥操作员完成同一动作的速度不一样。如果按 timestamp 硬对齐 $a_t^{\text{pred}}\leftrightarrow a_t^{\text{demo}}$，等于把「这个人操作得快还是慢」当成了 ground truth。DM0.5 把预测轨迹和 demo 做**单调匹配**，关注任务进度和动作语义是否一致，而不是逐时刻一致。

### 数据：榜首里最明确的 Ego 案例

$$50\text{K h robot}\ +\ 100\text{K h egocentric}$$

另有约 **100 万平方米场景重建数据**。这里的 Ego 被明确描述为**人类原生第一人称视角**，不是机器人腕部相机。

$$\boxed{\text{DM0.5 = VLA + Memory + Reasoning + 更好的动作对齐}}$$

## 2. 小米：按 foundation model 的方式 scale

小米不能只看一个模型，它是两条互补的线：

| 模型 | 角色 |
|---|---|
| **Xiaomi-Robotics-1** | Policy Foundation Model |
| **Xiaomi-Robotics-U0** | World Foundation Model / Data Engine |

### Robotics-1：架构传统，创新在数据 recipe

架构是 **Mixture-of-Transformers 风格的 VLM + DiT**：

- Qwen3-VL（2B / 5B / 10B 多档）编码图像和语言
- 小型 DiT 读取 **VLM 的 KV cache + proprioception**
- Flow Matching 生成 action chunk，推理用 **5 步 Euler 积分**

核心思想是：**不要一开始就在每种机器人上收数据。**

```text
Pretrain：100K+ h UMI
    人拿 UMI 装置在真实世界操作，环境和任务多样，但不绑定具体本体
    强 VLM 自动给 clip 打标：「夹爪把杯子从桌上移到架子上」
    学的是 (s_t, 期望的状态转移) → action
        ↓
Post-train：10K+ h 跨本体数据（其中 7,200+ h 真机，另有高质量 UMI 与 Bridge / RT-1 / DROID 等）
    对齐 UMI action → robot action
    对齐「状态转移描述」→ 祈使句指令
```

监督从简单的任务名变成了**语言描述的状态转移**。这和 LLM 的套路完全同构：

$$\boxed{\text{Web-scale pretraining}\rightarrow\text{instruction tuning}}\quad\Longleftrightarrow\quad\boxed{\text{UMI-scale pretraining}\rightarrow\text{本体 / 指令后训练}}$$

### Robotics-U0：世界模型负责 scale 数据

U0 是一个 **38B 自回归世界基础模型**，出发点是通用图像 / 视频生成，而不是机器人 policy。它统一做：

- Text → Image
- Image → Image 编辑
- Text → 多视角机器人场景
- 场景 → 具身迁移
- observation + 语言 / 动作 → 未来视频

经过具身持续训练后，通用视觉基础模型变成机器人世界模型和数据生成器。一个关键用法是**增广已有真机 demo**：重新生成背景、光照、无关物体、目标物体、工作台，同时**保持机器人位姿和交互状态**，一条真实数据变出大量 OOD 变体。

官方实验：把 U0 生成的增广加进 policy 后训练，在 held-out 背景 / 光照干扰设置下，平均成功率 **36.9% → 63.2%**。

### 小米的飞轮

```text
   100K+ h UMI                     通用图像/视频基础模型
        ↓                                   ↓
 Xiaomi-Robotics-1 (policy)        Xiaomi-Robotics-U0 (world model)
        ↓                                   ↓
   真机部署 → 真机数据 ─────────→   场景生成 / 迁移 / rollout
        ↑                                   ↓
        └──────── 合成具身数据 ←────────────┘
```

$$\boxed{\text{小米 = Policy Foundation Model + World Foundation Model + Data Scaling}}$$

## 3. 两家对比

| 问题 | DM0.5 | Xiaomi |
|---|---|---|
| VLA 架构 | VLM + Action Expert | VLM + DiT |
| 连续控制 | ✅ Flow Matching | ✅ Flow Matching |
| 长历史 | **重点** | 不是当前最核心创新 |
| 具身推理 | **重点** | 有，但不是论文主轴 |
| 失败 / 干扰鲁棒性 | **重点** | 更多靠 scale、后训练、增广 |
| 大规模 UMI | 有多源数据，但不是最鲜明的主张 | **核心：100K+ h** |
| Cross-embodiment | ✅ | **后训练核心** |
| Scaling law | 次要 | **核心问题** |
| World Model | 暂时不是主线 | **U0 独立主线** |
| 合成数据引擎 | 较弱 | **非常明确** |
| Policy ↔ WM 飞轮 | 尚不明显 | **已经形成路线** |

两家在回答不同的问题：

$$\boxed{\text{DM0.5：How do we make VLA itself smarter?}}$$

$$\boxed{\text{Xiaomi：How do we scale embodied foundation models?}}$$

## 4. RoboDojo 前 10 的技术路线

2026-09-10 快照，前三综合分分别是 **24.90 / 20.23 / 20.07**。

| 排名 | 模型 | 核心技术路线 | Ego / UMI 数据 |
|---|---|---|---|
| 1 | **DM0.5** | VLM + Flow Action Expert + 长记忆 + 具身推理 | ✅ 明确大量 Ego |
| 2 | **G0.5** | 统一 AR VLA：VLM 直接生成推理 + ActionCodec token | ❌ 未把 Ego 作为主要数据路线 |
| 3 | **Xiaomi-Robotics-1** | 100K h UMI 预训练 → 机器人后训练；VLM + Flow DiT | ✅ UMI（广义 Ego） |
| 4 | **OpenWAM-α** | Video DiT 世界模型 + ActionDiT，联合 diffusion / flow | ✅ 明确人类 egocentric |
| 5 | **Meituan-Robotics-0** | 公开技术材料非常有限 | ？无法可靠确认 |
| 6 | **Hy-Embodied-0.5-VLA** | MoT VLM + Flow Action Expert + Memory + FlowPRO | ✅ 大规模 egocentric UMI |
| 7 | **Spatial Forcing** | π0 / OpenVLA + VGGT 3D 特征对齐 | ❌ 不是 Ego scaling 路线 |
| 8 | **π0.5** | VLM + 高层 AR 推理 + Flow Action Expert | 原版 ❌；PI 后续有单独的 π0.5 + ego 研究 |
| 9 | **InternVLA-A1.5** | VLM + Flow Expert + 视频生成器监督的 latent foresight | ⚠️ 数据池包含 UMI |
| 10 | **VLAct** | Qwen3-VL 继续预训练 + 多种 action head | ❌ 主要是机器人轨迹 |

DM0.5 和 Xiaomi-Robotics-1 见上面第 1、2 节。下面逐个看其余模型。

### G0.5：重点不是 Ego，而是统一 action token

前排里最另类的一个。不是 VLM → Action Expert，而是**单个自回归 Transformer**：

$$\text{Image + Language + State}\rightarrow\boxed{\text{single AR transformer}}\rightarrow\text{Subtask / BBox / Trace / ActionHint}\rightarrow\text{ActionCodec tokens}$$

ActionCodec 用 **RVQ 类的学习式 action tokenizer**，把 **14 种本体**的异构动作统一到共享的 action 词表。推理和动作用**同一个 next-token prediction loss**。

数据是机器人演示 + web VQA + embodied QA，公开报告没有把大规模人类 egocentric 预训练作为核心数据源。它赌的是 **action 模态本身统一进 foundation model**，而不是去找 10 万小时 Ego 扩数据。细节见 [G0.5](04-g05-architecture.md)。

### OpenWAM-α：Ego → dynamics → action

这是真正的 **WAM**，不是普通 VLA：

$$\text{Wan2.2 Video DiT}\ +\ \text{ActionDiT}$$

- world stream 预测未来观测 $o_{t+1:t+H}$
- action stream 预测动作 $a_{t:t+H}$
- 两个 DiT 通过 **joint self-attention / mutual visibility** 互相读取，**同步去噪**

数据共 **6369 h**，其中 **70% robot + 30% human egocentric**，也就是约 **1900 h** 人类 Ego。作者专门做实验得出：**一阶段把人类 egocentric 数据和机器人数据一起 co-train**，有利于把 world coverage 和 action grounding 结合起来。

它不是用 Ego 让 VLM「看懂物体」，而是：

$$\text{Ego video}\rightarrow\text{学 world dynamics}\rightarrow\text{World Stream}\leftrightarrow\text{Action Stream}$$

### Hy-Embodied-0.5-VLA：明确的 UMI 派

腾讯出品，架构：

$$\text{Hy-Embodied MoT}\ +\ \text{Flow Matching Action Expert}\ +\ \text{Memory Encoder}$$

再加 **delta-action chunk** 和 **FlowPRO** 后训练。

数据：**10,000+ h UMI**，官方直接称为 egocentric UMI data，采集用自研**指尖接口 + 光学动作捕捉**，目前开放约 **2,000+ h**。

### Spatial Forcing：补 3D 表示，不是拼数据

出发点：普通 VLM 的 2D 视觉表示，对机器人来说 3D 几何太差。做法是拿 **VGGT** 这类预训练 3D 基础模型当老师，做中间层特征对齐：

$$f_{\text{VLA}}(\text{image})\approx f_{\text{VGGT}}(\text{image})$$

底座是 π0 / OpenVLA。

$$\boxed{\text{Spatial Forcing 是 representation alignment，不是 data scaling}}$$

它背后的判断是：问题可能不只是 VLM 没见过机器人数据，而是 **VLM latent 本身缺好的空间结构**。

### π0.5：原版不是 Ego 路线

原版 π0.5 的数据混合：移动操作、多环境机器人数据、跨本体机器人数据、web 图像 / VQA、高层语义标签、口头指令。其中约 **400 h** 移动操作数据分散在约 **100 个家庭**环境。

但 PI 后来专门做了 **human-to-robot transfer** 研究，直接对比 π0.5 与 **π0.5 + ego**。比如 UR5 迁移的 bussing 任务上，人类 Ego 数据确实带来明显迁移收益，**但仍不如目标本体数据**。

$$\boxed{\text{Human Ego}\approx\text{一种新的 cross-embodiment 数据源}}$$

### InternVLA-A1.5：在 latent 里学 dynamics

$$\text{VLM}\ +\ \text{Unified Action Expert}\ +\ \text{Foresight Tokens}$$

训练时拿一个**冻结的预训练视频生成器**当老师，让 foresight latent 对齐未来视频的语义；**推理时直接把视频生成器扔掉**，只保留学到的 latent foresight。所以它不是纯 WAM，推理不生成视频。

数据：**1.2M robot episodes + 3M 多模态**，真机数据池含 AgiBotWorld、UMI、DROID、Galaxea、RoboMind。算是「部分吃到了 UMI」，但 UMI scaling 不是整篇工作的主张。

### VLAct 与 Meituan-Robotics-0

- **VLAct**：Qwen3-VL 继续预训练 + 多种 action head，数据主要是机器人轨迹。
- **Meituan-Robotics-0**：公开技术材料非常有限，暂时无法可靠判断路线和数据，不做臆测。

## 5. 三种「第一视角」必须分开

说「用了 ego 数据」时，至少有三种完全不同的东西：

| 类型 | 相机在哪 | 谁在操作 | 有没有动作 | 代表 |
|---|---|---|---|---|
| **人类 Ego** | 人头上（头戴相机 / 眼镜） | 人用自己的手 | ❌ 没有机器人动作 | DM0.5、OpenWAM-α |
| **UMI** | 手持夹爪装置上（夹爪中心的 ego） | 人拿着类似末端执行器的采集器 | ⚠️ 位姿 + 开合，是 proxy action | Xiaomi-Robotics-1、Hy-VLA |
| **机器人自身相机** | 机器人头部 / 腕部 | 机器人（遥操作或自主） | ✅ 真实机器人动作 | 普通 teleop 数据 |

UMI 的相机更接近「夹爪视角」，和人戴头盔拍自己手的 Ego 视频携带的信息不一样，两者 scale 的东西也不一样。数据类型的完整对照见 [具身数据类型](../04-data-sim/02-data-types.md)。

## 6. 总结：四个派系

| 派系 | 代表 | 押注 |
|---|---|---|
| **Ego + memory / reasoning** | DM0.5 | 让 policy 本身更聪明：长历史、具身推理、对齐 |
| **UMI scaling + Flow VLA** | Xiaomi-Robotics-1、Hy-Embodied-0.5-VLA | 便宜的人类操作数据堆规模，再对齐到机器人 |
| **Action token / 统一 AR** | G0.5 | action 作为原生 token 模态进 foundation model |
| **Video dynamics / world prior 引入 action** | OpenWAM-α、InternVLA-A1.5 | 从视频里学物理动力学，再接到动作 |

Ego 类数据的使用者分两类：

- **通用人类 Ego**：DM0.5（100K h）、OpenWAM-α（30%，约 1900 h）
- **UMI / 与本体无关的人类操作数据**：Xiaomi-Robotics-1（100K+ h）、Hy-Embodied-0.5-VLA（10,000+ h）

$$\boxed{\text{趋势：纯机器人轨迹不再是唯一主线，Human / UMI 数据正在成为具身基础模型的预训练数据源}}$$

## 7. 面试一分钟版

> 现在前排 VLA 的 action head 基本收敛到「VLM + flow matching 生成 action chunk」，差距主要拉在数据和系统层面。
>
> DM0.5 走的是让 policy 本身更聪明：Gemma3 加 680M action expert，重点是约 60 秒的历史压缩、具身推理任务、以及用单调轨迹匹配代替按时间戳硬对齐；数据上是 50K 小时机器人加 100K 小时人类第一人称视频。
>
> 小米走的是 foundation model scaling：Robotics-1 先用 100K+ 小时 UMI 预训练、用 VLM 自动标状态转移描述，再用 10K+ 小时跨本体数据后训练，类比 pretrain → instruction tuning；另有 38B 的 U0 世界模型当数据引擎，改背景光照做增广。
>
> 另一条分叉是 action 表示：G0.5 用 RVQ ActionCodec 把 14 种本体的动作变成共享 token，推理和动作同一个 next-token loss。还有 OpenWAM 这种 Video DiT + ActionDiT 联合去噪的 WAM 路线。
>
> 讲 ego 数据时要分清：人类头戴 Ego、UMI 夹爪视角、机器人自身相机，三者离真实机器人动作的距离完全不同。

## 自测

**1.** 为什么说前排 VLA 的 action head 已经收敛？那差距拉在哪？

> **答：** DM0.5、Xiaomi-Robotics-1、Hy-Embodied-0.5-VLA、π0.5、InternVLA-A1.5 基本都是「VLM + flow matching 生成连续 action chunk」。
> 差距主要在三处：① **数据怎么 scale**（Ego / UMI / 真机 / 合成）；② 是否显式做**长历史和具身推理**；③ 是否把**世界模型当独立的数据引擎**。另有一条表示层分叉：连续 flow expert vs G0.5 的离散 ActionCodec token。

**2.** DM0.5 的架构和三个核心增强？

> **答：** 架构：Gemma3 ~4B + ~680M Action Expert，flow matching 输出连续 action chunk（VLM prefix → expert suffix）。
> 三个增强：① **Context Abstraction Layer** 把历史视觉压成 token，最长约 60 秒；② **具身推理**：在机器人数据上加任务阶段、环境变化、未来事件、动作意图等自回归任务；③ **动态轨迹匹配**：预测轨迹和 demo 做单调匹配。

**3.** ⭐ 为什么不能按 timestamp 硬对齐预测动作和 demo？

> **答：** 不同遥操作员完成同一动作的速度不同。按时间戳逐点对齐 $a_t^{\text{pred}}\leftrightarrow a_t^{\text{demo}}$，等于把「这个操作员快还是慢」当成 ground truth 去监督，模型要去拟合与任务无关的速度差异。单调匹配关注任务进度和动作语义一致，而不是逐时刻一致。

**4.** Xiaomi-Robotics-1 的训练 recipe？和 LLM 的什么过程同构？

> **答：** 架构：Qwen3-VL（2B/5B/10B）+ 小 DiT，DiT 读 VLM 的 KV cache 和本体状态，flow matching，推理 5 步 Euler。
> Recipe：先用 **100K+ h UMI** 预训练，由强 VLM 自动标注「状态转移描述」，学 $(s_t,\text{状态转移})\rightarrow a$；再用 **10K+ h 跨本体数据**（含 7,200+ h 真机）后训练，对齐 UMI action → robot action、状态转移描述 → 祈使指令。
> 同构于 **web-scale pretraining → instruction tuning**。

**5.** Xiaomi-Robotics-U0 是什么？怎么帮 policy？

> **答：** 38B 自回归世界基础模型，从通用图像 / 视频生成出发，统一做文生图、图像编辑、多视角机器人场景生成、具身迁移、根据观测和语言 / 动作预测未来视频。
> 用法之一是增广真机 demo：换背景、光照、物体、工作台，同时保持机器人位姿和交互状态。加进后训练后，held-out 背景 / 光照干扰设置下平均成功率 36.9% → 63.2%。

**6.** ⭐ 「人类 Ego」「UMI」「机器人腕部相机」三者有什么区别？

> **答：** **人类 Ego**：相机在人头上，人用自己的手操作，**没有机器人动作**，但最容易 scale（DM0.5、OpenWAM-α）。
> **UMI**：人手持带相机和位姿追踪的夹爪装置，相机是夹爪中心视角，有位姿 + 开合的 **proxy action**（Xiaomi-Robotics-1、Hy-VLA）。
> **机器人自身相机**：机器人头部 / 腕部，配合遥操作或自主执行，有**真实机器人动作**。三者离 robot action 的距离依次变近。

**7.** OpenWAM-α 的架构和数据结论？

> **答：** Wan2.2 Video DiT（world stream，预测未来观测）+ ActionDiT（action stream，预测动作），两个 DiT 通过 joint self-attention 互相可见、同步去噪。
> 数据 6369 h，70% robot + 30% human egocentric（约 1900 h）。结论：一阶段把人类 egocentric 和机器人数据 co-train，能把 world coverage 和 action grounding 结合起来。

**8.** Spatial Forcing 和 InternVLA-A1.5 分别在补 VLA 的什么？

> **答：** **Spatial Forcing** 补 **3D 表示**：用 VGGT 当老师做中间特征对齐 $f_{\text{VLA}}(\text{image})\approx f_{\text{VGGT}}(\text{image})$，底座是 π0 / OpenVLA，属于表示对齐而非数据 scaling。
> **InternVLA-A1.5** 补 **dynamics / 预见**：加 foresight tokens，训练时用冻结的视频生成器监督其对齐未来视频语义，推理时扔掉视频生成器，只保留 latent foresight。

**9.** π0.5 用了 Ego 数据吗？

> **答：** **原版没有。** 原版数据是移动操作（约 400 h、约 100 个家庭）、多环境与跨本体机器人数据、web 图像 / VQA、高层语义标签和口头指令。
> PI 后续的 human-to-robot transfer 研究对比了 π0.5 与 π0.5 + ego：人类 Ego 数据有明显迁移收益，但仍不如目标本体数据。说明人类 Ego 可以看作一种新的 cross-embodiment 数据源。

**10.** RoboDojo 前排可以归成哪四个派系？

> **答：** ① **Ego + memory / reasoning**：DM0.5；② **UMI scaling + Flow VLA**：Xiaomi-Robotics-1、Hy-Embodied-0.5-VLA；③ **Action token / 统一 AR**：G0.5；④ **Video dynamics / world prior 引入 action**：OpenWAM-α、InternVLA-A1.5。
> 另外 Spatial Forcing 是 3D 表示对齐，Meituan-Robotics-0 公开材料有限无法归类。
