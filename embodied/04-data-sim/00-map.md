# 数据与仿真总图

| 篇 | 内容 |
|---|---|
| [01 数据与 sim2real](01-data-and-sim2real.md) | 四类数据源的取舍、遥操作的质量问题、用人类视频的三条路线、仿真器、DR vs sysid、和 LLM 数据的对照 |
| [02 数据类型与采集](02-data-types.md) | UMI / Ego / Teleop / Leader-follower / MoCap / 仿真 / 神经合成 / Play 不是同一维度、数据金字塔、UMI 为什么只有腕部相机、IMU、采集演示视频 |
| [03 仿真器与环境栈](03-simulators.md) | 物理引擎 ≠ 任务环境 ≠ 算法、Gymnasium / dm_control / robosuite / Meta-World / LIBERO、Isaac Sim vs Isaac Lab、RoboDojo 为什么要 RT Core |

$$\boxed{\text{核心矛盾：有动作标注且贴近部署的数据恰恰最贵}}$$

具身的几乎全部数据技术，都是在想办法**用便宜数据替代真机遥操作数据**。
