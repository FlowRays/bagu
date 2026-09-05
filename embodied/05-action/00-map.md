# Action 总图

> **VLA 和 LLM 差别最大的一段。** LLM 输出离散 token，机器人输出连续、有物理量纲、有安全后果、还必须实时的动作。

| 篇 | 内容 |
|---|---|
| [01b 动作空间从零](01b-action-space-from-zero.md) | **机器人前置**：DoF / 关节与连杆、FK 与 IK、四种旋转表示手算、**维度不等于自由度** |
| [01 动作空间与表示](01-action-space.md) | 关节 / 末端 / 高层三层、**旋转为什么要 6D 表示**、delta vs absolute、归一化、wrist camera |
| [02 Action head](02-action-head.md) | 回归 / 离散 tokenization / diffusion / **flow matching** 四种做法的取舍、FAST、action expert |
| [03 Chunking 与实时性](03-chunking-and-realtime.md) | ACT 的 action chunking、temporal ensembling、三个频率、异步推理的延迟对齐 |

## 一条主线

$$\boxed{\text{动作分布是多模态的} \Rightarrow \text{必须生成式建模} \Rightarrow \text{但要实时} \Rightarrow \text{flow matching + chunk + 小 expert}}$$

整条 VLA 动作侧的设计，几乎都是这一句话推出来的：

1. **为什么不能回归**：MSE 的最优解是条件均值，绕障碍的"平均动作"是直接撞上去。
2. **那就生成式**：diffusion 能建模多模态，但要 50 步。
3. **50 步跑不动**：机器人 10–50 Hz 闭环，预算只有 20–100 ms。
4. **所以 flow matching**：直线路径，1–10 步。
5. **还是不够**：大 VLM 跑 K 次太贵 → **小 action expert** 只跑那 K 步。
6. **还是不够**：一次推理只出一个动作太浪费 → **chunk** 一次出 H 步，把推理频率和控制频率解耦。

## 三个高频面试题

**Q：为什么 VLA 要用 diffusion / flow 而不是直接回归动作？**
多模态。MSE 输出条件均值，两个合理模式的平均可能哪个都不是。

**Q：action chunking 解决什么？**
① compounding error（决策点少 H 倍）；② 多模态抖动（chunk 级一致）；③ 实时性（推理频率和控制频率解耦）。

**Q：旋转怎么表示？**
6D（旋转矩阵前两列 + Gram-Schmidt）。因为 $SO(3)$ 到 $\le4$ 维实空间不存在连续双射，欧拉角和四元数必然在某处不连续或多值，回归会撕裂。

## 相关

- [视觉生成](../02-visual-generation/00-map.md) — diffusion 和 flow matching 的推导
- [VLA 模型](../06-vla/00-map.md) — 这些 head 装在什么 backbone 上
- [推理基础](../../notes/02-inference-serving/01-inference-basics.md) — LLM 侧的延迟/吞吐对照
