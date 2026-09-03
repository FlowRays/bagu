# VLA 模型：架构与训练

## 1. 演进线

| 模型 | 关键贡献 |
|---|---|
| **RT-1** | Transformer 做机器人策略，动作离散化，大规模真机数据 |
| **RT-2** | **把动作当 token 接到 VLM 上**，第一次证明网络知识能迁移到动作（"把香蕉放到德国国旗上"这类语义泛化） |
| **OpenVLA** | 开源基线，7B、Prismatic VLM + 动作 tokenization，成为大家的对照组 |
| **π0 / π0.5** | **flow matching action expert**，高频、多模态、跨本体；π0.5 强调开放世界泛化 |
| **GR00T** | 开放的人形机器人基础模型 + 合成数据管线 |

$$\boxed{\text{主线：动作从「单独的 head」变成「VLM 的一部分」，再变成「挂在 VLM 上的专用 expert」}}$$

## 2. 典型架构

```text
多相机图像 ──→ vision encoder ──┐
语言指令   ──→ tokenizer      ──┼──→  VLM backbone  ──→  hidden
本体状态   ──→ proj           ──┘                          │
                                                           ▼
                                                    action expert
                                              （flow / diffusion / token head）
                                                           │
                                                           ▼
                                                  未来 H 步动作 chunk
```

三个设计位：

| 位置 | 常见做法 |
|---|---|
| **视觉** | 多相机（第三人称 + **腕部**），历史几帧；沿用 VLM 的 encoder |
| **本体状态注入** | 投影成若干 token 拼进序列，或作为 action expert 的额外条件 |
| **动作解码** | 见 [action head](../05-action/02-action-head.md) |

## 3. VLM backbone 怎么处理

| 做法 | 说明 |
|---|---|
| 全参训练 | 效果好但容易灾难性遗忘语义能力 |
| **小 lr + co-training** | 主流：backbone 用小 lr，同时混入网络图文数据防遗忘 |
| 冻结 backbone | 保语义但适配能力弱 |

**co-training** 是关键：只用机器人数据微调会让 VLM 迅速丢掉语言和视觉常识（这正是 RT-2 想利用的东西）。混入原始图文数据，本质是 [灾难性遗忘的 replay 手段](../../notes/06-post-training/sft/01-sft-basics.md#6-灾难性遗忘)。

## 4. 训练范式：两阶段

$$\boxed{\text{大规模跨本体预训练}\ \to\ \text{特定本体 / 任务后训练}}$$

**预训练**：Open X-Embodiment 这类混合了几十种机器人的数据，学通用的视觉-语言-动作对应关系。
难点是**动作空间不统一**（自由度、量纲、坐标系都不同），常见做法是统一到末端空间 + 归一化，或者给每个 embodiment 一个 embedding。

**后训练**：在目标本体的少量数据上微调，让动作分布对上具体硬件。

这个结构和 LLM 的 `pretrain → SFT` 完全同构，区别是这里的"预训练"数据也只有几千小时级别。

## 5. 数据配比

典型的金字塔：

```text
网络图文（语义、常识）        —— 最多，最便宜
     ↓
人类视频（动作先验、物理直觉）  —— 多，需要 latent action 之类的手段利用
     ↓
仿真数据（可无限生成）         —— 中，有 sim2real gap
     ↓
真机遥操作（最贴近部署分布）    —— 最少，最贵
```

配比是要调的超参。经验和 [SFT 数据配比](../../notes/06-post-training/sft/02-data-and-cot.md#6-数据配比) 一致：想要的能力数据占比要够，但某类过高会挤掉别的。

## 6. 推理效率

见 [chunking 与实时性](../05-action/03-chunking-and-realtime.md)。要点：

- **action chunk** 摊薄推理成本（最重要）
- **小 action expert**：大 backbone 跑一次，小 expert 跑 K 步
- **flow matching** 代替 diffusion，K 从 50 降到 4–10
- **异步推理**隐藏延迟，注意对齐延迟偏移
- 量化、蒸馏、视觉特征缓存

## 7. 面试一分钟版

> VLA 就是把 VLM 当 backbone、在上面接一个动作解码器的架构。RT-2 最早证明了把动作当 token 接到 VLM 上能让网络知识迁移到操作上；但离散 tokenization 精度和速度都受限，所以 π0 这一代改成在 VLM 上挂一个小的 **flow matching action expert**，一次输出未来几十步的动作 chunk。
>
> 训练是两阶段：先在 Open X-Embodiment 这类跨本体数据上预训练，再在目标本体上后训练；全程 co-train 网络图文数据防止 VLM 的语义能力被冲掉。
>
> 三个核心工程约束是：**动作多模态**（所以要生成式 head）、**实时性**（所以要 chunk + 小 expert + flow）、**数据极贵**（所以要跨本体预训练和用无标注视频）。

## 自测

**1.** RT-2 的关键贡献是什么？

> **答：** **把动作当 token 接到 VLM 上**，第一次证明网络预训练的知识能迁移到动作上（能理解"把香蕉放到德国国旗上"这种训练集里没有的语义组合）。

**2.** 画出 VLA 的典型架构，指出三个设计位。

> **答：** `多相机图像 → vision encoder` + `语言 → tokenizer` + `本体状态 → proj` → **VLM backbone** → hidden → **action expert** → 未来 H 步动作 chunk。
> 三个设计位：**视觉**（多相机含腕部、历史帧）、**本体状态注入**（投影成 token 拼进序列或作 expert 的额外条件）、**动作解码**（回归/token/diffusion/flow）。

**3.** VLM backbone 怎么处理？co-training 在解决什么？

> **答：** 三种：全参训练（效果好但灾难性遗忘）、**小 lr + co-training**（主流）、冻结（保语义但适配弱）。
> **co-training** 解决的是：只用机器人数据微调会让 VLM 迅速丢掉语言和视觉常识 —— 而那恰恰是 RT-2 想利用的东西。混入原始图文数据本质就是 replay。

**4.** 两阶段训练是什么？预训练的难点在哪？

> **答：** **大规模跨本体预训练 → 特定本体/任务后训练**，和 LLM 的 `pretrain → SFT` 同构。
> 预训练的难点是**动作空间不统一**（自由度、量纲、坐标系都不同），常见做法是统一到末端空间 + 归一化，或给每个 embodiment 一个 embedding。

**5.** 数据金字塔是什么？

> **答：** 从多到少、从便宜到贵：**网络图文**（语义常识）→ **人类视频**（动作先验，需要 latent action 之类手段利用）→ **仿真**（可无限生成但有 gap）→ **真机遥操作**（最贴近部署分布，最贵）。

**6.** VLA 的三个核心工程约束，各自导出什么设计？

> **答：** ① **动作多模态** → 必须生成式 head（不能回归）；② **实时性**（10–50 Hz） → action chunk + 小 expert + flow matching 代替 diffusion；③ **数据极贵** → 跨本体预训练、用无标注视频、仿真。
