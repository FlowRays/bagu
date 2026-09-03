# 蒸馏主线：SFT → KD → OPD → OPSD（总图 + 卡点索引）

> 这一目录讲 **on-policy distillation（OPD）和 on-policy self-distillation（OPSD）**，以及它们依赖的信息论地基（熵 / 交叉熵 / KL / JSD）。
> 全篇只围绕一个问题展开：**teacher 到底在什么状态上、用什么方式、给 student 多少监督。**

## 一、最重要的一句话

大量八股材料会把 OPD 直接说成"reverse KL 蒸馏"，这是**错的**，而且是面试里最容易被追问翻车的地方：

$$\boxed{\text{OPD}\ne\text{reverse KL}}$$

正确的拆法是**三个互相正交的维度**：

```text
维度 1：prefix / state 从哪来？        ← 这个才是 "on-policy" 的定义
        teacher/dataset  vs  student rollout

维度 2：在这个 state 上怎么对齐两个分布？
        forward KL  vs  reverse KL  vs  JSD

维度 3：这个 divergence 用几个 token 来估？
        sampled-token  vs  top-k  vs  full-vocabulary
```

**On-policy 只管维度 1。** 所以 "on-policy + forward KL" 完全成立，而且最新 OPSD 论文的主实验用的正是 forward KL。

## 二、维度 1 × 维度 2：四象限表

| | Forward KL $D_{KL}(T\|S)$ | Reverse KL $D_{KL}(S\|T)$ |
|---|---|---|
| **teacher/data prefix**（off-policy） | 传统 logit KD；hard-label 特例就是 **SFT** | off-policy reverse-KL KD（少见） |
| **student prefix**（on-policy） | **GKD / OPSD 主 recipe** | **Thinking Machines 的 OPD recipe** |

再把"target 到底长什么样"摊开，就是最适合背的一张表：

| prefix | target | 叫什么 |
|---|---|---|
| teacher/data state | hard GT token | **SFT** |
| teacher/data state | teacher soft distribution | **普通 logit KD** |
| **student state** | teacher soft distribution | **Forward-KL OPD** |
| **student state** | student 采样的 action + teacher 打分 | **Reverse-KL OPD** |

## 三、两层采样（最值得记的一个区分）

任何一个蒸馏 loss 都能写成两层期望：

$$\mathcal L=\underbrace{\mathbb E_{s\sim d^{?}}}_{\text{prefix 谁生成}}\Big[\ \underbrace{\mathbb E_{a\sim ?}}_{\text{KL 期望对谁取}}[\cdots]\ \Big]$$

- Reverse-KL OPD：$s_t\sim d^{\pi_S},\ a_t\sim\pi_S$ — **两层都是 student**，所以和 RL pipeline 天然同构
- Forward-KL OPD：$s_t\sim d^{\pi_S},\ a_t\sim\pi_T$ — **状态是 student 的，监督方向是 teacher → student**

以后看到任何一篇蒸馏论文，先问两个问题就能定位它：

$$\boxed{1.\ \text{prefix 谁生成？}}\qquad\boxed{2.\ \text{KL 是 }T\|S\text{ 还是 }S\|T?}$$

## 四、方法时间线

$$\text{GKD (ICLR'24)}\rightarrow\text{Thinking Machines OPD}\rightarrow\text{OPSD (2026.01)}\rightarrow\text{Rethink OPD (2026.04)}\rightarrow\text{MOPD}\rightarrow\text{U-OPSD (2026.08)}$$

| 方法 | 一句话 |
|---|---|
| GKD | 第一次把 divergence（forward / reverse / JSD）做成可选项，配 student-generated output |
| Thinking Machines OPD | sampled-token reverse KL，$A_t=\log\pi_T-\log\pi_S$ 直接当 per-token advantage 塞进 RL trainer |
| OPSD | 去掉外部 teacher，用**同一个模型 + privileged GT context** 当 teacher；主实验 full-vocab forward KL |
| Rethink OPD | 解释 OPD 何时有效：teacher 强 ≠ 好蒸；要 thinking pattern compatible **且** 真有新能力 |
| MOPD | 多个 domain RL specialist teacher → 蒸回一个统一 student，把"能力生产"和"能力集成"解耦 |

## 五、卡点索引

| # | 卡点 | 在哪 |
|---|---|---|
| 1 | 熵 / 交叉熵 / KL 三者到底什么关系 | [01](01-entropy-ce-kl-jsd.md#3-交叉熵--熵--kl最该背的一条恒等式) |
| 1b | $\delta_{vk}$ 是什么、softmax 的导数为什么是个矩阵 | [01b](01b-softmax-gradient.md#1-为什么会出现两个下标-v-和-k) |
| 2 | 为什么 CE 对 logits 的梯度就是 $p-q$ | [01](01-entropy-ce-kl-jsd.md#5-一个统一的梯度logits-梯度等于-p-减-q) |
| 3 | JSD 是什么，为什么要绕一个 mixture $m$ | [01](01-entropy-ce-kl-jsd.md#6-jsd把两个-kl-的爆炸都拆掉) |
| 4 | SFT 为什么可以看成 forward KL / 交叉熵 | [02](02-sft-and-kd.md#2-sft-其实是-one-hot-cross-entropy) |
| 5 | 从 teacher 采样做 SFT 和 soft CE 是什么关系 | [02](02-sft-and-kd.md#4-从-teacher-采样做-sft--forward-kl-的蒙特卡洛估计) |
| 6 | OPD 的定义到底是什么（不是 reverse KL） | [03](03-opd.md#1-定义on-policy-说的是-state-distribution) |
| 7 | forward-KL OPD 会不会退化成 SFT | [03](03-opd.md#卡点forward-kl-opd--sft-吗) |
| 8 | "OPD 只能强化已有能力"这句话错在哪 | [03](03-opd.md#5-support-这个词要小心用) |
| 9 | 为什么 reverse KL 的梯度**就是** policy gradient | [04](04-reverse-kl-as-pg.md#2-把梯度真正推一遍) |
| 10 | $A_t=\log\pi_T-\log\pi_S$ 到底是什么含义 | [04](04-reverse-kl-as-pg.md#3-这个-advantage-到底在说什么) |
| 11 | 为什么说 KL-regularized RL 本身就是 reverse KL | [04](04-reverse-kl-as-pg.md#6-更深的一层llm-rl-本来就可以写成-reverse-kl) |
| 12 | 整个词表算 KL 不是很贵吗，有没有 top-k | [05](05-kl-estimation.md#3-full-vocabulary贵在哪里) |
| 13 | sampled-token 的 loss 值和实际梯度实现是两回事 | [05](05-kl-estimation.md#卡点loss-的估计量--梯度的估计量) |
| 14 | 同一个模型怎么可能教得动自己 | [06](06-opsd.md#2-为什么自己教自己会有信号) |
| 15 | teacher 拿到 GT 后会不会只是照抄 GT | [06](06-opsd.md#4-teacher-会不会只是照抄-gt) |
| 16 | OPSD 主实验为什么反而用 forward KL | [06](06-opsd.md#6-论文的-divergence-消融主实验用-forward-kl) |
| 17 | teacher 更强为什么反而蒸不动 | [07](07-rethink-and-mopd.md#2-成功-opd-的两个条件) |
| 18 | OPD 失败时为什么解法反而是先做一段 SFT | [07](07-rethink-and-mopd.md#4-不-compatible-怎么办) |

## 六、阅读顺序

1. [01 熵 / 交叉熵 / KL / JSD](01-entropy-ce-kl-jsd.md) — 信息论地基，后面全部依赖它
2. [01b softmax 求导的三个式子](01b-softmax-gradient.md) — **数学前置**，看不懂梯度推导先补这篇
3. [02 SFT 与传统 KD](02-sft-and-kd.md) — 建立 off-policy 那一侧的基准
4. [03 OPD](03-opd.md) — 核心定义与四象限
5. [04 reverse KL 就是 policy gradient](04-reverse-kl-as-pg.md) — 把蒸馏和 RL 彻底接上
6. [05 KL 怎么估：sampled / top-k / full-vocab](05-kl-estimation.md) — 工程维度
7. [06 OPSD](06-opsd.md)
8. [07 Rethink OPD 与 MOPD](07-rethink-and-mopd.md)
9. [自测题](self-test.md)

前置：[RL 目录](../rl/00-map.md)（policy gradient、advantage、PPO ratio），尤其 [07 KL 散度](../rl/07-kl.md)（forward/reverse 的记忆方法、$\pi_{old}$ vs $\pi_{ref}$）。
