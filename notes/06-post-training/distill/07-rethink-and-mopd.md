# Rethink OPD 与 MOPD：OPD 什么时候有效，以及多教师版本

> Rethink OPD 回答的是 **"OPD 为什么有效 / 为什么失败"**；MOPD 回答的是 **"多个能力怎么合并进一个模型"**。
> 两者串起来会得到一个很漂亮的逻辑闭环。

## 一、Rethink OPD

*Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe*（2026-04）。

### 1. 最重要的一句结论

$$\boxed{\text{Teacher capability}\ \ne\ \text{Distillability}}$$

> **teacher 强，不代表 OPD 就一定有效。**

### 2. 成功 OPD 的两个条件

#### 条件 1：thinking pattern 要 compatible

在 student 当前访问到的状态 $s_t=(x,y^{\text{student}}_{<t})$ 上，两者对"下一步该往哪几个方向走"要有一定共识。

兼容的情况：

```text
Student:                Teacher:
therefore  0.30         therefore  0.35
so         0.20         thus       0.25
because    0.15         so         0.15
thus       0.10         ...
```

概率不同，但**考虑的是差不多的一组 continuation**。

不兼容的情况：

```text
Teacher:
however    0.4
let        0.3
suppose    0.2
```

student 和 teacher 的高概率区域几乎不 overlap，reverse KL 的梯度就会变成"你现在做的这些事情 teacher 全都不喜欢"，大量 token 拿到负 advantage，训练反而可能 collapse。

**实验证据**（MOPD 论文里的对照）：把同源 RL teacher 换成更强的 Qwen3-235B teacher 后，**teacher benchmark 更强，但 MOPD 反而变差** —— 初始 KL 从约 0.04 上升到约 0.19，policy-gradient 版本出现 entropy 收缩，top-k 版本甚至训练发散。

#### 条件 2：teacher 得真的提供新能力

反例：`student = Qwen 1.5B`，`teacher = Qwen 7B`，但 teacher 没有针对该任务额外 post-training。两者 thinking pattern 很像，可是 teacher 在 student rollout 访问到的区域里并没有提供什么新东西 → **OPD 收益很小**。

论文甚至做了 **weak-to-strong reverse distillation**，发现某些同系列 1.5B / 7B 模型，从 student 的视角看，它们在相关状态上的分布可能几乎 indistinguishable。

$$\boxed{\text{大模型}\ne\text{自动成为有效 teacher}}$$

两个条件缺一不可：

```text
        thinking compatible ✓
                  \
                   OPD 成功
                  /
        new capability ✓
```

### 3. OPD 的学习到底发生在哪里

这是最值得记的实证结果。看 student 和 teacher 的 **Top-K token overlap**：

```text
Student Top-5:  {A, B, C, D, E}
Teacher Top-5:  {A, B, C, F, G}
overlap:        {A, B, C}
```

成功 OPD 的现象是：

```text
初期：已有较大 overlap  →  训练  →  overlap 越来越高  →  student 越来越像 teacher
```

而且这个**很小的共享 token 集合能包含 97%–99% 的 probability mass**。

也就是说：

> OPD 并不是靠 teacher 把 student 完全不知道的 token 硬拽进来，更多是在 **student 和 teacher 已经共同认为 plausible 的 continuation** 里面重新分配概率。

```text
Student:            Teacher:            结果:
A 0.4               A 0.10              A 0.4 → 0.2
B 0.3               B 0.70              B 0.3 → 0.6
C 0.2               C 0.15              C 0.2 → 0.15
```

teacher 说的是：**你考虑的候选没错，但 B 才更该被强化。**

论文的说法：**progressive alignment on high-probability tokens at student-visited states.**

> 这也解释了为什么 [top-k OPD](05-kl-estimation.md#4-top-k折中) 是合理的：它训练的正是这个 high-probability region。top-k overlap 不是随便挑的分析指标。

### 4. 不 compatible 怎么办

#### 方法 1：Off-policy cold start

先别 OPD。让 teacher 生成 $y^T\sim\pi_T$，拿这些做**一小段 SFT**：

```text
Teacher rollout → 短暂 SFT → student 先学会 teacher 的基本思路 → 再切 OPD
```

$$\pi_S\xrightarrow{\ \text{SFT}\ }\pi_S'\xrightarrow{\ \text{OPD}\ }\pi_S''$$

这样 Top-K overlap 先被拉上去。

很有意思的一点：

> **OPD 失败时，解决方案反而是先来一点 off-policy SFT。**

因为 SFT 的 support-covering 能力恰好能把 student 带进 teacher 的区域，之后 reverse-KL OPD 才好发挥。这正好和 [03](03-opd.md#5-support-这个词要小心用) 的结论闭环：

$$\boxed{\text{SFT = support expansion}\qquad\text{OPD = on-policy refinement}}$$

#### 方法 2：Teacher-aligned prompt

teacher 做 RL 时一直看到的是

```text
Problem: ...
Please reason step by step...
```

结果 OPD 时你给的是

```text
Question: ...
Answer:
```

语义一样，但生成出来的 reasoning state distribution 可能不同。所以尽量让 $x_{\text{OPD}}\approx x_{\text{teacher-training}}$，让 teacher 处在自己熟悉的区域。

论文发现 teacher-aligned prompt 能提高 overlap 和 OPD 效果，但**过度使用过于 in-distribution 的 prompt 也会降低 entropy**，所以还需要保持一定多样性。

## 二、MOPD：Multi-Teacher OPD

> 注意歧义：**MOPD 这个缩写现在指三样东西**。
> - Xiaomi MiMo 的 **Multi-Teacher OPD** —— 本节讲的就是它
> - Microsoft 的 **Multi-Rollout OPD** —— 完全不同的东西
> - Kimi K3 也叫 **Multi-Teacher On-Policy Distillation** —— 同一个想法的另一次实现，见 [第三节](#三kimi-k3-的-mopd把-9-个-expert-合回一个模型)

### 1. 做法

从**同一个 SFT checkpoint** $\pi_0$ 出发，分别做 domain RL：

```text
                 ┌─ RL(math) ──> Math Teacher
SFT checkpoint ──┼─ RL(code) ──> Code Teacher
                 └─ RL(IF)   ──> IF Teacher
```

然后取 $\pi_\theta\leftarrow\pi_0$ 作为统一 student，混合采样各 domain 的 prompt：

```text
math question → student rollout → Math Teacher 打 logprob → OPD update
code question → student rollout → Code Teacher 打 logprob → OPD update
```

目标：

$$\min_\theta\ \mathbb E_{d,\ x,\ y\sim\pi_\theta}\ D_{KL}\big(\pi_\theta(\cdot|x,y_{<t})\ \big\|\ \pi_{\phi_d}(\cdot|x,y_{<t})\big)$$

$d$ 是 domain：**哪个 domain 的题，就用对应的 specialist teacher 监督 student。**

算法上和普通 OPD 完全一样，只是 teacher 按 domain 切换：

$$\hat A_{\text{MOPD},t}=\operatorname{sg}\big[\log\pi_{\phi_d}(y_t)-\log\pi_\theta(y_t)\big]$$

$$\boxed{\text{OPD = 一个 teacher；MOPD = 每个 domain 一个 teacher}}$$

### 2. 为什么不直接 Mix-RL

同时上 math / code / agent / IF 的 reward 一起 RL，问题是各 domain 的 reward 形式、rollout 长度、RL 难度、超参、收敛速度都不一样，互相干扰。

MOPD 把两件事拆开：

```text
能力生产：Math RL → Math specialist        能力集成：specialists
         Code RL → Code specialist                    ↓
         Agent RL → Agent specialist                MOPD
                                                      ↓
                                              Unified Student
```

这是 MOPD 最重要的工程意义：**每个领域团队可以独立 RL，最后统一 merge，而且 merge 的不是 parameter，而是 policy behavior。** 论文在 Qwen3-30B-A3B 上比 Mix-RL、Cascade RL、off-policy finetune 和 parameter merge 的整体集成效果都好。

### 3. 和 Rethink OPD 的漂亮闭环

MOPD 为什么强调每个 RL teacher 都从**同一个 SFT checkpoint** 出发？

```text
              SFT Model π_0
         /         |         \
      Math RL    Code RL    IF RL
        ↓          ↓          ↓
      T_math     T_code      T_IF
         \         |         /
                 MOPD
                   ↓
            Unified Student
```

因为 $T_d=\pi_0\xrightarrow{\text{domain RL}}\pi_{T_d}$，所以它们和 student：

- 天生 **thinking pattern highly compatible**（同源）✓
- RL 又赋予了 teacher **genuinely new domain capability** ✓

$$\boxed{\text{MOPD 的设计天然满足 Rethink OPD 总结出的两个成功条件}}$$

这也解释了为什么"找一个 benchmark 更强的巨大外部 teacher"未必比"从自己 checkpoint RL 出来的 teacher"更好。

## 三、Kimi K3 的 MOPD：把 9 个 expert 合回一个模型

> 同一个想法的第三次出现。K3 用它把 [3 domain × 3 effort 训出的 9 个 RL expert](../rl/11-kimi-k25-k3.md#10-改动七单一-rl-policy--9-个-specialized-policy) 合并成一个统一模型。
> teacher 的选择规则从"哪个 domain 的题"变成"**哪个 domain + 哪个 effort 档位**"，算法本身没变。

### 1. 只算 sampled token，不是全词表、也不是 top-k

K3 的公式里**没有对词表求和**，只访问 student 实际采到的那一维：

$$\boxed{r_t^{OPD}=\mathrm{clip}\Big(\mathrm{sg}\Big[\log\frac{\pi_T(y_t\mid x,y_{<t})}{\pi_S(y_t\mid x,y_{<t})}\Big],\ -R_{\max},\ R_{\max}\Big)}$$

teacher 只需要返回 $\log\pi_T(y_t|s_t)$，student 返回 $\log\pi_S(y_t|s_t)$，做个差。对比一下如果是全词表：

$$\sum_{v\in V}\pi_T(v|s_t)\log\frac{\pi_T(v|s_t)}{\pi_S(v|s_t)}\qquad\text{或 soft CE}\quad -\sum_{v\in V}\pi_T(v|s_t)\log\pi_S(v|s_t)$$

这需要 teacher/student 对整个 $|V|=160\text{K}$ 词表的分布做监督。**K3 的公式完全没有这个求和。** 在 2.8T 模型 + 超长 agent trajectory 上，sampled-token 便宜太多。

**top-k 呢？** 论文说他们**确实实验过** top-$k$ distillation objective，但在收敛速度和最终性能上**都没有观察到明显优势**，所以最终还是用 sampled-token。

$$\boxed{\text{whole vocab}\ \times}\qquad\boxed{\text{top-}k\ \times\ \text{（试过，没收益）}}\qquad\boxed{\text{student sampled token}\ \checkmark}$$

> 这也解释了为什么 K3 把它写成 **RL reward** 而不是普通的 soft distillation loss：$y_t\sim\pi_S$ 之后，teacher 只回答"**student 已经选了这个 token，我觉得这个选择有多好？**"，于是 $r_t=\log\pi_T(y_t)-\log\pi_S(y_t)$ 直接成为一个 **token-level dense reward**。

$$\boxed{\text{K3 MOPD}=\text{student rollout}+\text{teacher 给 sampled token 打分}+\text{log-prob ratio 当 reward}}$$

### 2. 为什么要 clip，clip 的是什么

$\tilde r_t=\log\pi_T-\log\pi_S$ 是**无界**的。两个方向都会爆：

- $\pi_S(y_t)=10^{-6}$、$\pi_T(y_t)=0.1$ $\Rightarrow$ $r_t=\log10^5\approx+11.5$
- $\pi_S(y_t)=0.1$、$\pi_T(y_t)=10^{-10}$ $\Rightarrow$ $r_t=\log10^{-9}\approx-20.7$

一个 token 就可能拿到 $|A_t|\approx20$，梯度远大于其他正常 token。而 K3 的 trajectory 有 $10^5\sim10^6$ 个 context token，出现这种 outlier 的概率很高：

```text
99.9% 的 token:   r = -1, 0.3, 0.7, 1.2 ...
某一个 token:     r = -25          ← 支配了整个 batch 的梯度
```

所以直接截断（下表的 $R_{\max}=5$ 只是举例，论文没公开具体取值）：

| 原始 OPD reward | clip 后 |
|---:|---:|
| $+0.3$ | $+0.3$ |
| $+2$ | $+2$ |
| $+13$ | **$+5$** |
| $-1$ | $-1$ |
| $-30$ | **$-5$** |

> teacher 可以说"这个 token 很好 / 很差"，但**不能因为一个 token 上的极端概率差异，让整个 batch 的梯度被它绑架**。

论文直接把它称作限制 **extreme advantage signals**。

### 3. 为什么 clip 的是 log-ratio 而不是 probability ratio

因为 OPD 本身来自 reverse KL：

$$D_{KL}(\pi_S\|\pi_T)=\mathbb E_{y\sim\pi_S}\Big[\log\frac{\pi_S(y)}{\pi_T(y)}\Big]$$

所以想要的 PG reward 天然就是 $r_t=-\log\frac{\pi_S}{\pi_T}=\log\frac{\pi_T}{\pi_S}$，**本来就在 log-probability space**。而且 log 已经把尺度压缩了一个量级：$\frac{\pi_T}{\pi_S}=10^5$ 时 ratio 是 100000，log-ratio 只有 11.5，再做 $\mathrm{clip}(\cdot,-R_{\max},R_{\max})$ 就非常稳。

### 4. clip 的代价

不 clip 时它更接近严格的 $D_{KL}(\pi_S\|\pi_T)$ 优化。clip 之后，$|\log\pi_T-\log\pi_S|>R_{\max}$ 的部分都被压平：

$$\boxed{\text{variance / instability}\downarrow}\qquad\text{但}\qquad\boxed{\text{bias}\uparrow}$$

teacher 和 student 分歧极大的 token 不会再获得无限大的纠正力量。本质是 $\boxed{\text{用一点 bias 换 stability}}$。

### 5. ⚠️ 别和 RL optimizer 的 clip 混

$$\boxed{\underbrace{\mathrm{clip}(\log\pi_T-\log\pi_S)}_{\text{MOPD：限制 teacher 的监督信号}}}\qquad\text{vs}\qquad\boxed{\underbrace{\mathrm{clip/mask}(\pi_\theta/\pi_{old})}_{\text{RL optimizer：限制 off-policy drift}}}$$

**完全不是一回事**，详见 [K3 的三层 clip](../rl/11-kimi-k25-k3.md#12--最容易混的两个-clip)。

## 四、面试版

**Rethink OPD**

> OPD 是否成功不主要取决于 teacher 有多强，而取决于 teacher/student 在 **student-visited states** 上是否有 compatible thinking pattern，以及 teacher 是否真的提供新的 capability。成功 OPD 本质上表现为两者高概率 token 的 progressive alignment；失败时可以通过 teacher trajectory SFT cold start 或 teacher-aligned prompt 提高初始 overlap。

**MOPD**

> 先从同一个 SFT checkpoint 分别训练多个 domain RL specialist，再用 multi-teacher OPD 把这些能力蒸馏回统一 student，把"能力生产"和"能力集成"解耦；同源 teacher 又天然保证较高的 policy overlap。

**Kimi K3 的 MOPD**

> K3 按 3 domain × 3 reasoning effort 训出 9 个 RL expert，再用 MOPD 合回一个模型。它的 token reward 只算 **student 实际采样到的那个 token** 的 teacher/student log-prob 差（不是全词表、也不是 top-$k$ —— 论文说 top-$k$ 试过但没有明显收益），并做 $\pm R_{\max}$ 截断来限制 extreme advantage signal。注意这个 clip 和 RL optimizer 里 clip $\pi_\theta/\pi_{old}$ 的那个是完全不同的两件事。

## 自测（口述版）

**1.** "teacher 越强蒸馏效果越好"错在哪？给一个实验证据。

> **答：** 错在把 **teacher capability 和 distillability 等同**了：$\text{Teacher capability}\ne\text{Distillability}$。
> 证据：MOPD 把同源 RL teacher 换成更强的 Qwen3-235B teacher 后，**teacher benchmark 更强但效果反而更差** —— 初始 KL 从约 0.04 升到约 0.19，policy-gradient 版本出现 entropy 收缩，top-k 版本甚至训练发散。

**2.** 成功 OPD 的两个条件是什么？各举一个失败的反例。

> **答：** ① **thinking pattern compatible**（在 student-visited states 上两者的高概率 continuation 有重叠）；② **teacher 真的提供新能力**。
> 反例一（有新能力、不兼容）：换一个完全异构的超强 reasoning teacher，两者高概率区域几乎不重合，reverse KL 下大量 token 拿到负 advantage，训练可能 collapse。
> 反例二（兼容、无新能力）：Qwen 1.5B ← Qwen 7B 但 teacher 没做针对性 post-training，论文的 weak-to-strong reverse distillation 显示两者在相关状态上的分布几乎 indistinguishable，收益很小。

**3.** Top-K overlap 的实验发现是什么？为什么说"OPD 的学习发生在共享的高概率区域"？

> **答：** 发现：成功的 OPD 表现为「**初期就已有较大 overlap → 训练中 overlap 持续升高**」，而且这个**很小的共享 token 集合覆盖 97%–99% 的 probability mass**。
> 所以 OPD 不是靠 teacher 把 student 完全不知道的 token 硬拽进来，而是在两者**共同认为 plausible 的候选**里**重新分配概率**（例：student `A .4 B .3 C .2`、teacher `A .1 B .7 C .15` → student 变成 `A .2 B .6 C .15`）。
> 论文称之为 progressive alignment on high-probability tokens at student-visited states。这也正是 top-k OPD 合理的原因。

**4.** teacher / student 不 compatible 时的两个 recipe 是什么？为什么 cold start 用的是 off-policy SFT 而不是继续 OPD？

> **答：** ① **off-policy cold start**：teacher 先生成 trajectory，拿这些做一小段 SFT 把 student 拉进 teacher 的区域、提高 Top-K overlap，再切 OPD；② **teacher-aligned prompt**：让 OPD 的 prompt 尽量贴近 teacher RL 时的格式（但过度 in-distribution 会降 entropy，仍需多样性）。
> 用 SFT 是因为它的 **support-covering** 能力正好补上 reverse-KL OPD 缺的那一环：**SFT = support expansion，OPD = on-policy refinement**。OPD 本身只在 student 已访问的区域学，不 compatible 时它根本走不出去。

**5.** MOPD 和普通 OPD 在算法上差多少？写出它的 advantage。

> **答：** **几乎没差**，只是 teacher 按 domain 切换：
> $$\hat A_{\text{MOPD},t}=\operatorname{sg}\big[\log\pi_{\phi_d}(y_t)-\log\pi_\theta(y_t)\big]$$
> $d$ 表示 domain —— 哪个 domain 的题就用哪个 specialist teacher 监督。算法上没有神秘的新东西。

**6.** 为什么不直接 Mix-RL？MOPD 解耦了哪两件事？

> **答：** 直接 Mix-RL 的问题：各 domain 的 reward 形式、rollout 长度、RL 难度、超参、收敛速度都不一样，混在一起互相干扰。
> MOPD 解耦了 **能力生产**（各 domain 独立 RL 出 specialist）和 **能力集成**（multi-teacher OPD 蒸回统一 student）。而且 merge 的是 **policy behavior 而不是 parameter**，各领域团队可以独立迭代。

**7.** 为什么 MOPD 的 teacher 必须从同一个 SFT checkpoint 出发？用 Rethink OPD 的结论解释。

> **答：** 因为 $T_d=\pi_0\xrightarrow{\text{domain RL}}\pi_{T_d}$，与 student（也来自 $\pi_0$）**天生 thinking pattern 高度兼容**（满足条件 ①）；同时 domain RL 又赋予了 teacher **真正的新能力**（满足条件 ②）。
> **恰好同时满足 Rethink OPD 总结出的两个成功条件**，所以设计上天然合理。这也解释了为什么「找一个 benchmark 更强的巨大外部 teacher」未必比「从自己 checkpoint RL 出来的 teacher」更好。

**8.** ⭐ K3 的 MOPD 是全词表、top-$k$ 还是 sampled token？

> **答：** **sampled token**。公式里没有对词表求和，只访问 student 实际采到的 $y_t$ 那一维：teacher 返回 $\log\pi_T(y_t|s_t)$、student 返回 $\log\pi_S(y_t|s_t)$，做差。
> 全词表要对 $|V|=160\text{K}$ 做监督，在 2.8T 模型 + 超长 agent trajectory 上太贵。**top-$k$ 他们确实试过，但收敛速度和最终性能都没有明显优势**，所以没用。
> 这也是为什么 K3 把它写成 RL reward 而不是 soft distillation loss：teacher 只回答"student 已经选了这个 token，这个选择有多好"。

**9.** ⭐⭐ MOPD 的 reward 为什么要 clip？clip 的是什么？代价是什么？

> **答：** 因为 $\log\pi_T-\log\pi_S$ **无界**，两个方向都会爆：$\pi_S=10^{-6}$、$\pi_T=0.1$ 给出 $+11.5$；$\pi_S=0.1$、$\pi_T=10^{-10}$ 给出 $-20.7$。K3 的 trajectory 有 $10^5\sim10^6$ token，出 outlier 的概率很高，一个 $r=-25$ 的 token 就能支配整个 batch 的梯度。
> clip 的对象是 **teacher/student 的 log-prob 差**（论文称之为限制 extreme advantage signals），$\mathrm{clip}(\mathrm{sg}[\cdot],-R_{\max},R_{\max})$。
> 代价：不 clip 时更接近严格的 $D_{KL}(\pi_S\|\pi_T)$ 优化，clip 后分歧极大的 token 不再获得无限纠正力量 —— **variance↓ 但 bias↑，用一点 bias 换 stability**。

**10.** 为什么 clip 的是 log-ratio 而不是 probability ratio？

> **答：** 因为 OPD 来自 reverse KL $D_{KL}(\pi_S\|\pi_T)=\mathbb E_{\pi_S}[\log\frac{\pi_S}{\pi_T}]$，想要的 reward 天然就是 $\log\frac{\pi_T}{\pi_S}$，**本来就在 log-probability space**。而且 log 先压了一个量级：$\pi_T/\pi_S=10^5$ 时 ratio 是 100000，log-ratio 只有 11.5，再截断就很稳。

**11.** ⭐⭐ 说出 K3 里两个（三个）clip 的区别。

> **答：** ① **MOPD reward clip**：$\mathrm{clip}(\mathrm{sg}[\log\pi_T-\log\pi_S],-R_{\max},R_{\max})$，限制 **teacher 给的监督信号**；② **off-policy mask**：$\mathrm{mask}(\pi_\theta/\pi_{old})$，限制 **当前 policy 与 rollout policy 的偏离**；③ $-\tau(\log\pi_\theta/\pi_{old})^2$ 软约束。
> ① 管 teacher 和 student 的差，②③ 管 $\theta$ 和 $old$ 的差，**目的完全不同**。
