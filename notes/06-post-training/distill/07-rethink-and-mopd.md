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

> 注意歧义：2026 年有两篇都叫 MOPD。这里说的是 Xiaomi MiMo 的 **Multi-Teacher OPD**；另一篇是 Microsoft 的 **Multi-Rollout OPD**。

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

## 三、面试版

**Rethink OPD**

> OPD 是否成功不主要取决于 teacher 有多强，而取决于 teacher/student 在 **student-visited states** 上是否有 compatible thinking pattern，以及 teacher 是否真的提供新的 capability。成功 OPD 本质上表现为两者高概率 token 的 progressive alignment；失败时可以通过 teacher trajectory SFT cold start 或 teacher-aligned prompt 提高初始 overlap。

**MOPD**

> 先从同一个 SFT checkpoint 分别训练多个 domain RL specialist，再用 multi-teacher OPD 把这些能力蒸馏回统一 student，把"能力生产"和"能力集成"解耦；同源 teacher 又天然保证较高的 policy overlap。

## 自测（口述版）

1. "teacher 越强蒸馏效果越好"错在哪？给一个实验证据。
2. 成功 OPD 的两个条件是什么？各举一个失败的反例。
3. Top-K overlap 的实验发现是什么？为什么说"OPD 的学习发生在共享的高概率区域"？
4. teacher / student 不 compatible 时的两个 recipe 是什么？为什么 cold start 用的是 off-policy SFT 而不是继续 OPD？
5. MOPD 和普通 OPD 在算法上差多少？写出它的 advantage。
6. 为什么不直接 Mix-RL？MOPD 解耦了哪两件事？
7. 为什么 MOPD 的 teacher 必须从同一个 SFT checkpoint 出发？用 Rethink OPD 的结论解释。
