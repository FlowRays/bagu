# OPD：On-Policy Distillation

> **高频考点。** 本篇只做一件事：把 OPD 的定义摆正，并把"on-policy"和"forward/reverse KL"这两个
> 被大量材料混在一起的概念彻底拆开。

## 1. 定义：on-policy 说的是 state distribution

OPD 的完整定义是：

> **让 student 自己 rollout，然后 teacher 在 student 真正访问到的状态上，提供 dense token-level supervision。**

形式化：student 先生成

$$\hat y\sim\pi_S(\cdot|x)$$

对 student trajectory 上第 $t$ 个 prefix

$$h_t=(x,\hat y_{<t})$$

同时算出 $\pi_S(\cdot|h_t)$ 和 $\pi_T(\cdot|h_t)$，然后优化

$$\boxed{\mathcal L_{\text{OPD}}=\mathbb E_{\hat y\sim\pi_S}\Big[\sum_t D\big(\pi_T(\cdot|h_t),\ \pi_S(\cdot|h_t)\big)\Big]}$$

**外层那个 $\hat y\sim\pi_S$ 才是 On-Policy。** 里面的 $D$ 是什么（forward KL / reverse KL / JSD）是另一个维度。

严格讲，on-policy 指的是

$$s_t\sim d^{\pi_S}$$

即 prefix 由 student 自己一步步生成。

> **teacher 的角色从"示范者"变成了"判卷者"。** 这句话很适合面试直接说。
> teacher 不重新写一份答案，只在 student 走到的每个位置回答"下一步应该怎么走"。

## 2. 为什么要 on-policy

见 [02](02-sft-and-kd.md#3-exposure-biassft-真正的病)：SFT 只训练"老师走对了前面所有步骤"的状态，student 在推理时自己走歪之后就是 OOD。

OPD 让 student 真的走到

```text
17 × 6 = 96, therefore
```

然后问 teacher 在这个**错误 prefix** 上的 next-token 分布，teacher 可能给出

```text
wait      0.4
actually  0.3
this      0.1
```

于是 student 学会的是：**在自己会犯的错误状态里怎么恢复。**

## 3. 第二个维度：这个 state 上怎么对齐

### Reverse KL：$D_{KL}(\pi_S\|\pi_T)$

$$D_{KL}(\pi_S\|\pi_T)=\mathbb E_{v\sim\pi_S}\big[\log\pi_S(v|h)-\log\pi_T(v|h)\big]$$

期望在 $\pi_S$ 下，所以 **student 自己认为重要的 token 才贡献主要 loss**。

| token | Student | Teacher |
|---|---:|---:|
| A | 0.8 | 0.001 |
| B | 0.1 | 0.6 |
| C | 0.1 | 0.399 |

A 的贡献 $0.8\log\frac{0.8}{0.001}$ 巨大，于是 student 被强烈要求：

> **不要把概率放在 teacher 认为很差的位置。**

这就是 **mode-seeking** 的准确说法（比"student 有值 teacher 没值"严谨得多）。

### Forward KL：$D_{KL}(\pi_T\|\pi_S)$

$$\min_\theta D_{KL}(\pi_T\|\pi_S)\iff\min_\theta -\mathbb E_{v\sim\pi_T}[\log\pi_S(v)]$$

即 teacher 分布对 student 做 soft CE。若 $\pi_T(v)>0$ 而 $\pi_S(v)\to0$，则 $\pi_T(v)\log\frac{\pi_T(v)}{\pi_S(v)}\to\infty$：

> **teacher 认为合理的东西，student 都必须留概率。** → **mode-covering**

### 一句话对照

$$\boxed{\text{Forward KL：Teacher 看 Student 有没有漏}}$$
$$\boxed{\text{Reverse KL：Student 看自己有没有乱跑}}$$

## 4. 两个维度合成四象限

| | Forward KL $D_{KL}(T\|S)$ | Reverse KL $D_{KL}(S\|T)$ |
|---|---|---|
| off-policy prefix | 传统 KD（hard 特例 = SFT） | off-policy reverse KD（少见） |
| **on-policy prefix** | **GKD / OPSD 主 recipe** | **Thinking Machines OPD recipe** |

四象限里 off-policy + reverse 理论完全成立，但少见 —— 因为它放弃了 OPD 最核心的优势（训练 student 真正会访问的状态）。

### 卡点：forward-KL OPD = SFT 吗？

**不会。** 两者内部 loss 都带 forward-KL / CE 的味道，但训练 state 完全不同：

$$L_{\text{SFT}}\approx\mathbb E_{h\sim d_{\text{expert}}}\big[D_{KL}(\pi_T\|\pi_S)\big]$$
$$L_{\text{fwd-OPD}}=\mathbb E_{h\sim d_{\pi_S}}\big[D_{KL}(\pi_T\|\pi_S)\big]$$

$$\boxed{d_{\text{expert}}\quad\text{vs}\quad d_{\pi_S}}$$

而且 target 也不同：SFT 是 hard token，forward-KL OPD 是 teacher 的完整 soft distribution。

$$\boxed{\text{SFT}=\text{teacher state}+\text{hard token}}$$
$$\boxed{\text{Forward OPD}=\text{student state}+\text{teacher soft distribution}}$$
$$\boxed{\text{Reverse OPD}=\text{student state}+\text{teacher 对 student action 打分}}$$

## 5. "support" 这个词要小心用

一个很常见的说法是：

> SFT 可以增加 support，reverse-KL OPD 只能强化 student 已有的能力。

这个**直觉有价值，但不能这么绝对地说**，面试里容易被抓：

1. LM 的 softmax 理论上让几乎每个 token 都有 $\pi(v)>0$，**严格数学意义上的 support 几乎总是全词表**。真正的问题不是 $p=0$，而是 **low-probability / 几乎不会被访问的 trajectory**。
2. 更准确的表述：
   - **SFT 能直接把 student 拉到当前 policy 很少访问的 expert trajectory 上，适合 capability injection / cold start。**
   - **On-policy OPD 主要在 student 当前 visitation distribution 上提供监督，更擅长 policy refinement 和纠正 inference-time states。**
3. 而且 Rethink OPD 发现，OPD 真正有效时 teacher **还必须提供 student 原先没有的新能力**（见 [07](07-rethink-and-mopd.md)）。所以不能说"OPD 不能传递新能力"。

## 6. 为什么现代 OPD 更常选 reverse KL

三条理由，按重要性排：

1. **便宜**。reverse KL 只需要 teacher 返回 student 实际采样 token 的 $\log\pi_T(a_t|s_t)$ 一个标量，不需要 100K+ 维完整 logits。精确 forward KL 则每个位置都要 teacher 的完整分布。
2. **小 student 覆盖不了大 teacher**。teacher 有 100 种合理表达，容量有限的 student 被 forward KL 逼着 mode-cover，可能把概率摊得很开、生成质量下降。reverse KL 的 mode-seeking 往往更聚焦。
3. **和 RL pipeline 天然同构**。$A_t=\log\pi_T-\log\pi_S$ 直接是 per-token advantage，见 [04](04-reverse-kl-as-pg.md)。

但 forward KL 有一个 reverse KL 没有的巨大优势：**能主动增加 support**。若 $\pi_T(B)=0.3$ 而 $\pi_S(B)\approx0$，reverse KL 下 student 几乎永远采不到 $B$，teacher 根本没机会说"B 其实很好"；forward KL 则直接产生 $-\pi_T(B)\log\pi_S(B)$ 的梯度硬推上去。

$$\boxed{\text{先用 SFT / forward-KL 增加 support，再用 reverse-KL OPD 在已有 support 内 mode-seek}}$$

## 7. 面试 1 分钟版

> **OPD 的核心不是某一种 KL，而是 on-policy trajectory。** Student 先自己 rollout，teacher 不重新生成答案，而是在 student 实际访问到的每个 prefix 上计算 next-token distribution，给 student dense token-level supervision。这样训练分布和 inference 时 student 自己的状态分布一致，可以减少 SFT 的 exposure bias。
>
> 如果用 reverse KL $D_{KL}(\pi_S\|\pi_T)$，期望是 student-weighted，主要惩罚 student 把概率放在 teacher 低概率区域，表现为 mode-seeking；如果用 forward KL $D_{KL}(\pi_T\|\pi_S)$，则要求 student 覆盖 teacher 的高概率区域，更 mode-covering。
>
> **SFT 和 OPD 更本质的区别不是 forward/reverse KL，而是 state distribution**：SFT 在 expert trajectory 上训练，OPD 在 student trajectory 上训练。

## 自测（口述版）

**1.** 给出 OPD 的定义，指出公式里哪一部分对应 "on-policy"。

> **答：** $$\mathcal L_{\text{OPD}}=\mathbb E_{\hat y\sim\pi_S}\Big[\sum_t D\big(\pi_T(\cdot|h_t),\ \pi_S(\cdot|h_t)\big)\Big],\quad h_t=(x,\hat y_{<t})$$
> **外层的 $\hat y\sim\pi_S$ 才是 on-policy**（严格讲是 $s_t\sim d^{\pi_S}$，prefix 由 student 自己一步步生成）。里面的 $D$ 是什么是另一个维度。
> 一句话：teacher 从「示范者」变成「判卷者」，不重新写答案，只在 student 走到的每个位置回答「下一步该怎么走」。

**2.** "OPD 就是 reverse KL 蒸馏"错在哪？画出四象限表。

> **答：** 错在把两个**正交**维度混为一谈：on-policy 描述 **prefix/state 从谁来**，forward/reverse KL 描述 **在这个 state 上两个分布怎么比**。
> 四象限：
> | | Forward $D_{KL}(T\|S)$ | Reverse $D_{KL}(S\|T)$ |
> |---|---|---|
> | off-policy prefix | 传统 KD（hard 特例 = SFT） | 少见 |
> | **on-policy prefix** | **GKD / OPSD 主 recipe** | **Thinking Machines OPD recipe** |
> 最直接的反例：**OPSD 论文的主实验用的正是 forward KL**。

**3.** forward-KL OPD 和 SFT 的区别是什么？写出两个 loss 的期望形式做对比。

> **答：** $$L_{\text{SFT}}\approx\mathbb E_{h\sim d_{\text{expert}}}\big[D_{KL}(\pi_T\|\pi_S)\big],\qquad L_{\text{fwd-OPD}}=\mathbb E_{h\sim d_{\pi_S}}\big[D_{KL}(\pi_T\|\pi_S)\big]$$
> ① **state 不同**：$d_{\text{expert}}$ vs $d_{\pi_S}$；② **target 不同**：SFT 是 hard token，forward-KL OPD 是 teacher 的完整 soft distribution。
> 所以 forward-KL OPD 仍然是 OPD，不会退化成 SFT。

**4.** 用表格例子解释 reverse KL 为什么是 mode-seeking。

> **答：** student $\{A:0.8,B:0.1,C:0.1\}$，teacher $\{A:0.001,B:0.6,C:0.399\}$。
> reverse KL $=\mathbb E_{v\sim\pi_S}[\log\frac{\pi_S}{\pi_T}]$，期望在 student 下，A 的贡献是 $0.8\log\frac{0.8}{0.001}$，**极大**。
> 于是 student 被强烈要求「**不要把概率放在 teacher 认为很差的位置**」，概率质量从 A 撤回到 teacher 的高概率 mode。
> 严谨表述是「student-weighted」，不要说成「student 有值 teacher 没值」。

**5.** "reverse-KL OPD 只能强化已有能力"这句话怎么改才严谨？

> **答：** 三点修正：
> ① LM 的 softmax 让几乎每个 token 都有 $\pi(v)>0$，**严格数学意义上的 support 几乎是全词表**，真正的问题不是 $p=0$ 而是 **low-probability / 几乎不会被访问的 trajectory**；
> ② 准确说法是「**SFT 适合 capability injection / cold start；on-policy OPD 主要在 student 当前 visitation distribution 上做 policy refinement**」；
> ③ Rethink OPD 发现 OPD 真正有效时 teacher **还必须提供新能力**，所以不能说「OPD 传不了新能力」。

**6.** 现代 OPD 偏爱 reverse KL 的三个理由，以及 forward KL 唯一的杀手锏是什么。

> **答：** 三个理由：① **便宜** —— 只需 teacher 返回 student 采样 token 的一个标量 logprob，不用 100K+ 维完整 logits；② 小 student 容量有限，被 forward KL 逼着 mode-cover 会把概率摊开、生成质量下降，reverse 的 mode-seeking 更聚焦；③ **和 RL pipeline 同构**，$A_t=\log\pi_T-\log\pi_S$ 直接当 per-token advantage。
> forward KL 的杀手锏：**能主动增加 support**。若 $\pi_T(B)=0.3$ 而 $\pi_S(B)\approx0$，reverse KL 下 student 几乎永远采不到 B，teacher 没机会说「B 很好」；forward KL 直接产生 $-\pi_T(B)\log\pi_S(B)$ 的梯度硬推上去。
> 所以典型 recipe 是**先 SFT/forward 扩 support，再 reverse-KL OPD 在已有 support 内 mode-seek**。

