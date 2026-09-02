# OPSD：On-Policy Self-Distillation

> **一句话**：OPSD 就是 OPD，但 teacher 不再是另一个更强的模型，而是**"拿到了答案的自己"**。
>
> 原始论文：*Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models*，Zhao et al.，arXiv v1 **2026-01-26**。

## 1. 设定：同一个模型，不同 context

对一道题 $x$ 和它的 GT solution $y^*$：

$$\text{Student}:\ p_S(\cdot\mid x)\qquad\text{Teacher}:\ p_T(\cdot\mid x,\ y^*)$$

参数本质相同，只是 **context 不同**：一个闭卷考试，一个拿着参考答案。

$$\boxed{\text{Privileged Information (PI)}}$$

所以 OPD 和 OPSD 的 teacher 优势来源完全不同：

$$\boxed{\text{OPD：能力优势（7B }\leftarrow\text{ 72B）}}\qquad\boxed{\text{OPSD：信息优势（}p_\theta(\cdot|x)\leftarrow p_\theta(\cdot|x,y^*)\text{）}}$$

teacher 本身未必更强，只是**多看了答案**。这才是 "privileged information" 最准确的含义。

## 2. 为什么"自己教自己"会有信号

因为

$$p_\theta(\cdot\mid x)\ \ne\ p_\theta(\cdot\mid x,y^*)$$

同一个模型，**信息不同**。背后利用的是一个很重要的现象：

> **知道正确答案以后解释"应该怎么推"，通常比完全不知道答案时自己找出来容易得多。**

类比：你闭卷做一道数学题可能不会；但给你标准答案后再问"你做到这一步，下一步是不是应该这样"，你往往就能判断出来。

OPSD 本质上是把

$$\boxed{\text{拥有 privileged information 时的能力}}\ \longrightarrow\ \boxed{\text{没有 privileged information 时的自己}}$$

论文把这个过程叫作利用 GT 做 **implicit rationalization**。

如果没有 GT，teacher 就是 $p_\theta(\cdot|x,\hat y_{<t})$，和 student 一模一样，**没有任何额外信息**，训练信号为 0。

## 3. PI 具体怎么给 teacher：直接拼上下文

最直接的实现就是**把 privileged information 拼进 teacher 的 context，student 看不到它**。

题目：`Solve: 2x + 3 = 7`，GT：`2x = 4，所以 x = 2`。

**Student 输入**（只有题目 + 自己已经写出来的部分）：

```text
Solve: 2x + 3 = 7

We subtract 3 from both sides.
2x = 4.
Then x =
```

**Teacher 输入**（多塞一份 GT）：

```text
Problem:
Solve: 2x + 3 = 7

Reference solution:
Subtract 3 from both sides:
2x = 4
Therefore x = 2

Student attempt:
We subtract 3 from both sides.
2x = 4.
Then x =
```

于是

$$p_S:\ P(2)=0.35,\ P(4)=0.30,\ P(1)=0.10$$
$$p_T:\ P(2)=0.92,\ P(4)=0.01,\ P(1)=0.01$$

蒸馏把 $p_S(2)$ 推上去。

**关键：teacher 不继续 generate。** 它只做一次 forward，看这个位置上词表的概率分布。rationalization 是通过 forward pass 隐式完成的。

## 4. teacher 会不会只是照抄 GT

这是最核心的 intuition，也是最容易被追问的地方。

**不会**，因为 teacher 做的不是

$$p_T(\cdot\mid x,y^*)$$

而是

$$\boxed{p_T(\cdot\mid x,\ y^*,\ \hat y_{<t})}$$

**三样东西都给 teacher**：题目 $x$、privileged GT $y^*$、**student 当前 trajectory prefix $\hat y_{<t}$**。

看 student 走错的情况。GT 是 `2x = 4 → x = 2`，但 student rollout 成：

```text
Subtract 3 from both sides.
2x = 10.
Therefore ...
```

如果 teacher 只是 copy GT，它应该无脑输出 `2x = 4`。但当前 prefix 已经是 `2x = 10. Therefore ...`，直接接一句 `2x = 4` **语言上都不自然**。

teacher 真正做的是：

> 我知道正确解法是 $x=2$，同时我也知道 student 现在已经写成了 $2x=10$。在这个上下文下，接下来最合理的 token 是什么？

于是它更倾向于 `this is incorrect` / `however` / `we made an error`，然后再修正。

$$\boxed{\text{GT-aware trajectory correction}\quad\ne\quad\text{GT imitation}}$$

### 开车类比

- **SFT**：驾校老师给你一条标准路线（出门左转 → 第二个路口右转）。你只学"正确状态 → 正确动作"。真实开车走错了就没辙。
- **OPSD**：你自己开，拐错了；老师坐副驾而且知道目的地。老师不说"你刚才本来应该左转"，而是说"**既然你现在已经到了这个路口，现在该怎么走才能回到正确方向**"。

$$\boxed{\text{learn recovery / correction on your own state distribution}}$$

## 5. 训练目标

$$\boxed{\mathcal L=\mathbb E_{\hat y\sim p_S}\Big[\frac1T\sum_t D\big(p_T(\cdot|x,y^*,\hat y_{<t})\ \big\|\ p_S(\cdot|x,\hat y_{<t})\big)\Big]}$$

论文的 Algorithm 1 / Eq. 6–8 做的是 **full-vocabulary** 计算（对整个词表做 full softmax），并用 [per-entry KL clipping](05-kl-estimation.md#5-full-vocab-的稳定性问题per-entry-kl-clipping) 稳定训练。

它同时给出一个 **sampled-token / policy-gradient 的 alternative objective**：对 student 实际采到的 $a_t\sim p_S$，

$$r_t=\log p_T(a_t\mid x,y^*,s_t)-\log p_S(a_t\mid x,s_t),\qquad \nabla J\approx\sum_t r_t\nabla\log p_S(a_t|s_t)$$

即 [04](04-reverse-kl-as-pg.md) 那一套。论文的结论是 **full-vocabulary > sampled-token**，代价是要保存 vocab-sized logits，**peak memory 更高**，这是一个明确的 performance–memory tradeoff。

### 一个必须修正的细节

teacher 虽然"来自同一个模型"，但训练时并不是每一步都让最新 student 同时充当 teacher。实际做法是：

$$\boxed{\text{teacher branch stop-gradient，并固定在 initial policy snapshot}}$$

student 持续更新，teacher 不动。这是为了训练稳定。所以更准确的说法是"**同源模型，不同 context**"，而不是"实时的自己"。

## 6. 论文的 divergence 消融：主实验用 forward KL

这是最反直觉、也最值得当细节讲的一点。论文 §4.3.1 比较三种 divergence（Qwen3-1.7B，AIME25）：

| divergence | Base | Step 50 | Step 100 |
|---|---:|---:|---:|
| **Forward KL** $D_{KL}(p_T\|p_S)$ | 36.7 | **43.9** | 41.1 |
| Reverse KL $D_{KL}(p_S\|p_T)$ | 36.7 | 37.5 | 35.0 |
| JSD | 36.7 | 36.9 | 39.0 |

然后作者写：**"We therefore adopt forward KL in all remaining experiments."**

所以：

$$\boxed{\text{OPSD 主实验 = on-policy prefix + full-vocab \textbf{forward} KL}}$$

这正好印证了 [00-map](00-map.md#一最重要的一句话) 的那句话：**on-policy 和 forward/reverse KL 是两个正交维度**，OPSD 是 "on-policy + forward KL" 的活生生的例子。论文本身也明确说框架的 divergence 可以是 forward KL / reverse KL / JSD，不把算法定义绑在某一种 KL 上。

> 常见误区：把 Thinking Machines 的 **sampled-token reverse-KL recipe** 直接当成 "OPSD 的主 objective"。它在 OPSD 论文里是被当作 *Alternative objective: sampled-token distillation through policy gradient* 讨论的。

## 7. 历史定位（别说过头）

OPSD **不是第一个**提出 self-distillation 或 privileged-information distillation 的工作 —— learning with privileged information、teacher-privileged distillation、各种 LLM self-training 都更早（2025 年就已有明确叫 Teacher Privileged Distillation 的工作）。

它真正新的是这个**组合**：

$$\boxed{\text{same model}+\text{privileged teacher context}+\text{student on-policy rollout}+\text{token-level distribution distillation}}$$

并把这一套正式命名为 OPSD。

时间线：

$$\text{GKD / earlier OPD}\rightarrow\text{Thinking Machines OPD}\rightarrow\boxed{\text{OPSD, 2026-01}}\rightarrow\text{Rethinking OPSD, 2026-07}\rightarrow\text{U-OPSD, 2026-08}$$

（U-OPSD 进一步去掉了 GT supervision。）

## 8. 面试版

> **OPSD 的核心不是用一个更强的 teacher，而是通过 privileged conditioning 构造 teacher**：student 只看 question 并做 on-policy rollout，teacher 是同源模型但在同一个 student prefix 上额外看到 GT solution，因此能给出 GT-aware 的 next-token distribution。这样模型不是只模仿标准答案，而是在自己真实会访问到的状态上学习如何朝正确方向修正。实现上 teacher 做 stop-gradient 并固定在 initial snapshot；主实验用 full-vocabulary forward KL，并对每个 vocabulary entry 的 KL 贡献做 clipping 来避免 style token 淹没 reasoning token。

## 自测（口述版）

1. 写出 OPSD 的 student / teacher 两个条件分布，指出 teacher 的优势来源和 OPD 有什么本质不同。
2. 为什么同一个模型能教自己？如果不给 GT 会怎样？
3. 用 $2x+3=7$ 的例子，把 student 和 teacher 的实际 context 各写一遍。
4. teacher 看到 GT 后会不会只是照抄 GT？为什么公式必须写成 $p_T(\cdot|x,y^*,\hat y_{<t})$？
5. OPSD 训练时 teacher 是实时跟着 student 更新的吗？为什么？
6. 论文主实验用的是 forward KL 还是 reverse KL？这和"OPD 是 reverse KL"矛盾吗？
7. full-vocabulary 和 sampled-token 两个 objective 的取舍是什么？
8. OPSD 是第一个提出 self-distillation 的工作吗？它真正的新意是什么？
