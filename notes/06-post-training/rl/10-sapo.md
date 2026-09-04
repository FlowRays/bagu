# SAPO：Soft Adaptive Policy Optimization

> Qwen 团队 2025-12-04 提出（注意区分后来出现的其他同名 SAPO）。它把 GRPO 和 GSPO 都当 baseline。
> 一句话动机：**GSPO 的 sequence-level hard clipping 太"硬"了，能不能既保留 sequence-level 的稳定性，又别把整条轨迹的梯度直接砍成 0？**

$$\boxed{\text{SAPO}=\text{GRPO 的 token-level 自适应}+\text{GSPO 的稳定性}+\text{soft trust region}}$$

## 1. 先说 GSPO 留下的问题

[GSPO](09-gspo.md) 一旦 sequence 越界（$A_i>0$ 且 $s_i>1+\epsilon$），整条的梯度归零：

$$\boxed{\nabla J_i=0}$$

假设一条 1000-token 的 response，里面只有十几个 token 因为 MoE routing / 数值误差而非常 off-policy。GSPO 看的是整条的 ratio，一旦越界：

$$\boxed{1000\ \text{个 token 一起没梯度}}$$

**一颗老鼠屎，整锅汤倒掉。** 这浪费了大量本来还不错的训练信号。

## 2. 核心想法：hard clip → soft gate

GRPO / GSPO 是 0/1 式的：

```text
ratio 正常  →  100% gradient
ratio 越界  →    0% gradient      ← 突变
```

SAPO 改成连续衰减：

```text
on-policy         →  100%
稍微 off-policy   →   80%
更 off-policy     →   30%
特别 off-policy   →   ≈0%
```

$$\boxed{\text{hard clipping}\ \rightarrow\ \text{soft gating}}$$

也就是构造一个连续的 **soft trust region**。

## 3. 注意：SAPO 又退回 token-level ratio 了

这里最容易想当然。**不要**以为"GSPO 是 sequence ratio，所以 SAPO 肯定也是"。SAPO 用的是：

$$\boxed{r_{i,t}=\frac{\pi_\theta(y_{i,t}\mid s_{i,t})}{\pi_{old}(y_{i,t}\mid s_{i,t})}}$$

和 GRPO 一模一样的 **token-level ratio**。区别只在后面接什么：

$$\boxed{\text{GRPO：token ratio}+\text{hard clip}}$$
$$\boxed{\text{GSPO：sequence ratio}+\text{hard clip}}$$
$$\boxed{\text{SAPO：token ratio}+\text{soft gate}}$$

## 4. 公式

$$J(\theta)=\mathbb E\Big[\frac1G\sum_i\frac1{T_i}\sum_t f(r_{i,t})\,\hat A_{i,t}\Big]$$

其中

$$\boxed{f(r)=\frac4\tau\,\sigma\big(\tau(r-1)\big)},\qquad \sigma(x)=\frac1{1+e^{-x}}$$

先别纠结 $4/\tau$ 这个常数。要抓住的是 $r-1$ 表示**当前 policy 离 rollout policy 有多远**：$r=1$ 就是 $\pi_\theta=\pi_{old}$，完全 on-policy。

## 5. 真正重要的不是 $f(r)$，是它的导数

令 $p=\sigma(\tau(r-1))$，则 $f(r)=\frac4\tau p$。sigmoid 的导数是 $\frac{dp}{dr}=\tau\,p(1-p)$，所以

$$\frac{df}{dr}=\frac4\tau\cdot\tau p(1-p)$$

$$\boxed{w(r)=4p(1-p)}$$

于是 policy gradient 变成

$$\boxed{\nabla J\ \sim\ w(r_t)\,r_t\,A_t\,\nabla_\theta\log\pi_\theta(y_t)}$$

这个 $w(r)$ 就是 SAPO 的灵魂。对比 GRPO 的有效权重是 $r_tA$，SAPO 是 $\boxed{w(r_t)r_tA}$。

## 6. $w(r)$ 长什么样

- $r=1$：$p=0.5$，$w=4\times0.5\times0.5=\boxed{1}$ —— on-policy 时保留完整梯度
- $r\gg1$：$p\to1$，$w\to0$
- $r\ll1$：$p\to0$，$w\to0$

```text
gradient weight

1.0             /\
               /  \
              /    \
             /      \
0 ──────────/        \──────────
                r=1
```

$$\boxed{r\approx1\Rightarrow w\approx1},\qquad \boxed{|r-1|\uparrow\ \Rightarrow\ w\downarrow}$$

和 clip 的区别是：不是在边界处从 1 突降到 0，而是平滑下降。

```text
GRPO（A>0）                    SAPO
gradient                      gradient
1 ──────────────┐             1          /\
                │                       /  \
                │                      /    \
0               └────────      0 ─────/      \─────
              1+ε                        r=1
```

$$\boxed{\text{GRPO：hard trust region}}\qquad\text{vs}\qquad\boxed{\text{SAPO：continuous / soft trust region}}$$

名字里的 **Soft Adaptive** 就是这么来的。

## 7. 为什么这样就比 GSPO 划算

一条 sequence 的 ratio 序列：

$$r_t=[1.01,\ 1.02,\ 1.00,\ 1.03,\ \boxed{3.0},\ 1.01,\ 0.99,\dots]$$

只有一个 token 特别离谱。

- **GSPO**：$\{r_t\}\to s_{seq}\to$ 整条一个决定。一旦触发 clipping，**全部 token 梯度 = 0**。
- **SAPO**：正常 token $r_t\approx1\Rightarrow w_t\approx1$；异常 token $r_t=3\Rightarrow w_t\approx0$。

$$\boxed{\text{SAPO：只压异常 token，不牺牲整条 sequence}}$$

这就是论文说的 **token-adaptive**。

## 8. 那不就退回 GRPO 的老问题了吗

这是 SAPO 最值得想清楚的一步。你会问：

> GSPO 不就是因为 token-level ratio 太 noisy 才改成 sequence ratio 的吗？SAPO 怎么又退回去了？

关键在于：**GRPO 真正的危险不是"用了 token ratio"本身**，而是 $r_tA$ 直接进梯度，只有一道脆弱的 hard clip 兜底 —— 异常 ratio 会产生很不稳定的 token-level 贡献，而且到了边界又是突变。

SAPO 给它加了一个 $w(r_t)$，$r_t$ 越异常 $w(r_t)\to0$。最终有效权重是 $w(r_t)r_tA$ 而不是 $r_tA$。所以它**保留了 token-level adaptivity，同时把极端 token 自动压掉**。

## 9. 它凭什么说自己也有 sequence coherence

**不是因为它显式算了 sequence ratio 。它没有。**

GSPO 的 $s_i=\exp(\frac1T\sum_t\log r_t)$ 是显式的 sequence ratio。SAPO 论文证明的是一个**统计性质**：当 policy update 较小、且同一 sequence 内 token log-ratio 的方差不太大时，整条 sequence 的平均 gate

$$\frac1T\sum_t w(r_t)$$

可以近似成 sequence log-ratio $\log s_i=\frac1T\sum_t\log r_t$ 的一个平滑函数：

$$\boxed{g(\log s_i)\approx\operatorname{sech}^2\Big(\frac\tau2\log s_i\Big)}$$

所以整条 sequence 越 off-policy（$|\log s_i|$ 越大），它整体的有效梯度也会下降。

$$\boxed{\text{形式上 token-level}}\qquad\text{但}\qquad\boxed{\text{统计行为上有 sequence-level coherence}}$$

这才是"SAPO 同时具有 GSPO 的 sequence coherence 和 GRPO 式的 token adaptivity"这句话的真正含义。注意它是**近似成立、有条件的**，不是恒等式。

## 10. 一个容易漏掉的设计：正负 advantage 用不同温度

SAPO 不是只有一个 $\tau$，而是：

$$A>0\ \Rightarrow\ \tau_{pos},\qquad A<0\ \Rightarrow\ \tau_{neg},\qquad \boxed{\tau_{neg}>\tau_{pos}}$$

$\tau$ 越大，gate 越窄：

```text
小 τ：            大 τ：
     /------\           /\
   /          \        /  \
──/            \──  ──/    \──
```

所以**负 advantage 更容易被迅速 downweight**。

### 为什么负 advantage 更危险

从 softmax 的约束看。一个 sampled token $y_t$ 是坏的（$A<0$），RL 要做 $\pi(y_t)\downarrow$。但 softmax 必须满足

$$\sum_v\pi(v)=1$$

你把一个 token 的概率压下去，这部分概率质量**必须分给整个巨大词表的其他 token**。也就是负更新会间接推高大量其他 token 的 logits —— 在几十万词表 + MoE 的情况下会更 noisy。

相比之下正 advantage 主要在说"把这个 sampled good token 的概率提上去"，方向集中得多。

所以 SAPO 对负样本更保守：$\tau_{neg}>\tau_{pos}$。这是论文里很重要的一个稳定性来源。

## 11. 把 GRPO → GSPO → SAPO 串起来

```text
GRPO :  token ratio r_t  →  r_t·A  →  hard clip
          问题：token ratio 方差大，MoE 上尤其严重
              ↓
GSPO :  压成 s = exp(1/T Σ log r_t)  →  s·A  →  sequence hard clip
          解决：稳定性
          代价：一条 sequence 越界 → 整条 gradient = 0
              ↓
SAPO :  重新用 r_t，但乘一个 soft gate w(r_t) = 4p(1-p)
          得到：w(r_t)·r_t·A·∇log π_t
          异常 token w→0，正常 token w≈1
          = GSPO 的稳定性 + token-level 的样本利用率
```

Qwen 报告 SAPO 在 Qwen3-30B-A3B 的数学 RL 和 Qwen3-VL 系列的大规模 RL 上，比 GSPO 和带 Routing Replay 的 GRPO 都更稳、最终性能也更好。

## 面试版

> GSPO 用 sequence-level importance ratio 和 hard clipping 保证稳定，但一旦序列越界会丢掉整条序列的梯度；SAPO 回到 token-level ratio，用一个随 off-policy 程度平滑衰减的 sigmoid gate $w(r)=4\sigma(\tau(r-1))(1-\sigma(\tau(r-1)))$ 替代 hard clipping，只 downweight 异常 token；并通过整体 gate 的统计行为保留 sequence-level coherence，同时对负 advantage 使用更高的 temperature 进一步增强稳定性。

## 自测

**1.** SAPO 想解决 GSPO 的什么问题？举一个具体场景。

> **答：** GSPO 一旦 sequence ratio 越界，**整条 response 的梯度全部归零**。一条 1000-token 的回答里可能只有十几个 token 因为 MoE routing 或数值误差非常 off-policy，却害得 1000 个 token 一起没梯度，浪费大量有效训练信号。

**2.** ⭐ SAPO 用的是 token-level 还是 sequence-level ratio？

> **答：** **token-level**，$r_{i,t}=\pi_\theta(y_{i,t})/\pi_{old}(y_{i,t})$，和 GRPO 一样。三者的区别是：GRPO = token ratio + hard clip；GSPO = sequence ratio + hard clip；**SAPO = token ratio + soft gate**。

**3.** ⭐ 写出 $f(r)$，推出 $w(r)$，说明 $r=1$ 时 $w$ 等于多少。

> **答：** $f(r)=\frac4\tau\sigma(\tau(r-1))$。令 $p=\sigma(\tau(r-1))$，$\frac{dp}{dr}=\tau p(1-p)$，故 $\frac{df}{dr}=\frac4\tau\cdot\tau p(1-p)=4p(1-p)=w(r)$。
> $r=1$ 时 $p=0.5$，$w=4\times0.5\times0.5=1$ —— on-policy 时完整保留梯度。$r\gg1$ 或 $r\ll1$ 时 $p\to1$ 或 $0$，$w\to0$。
> 最终 $\nabla J\sim w(r_t)r_tA_t\nabla\log\pi_\theta(y_t)$。

**4.** SAPO 退回 token ratio，不就又回到 GSPO 批评的问题了吗？

> **答：** 不是。GRPO 真正的危险不是"用 token ratio"本身，而是 $r_tA$ **直接**进梯度、只有一道 hard clip 兜底，异常 ratio 会产生很不稳定的贡献且在边界处突变。SAPO 的有效权重是 $w(r_t)r_tA$，$r_t$ 越异常 $w\to0$，极端 token 被自动压掉，同时保留了 token 级的自适应能力。

**5.** SAPO 说自己有 sequence coherence，是因为它算了 sequence ratio 吗？

> **答：** **不是，它没有显式算 sequence ratio。** 论文证的是一个统计性质：policy update 较小、且同一 sequence 内 token log-ratio 方差不太大时，整条的平均 gate $\frac1T\sum_t w(r_t)$ 可近似成 $\log s_i$ 的平滑函数 $\operatorname{sech}^2(\frac\tau2\log s_i)$，所以整条越 off-policy 有效梯度越低。**形式上 token-level，统计行为上有 sequence coherence**，且是有条件的近似。

**6.** ⭐ 为什么 $\tau_{neg}>\tau_{pos}$？

> **答：** $\tau$ 越大 gate 越窄，所以负 advantage 会被更快 downweight。原因在 softmax 的归一化约束：$\sum_v\pi(v)=1$，把一个 bad token 的概率压下去，这些概率质量**必须分摊给整个巨大词表的其他 token**，等于间接推高大量无关 token 的 logits，在几十万词表 + MoE 下非常 noisy。正 advantage 只是"把这个 token 提上去"，方向集中得多。所以对负样本更保守。
