# GSPO：Group Sequence Policy Optimization

> Qwen 团队 2025-07-27 提出，官方说明已用于当时最新的 Qwen3 系列（Instruct / Coder / Thinking）。
> 一句话：**GRPO 在 token level 做 importance ratio + clipping，GSPO 改成 sequence level。**

## 1. 核心质疑：reward 的粒度和 correction 的粒度对不上

GRPO 每个 token 一个 ratio：

$$\rho_{i,t}=\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{old}(y_{i,t}\mid x,y_{i,<t})}$$

但 GRPO 的 advantage 是**整条 response 一个**：

$$A_{i,1}=A_{i,2}=\cdots=A_{i,T}=A_i=\frac{R_i-\bar R}{\sigma_R}$$

于是出现一个不协调：

$$\boxed{\text{sequence-level reward}}\quad\text{却用}\quad\boxed{\text{token-level importance correction}}$$

而做 importance sampling 时，真正被采样的对象是**一整条 sequence**：

$$\mathbb E_{y\sim\pi_\theta}[f(y)]=\mathbb E_{y\sim\pi_{old}}\Big[\frac{\pi_\theta(y|x)}{\pi_{old}(y|x)}f(y)\Big],\qquad \frac{\pi_\theta(y|x)}{\pi_{old}(y|x)}=\prod_t\rho_{i,t}$$

GSPO 认为 GRPO 恰恰在这里把优化单位弄错了。

## 2. 一个三 token 的例子，一眼看出问题在哪

答案只有三个 token，$y=[A,B,C]$，整条答对：$A_i=+1$。更新几步后三个 ratio 变成：

$$\rho_1=0.5,\qquad \rho_2=1,\qquad \rho_3=2$$

GRPO 给三个 token 的梯度系数就是 $0.5,\;1,\;2$：

```text
token A : 只强化 0.5 倍
token B : 强化 1 倍
token C : 强化 2 倍   ← 是 token A 的 4 倍
```

问题在于：

> reward 只说了「**这整条回答不错**」。凭什么因为 token C 的 policy ratio 恰好大一点，就认为它该拿到 4 倍于 token A 的学习信号？

这个差异**不是 reward 给的信息**，纯粹来自 $\pi_\theta/\pi_{old}$ 的波动。长 CoT 几千甚至几万 token 时，这种噪声会一路累积。

## 3. GSPO 怎么改

先把整条序列压成一个数：

$$\boxed{s_i(\theta)=\Big(\frac{\pi_\theta(y_i|x)}{\pi_{old}(y_i|x)}\Big)^{1/T}=\exp\Big[\frac1T\sum_t\big(\log\pi_\theta(y_{i,t})-\log\pi_{old}(y_{i,t})\big)\Big]}$$

即**所有 token ratio 的几何平均**。上面的例子：

$$s=(0.5\times1\times2)^{1/3}=1$$

三个 token 于是都拿 $1\cdot A$，噪声被平均掉了。

目标函数：

$$\boxed{J_{GSPO}=\frac1G\sum_i\min\Big[s_iA_i,\ \mathrm{clip}(s_i,1-\epsilon,1+\epsilon)A_i\Big]}$$

**整条 response 只有一个 ratio、一个 advantage、一次 clipping。**

### 为什么要 $1/T$ 次方

如果直接用 $\prod_t\rho_t$，序列越长乘积越容易爆炸或趋近 0：几千个 token 的 log-ratio 累加起来是线性增长的，同一个 $\epsilon$ 对长短序列意义完全不同。做长度归一化后，$s_i$ 的尺度和序列长度基本无关，才能用统一的 clip 区间。

⚠️ 但要知道代价：**取了 $1/T$ 次方之后，$s_i$ 已经不是 $\pi_\theta(y)/\pi_{old}(y)$ 这个严格意义上的 sequence importance ratio 了。** GSPO 更应该理解成一个**更稳定的 surrogate objective**，而不是"理论上更正确的 importance sampling"。这个点面试很适合深问。

## 4. 梯度：两者的差别就一个系数

$$J=s A,\qquad s=\exp\Big[\frac1T\sum_t(\log\pi_\theta(y_t)-\log\pi_{old}(y_t))\Big]$$

$\pi_{old}$ 与 $\theta$ 无关，没有梯度。用 $\nabla e^f=e^f\nabla f$：

$$\nabla_\theta s=s\cdot\nabla_\theta\Big[\frac1T\sum_t\log\pi_\theta(y_t)\Big]=\frac sT\sum_t\nabla_\theta\log\pi_\theta(y_t)$$

所以：

$$\boxed{\nabla_\theta J_{GSPO}=\frac{s A}{T}\sum_t\nabla_\theta\log\pi_\theta(y_t)}$$

对照 GRPO：

$$\boxed{\nabla_\theta J_{GRPO}\sim\frac AT\sum_t\rho_t\nabla_\theta\log\pi_\theta(y_t)}$$

逐 token 摆开：

| | token 1 | token 2 | token 3 | … |
|---|---|---|---|---|
| GRPO | $\rho_1A$ | $\rho_2A$ | $\rho_3A$ | … |
| GSPO | $sA$ | $sA$ | $sA$ | … |

$$\boxed{\text{GRPO：每个 token 自己决定 importance weight}}$$
$$\boxed{\text{GSPO：整条 response 共同决定 importance weight}}$$

**注意 GSPO 并不是"不对 token 求梯度"。** 每个 token 仍然走 $y_t\to\log\pi_\theta(y_t)\to\text{logits}\to\theta$ 正常反传，只是前面的系数从 $\rho_tA$ 统一成了 $sA$。

## 5. clip 的四象限（和 PPO 完全一致，只是单位变成 response）

被 clip 的条件**取决于 advantage 的正负**，不是"越界就 clip"。这一点和 [PPO 的 min](03-clip-and-min.md#3-为什么必须有-min) 是同一个逻辑：

| Advantage | ratio | 意义 | 是否 clip |
|---|---:|---|---|
| $A>0$ | $s>1+\epsilon$ | 好答案已经涨太多 | ✅ 停 |
| $A>0$ | $s<1-\epsilon$ | 好答案反而降了 | ❌ 赶紧救 |
| $A<0$ | $s<1-\epsilon$ | 坏答案已经降很多 | ✅ 停 |
| $A<0$ | $s>1+\epsilon$ | 坏答案反而涨了 | ❌ 赶紧压 |

举例，$A=+1$、$\epsilon=0.2$：

- $s=1.1$：$\min(1.1,1.1)=1.1$，有梯度
- $s=1.5$：$\min(1.5,1.2)=1.2$，是常数 → $\nabla J=0$
- $s=0.5$：虽然 $<0.8$，但 $\min(0.5,0.8)=0.5$，**仍有梯度** —— 一个好答案的概率居然掉了，当然要赶紧拉回来

$$\boxed{\text{clip 的本质不是"限制 ratio 在区间内"，而是"只阻止策略往 reward 希望的方向走得太远"}}$$

### 被 clip 之后是不是整条都没梯度

**是。** $A>0$ 且 $s>1+\epsilon$ 时目标取到 $(1+\epsilon)A$，这是个常数，$\nabla_\theta J=0$，**整条 response 的所有 token 都不再被强化**。

这既是 GSPO 的稳定性来源，也是它最大的代价 —— 后面 [SAPO](10-sapo.md) 就是冲着这一点来的。

## 6. 为什么对 MoE 尤其重要

MoE 每个 token 都要过 router：

$$h_t\ \rightarrow\ \text{router}\ \rightarrow\ \{\text{Expert 3, Expert 17}\}$$

哪怕参数几乎没变，一点数值误差就可能让重新 forward 时选到不同的专家：

```text
rollout 时 π_old :  Expert 3, 17
训练时  π_θ     :  Expert 3, 18   ← 只差一个专家
```

于是某个 token 的概率突然 $0.10\to0.08$（$\rho_t=0.8$），另一个 $0.10\to0.13$（$\rho_{t'}=1.3$）。

**GRPO** 直接把这些抖动乘进梯度，甚至触发 clipping：

$$\text{routing 变化}\rightarrow\text{token prob 变化}\rightarrow\rho_t\text{ 变化}\rightarrow\text{梯度权重变化}$$

Qwen 说 GRPO 训 MoE 因此可能无法正常收敛，以前得靠 **Routing Replay**（缓存 rollout 时激活的专家，训练 forward 时强行复现同样的 routing）才能训下去。

**GSPO** 则把这些波动在 $\log s_i=\frac1T\sum_t\log\rho_t$ 里平均掉了：

$$\rho=[0.8,\,1.2,\,0.9,\,1.1,\,1.05,\,0.95]\ \Rightarrow\ s\approx1$$

GRPO 看见六个不同的 ratio，GSPO 看见的基本就是 1。所以 GSPO **不需要 Routing Replay**，对 rollout engine 和 training engine 之间的数值差异也更 tolerant。

## 7. 一个反直觉的实验现象

Qwen 报告：

> **GSPO 被 clip 掉的 token 比例反而比 GRPO 高约两个数量级，但训练效率和最终性能都更好。**

这里 GSPO 的"clipped token"其实是：sequence 被 clip → 这条 sequence 的所有 token 都计为 clipped，所以比例天然就高。

它支持的结论是：

$$\boxed{\text{更多 gradient}\ \ne\ \text{更有效 gradient}}$$

GRPO 表面上保留了大量 token gradient，但里面混着很多 noisy 的 token-level importance weight；GSPO 更激进地以 sequence 为单位过滤，信号反而更干净。

## 8. 一个必须避开的八股坑

⚠️ **不要在面试里说「PPO 的 token-level importance sampling 是错的」。** 这太绝对。

标准 PPO 里每个 state/action 有**自己的** advantage $A_1,A_2,\dots,A_T$，token-level ratio 配 token-level advantage 是自洽的。

而 GRPO 的设定是 $A_1=A_2=\cdots=A_T=A_{seq}$。

$$\boxed{\text{GSPO 针对的是「sequence-level reward + token-level ratio」这个特定组合下的 mismatch}}$$

论文自己的表述是：GRPO 的 objective **在这种场景下**存在 importance sampling 的误用，并因此产生高方差噪声。

## 9. 两张脑图

```text
      GRPO                          GSPO
  R_seq → A_seq                 R_seq → A_seq
       ↓                             ↓
  token ratio ρ_t               sequence ratio s
       ↓                             ↓
 ρ_t·A_seq·∇log π_t          s·A_seq·(1/T)Σ_t ∇log π_t
```

$$\boxed{\text{GRPO: token ratio + sequence advantage}}$$
$$\boxed{\text{GSPO: sequence ratio + sequence advantage}}$$

## 10. 四者的最终对照

$$\boxed{PPO\rightarrow GRPO}\ \text{改：advantage 怎么估计（砍 critic）}$$
$$\boxed{GRPO\rightarrow DAPO}\ \text{改：怎么让 reasoning RL 稳定高效地 scale}$$
$$\boxed{GRPO/DAPO\rightarrow GSPO}\ \text{改：ratio 的粒度}$$

> **LLM RL 的 policy ratio，到底该把 token 当 action，还是把整个 sequence 当 action？**

注意 GSPO 和 [DAPO 的 token-level loss](08-dapo.md#3-token-level-policy-gradient-loss) **不冲突**：一个改的是 **ratio 的粒度**（token vs sequence），一个改的是 **loss 在 batch 里怎么聚合**（除 $\frac1{|y_i|}$ 还是除总 token 数 $N$）。两件事在公式里的位置不一样。

## 面试版

> GSPO 认为 GRPO 存在 reward 粒度和 importance correction 粒度的错配：reward 和 advantage 都是 sequence-level 的，ratio 和 clip 却在 token-level 做，这会让同一条 response 内的 token 因为 $\pi_\theta/\pi_{old}$ 的随机波动拿到差异很大的梯度权重，长 CoT 上噪声不断累积。GSPO 改成用所有 token ratio 的几何平均作为 sequence-level ratio，并在 sequence level 做 clip，一条 response 内所有 token 共享同一个权重 $s_iA_i$。这对 MoE 尤其关键 —— router 的微小抖动在 GRPO 下会直接变成 token-level 的梯度权重噪声，GSPO 把它平均掉，因此不再需要 Routing Replay。代价是一旦 sequence 越界，整条的梯度都归零。

## 自测

**1.** GSPO 质疑 GRPO 的什么？

> **答：** 质疑 **reward/advantage 是 sequence-level 的，importance ratio 和 clip 却在 token-level 做**。同一条 response 内所有 token 共享一个 $A_i$，但 GRPO 让它们各自带一个 $\rho_{i,t}$，token 之间的权重差异完全来自 $\pi_\theta/\pi_{old}$ 的波动，不是 reward 给的信息。

**2.** 用 $\rho=[0.5,1,2]$ 这个例子说明 GRPO 的问题，以及 GSPO 怎么处理。

> **答：** 整条答对 $A=+1$，GRPO 下三个 token 的梯度系数是 $0.5/1/2$，token 3 拿到 token 1 的 4 倍学习信号 —— 但 reward 只说了"整条不错"。GSPO 先算几何平均 $s=(0.5\cdot1\cdot2)^{1/3}=1$，三个 token 统一乘 $1\cdot A$。

**3.** 写出 $s_i$，说明为什么要 $1/T$ 次方，以及这带来什么理论上的让步。

> **答：** $s_i=\exp\big[\frac1T\sum_t(\log\pi_\theta(y_t)-\log\pi_{old}(y_t))\big]$，即 token ratio 的几何平均。$1/T$ 是长度归一化：log-ratio 之和随长度线性增长，不归一化时长序列的 ratio 会指数偏离 1，统一的 $\epsilon$ 就没有意义。让步是：开了 $1/T$ 次方之后它**不再是严格的 sequence importance ratio** $\pi_\theta(y)/\pi_{old}(y)$，GSPO 本质是一个更稳定的 surrogate objective。

**4.** ⭐ 写出 GSPO 的梯度，和 GRPO 并排比较。

> **答：** $\nabla_\theta J_{GSPO}=\frac{sA}{T}\sum_t\nabla_\theta\log\pi_\theta(y_t)$；$\nabla_\theta J_{GRPO}\sim\frac AT\sum_t\rho_t\nabla_\theta\log\pi_\theta(y_t)$。
> 推导：$\nabla e^f=e^f\nabla f$，且 $\pi_{old}$ 与 $\theta$ 无关，所以 $\nabla s=\frac sT\sum_t\nabla\log\pi_\theta(y_t)$。
> 差别只有系数：GRPO 每个 token 是 $\rho_tA$，GSPO 每个 token 都是 $sA$。**GSPO 仍然逐 token 反传**，不是不对 token 求梯度。

**5.** $A>0$、$\epsilon=0.2$、$s=0.5$，会被 clip 吗？

> **答：** **不会**。$\min(0.5A,\,0.8A)=0.5A$，仍有梯度。一个好答案的概率反而掉到 0.5 倍，正是最该被拉回来的情况。clip 只阻止"往 reward 希望的方向走太远"。

**6.** ⭐ 为什么 GSPO 对 MoE 收益特别大？以前怎么解决的？

> **答：** MoE 每个 token 过 router，微小数值误差就可能让训练时 forward 选到与 rollout 不同的专家（如 Expert 17→18），token 概率 $0.10\to0.08$ 这类抖动直接变成 $\rho_t=0.8$ 的梯度权重噪声，甚至触发 clip。GSPO 把这些波动在 $\frac1T\sum_t\log\rho_t$ 里平均掉，$s\approx1$。
> 以前的解法是 **Routing Replay**：缓存 rollout 时激活的专家，训练 forward 时强行复现同样的 routing。GSPO 不需要它。

**7.** GSPO 被 clip 的 token 比例反而更高，为什么效果还更好？

> **答：** 那个比例是"sequence 被 clip → 整条所有 token 都计为 clipped"算出来的，天然偏高。结论是 **更多 gradient ≠ 更有效 gradient**：GRPO 保留的大量 token gradient 里混着很多 noisy 的 token-level importance weight，GSPO 以 sequence 为单位整条过滤，剩下的信号更干净。

**8.** 为什么不能说"PPO 的 token-level importance sampling 是错的"？

> **答：** 标准 PPO 里每个 timestep 有自己的 advantage $A_t$，token ratio 配 token advantage 是自洽的。GSPO 针对的是 **GRPO 这种「sequence reward → 所有 token 共享一个 $A$」+ token-level ratio** 的特定组合。说死"PPO 的 token IS 是错的"会被追问。

**9.** GSPO 的 sequence-level ratio 和 DAPO 的 token-level loss 冲突吗？

> **答：** **不冲突**，改的是公式里两个不同位置。GSPO 改 **ratio 的粒度**（token ratio → sequence ratio）；DAPO 改 **loss 在 batch 内怎么聚合**（$\frac1G\sum_i\frac1{|y_i|}\sum_t$ → $\frac1N\sum_{i,t}$，$N$ 是 batch 总生成 token 数）。
