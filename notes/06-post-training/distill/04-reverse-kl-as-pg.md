# Reverse KL 就是 policy gradient：蒸馏和 RL 的接口

> "reverse KL 和 RL pipeline 很搭"不是类比，**是梯度形式真的一样**。
> 这一篇把它推一遍，推完之后 SFT / OPD / PPO / GRPO 就彻底串成一条线了。

## 1. 目标函数长得就像 RL

只看**一个状态 $s$、一个 token action $a$**，记

$$p_\theta(a)=\pi_S(a|s),\qquad q(a)=\pi_T(a|s)$$

OPD 要最小化 $D_{KL}(p_\theta\|q)$，等价于最大化

$$J(\theta)=-D_{KL}(p_\theta\|q)=\sum_a p_\theta(a)\big[\log q(a)-\log p_\theta(a)\big]$$

写成期望：

$$\boxed{J(\theta)=\mathbb E_{a\sim p_\theta}\big[\log q(a)-\log p_\theta(a)\big]}$$

对比 RL 目标 $J_{\text{RL}}=\mathbb E_{a\sim\pi_\theta}[R(a)]$，只要令 $R(a)=\log q(a)-\log p_\theta(a)$ 就一模一样。

**但有个问题**：这个 "reward" 里面自己带着 $\theta$，还能直接用 policy gradient 吗？可以，而且有一个漂亮的消项。

## 2. 把梯度真正推一遍

$$\nabla_\theta J=\nabla_\theta\sum_a p_\theta(a)\big[\log q(a)-\log p_\theta(a)\big]$$

product rule：

$$=\underbrace{\sum_a \nabla p_\theta(a)\big[\log q(a)-\log p_\theta(a)\big]}_{\text{(I)}}\ -\ \underbrace{\sum_a p_\theta(a)\nabla\log p_\theta(a)}_{\text{(II)}}$$

**第二项直接消失**：

$$p_\theta(a)\nabla\log p_\theta(a)=\nabla p_\theta(a)\ \Rightarrow\ \sum_a p_\theta(a)\nabla\log p_\theta(a)=\sum_a\nabla p_\theta(a)=\nabla\Big(\sum_a p_\theta(a)\Big)=\nabla 1=0$$

**第一项**再用 log trick $\nabla p_\theta(a)=p_\theta(a)\nabla\log p_\theta(a)$：

$$\boxed{\nabla J=\mathbb E_{a\sim p_\theta}\Big[\underbrace{\big(\log q(a)-\log p_\theta(a)\big)}_{A(a)}\ \nabla\log p_\theta(a)\Big]}$$

和 policy gradient $\nabla J_{\text{PG}}=\mathbb E[A(a)\nabla\log\pi_\theta(a|s)]$ **完全一样**。于是可以定义

$$\boxed{A(a)=\log\pi_T(a|s)-\log\pi_S(a|s)}$$

这不是启发式：它在期望下就是 $-D_{KL}(S\|T)$ 的梯度。

> 代价是：这是一个单样本 Monte-Carlo estimator，**梯度方差较大**（这也是 top-k / full-vocab 变体的动机，见 [05](05-kl-estimation.md)）。

## 3. 这个 advantage 到底在说什么

| 情况 | $\pi_S(a)$ | $\pi_T(a)$ | $A$ | 效果 |
|---|---:|---:|---:|---|
| teacher 比 student 更喜欢 | 0.1 | 0.4 | $\log 4\approx1.386$ | $\pi_S(a)\uparrow$ |
| student 很喜欢但 teacher 不喜欢 | 0.4 | 0.1 | $-1.386$ | $\pi_S(a)\downarrow$ |
| 两边一样 | — | — | $0$ | 不更新 |

所以它是

$$\boxed{A_t=\log\frac{\pi_T(a_t|s_t)}{\pi_S(a_t|s_t)}}$$

也就是 **teacher 相对于 student 自己**有多喜欢这个 token —— **不是看 teacher 概率绝对值高不高**，而是看比值。

teacher 在说：

- $A>0$：你刚才这个 token 其实挺好的，你自己还不够相信它
- $A<0$：你特别喜欢这个 token，但我认为它不该出现

## 4. Thinking Machines 的实现：几乎不用改 RL trainer

Student rollout 时已经记录了 $\log\pi_{\text{old}}(a_t|s_t)$，teacher 对**同一个 student token** 再算一次 $\log\pi_T(a_t|s_t)$：

$$\hat k_t=\log\pi_{\text{old}}(a_t|s_t)-\log\pi_T(a_t|s_t)\qquad\text{(sample reverse KL)}$$

$$\boxed{A_t=-\hat k_t=\log\pi_T(a_t|s_t)-\log\pi_{\text{old}}(a_t|s_t)}$$

然后直接送进现成的 importance-sampling RL loss：

$$\rho_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)},\qquad L=-\mathbb E\big[\rho_t(\theta)A_t\big]$$

rollout 刚结束时 $\theta=\theta_{\text{old}}$，$\rho=1$，梯度正是上面推出来的 $-\mathbb E[A_t\nabla\log\pi_\theta]$。

工程上的对照非常干净：

```text
普通 RL :  rollout → reward → advantage → PG
OPD     :  rollout → teacher logprob → KL advantage → PG
```

对一个已有的 KL-regularized RL trainer 来说，**近似就是"把 regularizer model 换成 teacher"**。

> 注意：这里的 $\pi_{\text{old}}$ 是 rollout policy 快照，和 $\pi_{\text{ref}}$ 不是一回事，见 [RL/07](../rl/07-kl.md#再次强调pi_old-和-pi_ref-不是一回事)。

## 5. 为什么比普通 RL 的 reward 更 dense

一道数学题生成 1000 个 token，outcome-based RL 是：

```text
token 1 … token 1000  →  最终答案错误  →  reward = 0
```

然后 GRPO/PPO 要自己想办法做 credit assignment：到底哪个 token 错了？

OPD 则是每个 token 都有监督：

$$A_1=\log\frac{\pi_T(a_1)}{\pi_S(a_1)},\quad A_2=\log\frac{\pi_T(a_2)}{\pi_S(a_2)},\quad\dots,\quad A_{1000}=\log\frac{\pi_T(a_{1000})}{\pi_S(a_{1000})}$$

$$\boxed{\text{token-level dense advantage}}$$

而且实现上取 $\gamma=0$：第 $t$ 个 token **只优化当前位置的 teacher mismatch**，不把未来 reward 折回来，即

$$\boxed{A_t=r_t}$$

不需要传统的 $G_t=r_t+\gamma r_{t+1}+\cdots$。（Thinking Machines 报告更大的 discount factor 没带来改善。）

## 6. 更深的一层：LLM RL 本来就可以写成 reverse KL

这一段面试里很加分。常见的 KL-regularized RL 目标：

$$\max_\pi\ \mathbb E_{y\sim\pi}[R(y)]-\beta D_{KL}(\pi\|\pi_{\text{ref}})$$

定义一个**隐式目标分布**（base model 越喜欢 + reward 越高 → 概率越大）：

$$\boxed{\pi_R^*(y)=\frac1Z\pi_{\text{ref}}(y)e^{R(y)/\beta}}$$

算 $D_{KL}(\pi\|\pi_R^*)$：

$$D_{KL}(\pi\|\pi_R^*)=\mathbb E_\pi\Big[\log\pi-\log\pi_{\text{ref}}-\frac{R}{\beta}+\log Z\Big]$$

两边乘 $-\beta$：

$$-\beta D_{KL}(\pi\|\pi_R^*)=\mathbb E_\pi[R]-\beta D_{KL}(\pi\|\pi_{\text{ref}})-\beta\log Z$$

最后一项与 $\pi$ 无关，于是

$$\boxed{\max_\pi\big\{\mathbb E_\pi[R]-\beta D_{KL}(\pi\|\pi_{\text{ref}})\big\}\iff\min_\pi D_{KL}(\pi\|\pi_R^*)}$$

也就是说：

> **KL-regularized RL 本身就在做一个 sequence-level reverse KL**，只是它的 "teacher distribution" 没有显式模型，而是由 reward model + reference model 隐式定义。

OPD 相当于把这个隐式 teacher $\pi_R^*$ 换成一个**真实存在、而且能对每个 token 给 logprob** 的 $\pi_T$。这就是 "reverse KL 与 RL 有天然 synergy" 的准确含义。

## 7. 反过来：forward KL 为什么不走 PG

forward KL 要优化的是

$$-\sum_v\pi_T(v|s)\log\pi_S(v|s)$$

这本质上是一个 **supervised / distillation loss**，直接 backprop CE 就行，不需要 policy gradient 那一套。

$$\boxed{\text{Forward KL}\to\text{supervised CE}}\qquad\boxed{\text{Reverse KL}\to\text{policy gradient}}$$

## 自测（口述版）

**1.** **纸上推导** $\nabla_\theta\big[-D_{KL}(p_\theta\|q)\big]$，说明哪一项为什么会消失。

> **答：** $J=\sum_a p_\theta(a)[\log q(a)-\log p_\theta(a)]$，product rule 得两项：
> (I) $\sum_a\nabla p_\theta(a)[\log q-\log p_\theta]$；(II) $-\sum_a p_\theta(a)\nabla\log p_\theta(a)$。
> **(II) 消失**：$p_\theta\nabla\log p_\theta=\nabla p_\theta$，故 $\sum_a\nabla p_\theta(a)=\nabla\big(\sum_a p_\theta(a)\big)=\nabla 1=0$。
> (I) 再用 log trick：
> $$\nabla J=\mathbb E_{a\sim p_\theta}\big[\underbrace{(\log q(a)-\log p_\theta(a))}_{A(a)}\nabla\log p_\theta(a)\big]$$
> 和 policy gradient 形式完全一样。

**2.** 为什么 "reward 里含 $\theta$" 不影响这个推导？

> **答：** 因为 $\theta$ 出现在 reward 里额外贡献的那一项恰好是 score function 的期望 $-\mathbb E[\nabla\log p_\theta]$，它等于 $\nabla\sum_a p_\theta(a)=\nabla1=0$（归一化常数求导为 0）。
> 所以可以直接把 $A(a)=\log q(a)-\log p_\theta(a)$ 当作（stop-gradient 的）advantage 使用，不需要额外的修正项。

**3.** 写出 $A_t$，并用 $(\pi_S,\pi_T)=(0.1,0.4)$ 和 $(0.4,0.1)$ 各算一次，解释含义。

> **答：** $$A_t=\log\pi_T(a_t|s_t)-\log\pi_S(a_t|s_t)=\log\frac{\pi_T}{\pi_S}$$
> $(0.1,0.4)$：$A=\log4\approx+1.386$ → 提高该 token 概率。teacher 在说「你这个 token 其实挺好的，你自己还不够相信它」。
> $(0.4,0.1)$：$A=\log0.25\approx-1.386$ → 降低。teacher 在说「你特别喜欢它，但我认为它不该出现」。
> 相等则 $A=0$ 不更新。**关键：看的是比值，不是 teacher 概率的绝对值。**

**4.** 描述 Thinking Machines 的实现流程：rollout → ? → ? → PG。为什么 RL trainer 几乎不用改？

> **答：** `student rollout → teacher 算同一个 token 的 logprob → $A_t=\log\pi_T-\log\pi_{\text{old}}$（负的 sampled reverse KL）→ 送进现成的 importance-sampling policy loss`。
> rollout 时已经记录了 $\log\pi_{\text{old}}$，loss 是 $L=-\mathbb E[\rho_tA_t]$、$\rho_t=\pi_\theta/\pi_{\text{old}}$；rollout 刚结束时 $\rho=1$，梯度正是推导出的 $-\mathbb E[A_t\nabla\log\pi_\theta]$。
> 不用改是因为整个 pipeline 只是把 `reward → advantage` 换成了 `teacher logprob → KL advantage`；对已有的 KL-regularized RL trainer 来说约等于「把 regularizer model 换成 teacher」。

**5.** OPD 为什么比 outcome RL 更 dense？为什么可以取 $\gamma=0$？

> **答：** outcome RL 一条 1000 token 的 CoT 只有最后一个 0/1 reward，credit assignment 极难；OPD **每个 token 都有 $A_t$**，是 token-level dense advantage。
> $\gamma=0$ 表示第 $t$ 个 token 只优化当前位置的 teacher mismatch，不把未来 reward 折回来，即 $A_t=r_t$，不需要 $G_t=r_t+\gamma r_{t+1}+\cdots$。这样做合理是因为 teacher 已经在**每个位置**给出了正确信号，不需要靠未来回报来推断当前动作好坏；实验上更大的 discount 也没带来改善。

**6.** **推导** $\max_\pi\{\mathbb E[R]-\beta KL(\pi\|\pi_{\text{ref}})\}\iff\min_\pi KL(\pi\|\pi_R^*)$，写出 $\pi_R^*$。

> **答：** 定义 $\boxed{\pi_R^*(y)=\frac1Z\pi_{\text{ref}}(y)e^{R(y)/\beta}}$（base 越喜欢 + reward 越高 → 目标概率越大）。
> $$D_{KL}(\pi\|\pi_R^*)=\mathbb E_\pi\Big[\log\pi-\log\pi_{\text{ref}}-\frac R\beta+\log Z\Big]$$
> 两边乘 $-\beta$：
> $$-\beta D_{KL}(\pi\|\pi_R^*)=\mathbb E_\pi[R]-\beta D_{KL}(\pi\|\pi_{\text{ref}})-\beta\log Z$$
> 最后一项与 $\pi$ 无关，所以两个优化问题等价。
> 含义：**KL-regularized RL 本身就在做 sequence-level reverse KL**，只是它的 teacher 分布没有显式模型，而由 reward model + reference model 隐式定义。OPD 相当于把这个隐式 teacher 换成真实存在、且能逐 token 给 logprob 的 $\pi_T$。

**7.** 为什么 forward-KL 蒸馏一般不写成 policy gradient？

> **答：** forward KL 要优化的是 $-\sum_v\pi_T(v|s)\log\pi_S(v|s)$，这是标准的 **supervised / distillation loss**，直接对它 backprop 就行，不涉及「对自己采样的分布求导」这个难点。
> 记：**Forward KL → supervised CE；Reverse KL → policy gradient。**

