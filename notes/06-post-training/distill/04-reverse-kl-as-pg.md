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

## 7. 卡点：SFT 的 loss 不用 PG，OPD 的 loss 要，差在哪

把两个 loss 写成**同一个形式**：

$$L=\sum_v \underbrace{w(v)}_{\text{权重}}\cdot \underbrace{f_\theta(v)}_{\text{被积函数}}$$

| | 权重 $w(v)$ | 被积函数 $f_\theta(v)$ |
|---|---|---|
| **SFT** | $\delta_{y^*}(v)$（one-hot 标签） | $-\log\pi_S(v)$ |
| **Forward KL 蒸馏** | $\pi_T(v)$ | $-\log\pi_S(v)$ |
| **Reverse KL（OPD）** | $\boxed{\pi_S(v)}$ | $\log\dfrac{\pi_S(v)}{\pi_T(v)}$ |

$$\boxed{\text{唯一的结构差别：reverse KL 的权重里也有 }\theta}$$

### 链式法则有两项

$$\nabla_\theta L=\underbrace{\sum_v \nabla_\theta w(v)\cdot f_\theta(v)}_{\text{(I) 权重在动}}\ +\ \underbrace{\sum_v w(v)\cdot\nabla_\theta f_\theta(v)}_{\text{(II) 被积函数在动}}$$

**SFT / forward KL**：权重是标签或 teacher 分布，**与 $\theta$ 无关** $\Rightarrow$ 第 (I) 项**压根不存在**，只剩 (II)。

**Reverse KL**：两项都在。而且对 reverse KL 还有个特别的事实 ——

$$\text{(II)}=\sum_v\pi_S(v)\nabla\log\frac{\pi_S(v)}{\pi_T(v)}=\sum_v\pi_S(v)\nabla\log\pi_S(v)=\nabla\sum_v\pi_S(v)=\nabla 1=0$$

$$\boxed{\text{reverse KL 的梯度信息\textbf{全部}在 (I) 里；SFT 的梯度信息全部在 (II) 里}}$$

实测（$V=5$，teacher 固定）：

| | (I) 权重项 | (II) 被积函数项 |
|---|---|---|
| Forward KL | `[0 0 0 0 0]`（不存在） | `[+0.5925 −0.1905 −0.5383 +0.1778 −0.0416]` |
| Reverse KL | `[+0.7289 −0.3269 −0.0872 −0.1851 −0.1297]` | `[0 0 0 0 0]`（恒为 0） |

**两者的梯度来源恰好互换了。** 这就是全部答案的来源。

### 采样把第 (I) 项弄丢了

reverse KL 实际是采样估计的（只要 teacher 给采样 token 的一个标量 logprob，$[B,L]$ 而不是 $[B,L,V]$）。写成代码：

```python
a = torch.multinomial(p, N)                # v_i 采出来只是一堆整数下标
loss = (logp[a] - q.log()[a]).mean()       # 表达式里已经没有 π_S(v) 这个因子了
```

$$\boxed{\text{权重 }\pi_S\text{ 被「吸收」进了「哪些下标被抽中」这件事}}$$

而下标 $v_i$ 在计算图里是**常数**。autograd 只看得见 $f_\theta(v_i)$，于是算出来的只有第 (II) 项 —— 而 (II) 恒等于 0。

实测：采样 + 天真 backprop 得到 `[+0.0000 −0.0000 +0.0000 +0.0001 −0.0001]`，**梯度全丢**。

$$\boxed{\text{“采样不可导”的准确含义：丢的不是采样这个操作，是「权重随 }\theta\text{ 变化」这一项}}$$

### PG 就是把第 (I) 项从样本里捞回来

$\sum_v\nabla\pi_S(v)f(v)$ 没法直接用样本估，因为它是「梯度的加权和」而不是「某个分布下的期望」。log-derivative trick 把它改写成期望：

$$\sum_v\nabla\pi_S(v)\,f(v)=\sum_v\pi_S(v)\,\nabla\log\pi_S(v)\,f(v)=\mathbb E_{v\sim\pi_S}\big[f(v)\,\nabla\log\pi_S(v)\big]$$

右边是 $\pi_S$ 下的期望，**可以用样本估**。这就是 policy gradient。

实测：采样 + PG 得到 `[+0.7289 −0.3269 −0.0874 −0.1850 −0.1296]`，和精确值差 $1.8\times10^{-4}$。

### 而 SFT / forward KL 采样时什么都不会丢

$$-\sum_v\pi_T(v)\log\pi_S(v)=\mathbb E_{v\sim\pi_T}\big[-\log\pi_S(v)\big]$$

从 $\pi_T$ 采样，**这个分布与 $\theta$ 无关**。本来就没有第 (I) 项，所以"吸收进采样"没有吸收掉任何东西，autograd 拿到的 (II) 就是完整梯度。

实测：forward KL 从 teacher 采样 + 直接 backprop，和全词表精确值差 $2.9\times10^{-4}$，**无偏**。

SFT 是这里的极端情形：权重是 one-hot，求和只剩一项，直接就是 $-\log\pi_S(y^*)$。

### 旁证：连续动作里就不需要 PG

diffusion / flow policy 的动作连续，可以**重参数化**（$x=\mu_\theta+\sigma_\theta\epsilon$）：随机性被挪到与 $\theta$ 无关的 $\epsilon$ 上，权重对 $\theta$ 的依赖重新变成表达式里的显式可导路径，第 (I) 项又能被 autograd 看见。

$$\boxed{\text{离散 token 无法重参数化，才是 LLM 侧非 PG 不可的根本原因}}$$

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

**7.** ⭐⭐ SFT 的 loss 不用 PG，OPD 的 reverse KL 要，结构差别在哪？

> **答：** 把两个都写成 $L=\sum_v w(v)f_\theta(v)$，**唯一差别是 reverse KL 的权重 $w=\pi_S$ 里也有 $\theta$**（SFT 的权重是 one-hot 标签、forward KL 的是 $\pi_T$，都与 $\theta$ 无关）。
> 链式法则两项：$\nabla L=\underbrace{\sum\nabla w\cdot f}_{(I)}+\underbrace{\sum w\cdot\nabla f}_{(II)}$。
> SFT / forward KL：**(I) 不存在**，只剩 (II)，普通 backprop 就是完整梯度。
> reverse KL：两项都在，而且 $(II)=\sum\pi_S\nabla\log\pi_S=\nabla 1=0$，**全部信息都在 (I)**。
> **枚举时**代码里 `p * (p.log()-q.log())` 的 $\pi_S$ 是显式可导因子，autograd 按乘法法则把 (I)(II) 都算了 → 不用 PG。
> **采样时**代码变成 `mean(logp[a]-logq[a])`，权重 $\pi_S$ 被「吸收进哪些下标被抽中」，而下标是常数 → autograd 只算得到 (II) ≡ 0 → **梯度几乎全零**。
> PG 就是用 log-derivative 把 (I) 改写成可采样的期望：$\sum\nabla\pi_S f=\mathbb E_{v\sim\pi_S}[f\nabla\log\pi_S]$。
> forward KL 采样时从 $\pi_T$ 采（θ 无关），本来就没有 (I)，所以什么都不丢。

