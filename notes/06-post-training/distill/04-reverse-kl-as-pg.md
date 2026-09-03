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

## 7. SFT 与 OPD 的梯度：为什么一个要 PG，一个不要

> 这一节全程停在 $\nabla_\theta$ 层面，**不需要任何指标运算**。
> logits 层面的闭式解放在 7.9 附录，先看懂这里再去看那里。

### 7.1 用两个 token 把话说完

词表只有 $A,B$ 两个 token：

$$\pi_S=(0.8,\ 0.2),\qquad \pi_T=(0.6,\ 0.4)$$

**SFT**（假设数据集给的正确答案是 $B$）：

$$L=-\log\pi_S(B)=-\log 0.2$$

把 $B$ 的概率从 0.2 往上拉就完事了。关键在于 **$B$ 是数据集告诉你的，不是 student 自己抽出来的**。

```python
loss = -log_p_student[B]
loss.backward()          # 完事
```

**Forward KL**（忽略与 student 无关的常数项，就是交叉熵）：

$$L=-\sum_v\pi_T(v)\log\pi_S(v)=-0.6\log 0.8-0.4\log 0.2$$

前面的权重 $0.6,\ 0.4$ 是谁给的？**teacher。student 参数怎么变，这两个数都不变。** 所以它本质就是 soft-label 版的 SFT：

$$\nabla_z L=\pi_S-\pi_T$$

一样直接 backward，不需要 PG。

**Reverse KL（OPD）**：

$$L=\sum_v\pi_S(v)\log\frac{\pi_S(v)}{\pi_T(v)}=0.8\log\frac{0.8}{0.6}+0.2\log\frac{0.2}{0.4}$$

现在盯住被求和的这一项：

$$\pi_S(v)\ \log\frac{\pi_S(v)}{\pi_T(v)}$$

$$\boxed{\text{student 出现了两次：一次在外面当权重，一次在 log 里面}}$$

**这就是全部差别的来源。** SFT 和 forward KL 里，外面那个权重是数据集或 teacher 给的死数；reverse KL 里，外面那个权重就是 student 自己。

### 7.2 出现两次，就必须用乘法法则

记对数比 $r(v)=\log\dfrac{\pi_S(v)}{\pi_T(v)}$，则 $L=\sum_v\pi_S(v)\,r(v)$。两个因子都含 $\theta$，求导必须拆成两项：

$$\nabla\big[\pi_S(v)\,r(v)\big]=\underbrace{\nabla\pi_S(v)\cdot r(v)}_{\text{(I) 权重在动}}+\underbrace{\pi_S(v)\cdot\nabla r(v)}_{\text{(II) log 里面在动}}$$

而 SFT / forward KL 只有 (II) 那一项 —— 因为它们的权重是常数，$\nabla(\text{常数})=0$，(I) 压根不存在。

### 7.3 (II) 恒等于 0，所以梯度全在 (I)

因为 $\pi_T$ 是常数，$\nabla r(v)=\nabla\log\pi_S(v)$，于是

$$\text{(II)}=\sum_v\pi_S(v)\nabla\log\pi_S(v)$$

用 $\nabla\log\pi=\dfrac{\nabla\pi}{\pi}$，把 $\pi_S(v)$ 约掉：

$$\text{(II)}=\sum_v\nabla\pi_S(v)=\nabla\Big(\underbrace{\sum_v\pi_S(v)}_{\equiv\,1}\Big)=\nabla 1=0$$

$$\boxed{\text{概率永远加起来等于 1，所以「所有概率的总变化量」必然是 0}}$$

于是：

$$\boxed{\text{Reverse KL 真正有用的梯度全部来自 (I)}=\sum_v\nabla\pi_S(v)\,r(v)}$$

**和 SFT 恰好相反** —— SFT 的梯度全部来自 (II)，因为它没有 (I)。

### 7.4 全词表：autograd 两项都看得见，不需要 PG

如果你真的枚举整个词表：

```python
p = softmax(student_logits)
q = softmax(teacher_logits)
loss = (p * (p.log() - q.log())).sum()
loss.backward()
```

PyTorch 看得见完整的计算图：

```text
        θ
        ↓
   student logits
        ↓
        p
    ┌───┴───┐
    ↓       ↓
  外面的 p   log p        ← p 沿两条路都通向 loss
    └───┬───┘
        ↓
       loss
```

**$p$ 通向 loss 有两条路**，乘法法则是 autograd 自己会做的事，(I) 和 (II) 它都算得出来（只是 (II) 最后正好抵消为 0）。

$$\boxed{\text{全词表 reverse KL 根本不需要 policy gradient}}$$

### 7.5 采样：权重被藏进「抽到了哪个 token」

现实里 OPD 不想让 teacher 返回整个 $[B,L,V]$ 的 logits（这正是它相对 forward KL 的成本优势，见 [KL 估计粒度](05-kl-estimation.md)）。做法是：student 自己生成一个 token，teacher 只回一个标量 $\log\pi_T(A)$。

```python
a = sample(p)                       # 抽到 A，是个整数 token id，比如 4392
loss = log_p_student[a] - log_p_teacher[a]
```

计算图变成：

```text
        θ
        ↓
   student logits
        ↓
        p ──→ sample ──→ A = 4392
        ↓                  ✗  ← 这条边断了
     log p[A] ────────→ loss
```

断掉的是哪一条？**「为什么刚才更容易抽到 A」这件事。** 采样把整个分布 $(0.8,0.2)$ 压成了一个离散结果 $A$，而 $A$ 是个常数整数，autograd 无从知道 $\theta$ 变化会让 $A$ 更容易还是更不容易被抽到。

对照 7.3 的两项：

$$\text{原本要算}\quad\sum_v\underbrace{\nabla\pi_S(v)}_{\text{抽到谁的概率在变}}\,r(v)$$

采样之后你只看得见 $r(v)$，**原来那个 $\pi_S(v)$ 被隐藏进了"这个 token 被抽到的概率"里面**，autograd 看不到它。

$$\boxed{\text{“采样不可导”不是说 loss 不可导，是「采样概率随参数变化」这条梯度丢了}}$$

丢掉的恰好是 (I) —— 也就是**全部有用的梯度**。剩下的 (II) 恒为 0。

### 7.6 天真 backward 为什么期望是零

抽到 $A$ 后直接对 $L=\log\pi_S(A)-\log\pi_T(A)$ 求导，autograd 给你的只有

$$\nabla\log\pi_S(A)$$

但这个量在 $\pi_S$ 下的平均，正是 7.3 里已经证过的那个 0：

$$\mathbb E_{A\sim\pi_S}\big[\nabla\log\pi_S(A)\big]=\sum_a\pi_S(a)\nabla\log\pi_S(a)=\sum_a\nabla\pi_S(a)=\nabla 1=0$$

$$\boxed{\text{loss 的\textbf{数值}是对的（无偏），但它的 backward 不是 reverse KL 的梯度，期望还是 0}}$$

这是整节最反直觉的地方，也是这个 bug 最阴险的地方：**日志里 loss 打印得完全正常，模型就是不动。**

### 7.7 PG：把藏起来的那条梯度估回来

我们要算 $\sum_v\nabla\pi_S(v)\,r(v)$，但手上只有样本。用 log-derivative 恒等式

$$\nabla\pi(v)=\pi(v)\,\nabla\log\pi(v)$$

代进去：

$$\sum_v\nabla\pi_S(v)\,r(v)=\sum_v\pi_S(v)\,r(v)\,\nabla\log\pi_S(v)=\boxed{\mathbb E_{v\sim\pi_S}\big[r(v)\,\nabla\log\pi_S(v)\big]}$$

$$\boxed{\sum_v(\cdots)\ \text{变成了}\ \mathbb E_{v\sim\pi_S}[\cdots]\text{，于是可以采样估}}$$

抽 $v_1,\dots,v_N\sim\pi_S$：

$$\widehat{\nabla L}=\frac1N\sum_i r(v_i)\,\nabla\log\pi_S(v_i)$$

这就是 **policy gradient**。一句话背下来：

> PG 是用 $\nabla\pi(a)=\pi(a)\nabla\log\pi(a)$，把「采样概率本身的梯度」转换成对**已采样 action** 的 $\nabla\log\pi(a)$，从而让 Monte Carlo 能估。

这就是 log trick / score-function estimator。

### 7.8 surrogate loss 和 detach 是什么

autograd 没法直接"算一个 $\nabla\log\pi$ 的加权和"，只能对某个标量调 `.backward()`。所以人工构造一个 loss，让它的梯度恰好等于我们要的东西：

$$\tilde L=\frac1N\sum_i\operatorname{sg}\big[r(v_i)-b\big]\cdot\log\pi_S(v_i)$$

$\operatorname{sg}[\cdot]$ 就是 stop-gradient，代码里的 `.detach()`。对 $\tilde L$ 求导，$\operatorname{sg}$ 里的东西被当常数，剩下的正好是 $r\cdot\nabla\log\pi_S$。

$$\boxed{\tilde L\ \text{不是 KL loss，它的\textbf{数值没有意义}，只有它的梯度有意义}}$$

这是所有 PG 实现里最容易看懵的一行代码。$b$ 是 baseline，因为 $\mathbb E[\nabla\log\pi_S]=0$，减任何与样本无关的常数都不改变期望，只降方差。

### 7.9 一张总图

```text
                    SFT
          label y 来自数据集（死数）
                    ↓
              -log πS(y)
                    ↓
              直接 backward


                Forward KL
          权重 πT 来自 teacher（死数）
                    ↓
          Σ πT(v)·[-log πS(v)]
                    ↓
              直接 backward


                Reverse KL
          权重 πS 来自 student（会动）
                    ↓
        Σ πS(v)·log[πS(v)/πT(v)]
             ↓              ↓
         权重也会动      log 里面会动
           (I)              (II)
            ↓                ↓
        真正的梯度            0
            ↓
  全词表   → autograd 两条路都看得见 → 直接 backward
            ↓
  student 采样 → πS 被藏进「抽到了哪个 token」
            ↓
        autograd 看不到 → 丢掉 (I) → 梯度期望为 0
            ↓
        用 ∇π = π∇logπ 重写成期望
            ↓
          Policy Gradient
```

**先不用记 7.11 附录那个 logits 公式**，记住三句就够：

$$\boxed{\text{Reverse KL 的权重本身就是 student 的分布}}$$
$$\boxed{\text{一旦用采样替代全词表求和，这个权重的梯度就被采样藏起来了}}$$
$$\boxed{\text{PG 就是把这条被藏起来的梯度重新估出来}}$$

### 7.10 序列级：OPD 就长成了 RL

上面是单步。整条序列 $y=(y_1,\dots,y_T)$ 时，$\log\pi_S(y)=\sum_t\log\pi_S(y_t|y_{<t})$，逐 token 对数比记 $r_t=\log\dfrac{\pi_S(y_t|y_{<t})}{\pi_T(y_t|y_{<t})}$，于是

$$L=\mathbb E_{y\sim\pi_S}\Big[\sum_t r_t\Big],\qquad \nabla_\theta L=\mathbb E_{y\sim\pi_S}\Big[\Big(\sum_t r_t\Big)\nabla_\theta\log\pi_S(y)\Big]$$

再用因果性（$t$ 时刻的 token 影响不到 $t'<t$ 的 $r_{t'}$）拆成 reward-to-go：

$$\boxed{\nabla_\theta L=\mathbb E\left[\sum_t\nabla_\theta\log\pi_S(y_t|y_{<t})\Big(\sum_{t'\ge t}r_{t'}-b_t\Big)\right]}$$

标准的 policy gradient，**reward 为 $-r_t$**（因为我们最小化 $L$）。到这一步 OPD 和 PPO/GRPO 的 trainer 已经可以共用（第 4 节）。

### 7.11 附录（选读）：logits 层面的闭式解

> 上面全程停在 $\nabla_\theta$，够用了。这里补 logits 层面的精确表达式 —— 面试被追问"写出梯度"时用。
> 需要三个前置式子，不熟的先看 [01b softmax 求导](01b-softmax-gradient.md)：
> $\dfrac{\partial\log\pi_S(v)}{\partial z_k}=\delta_{vk}-\pi_S(k)$，$\dfrac{\partial\pi_S(v)}{\partial z_k}=\pi_S(v)(\delta_{vk}-\pi_S(k))$，$\sum_v\pi_S(v)(\delta_{vk}-\pi_S(k))=0$。

**SFT**：$L=-\log\pi_S(y^*)$，直接套第一式：

$$\frac{\partial L}{\partial z_k}=-\big(\delta_{y^*k}-\pi_S(k)\big)\quad\Longrightarrow\quad\boxed{\nabla_z L=\pi_S-e_{y^*}}$$

**Forward KL**：$L=-\sum_v\pi_T(v)\log\pi_S(v)$，

$$\frac{\partial L}{\partial z_k}=-\sum_v\pi_T(v)\big(\delta_{vk}-\pi_S(k)\big)=-\pi_T(k)+\pi_S(k)\underbrace{\sum_v\pi_T(v)}_{=1}\quad\Longrightarrow\quad\boxed{\nabla_z L=\pi_S-\pi_T}$$

SFT 是它在 $\pi_T=e_{y^*}$ 时的特例。两个推导里 $\pi_T$（或 one-hot）**始终作为常数被拎出求和号，从未被求导** —— 这就是 7.2 说的"(I) 不存在"。

**Reverse KL**：$L=\sum_v\pi_S(v)r(v)$。(II) 项就是第三式，精确为 0；(I) 项代入第二式：

$$\text{(I)}=\sum_v\pi_S(v)\big(\delta_{vk}-\pi_S(k)\big)r(v)=\pi_S(k)r(k)-\pi_S(k)\sum_v\pi_S(v)r(v)$$

而 $\sum_v\pi_S(v)r(v)=D_{KL}(\pi_S\|\pi_T)=L$ 本身，所以

$$\boxed{\frac{\partial L}{\partial z_k}=\pi_S(k)\Big(\log\frac{\pi_S(k)}{\pi_T(k)}-D_{KL}(\pi_S\|\pi_T)\Big)}$$

两个值得注意的点：

1. **括号里自动出现了一个 baseline** $-D_{KL}$，即"该 token 的对数比减去平均对数比"。这不是谁手工加的方差缩减项，是 softmax Jacobian 里的 $-\pi_S(k)\sum_v\pi_S(v)(\cdot)$ 自己长出来的 —— 也就是 7.8 里那个 $b$ 的来历。全词表版本免费自带它，采样版本得自己估（用 batch 内均值）。
2. **合法性自检**：$\sum_k\frac{\partial L}{\partial z_k}=\sum_k\pi_S(k)r(k)-D_{KL}\sum_k\pi_S(k)=0$。梯度与 $\mathbf 1$ 正交 —— 必须如此，因为所有 logits 同加一个常数不改变 softmax。$\nabla_z L=\pi_S-\pi_T$ 也满足（$1-1=0$）。

**验证 PG 估计量确实收敛到这个闭式解**（取 $b=0$）：

$$\mathbb E_{v\sim\pi_S}\big[r(v)(\delta_{vk}-\pi_S(k))\big]=\pi_S(k)r(k)-\pi_S(k)\sum_v\pi_S(v)r(v)=\pi_S(k)\big(r(k)-D_{KL}\big)\ \checkmark$$

### 7.12 旁证：连续动作为什么不需要 PG

diffusion / flow policy 的动作是连续的，可以**重参数化**：$x=\mu_\theta+\sigma_\theta\epsilon$，$\epsilon\sim\mathcal N(0,I)$。随机性被挪到与 $\theta$ 无关的 $\epsilon$ 上，$x$ 重新变成 $\theta$ 的可导函数，7.5 里那条断掉的边就接回来了，autograd 又能看见 (I)。

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

