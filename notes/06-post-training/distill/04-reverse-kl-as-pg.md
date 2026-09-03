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

## 7. 完整梯度推导：SFT 与 OPD 从 logits 一路推到底

第 2 节是在抽象的 $\nabla_\theta$ 层面推的。这一节推到**logits 层面**，把 SFT 和 OPD 并排放一起，就能看清"为什么一个要 PG 一个不要"这件事到底发生在哪一步。

约定：学生网络输出 logits $z\in\mathbb R^{|\mathcal V|}$，

$$\pi_S(v)=\frac{e^{z_v}}{\sum_u e^{z_u}},\qquad \pi_T(v)\ \text{固定，与}\ \theta\ \text{无关}$$

所有推导都对 $z_k$ 求偏导；最后统一乘 Jacobian 即可回到参数：$\nabla_\theta L=\left(\frac{\partial z}{\partial \theta}\right)^{\!\top}\nabla_z L$。

### 7.1 预备：softmax 的两个导数

> 下面三个式子如果不熟（尤其 $\delta_{vk}$ 是什么、为什么会有 $v$ 和 $k$ 两个下标），先看 [01b softmax 求导的三个式子](01b-softmax-gradient.md)，那里从零推了一遍。

**log-softmax 的导数**（最常用的那个）：

$$\log\pi_S(v)=z_v-\log\sum_u e^{z_u}$$

$$\frac{\partial\log\pi_S(v)}{\partial z_k}=\delta_{vk}-\frac{e^{z_k}}{\sum_u e^{z_u}}=\boxed{\delta_{vk}-\pi_S(k)}$$

**softmax 本身的导数**，由上式乘 $\pi_S(v)$ 得到：

$$\frac{\partial\pi_S(v)}{\partial z_k}=\pi_S(v)\frac{\partial\log\pi_S(v)}{\partial z_k}=\boxed{\pi_S(v)\big(\delta_{vk}-\pi_S(k)\big)}$$

**一个贯穿全文的恒等式**（后面反复用）：

$$\sum_v\pi_S(v)\big(\delta_{vk}-\pi_S(k)\big)=\pi_S(k)-\pi_S(k)\underbrace{\sum_v\pi_S(v)}_{=1}=0$$

即 $\mathbb E_{v\sim\pi_S}[\nabla\log\pi_S(v)]=0$。**score function 的期望为零**，这是后面所有事情的枢纽。

### 7.2 SFT 的梯度

$$L_{\text{SFT}}=-\log\pi_S(y^*)$$

直接套 7.1 第一式，$v=y^*$：

$$\frac{\partial L_{\text{SFT}}}{\partial z_k}=-\big(\delta_{y^*k}-\pi_S(k)\big)$$

$$\boxed{\nabla_z L_{\text{SFT}}=\pi_S-e_{y^*}}$$

就是那个人人都背过的 **"预测分布减 one-hot"**。推导里**从头到尾没有出现过对权重求导这一步**，因为权重（one-hot 标签）是常数。

### 7.3 Forward KL 蒸馏的梯度

$$L_{\text{FKD}}=-\sum_v\pi_T(v)\log\pi_S(v)$$

（这是交叉熵；$D_{KL}(\pi_T\|\pi_S)=L_{\text{FKD}}-H(\pi_T)$ 只差一个与 $\theta$ 无关的常数，梯度相同。）

$$\frac{\partial L_{\text{FKD}}}{\partial z_k}=-\sum_v\pi_T(v)\big(\delta_{vk}-\pi_S(k)\big)=-\pi_T(k)+\pi_S(k)\underbrace{\sum_v\pi_T(v)}_{=1}$$

$$\boxed{\nabla_z L_{\text{FKD}}=\pi_S-\pi_T}$$

SFT 是它在 $\pi_T=e_{y^*}$ 时的特例。同样地，$\pi_T$ 全程作为常数被拎出求和号，**从未被求导**。

### 7.4 Reverse KL（OPD）全词表的梯度

记对数比

$$r(v):=\log\frac{\pi_S(v)}{\pi_T(v)},\qquad L_{\text{RKL}}=\sum_v\pi_S(v)\,r(v)$$

注意 $r$ 里也含 $\theta$。乘法法则，两项都要留：

$$\frac{\partial L_{\text{RKL}}}{\partial z_k}=\underbrace{\sum_v\frac{\partial\pi_S(v)}{\partial z_k}\,r(v)}_{\text{(I) 权重在动}}+\underbrace{\sum_v\pi_S(v)\frac{\partial r(v)}{\partial z_k}}_{\text{(II) 被积函数在动}}$$

**先算 (II)**。因为 $\pi_T$ 是常数，$\dfrac{\partial r(v)}{\partial z_k}=\dfrac{\partial\log\pi_S(v)}{\partial z_k}=\delta_{vk}-\pi_S(k)$，于是

$$\text{(II)}=\sum_v\pi_S(v)\big(\delta_{vk}-\pi_S(k)\big)=0$$

正是 7.1 的那个恒等式。**(II) 精确为零，不是近似。**

**再算 (I)**。代入 softmax 导数：

$$\text{(I)}=\sum_v\pi_S(v)\big(\delta_{vk}-\pi_S(k)\big)r(v)=\pi_S(k)r(k)-\pi_S(k)\sum_v\pi_S(v)r(v)$$

而 $\sum_v\pi_S(v)r(v)=D_{KL}(\pi_S\|\pi_T)=L_{\text{RKL}}$ 本身。所以

$$\boxed{\frac{\partial L_{\text{RKL}}}{\partial z_k}=\pi_S(k)\Big(\log\frac{\pi_S(k)}{\pi_T(k)}-D_{KL}(\pi_S\|\pi_T)\Big)}$$

三件事值得注意：

1. **梯度全部来自 (I)**，和 SFT / forward KL 恰好互换（那两个的梯度全部来自 (II)，因为它们的 (I) 压根不存在）。
2. **括号里自动出现了一个 baseline** $-D_{KL}$，即"该 token 的对数比减去平均对数比"。这不是谁手工加的方差缩减项，是 softmax Jacobian 里的 $-\pi_S(k)\sum_v\pi_S(v)(\cdot)$ 自己长出来的。这正是 PG 里 advantage 的雏形（第 3 节讲的就是它）。
3. **合法性自检**：$\sum_k\frac{\partial L}{\partial z_k}=\sum_k\pi_S(k)r(k)-D_{KL}\sum_k\pi_S(k)=0$。梯度与 $\mathbf 1$ 正交 —— 必须如此，因为所有 logits 同加一个常数不改变 softmax。$\nabla_z L_{\text{FKD}}=\pi_S-\pi_T$ 也满足（$1-1=0$）。

**到这一步为止，全词表 reverse KL 是个普通的可导表达式，`.backward()` 就能算，不需要 PG。**

### 7.5 采样之后：天真 backprop 得到的是零

实践中词表枚举太贵（reverse KL 的卖点就是只要 teacher 给采样 token 的一个标量 logprob，$[B,L]$ 而不是 $[B,L,V]$）。于是把求和换成采样：$v_1,\dots,v_N\sim\pi_S$，

$$\hat L=\frac1N\sum_{i=1}^N r(v_i)$$

**这个 $\hat L$ 的值是无偏的**：$\mathbb E[\hat L]=\sum_v\pi_S(v)r(v)=L$。日志里打印出来的 loss 完全正确。

但对它求导：$v_i$ 是采出来的整数下标，在计算图里是常数，autograd 只能沿 $r(v_i)=\log\pi_S(v_i)-\log\pi_T(v_i)$ 这条路走：

$$\frac{\partial\hat L}{\partial z_k}=\frac1N\sum_i\frac{\partial\log\pi_S(v_i)}{\partial z_k}=\frac1N\sum_i\big(\delta_{v_ik}-\pi_S(k)\big)$$

取期望：

$$\mathbb E\left[\frac{\partial\hat L}{\partial z_k}\right]=\mathbb E_{v\sim\pi_S}[\delta_{vk}]-\pi_S(k)=\pi_S(k)-\pi_S(k)=0$$

$$\boxed{\text{天真采样梯度是「零」的无偏估计 —— 它整个就是噪声}}$$

对照 7.4 就知道发生了什么：把 $\sum_v\pi_S(v)(\cdot)$ 换成 $\frac1N\sum_i(\cdot)$ 之后，**权重 $\pi_S(v)$ 不再是表达式里的一个可导因子，而是被"哪些下标被抽中"这件事吸收掉了**。autograd 看不见它，于是只算出 (II)，而 (II) 恒等于 0。

$$\boxed{\text{"采样不可导"丢的不是采样这个操作，是「权重随 }\theta\text{ 变化」这一整项}}$$

值得记住的是这个 bug 的形态：**loss 数值正常，梯度接近零，模型不动。**

### 7.6 PG：把丢掉的 (I) 变成可采样的期望

问题在于 (I) $=\sum_v\nabla\pi_S(v)\,r(v)$ 是"梯度的加权和"，不是"某个分布下的期望"，没法直接用样本估。log-derivative（score function）恒等式

$$\nabla\pi_S(v)=\pi_S(v)\,\nabla\log\pi_S(v)$$

把它改写成 $\pi_S$ 下的期望：

$$\nabla_\theta L_{\text{RKL}}=\underbrace{\mathbb E_{v\sim\pi_S}\big[r(v)\,\nabla\log\pi_S(v)\big]}_{\text{(I)}}+\underbrace{0}_{\text{(II)}}$$

估计量（含任意常数 baseline $b$，因为 $\mathbb E[\nabla\log\pi_S]=0$ 所以不引入偏差）：

$$\widehat{\nabla L}=\frac1N\sum_i\big(r(v_i)-b\big)\nabla\log\pi_S(v_i)$$

**验证它确实收敛到 7.4 的闭式解**（取 $b=0$，对 $z_k$）：

$$\mathbb E\left[r(v)\big(\delta_{vk}-\pi_S(k)\big)\right]=\pi_S(k)r(k)-\pi_S(k)\sum_v\pi_S(v)r(v)=\pi_S(k)\big(r(k)-D_{KL}\big)\ \checkmark$$

和 $\boxed{\pi_S(k)(r(k)-D_{KL})}$ 完全一致。而且这里能看出 **$b=D_{KL}(\pi_S\|\pi_T)$ 是最自然的 baseline** —— 全词表版本免费自带它，采样版本得自己估（用 batch 内均值）。

**写成代码的形式**：autograd 没法直接算 $\nabla\log\pi_S$ 加权和，所以构造一个 surrogate loss，让它的梯度恰好等于上式：

$$\tilde L=\frac1N\sum_i \text{sg}\big[r(v_i)-b\big]\cdot\log\pi_S(v_i)$$

$\text{sg}[\cdot]$ 是 stop-gradient（`.detach()`）。对 $\tilde L$ 调 `.backward()` 得到的就是 PG 估计量。**$\tilde L$ 的数值没有意义，只有它的梯度有意义** —— 这是所有 PG 实现里最容易看懵的一行。

（注：也可以不 detach $r$，把 (II) 一并留在图里。因为 (II) 期望为零，这样做不引入偏差，只是多一项零均值噪声。两种写法都对。）

### 7.7 序列级：OPD 就长成了 RL

上面是单步。整条序列 $y=(y_1,\dots,y_T)$ 时，$\log\pi_S(y)=\sum_t\log\pi_S(y_t|y_{<t})$，逐 token 对数比记 $r_t=\log\dfrac{\pi_S(y_t|y_{<t})}{\pi_T(y_t|y_{<t})}$，则

$$L=\mathbb E_{y\sim\pi_S}\Big[\sum_t r_t\Big],\qquad \nabla_\theta L=\mathbb E_{y\sim\pi_S}\Big[\Big(\sum_t r_t\Big)\nabla_\theta\log\pi_S(y)\Big]$$

再用因果性（$t$ 时刻的动作影响不到 $t'<t$ 的 reward）把它拆成 reward-to-go 形式：

$$\boxed{\nabla_\theta L=\mathbb E\left[\sum_t\nabla_\theta\log\pi_S(y_t|y_{<t})\Big(\sum_{t'\ge t}r_{t'}-b_t\Big)\right]}$$

这就是标准 policy gradient，**reward 为 $-r_t$**（因为我们最小化 $L$）。到这一步 OPD 和 PPO/GRPO 的 trainer 已经可以共用（第 4 节）。

### 7.8 并排小结

| | 权重 $w(v)$ | 权重含 $\theta$？ | 梯度来自 | 采样时 |
|---|---|---|---|---|
| **SFT** | $e_{y^*}$ | 否 | (II)，$\nabla_z L=\pi_S-e_{y^*}$ | 从数据采，与 $\theta$ 无关，**无损** |
| **Forward KL** | $\pi_T$ | 否 | (II)，$\nabla_z L=\pi_S-\pi_T$ | 从 $\pi_T$ 采，与 $\theta$ 无关，**无损** |
| **Reverse KL** | $\pi_S$ | **是** | (I)，$\nabla_z L=\pi_S(k)(r(k)-D_{KL})$ | 从 $\pi_S$ 采，(I) 被吸收，**必须 PG** |

$$\boxed{\text{分水岭是「}L=\sum_v w(v)f_\theta(v)\text{ 里的权重 }w\text{ 含不含 }\theta\text{」}}$$

### 7.9 旁证：连续动作为什么不需要 PG

diffusion / flow policy 的动作是连续的，可以**重参数化**：$x=\mu_\theta+\sigma_\theta\epsilon$，$\epsilon\sim\mathcal N(0,I)$。随机性被挪到与 $\theta$ 无关的 $\epsilon$ 上，$x$ 重新变成 $\theta$ 的可导函数，于是"权重随 $\theta$ 变"这件事重新回到表达式里的显式可导路径上，autograd 又能看见 (I) 了。

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

