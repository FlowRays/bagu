# 从 J 到 Loss：policy gradient 的四层概念

> 这一篇是整条 RL 主线的地基。学习时最大的困惑就出在这里：**J、E、Loss、gradient 混成一团**。
> 后面 PPO 的 $rA$、clip、min 全都建立在"什么是 surrogate objective"这个理解上。

## 卡点 1（核心）：J / E / L / ∇L 到底是什么关系

> **常见卡点**：J、E、Loss、gradient 这几层推导混在一起，单看每一步都懂，串起来就说不清。

先把四个东西彻底分开：

| 概念 | 是什么 | 谁需要它 |
|---|---|---|
| $J(\theta)$ | 真正想优化的 RL 目标（期望回报） | 我们心里的目标 |
| $\mathbb E[\cdot]$ | 对随机样本取平均，代码里就是 batch average | 只是记号，不是新目标 |
| $L(\theta)$ | 为了方便训练**人为构造**的 objective / loss | autograd |
| $\nabla_\theta L$ | optimizer 实际用来更新参数的东西 | optimizer |

整条逻辑链是：

$$J \;\longrightarrow\; \nabla J \;\longrightarrow\; \text{构造 } L \text{ 使 } \nabla L = \nabla J \;\longrightarrow\; \theta \leftarrow \theta + \alpha\nabla L$$

**最需要修正的一句话**：

- ❌ "$\mathbb E[A\log\pi]$ 等价于 $J$。"
- ✅ "$J$ 是真正目标；policy gradient 定理给出 $\nabla J$ 的形式；然后我们构造 $A\log\pi$ 这个 surrogate，让它的**梯度**正好是我们想要的 policy gradient。"

## 1. J 是真正目标

一个 trajectory $\tau=(s_0,a_0,r_0,s_1,\dots)$，policy 是 $\pi_\theta(a|s)$，目标：

$$\boxed{J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\Big[\sum_t \gamma^t r_t\Big]}$$

含义：用当前 policy 玩很多局，最大化平均总 reward。

## 2. 为什么不能直接对 J 求导

因为 $\theta$ 不直接出现在 reward 里，而是：

$$\theta \rightarrow \pi_\theta \rightarrow \text{采样 action} \rightarrow \text{trajectory} \rightarrow \text{reward}$$

reward 由环境给，中间隔着一次采样，没法像监督学习那样直接 backprop。

## 3. 关键澄清：PG 定理给的是 ∇J，不是另一个 J

Policy Gradient Theorem：

$$\boxed{\nabla_\theta J=\mathbb E_{s,a\sim\pi_\theta}\big[A(s,a)\,\nabla_\theta\log\pi_\theta(a|s)\big]}$$

它说的是**梯度长什么样**，它**没有说** $J=\mathbb E[A\log\pi]$。这两个 objective 本身并不相等。

实际含义极其简单：

$$A>0 \Rightarrow \text{提高这个 action 的概率},\qquad A<0 \Rightarrow \text{降低这个 action 的概率}$$

PPO 后面所有花样，本质都没改变这一件事。

## 4. 那代码里的 $A\log\pi$ 是哪来的

因为 autograd 需要一个可以 `.backward()` 的标量。我们发现：

$$\nabla_\theta\big(A\log\pi_\theta(a|s)\big)=A\nabla_\theta\log\pi_\theta(a|s)$$

右边正好是 policy gradient。所以人为构造：

$$\boxed{L_{PG}=\mathbb E[A\log\pi_\theta]}$$

它的意义**不是** "$L$ 等于真正的 $J$"，而是"$L$ 的梯度等于我们想要的 $\nabla J$"。

$$\boxed{J \ne L},\qquad \text{但我们要} \boxed{\nabla J=\nabla L}$$

这就是 **surrogate objective（代理目标）**。

### 监督学习类比

真正目标是"考试分数最高"（$J$），但你不能对考试分数 backprop，于是设计一个交叉熵 loss（$L$）。你并不认为"考试分数 = 负交叉熵"，只是认为"降低这个 loss 会让模型朝提高分数的方向走"。

RL 的区别只是：这里有定理告诉你，这个 surrogate 的梯度**正好**对应 policy gradient。

## 5. E 只是"取平均"

$\mathbb E$ 不是一种新目标，只是"对很多可能出现的样本取平均"。代码里就近似成：

$$\mathbb E[\cdot]\approx\frac1N\sum_{i=1}^N(\cdot)$$

- $J=\mathbb E_{\tau\sim\pi_\theta}[G(\tau)]$：rollout 很多条 trajectory，平均 reward。
- $L=\mathbb E[A\log\pi]$：对 rollout batch 里很多个 $(s,a)$ 的 $A\log\pi$ 取平均。

## 6. 到这里的完整链条

$$J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[G] \quad\text{（真正目标）}$$

$$\Downarrow \text{ Policy Gradient Theorem}$$

$$\nabla J=\mathbb E_{\pi_\theta}[A\nabla\log\pi_\theta]$$

$$\Downarrow \text{ 构造 surrogate 让 autograd 产生它}$$

$$L_{PG}=\mathbb E_{\pi_\theta}[A\log\pi_\theta]$$

$$\Downarrow$$

$$\theta \leftarrow \theta+\alpha\nabla_\theta L_{PG}$$

## 7. 一个隐藏前提（PPO 的入口）

上面这个 $\nabla J=\mathbb E_{s,a\sim\pi_\theta}[\cdot]$ 有一个特别重要的条件：

$$\boxed{s,a \text{ 必须是由当前的 } \pi_\theta \text{ 采出来的}}$$

也就是：用 $\pi_\theta$ 跑环境 → 得到 $(s,a)$ → 算 $A$ → 做一次更新 → **把数据扔掉** → 重新 rollout。这样完全没问题。

问题出在 PPO 想重复利用同一批 rollout，此时 $\pi_\theta \ne \pi_{old}$，前提被打破。这就是下一篇 [importance sampling](02-importance-sampling-and-ratio.md) 的起点。

## 自测

1. $J$ 和 $L$ 是什么关系？为什么可以用 $L$ 训练？
2. 为什么说 "$J=\mathbb E[A\log\pi]$" 是错的？
3. 写出 policy gradient 定理，说明它给的是什么。
4. policy gradient 成立的隐藏前提是什么？PPO 在哪一步破坏了它？
