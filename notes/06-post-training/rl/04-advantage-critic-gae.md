# Advantage / Critic / GAE：A 到底从哪来

> PPO 的 actor 更新靠 $A_t$，但 $A(s,a)=Q(s,a)-V(s)$ 里 $Q,V$ 都不知道。这一篇讲怎么估出来。
> 卡点：critic 到底学什么、bias/variance 到底指什么、$V_{old}$ 是什么、为什么 target 叫 $\hat R$。

## 1. 为什么需要 baseline

最简单的 advantage 就是 $A_t=G_t$（Monte Carlo return），这就是 REINFORCE。但 variance 非常大：同一个 action，一局最后 reward 是 100，另一局是 20，可能根本不是这个 action 导致的，只是后面随机性不同。

引入 $V(s)$ 后：

$$A_t=G_t-V(s_t)$$

问的问题变成：**这次结果，相比"从这个 state 正常应该拿到多少"，到底好多少？**

- $G_t=100$ 但 $V(s_t)=95$ → $A=5$，其实只是略好
- $G_t=20$ 但 $V(s_t)=2$ → $A=18$，反而是非常优秀的 action

所以 baseline 极大降低 variance。这就是 critic 的核心作用：**给 return 提供 baseline（"本来应该有多好"）**，而不是直接选 action。

## 2. Actor 和 Critic 的分工

| | 网络 | 回答什么 |
|---|---|---|
| Actor | $\pi_\theta(a|s)$ | 做什么？ |
| Critic | $V_\phi(s)$ | 这个状态本身未来大概有多好？ |

可以共享 backbone 分两个 head（游戏 RL 常见）。LLM RL 通常更复杂：actor / critic / reference / reward 可能是四个不同模型。

## 3. Critic 到底学什么

> **常见卡点**："critic 这里讲得太含糊了。"

critic 学的是：

$$\boxed{V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s]}$$

即"在状态 $s$ 下，按当前 policy 继续玩，未来 return 的**期望**"。

比如从 $s_t$ 出发玩很多次，return 分别是 $8,12,10,9,11$，平均 10，那么真实 $V^\pi(s_t)=10$，critic 网络就是要逼近这个 10。

**问题是我们不知道真实的 $V^\pi(s_t)$**，因为那需要从同一个 $s_t$ 出发 rollout 无数次取平均，现实中不可能。所以必须构造一个 **target**。

最简单的 target 就是 $G_t$：这一次 rollout 后面真的拿到的 return。

单看一次当然不准（真实值是 10，这次可能是 12），但如果有很多 rollout $8,12,10,9,11,\dots$，critic 不断拟合这些 noisy target，平均下来就会逼近 10。

$$\boxed{\text{critic training} = \text{用单次 sampled return 去估计 expected return}}$$

所以 critic 本质就是**一个监督回归器**：输入 $s$，输出 $V(s)$，label 是 return target。

$$L_V=\tfrac12\big(V_\phi(s_t)-G_t\big)^2$$

## 4. 两种 target：MC vs TD

$G_t$ 虽然可用但很 noisy。另一种做法是不等 trajectory 结束，用 Bellman 方程：

$$V^\pi(s_t)=\mathbb E[r_t+\gamma V^\pi(s_{t+1})]$$

于是可以训练 $V_\phi(s_t)\approx r_t+\gamma V_\phi(s_{t+1})$。这叫 **bootstrapping**：用自己对下一状态的预测，来构造当前状态的 target。

| | target | 特点 |
|---|---|---|
| Monte Carlo | $G_t$ | 看完整 future，不依赖下一步预测，**方差大、bias 小** |
| TD(0) | $r_t+\gamma V(s_{t+1})$ | 只看一步真实 reward，后面全靠 critic，**方差低、bias 大** |

对应的 **TD error**：

$$\boxed{\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)}$$

它为什么近似 advantage？因为 $Q(s_t,a_t)\approx r_t+\gamma V(s_{t+1})$，而 $A=Q-V$，所以 $A_t\approx\delta_t$。

直觉：critic 原来预测这个状态值 $V(s_t)$，走一步后真实拿到 $r_t$ 并进入一个价值 $V(s_{t+1})$ 的状态，新旧估计做差就是 $\delta_t$。$\delta_t>0$ 说明这一步比 critic 预期更好。

## 卡点 8：bias 和 variance 到底在说什么

设真实 advantage 是 $A^*=10$（我们不知道，只能估计）。

**偏差 Bias**：一种方法估很多次得到 $6,7,6,7,6$，很稳定但平均只有 6.4，离真值 10 很远 → **高偏差、低方差**。
> 偏差关心：估计**平均而言**离真实答案有多远。

**方差 Variance**：另一种方法得到 $2,18,5,15,10$，平均正好 10（几乎无 bias），但每次都跳得厉害 → **低偏差、高方差**。
> 方差关心：同样的东西重复估计，结果有多飘。

> ⚠️ 关键澄清：**bias 不是"这个样本估错了多少"，variance 也不是"reward 大不大"**。两者都是"假设你对同一个量重复估计很多次"之后，描述这个**估计器**性质的概念。

放回来：

- $\lambda=0$（只用 $\delta_t$）：不看后面真正发生了什么，直接相信 critic → 随机性少，**低方差**；但 critic 有误就系统性被带偏，**高偏差**
- $\lambda=1$（完整 MC）：不用 critic bootstrap，critic 的错误不会一层层带进来 → **低偏差**；但后面所有随机事件都影响 $G_t$（同一 action 可能 $G=30/5/17$）→ **高方差**

## 5. GAE：λ=1 的完整推导（telescoping）

**GAE = Generalized Advantage Estimation（广义优势估计）**，在 TD(0) 和 MC 两个极端之间连续插值：

$$\boxed{\hat A_t^{GAE(\gamma,\lambda)}=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}=\delta_t+\gamma\lambda\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+\cdots}$$

### λ=1 时逐项展开

$$\hat A_t=\delta_t+\gamma\delta_{t+1}+\gamma^2\delta_{t+2}+\cdots$$

$$\delta_t=r_t+\gamma V_{t+1}-V_t$$
$$\gamma\delta_{t+1}=\gamma r_{t+1}+\gamma^2V_{t+2}-\gamma V_{t+1}$$
$$\gamma^2\delta_{t+2}=\gamma^2r_{t+2}+\gamma^3V_{t+3}-\gamma^2V_{t+2}$$

相加，看 value 项：$+\gamma V_{t+1}$ 与 $-\gamma V_{t+1}$ 抵消，$+\gamma^2V_{t+2}$ 与 $-\gamma^2V_{t+2}$ 抵消，后面同理**全部 telescope cancellation**。最后只剩：

$$\hat A_t=\underbrace{r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots}_{G_t}-V_t$$

$$\boxed{\hat A_t^{GAE(\lambda=1)}=G_t-V(s_t)}$$

有限 horizon 且 $s_T$ 非真正 terminal 时会保留一个 bootstrap 项 $\gamma^{T-t}V(s_T)$；若 $s_T$ 是真 terminal 则 $V(s_T)=0$，就是纯 MC。

$\lambda=0$ 时显然 $\hat A_t=\delta_t$，即 one-step TD。

### λ 的作用与取值

$$\lambda\downarrow \Rightarrow \text{更相信 critic 的短期预测} \Rightarrow \text{bias}\uparrow,\ \text{variance}\downarrow$$
$$\lambda\uparrow \Rightarrow \text{更相信真实长程 return} \Rightarrow \text{bias}\downarrow,\ \text{variance}\uparrow$$

经典默认：$\boxed{\gamma=0.99,\ \lambda=0.95}$。reward 很噪 / trajectory 很长可以降到 0.9；环境确定、critic 不可信可以升到 0.97~1.0。但实际上 lr、batch size、rollout length、clip range 通常比把 $\lambda$ 从 0.95 调到 0.97 更敏感。

## 卡点 10：V_old 是什么，为什么 target 叫 R hat

> **常见卡点**："为什么用 R，还有 V_old 又是什么？"

### $\hat R$ 不是单步 reward

要区分三个量：

$$\boxed{r_t=\text{单步 reward}} \qquad \boxed{G_t=\text{真实 MC return}} \qquad \boxed{\hat R_t=\text{用于训练 critic 的 return target}}$$

$\hat R_t$ 表示"从 $t$ 开始未来累计回报的估计"，代码里常叫 `returns` 或 `value_targets`，和环境每步给的 $r_t$ 完全不是一回事。

### $V_{old}$ 是 rollout 时刻的 critic 快照

rollout 那一刻，当时的 critic 给每个状态一个 value $V_{old}(s_t)$，**保存下来冻结**。之后 SGD 过程中 critic 参数不断变化，当前值是 $V_\phi(s_t)$。这跟 actor 的 $\pi_{old}$ vs $\pi_\theta$ 完全对应：

| actor | critic |
|---|---|
| $\pi_{old}$：产生 rollout 的 policy 快照 | $V_{old}$：rollout 时的 value 预测 |
| $\pi_\theta$：当前正在训练、每步都变 | $V_\phi$：当前正在训练、每步都变 |

**为什么算 GAE 要用 old value**：advantage 应该在 rollout 完成后先固定下来。

$$\delta_t=r_t+\gamma V_{old}(s_{t+1})-V_{old}(s_t)$$

如果一边更新 critic 一边用新 $V_\phi$ 重算 advantage，target 本身会一直漂，训练会非常混乱。所以顺序是：

$$\boxed{\text{rollout} \rightarrow \text{保存 } V_{old} \rightarrow \text{算出并冻结 } \hat A,\hat R \rightarrow \text{再开始 SGD}}$$

### 为什么 $\hat R_t=\hat A_t+V_{old}(s_t)$

因为 $A=Q-V \Rightarrow Q=A+V$，critic 想学的正是 value/return 这一类量。

具体例子：$V_{old}(s_t)=10$，GAE 算出 $\hat A_t=+3$，意思是"这次 trajectory 比这个状态原本预计的 10 好了 3"，那这次对应的 return target 自然约等于 $10+3=13$，于是训练 $V_\phi(s_t)\to13$。

$$\boxed{V_{old}=\text{原来的预测}},\quad \boxed{\hat A=\text{这次相对原预测好/差多少}},\quad \boxed{\hat R=V_{old}+\hat A}$$

### 两个极端验证（很漂亮）

$\lambda=1$：$\hat A_t=G_t-V_{old}(s_t)$，所以

$$\hat R_t=G_t-V_{old}+V_{old}=G_t \quad\Rightarrow\quad \boxed{\text{critic target = MC return}}$$

$\lambda=0$：$\hat A_t=r_t+\gamma V_{old}(s_{t+1})-V_{old}(s_t)$，所以

$$\hat R_t=r_t+\gamma V_{old}(s_{t+1}) \quad\Rightarrow\quad \boxed{\text{critic target = TD target}}$$

$0<\lambda<1$ 就是两者之间的折中。

## 6. 闭环

$$\boxed{\text{rollout} \rightarrow \text{reward} \rightarrow \text{critic 给 } V \rightarrow \text{算 TD error} \rightarrow \text{GAE 得 } \hat A \rightarrow \text{actor 用 PPO loss 更新}}$$

$$\boxed{\hat A + V_{old} \rightarrow \hat R \rightarrow \text{训练 critic}}$$

同一套 rollout，**actor 用 $\hat A$，critic 用 $\hat R$**：

- critic 提供 $V$，帮 actor 算更低方差的 $A$
- actor 按 $A$ 更新 policy
- 新 rollout 又给 critic 新数据继续拟合

$$V \rightarrow \delta \rightarrow \text{GAE} \rightarrow A \rightarrow \text{PPO actor loss}$$

## 自测

1. 为什么不能直接拿 $G_t$ 当 advantage？baseline 解决了什么？
2. critic 学的目标是什么？真实值不知道，用什么当 target？为什么 noisy target 也能训出来？
3. 用自己的话解释 bias 和 variance，说明它们描述的是**估计器**而不是单个样本。
4. **纸上推导**：$\lambda=1$ 时 GAE 如何 telescope 成 $G_t-V_t$。
5. $V_{old}$ 和 $V_\phi$ 有什么区别？为什么算 GAE 必须用 $V_{old}$？
6. $\hat R$、$G_t$、$r_t$ 三者的区别？为什么 $\hat R=\hat A+V_{old}$？
7. $\lambda=0$ 和 $\lambda=1$ 时 critic target 分别退化成什么？
