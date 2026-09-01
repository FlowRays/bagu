# RL 自测题库（关掉笔记用）

> 对应学习流程第三步：**跟 GPT 口述 / 纸上手写验证**。
> 用法：先关掉所有笔记，口述或纸上写；卡住的题记下来，回填对应笔记。
> 标 ⭐ 的是**高频考点**或最容易卡的题。
>
> 侧边栏顶部有 **「答案」开关**，可以全局显示 / 隐藏所有答案。自测时先隐藏。

## A. 概念地基（[01](01-from-J-to-loss.md)）

**1.** ⭐ $J$、$\mathbb E$、$L$、$\nabla L$ 四个东西分别是什么？关系是什么？

> **答：** $J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[G]$ 是真正想优化的 RL 目标；$\mathbb E$ 只是"对样本取平均"，代码里就是 batch average；$L$ 是为了能用 autograd 而**人为构造**的代理目标；$\nabla_\theta L$ 是 optimizer 实际使用的梯度。
> 链条：$J \to \nabla J$（policy gradient 定理）$\to$ 构造 $L$ 使 $\nabla L=\nabla J \to \theta\leftarrow\theta+\alpha\nabla L$。

**2.** ⭐ 为什么说 "$J=\mathbb E[A\log\pi]$" 是错的？正确的说法是什么？

> **答：** policy gradient 定理给出的是 **$\nabla J$ 的形式**，不是另一个与 $J$ 相等的表达式。$A\log\pi$ 只是一个 surrogate，它的**梯度**等于 $\nabla J$，函数值并不等于 $J$。
> 正确说法：$J\ne L$，但我们要求 $\nabla J=\nabla L$（至少在当前更新点上成立）。

**3.** 写出 policy gradient 定理。它给出的是 $J$ 还是 $\nabla J$？

> **答：** $\nabla_\theta J=\mathbb E_{s,a\sim\pi_\theta}\big[A(s,a)\nabla_\theta\log\pi_\theta(a|s)\big]$。给出的是 **$\nabla J$**（梯度），不是 $J$ 本身。实际含义就是：$A>0$ 提高该 action 概率，$A<0$ 降低。

**4.** 为什么代码里要写 $A\log\pi$ 这个 loss？它和真正目标是什么关系？

> **答：** 因为 autograd 需要一个可以 backward 的标量，而 $\nabla(A\log\pi)=A\nabla\log\pi$ 正好是想要的 policy gradient。它是"能产生正确梯度的代理"，不是 $J$ 本身。

**5.** 用监督学习类比解释 surrogate objective。

> **答：** 真正目标是"考试分数最高"（$J$），但不能对分数直接 backprop，于是设计一个交叉熵 loss（$L$）。我们并不认为"分数 = 负交叉熵"，只认为降低 $L$ 会朝提高分数的方向走。RL 的区别只是：这里有定理保证 $L$ 的梯度正好对应 $\nabla J$。

**6.** policy gradient 成立的隐藏前提是什么？

> **答：** $s,a$ 必须由**当前的** $\pi_\theta$ 采样。PPO 为了重复利用 $\pi_{old}$ 的 rollout，在第一次 SGD 之后就破坏了这个前提，这正是引入 importance sampling 的起点。

## B. Importance Sampling（[02](02-importance-sampling-and-ratio.md)）

**7.** PPO 为什么要重复利用 rollout？重用之后出了什么问题？（注意：不是"$J$ 变了"）

> **答：** rollout 很贵（LLM 尤其），一批数据只更新一次太浪费。重用之后 $J$ **一直没变**（始终是最大化期望回报），变的是**用来估计 $\nabla J$ 的样本分布错位了**：理论上需要 $a\sim\pi_\theta$，手里却是 $a\sim\pi_{old}$。

**8.** 写出 importance sampling 恒等式。

> **答：** $\mathbb E_{x\sim p}[f(x)]=\mathbb E_{x\sim q}\big[\frac{p(x)}{q(x)}f(x)\big]$。取 $p=\pi_\theta$（想估计的）、$q=\pi_{old}$（实际采样的）。

**9.** ⭐ 用 500L/500R → 900L/100R 的例子解释 $r$ 在做什么。

> **答：** old policy $P(L)=P(R)=0.5$，采 1000 次得 500L/500R；新 policy $P(L)=0.9,P(R)=0.1$，真采应得 900L/100R。
> $r_L=0.9/0.5=1.8$（旧数据里 L 太少，每个算 1.8 份）$\Rightarrow 500\times1.8=900$；$r_R=0.1/0.5=0.2 \Rightarrow 500\times0.2=100$，正好恢复。
> 所以 $r$ 就是**把 old-policy 样本重新加权，让它们看起来像 current-policy 采的**。

**10.** **纸上推导**：$\nabla r=r\nabla\log\pi$。

> **答：** $r=\pi_\theta/\pi_{old}$，$\pi_{old}$ 对当前 $\theta$ 是常数，所以 $\nabla r=\frac{1}{\pi_{old}}\nabla\pi_\theta$。又因为 $\nabla\pi_\theta=\pi_\theta\nabla\log\pi_\theta$，代入得 $\nabla r=\frac{\pi_\theta}{\pi_{old}}\nabla\log\pi_\theta=r\nabla\log\pi_\theta$。

**11.** ⭐⭐ 为什么 surrogate 是 $\mathbb E_{old}[rA]$ 而不是 $\mathbb E_{old}[rA\log\pi]$？把后者的梯度算出来，指出多了什么。

> **答：** 关键是 **IS 作用在梯度的期望上**，不是对原 surrogate 做代数改写：$\nabla J=\mathbb E_{\pi_{old}}[rA\nabla\log\pi_\theta]$。再找一个 $L$ 使 $\nabla L$ 等于它，利用 $\nabla r=r\nabla\log\pi$，取 $L=\mathbb E_{old}[rA]$ 即有 $\nabla L=\mathbb E_{old}[A\nabla r]=\mathbb E_{old}[rA\nabla\log\pi]$，正好对上。
> 而 $\nabla(rA\log\pi)=A[(\nabla r)\log\pi+r\nabla\log\pi]=Ar\nabla\log\pi\,(\log\pi+1)$，**多出一个 $(\log\pi+1)$ 因子**，不再是想要的梯度。
> 一句话：不是给 $A\log\pi$ 乘 $r$，而是**重新找一个能产生正确梯度的 surrogate**。

**12.** $A\log\pi$ 和 $rA$ 是代数等价的吗？它们真正的联系是什么？

> **答：** 不等价。它们是**两个不同采样条件下构造的 surrogate**：on-policy 用 $A\log\pi$，off-policy（数据来自 $\pi_{old}$）用 $rA$。真正联系是：在 $r=1$（即 $\pi_\theta=\pi_{old}$）处两者给出相同梯度 $A\nabla\log\pi$。

**13.** 为什么 rollout 刚结束时所有 $r=1$？

> **答：** 那一刻 $\theta=\theta_{old}$，所以 $\pi_\theta=\pi_{old}$，比值为 1。随着 SGD 才逐渐偏离 1。

## C. Clip 与 min（[03](03-clip-and-min.md)）

**14.** 只用 $rA$ 会出什么问题？

> **答：** 会无限把概率往极端推。$A>0$ 时 $L=rA$ 随 $r$ 单调增大，optimizer 会一路 $r:1\to2\to5$，目标一直变大，**没有任何"够了"的信号**。而这批数据始终来自 $\pi_{old}$，推太远后旧数据就不再代表当前 policy。

**15.** $r$ 和 clip 各自解决什么问题？（分清两层职责）

> **答：** $r$ 解决"old data **如何**用于 current policy"（分布修正，importance sampling）；clip 解决"就算能用，也**不能无限榨**同一批 old data"（更新幅度）。clip 不是在修正 IS，IS 已由 $r$ 完成。这就是 PPO 里 "Proximal" 的含义。

**16.** ⭐ clip 是怎么让参数停止更新的？说出 clip → loss → gradient → θ 的完整链条。

> **答：** clip 改变 $L$ 的形状，让某些区域变平。以 $A>0$、$r=1.5$ 为例，$\mathrm{clip}(r)=1.2$，于是 $L=1.2A$ 是一个**与 $\theta$ 无关的常数**（$A$ 在 rollout 后已固定），所以 $\partial L/\partial r=0$；再由链式法则 $\frac{\partial L}{\partial\theta}=\frac{\partial L}{\partial r}\frac{\partial r}{\partial\theta}=0$，该 sample 不再推动参数。
> 即：**clip surrogate 对 ratio 的依赖 → loss 变平 → 梯度为 0 → 停止更新**。不是 clip 参数，也不是 clip 梯度。

**17.** PPO 有硬约束 $r\le1.2$ 吗？更新后 $r=2$ 可能吗？

> **答：** 没有硬约束，$r=2$ 完全可能。PPO 是 **objective clipping**，不是 parameter hard constraint，只是超出有害方向后不再提供继续推远的梯度收益。（TRPO 才是真加约束 $D_{KL}(\pi_{old}\|\pi_\theta)\le\delta$。）

**18.** ⭐ 为什么不能只写 $\mathrm{clip}(r)A$？举一个具体反例（$A>0$、$r=0.5$）。

> **答：** 因为只想阻止"正确方向走太远"，不想阻止纠错。$A>0,r=0.5$ 表示好动作的概率反而被降了一半，是**错误方向**，应该继续纠正。若直接用 $\mathrm{clip}(r)A$，$\mathrm{clip}(0.5)=0.8$ 会把 0.5 假装成 0.8，掩盖错误严重程度、削弱纠错梯度。
> 用 min：$\min(0.5A,0.8A)=0.5A$，选未 clip 的分支，梯度保留，继续把概率拉回来。

**19.** ⭐⭐ **默写四象限表**，并对 $A=-1,r=0.5$ 和 $A=-1,r=1.5$ 手算 $\min$。

> **答：** （$\epsilon=0.2$，范围 $[0.8,1.2]$）
>
> | 情况 | 含义 | 结果 |
> |---|---|---|
> | $A>0,\ r>1+\epsilon$ | 好动作已提太多 | **clip**，梯度 0 |
> | $A>0,\ r<1-\epsilon$ | 好动作反被降概率（错方向） | 不 clip，继续纠正 |
> | $A<0,\ r<1-\epsilon$ | 坏动作已降太多 | **clip**，梯度 0 |
> | $A<0,\ r>1+\epsilon$ | 坏动作反被提概率（错方向） | 不 clip，继续纠正 |
>
> $A=-1,r=0.5$：$rA=-0.5$，$\mathrm{clip}(r)A=0.8\times(-1)=-0.8$，$\min(-0.5,-0.8)=-0.8$ → **clip**（注意负数下大小关系反过来）。
> $A=-1,r=1.5$：$rA=-1.5$，$\mathrm{clip}(r)A=1.2\times(-1)=-1.2$，$\min(-1.5,-1.2)=-1.5$ → **不 clip**，继续降低坏动作概率。

**20.** PPO 只 clip 哪两种情况？用两句话概括。

> **答：** 只 clip $A>0,\ r>1+\epsilon$ 和 $A<0,\ r<1-\epsilon$，也就是**朝正确方向走太远**。
> 两句话：$A>0$ 时 $r$ 太大才 clip；$A<0$ 时 $r$ 太小才 clip。"根据 $A$ 正负自动决定截哪一边"正是 min 实现的。

**21.** PPO ratio clip、gradient clipping、value clipping 三者的区别。

> **答：** PPO ratio clip 作用于 $r_t$，限制 surrogate objective 从而间接限制 policy 更新幅度；gradient clipping 作用于梯度范数 $\|g\|\le c$（如 `max_grad_norm=0.5`），防止 optimizer 梯度爆炸；value clipping 作用于 $V_\theta-V_{old}$，让 critic 不要一次更新太猛（各实现差异大，非核心）。

**22.** PPO 和 TRPO 是什么关系？

> **答：** TRPO 是 $\max_\theta L(\theta)$ subject to $D_{KL}(\pi_{old}\|\pi_\theta)\le\delta$，是真正的 trust-region 硬约束；PPO 用简单的 clipping 近似实现同样的"别走太远"效果，优势是简单、稳定、好实现。

## D. Advantage / Critic / GAE（[04](04-advantage-critic-gae.md)）

**23.** 为什么不能直接拿 $G_t$ 当 advantage？baseline 解决了什么？举 $G=100,V=95$ 和 $G=20,V=2$ 的例子。

> **答：** $A_t=G_t$ 就是 REINFORCE，variance 极大：同一个 action，一局最后 100、另一局 20，可能只是后面随机性不同，不是这个 action 导致的。
> 减去 baseline 后问题变成"相比这个 state 正常水平好多少"：$G=100,V=95\Rightarrow A=5$（其实只是略好）；$G=20,V=2\Rightarrow A=18$（反而非常优秀）。

**24.** ⭐ critic 学的目标是什么？真实值不知道，用什么当 target？为什么 noisy target 能训出来？

> **答：** critic 学 $V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s]$。真实值需要从同一个 $s$ 出发 rollout 无数次取平均，现实不可能，所以用**这一次实际拿到的 return $G_t$** 当 target。
> 单次当然不准（真值 10 这次可能是 12），但很多 rollout 给出 $8,12,10,9,11,\dots$，critic 不断拟合这些 noisy target，平均下来就逼近 10。本质上 critic 就是个**监督回归器**：输入 $s$，输出 $V$，label 是 return target。

**25.** 写出 TD error，解释它为什么近似 advantage。

> **答：** $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$。因为 $Q(s_t,a_t)\approx r_t+\gamma V(s_{t+1})$，而 $A=Q-V$，所以 $A_t\approx\delta_t$。
> 直觉：critic 原本预测这个状态值 $V(s_t)$，走一步后真实拿到 $r_t$ 并进入价值 $V(s_{t+1})$ 的状态，新旧估计之差就是"这一步比预期好多少"。

**26.** 什么是 bootstrapping？

> **答：** 用自己对下一状态的预测来构造当前状态的 target，例如用 $r_t+\gamma V_\phi(s_{t+1})$ 当 $V_\phi(s_t)$ 的监督信号（依据 Bellman 方程），而不是等整条 trajectory 结束用真实 $G_t$。

**27.** ⭐ 用自己的话解释 bias 和 variance。它们描述的是单个样本还是估计器？

> **答：** 设真值 $A^*=10$。若一种方法重复估计得到 $6,7,6,7,6$，很稳定但平均 6.4 → **高偏差、低方差**；另一种得到 $2,18,5,15,10$，平均正好 10 但极飘 → **低偏差、高方差**。
> 偏差关心"平均而言离真值多远"，方差关心"重复估计有多飘"。**两者描述的都是估计器的性质**，bias 不是"这个样本估错了多少"，variance 也不是"reward 大不大"。

**28.** MC target 和 TD(0) target 分别的 bias/variance 特点？

> **答：** MC（$G_t$）看完整 future、不依赖下一步预测 → **低 bias、高 variance**；TD(0)（$r_t+\gamma V(s_{t+1})$）只看一步真实 reward、后面全靠 critic → **低 variance、高 bias**。

**29.** ⭐⭐ GAE 全称是什么？写出定义。**纸上推导** $\lambda=1$ 时如何 telescope 成 $G_t-V_t$。

> **答：** Generalized Advantage Estimation（广义优势估计）。$\hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}$。
> $\lambda=1$ 时逐项展开：$\delta_t=r_t+\gamma V_{t+1}-V_t$；$\gamma\delta_{t+1}=\gamma r_{t+1}+\gamma^2V_{t+2}-\gamma V_{t+1}$；$\gamma^2\delta_{t+2}=\gamma^2r_{t+2}+\gamma^3V_{t+3}-\gamma^2V_{t+2}$。
> 相加后 value 项两两抵消（$+\gamma V_{t+1}$ 与 $-\gamma V_{t+1}$，$+\gamma^2V_{t+2}$ 与 $-\gamma^2V_{t+2}$，依此 telescope），只剩 $r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots-V_t=G_t-V_t$。
> （有限 horizon 且末状态非 terminal 时会保留 bootstrap 项 $\gamma^{T-t}V(s_T)$。）

**30.** $\lambda$ 控制什么？$\lambda\uparrow$ 时 bias 和 variance 分别怎么变？常用值？

> **答：** 控制"多相信 critic 的短期预测，还是多相信真实长程 return"。$\lambda\uparrow$ → 更接近 MC → **bias↓、variance↑**；$\lambda\downarrow$ → 更接近 TD(0) → bias↑、variance↓。常用 $\gamma=0.99,\lambda=0.95$。

**31.** ⭐ $V_{old}$ 是什么？和 $V_\phi$ 有什么区别？为什么算 GAE 必须用 $V_{old}$？

> **答：** $V_{old}$ 是 **rollout 那一刻冻结保存**的 critic 预测；$V_\phi$ 是当前正在训练、每个 optimizer step 都在变的 critic。对应关系同 actor 的 $\pi_{old}$ vs $\pi_\theta$。
> 必须用 $V_{old}$，因为 advantage 应在 rollout 完成后先固定下来。若一边更新 critic 一边用新 $V_\phi$ 重算 advantage，监督 target 本身一直漂，训练会非常混乱。顺序是：rollout → 保存 $V_{old}$ → 算出并冻结 $\hat A,\hat R$ → 再开始 SGD。

**32.** ⭐ $r_t$、$G_t$、$\hat R_t$ 三者的区别？为什么 $\hat R=\hat A+V_{old}$？

> **答：** $r_t$ 是环境每步给的**单步 reward**；$G_t$ 是**真实 MC return**；$\hat R_t$ 是**用于训练 critic 的 return target**（代码里常叫 `returns` / `value_targets`）。
> 因为 $A=Q-V\Rightarrow Q=A+V$，critic 要学的正是 value/return 这类量。例：$V_{old}=10$、$\hat A=+3$（这次比原预计好 3）→ 该样本的 return target 就该是 $13$。

**33.** $\lambda=0$ 和 $\lambda=1$ 时 critic target 分别退化成什么？

> **答：** $\lambda=1$ 时 $\hat A=G_t-V_{old}$，代入 $\hat R=\hat A+V_{old}$ 得 $\hat R=G_t$，即 **MC return**；$\lambda=0$ 时 $\hat A=\delta_t$，得 $\hat R=r_t+\gamma V_{old}(s_{t+1})$，即 **TD target**。$0<\lambda<1$ 是两者折中。

**34.** actor 用哪个量？critic 用哪个量？

> **答：** 同一套 rollout，**actor 用 $\hat A$**（决定 action 概率升降），**critic 用 $\hat R$**（回归 target）。

## E. PPO 工程（[05](05-ppo-engineering.md)）

**35.** 默写完整 PPO loss 三项，说明各自作用和符号方向。

> **答：** $L_{total}=L_{policy}+c_vL_{value}-c_eH$。
> policy：$-\mathbb E[\min(r_t\hat A_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)]$，按 advantage 调 action 概率同时不离 old policy 太远（代码做 gradient descent 故带负号）；value：$\mathbb E[(V_\phi-\hat R)^2]$，让 critic 更准；entropy：前面是 $-c_eH$，这样最小化总 loss 时会让 $H\uparrow$，保持探索、避免过早 collapse。

**36.** ⭐ 按时间顺序说出一次 PPO iteration 的完整步骤，指出哪些量在什么时候被冻结。

> **答：** ① 用 $\pi_{old}$ rollout；② 保存 $\log\pi_{old}(a_t|s_t)$ 和 $V_{old}(s_t)$；③ 算 reward；④ 算 $\delta_t$（用 $V_{old}$）；⑤ GAE 得 $\hat A_t$；⑥ $\hat R_t=\hat A_t+V_{old}$；⑦ **冻结** $\hat A,\hat R$；⑧ 对这批数据做多轮 minibatch SGD；⑨ 更新 actor/critic；⑩ 用新 policy 再 rollout。
> 冻结的是 $\log\pi_{old}$、$V_{old}$、$\hat A$、$\hat R$；每步都在变的是 $\pi_\theta$、$V_\phi$。

**37.** 用一句话概括 PPO 的时间结构。

> **答：** **先采样，再冻结监督信号，再训练。**

**38.** ⭐ 为什么用 `exp(log_prob - old_log_prob)` 而不是直接相除？

> **答：** 数学上完全等价，但数值上稳定得多。LLM 里概率可能小到 $10^{-20}$ 甚至更小，直接存概率会 underflow；而 $\log10^{-20}\approx-46$ 很好表示。所以工程上全程存 $\log\pi$，需要 ratio 时再取指数。

**39.** rollout batch 和 SGD minibatch 的区别？`ppo_epochs=4` 是什么意思？epoch 越多有什么代价？

> **答：** rollout batch 是环境采样得到的完整数据量（如 65536）；minibatch 是优化时切成的小块（如 1024，则每 epoch 有 64 个 optimizer step）。
> `ppo_epochs=4` 指**同一批 rollout 数据反复训练 4 遍**，不是 rollout 4 次。epoch 越多数据利用率越高，但 $\pi_\theta$ 离 $\pi_{old}$ 越远，clipfrac / KL 越大。PPO 本质是在数据利用率和 on-policy 稳定性之间取平衡。

**40.** 写出经典 PPO 默认超参（$\gamma,\lambda,\epsilon,c_v,c_e$, max_grad_norm）。LLM RL 的 lr 和经典 RL 差多少量级？

> **答：** $\gamma=0.99$、$\lambda=0.95$、$\epsilon=0.2$、$c_v=0.5$、$c_e\approx0.01$（LLM reasoning RL 常直接取 0）、`max_grad_norm=0.5`、`ppo_epochs=4`。
> 经典小型 RL 网络 lr 约 $3\times10^{-4}$，**LLM RL 约 $10^{-6}\sim10^{-5}$，差约两个数量级**，不能直接搬。

**41.** 为什么要做 advantage normalization？

> **答：** advantage 量级可能变化极大（某个 batch $A\in[-1000,1000]$，另一个 $A\in[-0.1,0.1]$），梯度尺度完全不同。归一化 $A\leftarrow(A-\mu_A)/(\sigma_A+\epsilon)$ 后 $\mathbb E[A]\approx0,\mathrm{std}(A)\approx1$，训练稳定很多。

**42.** 三个必看指标是什么？clipfrac 很高说明什么？列举可能原因。

> **答：** **KL、clipfrac、entropy**。clipfrac 是落入 clipping region 的样本比例，很高（如 0.8）说明当前 policy 已离 rollout policy 很远。可能原因：learning rate 太大、PPO epochs 太多、minibatch 太小、advantage 太极端、rollout/train policy mismatch。

**43.** ⭐ LLM 里 $s_t$、$a_t$ 分别是什么？reward 什么时候给？

> **答：** $s_t=(x,y_{<t})$（prompt + 已生成的 token），$a_t=y_t$（下一个 token），于是 $\pi_\theta(a_t|s_t)=P_\theta(y_t|x,y_{<t})$。一条 response 相当于一条 trajectory，但 **reward 通常只在整个 response 结束后给** $R(y)$（如数学题答对 1、答错 0）。

**44.** ⭐ $\pi_{old}$ 和 $\pi_{ref}$ 的区别？

> **答：** $\pi_{old}$ 是产生**这批 rollout** 的 actor 快照，用于算 importance ratio，每轮 rollout 都变，是**短期训练基准**；$\pi_{ref}$ 通常是 SFT model / RL 开始前固定的模型，用于 KL regularization，长期不动，是**长期行为锚点**。

**45.** PPO 和 DQN 的本质区别？

> **答：** PPO 直接参数化 policy $\pi_\theta(a|s)$ 并学习 policy 本身，属 policy-based / actor-critic；DQN 学 $Q(s,a)$，policy 由 $\arg\max_aQ$ 间接产生，属 value-based。

## F. GRPO（[06](06-grpo.md)）⭐ 高频考点

**46.** ⭐⭐ **一分钟讲清 GRPO 的原理和训练方式。**

> **答：** GRPO 保留 PPO 的 importance ratio 和 clip，**只把 advantage 的来源从 critic + GAE 换成同一个 prompt 多条 rollout 的组内相对 reward**。
> 训练方式：对 prompt $x$ 采 $G$ 条 response，各得 sequence-level reward $R_1..R_G$；组内标准化 $A_i=(R_i-\mu_R)/(\sigma_R+\epsilon)$；把这个 sequence-level 的 $A_i$ 广播到该条 response 的每个 token；再对每个 token 算自己的 ratio $r_{i,t}$，套 PPO 的 $\min(rA,\mathrm{clip}(r)A)$；最后聚合成 batch loss 反传。
> 一句话：$\text{GRPO}=\text{PPO actor update}-\text{critic}-\text{GAE}+\text{same-prompt group advantage}$。

**47.** ⭐ GRPO 的 advantage 怎么算？写出公式。

> **答：** $A_i=\dfrac{R_i-\mu_R}{\sigma_R+\epsilon}$，其中 $\mu_R,\sigma_R$ 在**同一个 prompt 的 group 内**统计。例：$R=[1,1,0,0]\Rightarrow\mu=0.5\Rightarrow A=[+0.5,+0.5,-0.5,-0.5]$，除以 $\sigma=0.5$ 后为 $[1,1,-1,-1]$。

**48.** ⭐ 为什么可以不用 critic？为什么这在 LLM reasoning 里可行、在 Atari 里不可行？

> **答：** critic 的作用是提供 baseline"这个状态正常能有多好"。LLM reasoning 里同一个 prompt 极容易采很多条 completion 并用 verifier 打分，**天然拥有 within-prompt baseline**，直接用组内均值即可。
> Atari / robot 里每个 state $s_t$ 很难反复采 16 条完全可比的 trajectory，所以必须训练一个 $V(s)$ 来估计。另外对 LLM 来说，32B actor 再配 32B critic，显存、forward、backward、optimizer state 都很贵。

**49.** ⭐ 为什么除标准差而不是方差？如果 reward 整体放大 10 倍分别会怎样？

> **答：** 标准化要得到**无量纲、尺度约为 1** 的量，关键是 scale invariance。$R'=10R$ 时 $R'-\mu'=10(R-\mu)$、$\sigma'=10\sigma$：
> 除标准差 $\frac{10(R-\mu)}{10\sigma}=\frac{R-\mu}{\sigma}$，**完全不变** ✅；
> 除方差（$\sigma'^2=100\sigma^2$）得 $\frac{10(R-\mu)}{100\sigma^2}=\frac1{10}\frac{R-\mu}{\sigma^2}$，reward 只是放大，normalized 值反而**缩小 10 倍**，量纲也不对 ❌。

**50.** 常见的 norm 有哪几种？为什么 advantage 偏好 z-score？

> **答：** z-score / standardization $\frac{x-\mu}{\sigma+\epsilon}$（mean≈0, std≈1）；只 center $x-\mu$；min-max $\frac{x-x_{\min}}{x_{\max}-x_{\min}}$（压到 $[0,1]$）。
> advantage 偏好 z-score，因为关心的正是**正负号 + 相对尺度**：高于平均为正、低于为负、偏离几个标准差决定 magnitude。$\frac{R_i-\mu}{\sigma}$ 可直接读作"比同组平均高/低几个标准差"。

**51.** ⭐⭐ 一条 response 的 sequence-level advantage 是怎么作用到每个 token 上的？**推导**。这是"替代"还是"分解"？

> **答：** 是**分解**，不是替代。自回归定义 $\pi_\theta(y|x)=\prod_t\pi_\theta(y_t|x,y_{<t})$，取 log 得 $\log\pi_\theta(y|x)=\sum_t\log\pi_\theta(y_t|x,y_{<t})$。
> 于是 sequence-level surrogate $L_{seq}=A\log\pi_\theta(y|x)=\sum_tA\log\pi_\theta(y_t|x,y_{<t})$，**自动就是一堆 token-level loss 之和**；梯度同理 $\nabla L_{seq}=\sum_tA\nabla\log\pi_\theta(y_t|\cdot)$。
> 例：$y=(A,B,C)$，Advantage $=+2$，则 $L=2\log P(A)+2\log P(B|A)+2\log P(C|AB)$，三个 token 各得 $+2$ 信号。这不是额外假设，由 $P(\text{seq})=\prod P(\text{token})$ 自动推出。

**52.** ⭐ GRPO 解决 credit assignment 了吗？为什么？

> **答：** **没有**。一条 100-token reasoning 最后答对，可能只有第 80~100 个 token 真正关键，但 vanilla GRPO 让前面所有 sampled token 共享同一个正 advantage（都"沾光"）；反过来答错时，即使中间有很多正确推理，所有 token 也一起被惩罚。
> 它只是说：我没有 token-level reward，那就把 trajectory-level 的信号摊给整条 trajectory。这正是 reasoning RL / long-horizon agent 的核心难题。

**53.** ⭐ 为什么必须同 prompt 分组，不能整个 batch 一起 norm？举难易两道题的例子。

> **答：** 因为要衡量的是"**相对这个问题的正常水平**有多好"。设 $x_1$ 易题 $R=[1,1,1,0]$、$x_2$ 难题 $R=[1,0,0,0]$。
> 整个 batch 一起算（$\mu=0.5$）会把**题目难度差异**混进来。按 prompt 分组后：$x_1$（$\bar R=0.75$）得 $A=[+0.25,+0.25,+0.25,-0.75]$，"这题本来几乎都会，答错一次非常差"；$x_2$（$\bar R=0.25$）得 $A=[+0.75,-0.25,-0.25,-0.25]$，"这题通常答不出，这次成功了非常值得强化"。都很合理。

**54.** GRPO 的 group baseline 对应 PPO 里的什么？

> **答：** 对应 $V(s)$，都是回答"在这个状态/这道题上正常能有多好"。GRPO 把**学出来的网络**换成了**同 prompt 多次 rollout 的经验均值** $b(x)\approx\frac1G\sum_iR(x,y_i)$。

**55.** GRPO 什么时候完全没有学习信号？

> **答：** 组内 reward 全一样时。$R=[1,1,1,1]$（全会）或 $R=[0,0,0,0]$（全不会）都会得到 $A_i=0$。真正有用的是 $0<\text{success rate}<1$。

**56.** ⭐ GRPO 相对 PPO 的收益和代价各是什么？

> **答：** 收益：不需要 critic model、少一套 forward/backward、少一份 optimizer state、对大模型 RL 的显存和工程复杂度友好很多。
> 代价：advantage 很粗（sequence-level）、credit assignment 更差、必须同 prompt 多采样、整组全对或全错时几乎没信号。
> 本质是：**牺牲更细粒度的 value estimation，换更简单更便宜的大模型 RL。**

**57.** GRPO loss 里有哪两种"不要走太远"？时间尺度有何不同？

> **答：** ① PPO clip（比较对象 $\pi_{old}$，即 rollout 时的快照）限制"**这一批 rollout 上单次更新别太猛**"；② reference KL（比较对象 $\pi_{ref}$，RL 开始前的模型）限制"**整个训练过程中不要累计偏离原模型太远**"。
> 训练到第 1000 step 时 $\pi_{old}$ 可能已是很强的 RL model，$\pi_\theta$ 只需离它很近；但它可能已离最初的 $\pi_{ref}$ 很远，所以 reference KL 仍提供长期约束。

## G. KL（[07](07-kl.md)）⭐ 高频考点

**58.** ⭐ 写出 KL 定义，说出三个性质。

> **答：** $D_{KL}(P\|Q)=\mathbb E_{x\sim P}[\log\frac{P(x)}{Q(x)}]=\sum_xP(x)\log\frac{P(x)}{Q(x)}$。
> 性质：① $\ge0$；② $=0\iff P=Q$；③ **不对称** $D_{KL}(P\|Q)\ne D_{KL}(Q\|P)$，所以不是"距离"。

**59.** ⭐⭐ **纸上证明** KL ≥ 0。这个不等式叫什么？

> **答：** 用 $\log x\le x-1$。令 $x=\frac{Q(x)}{P(x)}$，得 $\log\frac{P(x)}{Q(x)}\ge1-\frac{Q(x)}{P(x)}$。两边乘 $P(x)$：$P(x)\log\frac{P(x)}{Q(x)}\ge P(x)-Q(x)$。对所有 $x$ 求和：$D_{KL}(P\|Q)\ge\sum_xP(x)-\sum_xQ(x)=1-1=0$，等号当且仅当 $P=Q$。
> 这叫 **Gibbs' inequality**。

**60.** ⭐⭐ Forward KL 和 Reverse KL 的定义、区别、各自导致什么行为？

> **答：** Forward $=D_{KL}(P_{data}\|Q_{model})$，期望在 $P$ 下，只要 $P$ 有质量的地方 $Q$ 都不能漏（$P>0$ 而 $Q\approx0$ 时 $\log\frac PQ\to\infty$），所以 **mode covering**（把所有峰都罩住，哪怕中间概率很低）。
> Reverse $=D_{KL}(Q_{model}\|P_{data})$，期望在 $Q$ 下，最怕"自己给了高概率但 $P$ 在这里概率极低"，所以躲在 $P$ 的高概率区，表现为 **mode seeking**（可能只挑一个峰）。

**61.** ⭐⭐ **不许背名字**，只用"谁在 expectation 下面"现场推出 mode-covering / mode-seeking。

> **答：** $\mathbb E_P[\cdots]\Rightarrow$ 去 $P$ 出现的地方检查 $Q\Rightarrow$ 不能漏 $P$ 的 mode $\Rightarrow$ **cover**。
> $\mathbb E_Q[\cdots]\Rightarrow$ 去 $Q$ 自己出现的地方检查 $P\Rightarrow$ 不能跑到 $P$ 的低概率区 $\Rightarrow$ **seek**。
> 口诀：$D_{KL}(P\|Q)$ 是"P 负责出考题，Q 必须都会"；$D_{KL}(Q\|P)$ 是"Q 自己出门，只敢去 P 觉得安全的地方"。

**62.** ⭐ GRPO 的 reference KL 是 forward 还是 reverse？当场展开推一遍。

> **答：** 是 **reverse KL**（mode-seeking 那一边）。展开 $D_{KL}(\pi_\theta\|\pi_{ref})=\mathbb E_{a\sim\pi_\theta}[\log\frac{\pi_\theta(a|s)}{\pi_{ref}(a|s)}]$，expectation 下面是 $\pi_\theta$，说明检查的是"**当前 RL policy 自己会产生什么**"。
> 若 $\pi_\theta(a)=0.3$ 而 $\pi_{ref}(a)=0.001$，则 $\log\frac{0.3}{0.001}\approx5.7$ 很大，受强惩罚。含义：**你自己产生的东西，不要跑到 reference 认为非常离谱的区域。**

**63.** reference KL 在目标函数里怎么写？直觉是什么？

> **答：** $\text{objective}=\text{reward}-\beta D_{KL}(\pi_\theta\|\pi_{ref})$。policy 和 reference 很像时 $KL\approx0$ 基本不罚，改得越远罚得越重。
> 直觉：**reward 鼓励你变，reference KL 告诉你别变得离原模型太远**（保住语言能力、格式、行为分布）。

**64.** 为什么现在很多 reasoning RL 把 $\beta$ 设成 0？

> **答：** ① rule-based reward 很干净，不太需要靠 KL 兜底；② 强 KL 会限制探索；③ reasoning model 本来就希望允许明显偏离 SFT policy。DeepSeek-R1-style / DAPO-style setting 常把 KL penalty 弱化甚至去掉，核心只剩 group advantage + ratio + clip。

**65.** PPO 里即使不加 KL penalty 也要监控 KL，为什么？怎么近似算？

> **答：** 因为真正关心的是"policy 到底变化了多少"，KL 突然很大（如 0.5）说明已经跑飞。用 $\log r_t=\log\pi_\theta-\log\pi_{old}$ 近似（`approx_kl`）。有些实现据此 early stop PPO epochs、自适应 lr 或加 KL penalty。

## H. DAPO（[08](08-dapo.md)）

**66.** DAPO 全称？四个改动分别是什么？

> **答：** **D**ecoupled Clip and Dynamic s**A**mpling **P**olicy **O**ptimization（ByteDance）。四个改动：Clip-Higher、Dynamic Sampling、Token-level Policy Gradient Loss、Overlong Reward Shaping。

**67.** ⭐ Clip-Higher 改了什么？为什么只放宽上界？举 $0.001\to0.0012$ 的例子。

> **答：** 把对称 clip $[1-\epsilon,1+\epsilon]$ 改成**不对称** $[1-\epsilon_{low},1+\epsilon_{high}]$，且 $\epsilon_{high}>\epsilon_{low}$（如 $[0.8,1.28]$）。
> 例：某正确 reasoning token 在 old policy 下 $\pi_{old}=0.001$，偶然采到且答对，训练后 $\pi_\theta=0.0012$，则 $r=1.2$ 已撞上界——概率只从 0.1% 提到 0.12% 就不再推它，对探索太保守。
> 只放宽上界是因为：低概率但正确的新 reasoning pattern 是宝贵探索，值得更大胆强化；而过度压低坏 trajectory 一般没那么值得冒险。注意 min 逻辑保持不变。

**68.** Dynamic Sampling 解决什么问题？保留什么样的 prompt？

> **答：** 解决"组内全对 $[1,1,1,1]$ 或全错 $[0,0,0,0]$ 时 $A_i=0$、没有学习信号"的浪费。动态筛掉这类 prompt，保留 $0<\text{success rate}<1$ 的（如 $[1,1,0,0]$），本质是提高**有效 rollout / 总 rollout** 的比例。

**69.** ⭐ $A=0$ 时训练和不训练有区别吗？真正浪费的是什么？有什么例外？

> **答：** 从 policy update 看几乎没区别：$r_iA_i=0$，clip 后仍是 0，所以 $\nabla_\theta L_{policy}=0$。
> 但真正浪费的**不是 backward**，而是前面已经发生的：生成这些 response 的推理算力、reward/verifier 计算、logprob forward 和数据搬运。
> 例外：若训练目标里还有 **KL、entropy 或其他辅助 loss**，即使 $A=0$ 这些样本仍可能产生梯度，所以"等价于不训练"只对纯 policy-gradient 那部分严格成立。

**70.** ⭐⭐ **手算**：response A 有 2 个 token、B 有 10 个 token（每个 token loss 都是 1），GRPO 和 DAPO 的 batch loss 分别怎么算？权重比各是多少？

> **答：** GRPO（先序列内平均，再序列间平均）：$L_A=\frac{1+1}{2}=1$，$L_B=1$，$L=\frac{1+1}{2}=1$，**权重 A:B = 1:1**（尽管 B 的 token 数是 A 的 5 倍）。
> DAPO（全 batch token 平均）：$L=\frac{\sum\ell}{\sum T}=\frac{2+10}{12}=1$，但 B 贡献 10 个 token、A 只贡献 2 个，**B 对梯度的影响约是 A 的 5 倍**。
> 即：GRPO 每条 response 等权，DAPO 每个 token 等权。

**71.** ⭐ GRPO 和 DAPO 哪个倾向生成更长回答？为什么？这算"直接奖励长度"吗？

> **答：** **DAPO** 更容易鼓励长回答。GRPO 先做 sequence 内平均，50 token 和 500 token 的 response 总权重差不多；DAPO 全 batch token-level average 后，长 response 含更多 token 就贡献更多梯度，正 advantage 的长 reasoning 会有更多 token 被强化。
> **不算**直接奖励长度：如果长回答 reward 不高或触发 overlong penalty，一样会被抑制。准确说法是"长的**高质量** response 相对更容易获得更大的总训练影响"。

**72.** DAPO 的 token-level aggregation 是无条件更好吗？副作用是什么？

> **答：** 不是无条件更好。优点是每个 token 权重一致，长短回答不会被强行拉平，适合需要长 CoT 的任务；副作用是长回答天然占更多权重，reward 设计不够好时容易被长序列主导、出现"越写越长"。所以通常要配合 overlong penalty、长度控制、dynamic sampling 一起用。取决于你想让**什么成为基本训练单位**。

**73.** ⭐ 默写 Soft Overlong Punishment 的三段公式，算 $|y|=14336$ 时的 $R_{length}$（$L_{\max}=16384,L_{cache}=4096$）。

> **答：** $R_{length}(y)=0$ 当 $|y|\le L_{\max}-L_{cache}$；$=\frac{(L_{\max}-L_{cache})-|y|}{L_{cache}}$ 当 $L_{\max}-L_{cache}<|y|\le L_{\max}$；$=-1$ 当 $|y|>L_{\max}$。总 reward $R_{total}=R_{correct}+R_{length}$。
> 代入：安全区上界 $16384-4096=12288$，$14336$ 落在软惩罚区，$R_{length}=\frac{12288-14336}{4096}=-0.5$。若 $R_{correct}=1$ 则 $R_{total}=0.5$，即"答对了但太啰嗦，不给满分"。

**74.** overlong 处理的演化过程是怎样的？为什么叫 reward shaping？

> **答：** 粗暴给截断样本 $R=0$ → 发现 reward noise 很大 → **Overlong Filtering**（直接把截断样本的 loss mask 掉不训练），训练明显更稳 → 再进一步用**长度相关的连续 soft penalty**。
> 叫 reward shaping 是因为它不是二元判断"超长/不超长"，而是把接近上限时的 reward 做成一个**连续坡度**，避免 reward 在边界处不连续（7999 拿 1、8001 拿 0），同时抑制无限拉长 reasoning。

## I. GSPO（[09](09-gspo.md)）

**75.** GSPO 质疑 GRPO 的什么？

> **答：** 质疑 **reward 明明是 sequence-level 的，importance sampling 却在 token-level 做**。GRPO 给每个 token 单独算 ratio 并单独 clip，但真正获得 reward 的对象是整条 response。

**76.** sequence-level ratio 怎么写？为什么要长度归一化？可以理解成什么平均？

> **答：** $r_i^{seq}=\exp\big[\frac{1}{|y_i|}\sum_t\log\frac{\pi_\theta(y_{i,t}|\cdot)}{\pi_{old}(y_{i,t}|\cdot)}\big]$。长度归一化是为了防止长序列上连乘导致数值爆炸。可以理解成**每个 token ratio 的几何平均**。

**77.** GRPO 和 GSPO 的 clip 分别在什么粒度？

> **答：** GRPO 在**每个 token** 上算 ratio 并分别 clip（问"这个 token 变太多了吗"）；GSPO 在**整条 sequence** 上算一个 ratio 并在 sequence level clip（问"这整个回答相对 old policy 变化太大了吗"）。

**78.** 为什么说"把整条 sequence 当 action"更自然？

> **答：** 从 MDP 表述上把 token 当 action 也可以，但 reward $R(y)$ 往往只在整个 response 完成后才产生（数学答案对/错），真正获得 reward 的对象更像是整条 reasoning trajectory。既然如此，policy correction / clipping 也应该贴近 trajectory-level。Qwen 团队称这样训练更稳定，对 MoE RL 尤其明显。

## J. 整体串讲（压轴）

**79.** ⭐⭐ **把 PPO → GRPO → DAPO → GSPO 的演化线讲一遍**，说明每一步改的是什么。

> **答：** 四者只改三样东西：**A 怎么来、ratio 怎么算、loss 怎么聚合**。
> Policy Gradient → **PPO**（引入 ratio + clip，让旧数据可重用且更新不过猛）→ 但 critic 很贵、而 LLM 可以同 prompt 多采样 → **GRPO**（去掉 critic 和 GAE，用组内相对 reward 当 advantage）→ 真跑大后暴露 clip 太保守、无效 prompt 浪费、长度偏置、超长截断 → **DAPO**（Clip-Higher、Dynamic Sampling、token-level loss、overlong shaping）→ 再质疑 token-level ratio 本身是否合理 → **GSPO**（改成 sequence-level ratio 和 clip）。
> 即：PPO→GRPO 改的是 **advantage 怎么估**；GRPO→DAPO 改的是 **怎么稳定高效地 scale**；GRPO/DAPO→GSPO 改的是 **ratio 的粒度（token 还是 sequence 当 action）**。

**80.** ⭐ 默写五个必背公式。

> **答：** ① $A=Q-V$；② $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$；③ $\hat A_t^{GAE}=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}$；④ $r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$；⑤ $L^{PPO}=\mathbb E[\min(r_tA_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t)]$。

**81.** ⭐ 从 $J(\theta)$ 一路推到 $L^{CLIP}$，中间不跳步。

> **答：** ① $J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[G]$ 是真正目标。② policy gradient 定理：$\nabla J=\mathbb E_{\pi_\theta}[A\nabla\log\pi_\theta]$。③ 为让 autograd 产生这个梯度，构造 on-policy surrogate $L=\mathbb E[A\log\pi]$（因 $\nabla L=A\nabla\log\pi$，注意 $J\ne L$）。④ 想重用 $\pi_{old}$ 的 rollout，但采样分布错位。⑤ importance sampling 作用在梯度期望上：$\nabla J=\mathbb E_{\pi_{old}}[rA\nabla\log\pi]$。⑥ 再找新 surrogate，利用 $\nabla r=r\nabla\log\pi$ 得 $\nabla(rA)=rA\nabla\log\pi$，所以 $L=\mathbb E_{old}[rA]$。⑦ 但 $rA$ 会把概率无限推远，加 clip 让超出区间的 loss 变平、梯度为 0。⑧ 只想截断"正确方向走太远"、不想妨碍纠错，所以取更保守的分支：$L^{CLIP}=\mathbb E[\min(rA,\mathrm{clip}(r,1-\epsilon,1+\epsilon)A)]$。

**82.** 画出经典 RLHF PPO 的架构图（actor / critic / reward / reference 四个模型的关系）。

> **答：**
>
> ```text
> Actor ──rollout──> responses
>                       │
>    Reward Model ──────┼──> reward
>    Critic ────────────┼──> value V_old
>                       ▼
>                      GAE ──> advantage Â ──> PPO loss ──> 更新 actor
>                                    │
>                       Â + V_old = R̂ ──────> value loss ──> 更新 critic
>    Reference Model（固定）──> KL penalty ──> 加进 reward 或 loss
> ```
>
> actor / critic / reward / reference 可能是四个不同的模型，这正是 GRPO 想砍掉 critic 的动机。

## 记录卡住的题

> 每次自测把卡住的题号记在这里，回填笔记后划掉。

- [ ] （待填）
