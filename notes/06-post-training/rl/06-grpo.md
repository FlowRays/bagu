# GRPO：Group Relative Policy Optimization

> **高频考点**：GRPO 的原理 / 训练方式。核心涉及：advantage 怎么算、为什么可以不用 critic、group 内多条 response 如何利用 reward、与 PPO 的关系区别。
> 这块最容易"知道大概但讲不扎实"，目标是能完整口述。

## 一句话

$$\boxed{\text{GRPO}=\text{PPO actor update}-\text{critic}-\text{GAE}+\text{same-prompt group advantage}}$$

保留 PPO 的 ratio 和 clip，**只改 advantage 怎么估**。

## 1. 动机：critic 太贵

PPO 里 actor 已经是 32B，critic 再来一个 32B，显存、forward、backward、optimizer state 都很贵。于是 GRPO 问：

> 可不可以不用 $V(s)$，直接用同一个 prompt 的多条 rollout 建 baseline？

而 LLM reasoning 恰好有一个非常特殊的 setting 让这件事可行：同一个 prompt 极容易采很多条 completion，再用 verifier 检查得到 $R_i\in\{0,1\}$。这意味着**天然拥有一个 within-prompt baseline**。

对比传统 robot/Atari：每个 state $s_t$ 不容易反复采 16 条完全可比的 trajectory，所以训练 $V(s)$ 才有意义。

## 2. Group Relative Advantage

对 prompt $x$ 采 $G$ 条 response $y_1,\dots,y_G$，得 reward $R_1,\dots,R_G$：

$$\mu_R=\frac1G\sum_i R_i,\qquad \sigma_R=\sqrt{\frac1G\sum_i(R_i-\mu_R)^2}$$

$$\boxed{A_i=\frac{R_i-\mu_R}{\sigma_R+\epsilon}}$$

例：$R=[1,1,0,0]$ → $\mu=0.5$ → $A=[+0.5,+0.5,-0.5,-0.5]$；若再除 $\sigma=0.5$ → $A=[1,1,-1,-1]$。

答对的提概率，答错的降概率。核心直觉：

> **不用 critic 问"这个状态本来值多少"，而是直接拿同一道题的其他 rollout 当 baseline。**

两种 baseline 的对照：

$$A_{PPO}\approx R-V(s) \qquad\qquad A_{GRPO}\approx \frac{R-\mu_{group}}{\sigma_{group}}$$

## 卡点 12：为什么除标准差不是方差

因为标准化的目标是得到一个**无量纲、尺度约为 1** 的量。

关键性质是 **scale invariance**。若 reward 整体放大 10 倍（$R'=10R$）：

$$R'-\mu'=10(R-\mu),\qquad \sigma'=10\sigma$$

除标准差：

$$\frac{R'-\mu'}{\sigma'}=\frac{10(R-\mu)}{10\sigma}=\frac{R-\mu}{\sigma}$$

✅ **完全不变**，这正是我们想要的。

除方差（$\sigma'^2=100\sigma^2$）：

$$\frac{10(R-\mu)}{100\sigma^2}=\frac{1}{10}\cdot\frac{R-\mu}{\sigma^2}$$

❌ reward 只是整体放大，normalized 值反而**缩小 10 倍**，而且量纲也不对（缩放过强）。

### 一般的 norm 有哪几种

| 方法 | 公式 | 效果 |
|---|---|---|
| **z-score / standardization**（最常用） | $x'=\dfrac{x-\mu}{\sigma+\epsilon}$ | $\mathrm{mean}\approx0,\ \mathrm{std}\approx1$ |
| 只 center | $x'=x-\mu$ | 只去均值 |
| min-max | $x'=\dfrac{x-x_{\min}}{x_{\max}-x_{\min}}$ | 压到 $[0,1]$ |

RL 的 advantage 偏好 z-score，因为我们关心的正是 $\boxed{\text{正负号 + 相对尺度}}$：高于平均为正、低于平均为负、偏离平均几个标准差决定 magnitude。

$$\frac{R_i-\mu_R}{\sigma_R}\ \text{读作：这条 rollout 比同组平均水平高/低多少个标准差}$$

> 注：GRPO 的一些变体会讨论**是否应该除 group std**，因为除标准差会带来额外的 reward-scale 效应，尤其当某个 group 的 reward variance 很小时（分母接近 0）。

## 卡点 13：为什么 sequence-level 的 A 能乘到每个 token 上

> **常见卡点**："为什么可以直接用 seq-level loss 替代 token-level loss？"
>
> **关键澄清**：这不是"替代"。**整条 sequence 的 log-prob 本来就等于各 token log-prob 的和，所以 sequence-level 的 policy gradient 天然会分解成 token-level gradient。**

自回归模型的基本定义：

$$\pi_\theta(y|x)=\prod_{t=1}^T\pi_\theta(y_t|x,y_{<t})$$

取 log，利用 $\log(ab)=\log a+\log b$：

$$\boxed{\log\pi_\theta(y|x)=\sum_t\log\pi_\theta(y_t|x,y_{<t})}$$

把整个 response 当成一个 action，sequence-level surrogate 是 $L_{seq}=A\log\pi_\theta(y|x)$，展开：

$$\boxed{L_{seq}=\sum_t A\log\pi_\theta(y_t|x,y_{<t})}$$

**它自动就变成了一堆 token-level loss。** 梯度同理：

$$\nabla L_{seq}=A\nabla\log\pi_\theta(y|x)=\sum_t A\nabla\log\pi_\theta(y_t|x,y_{<t})$$

极小例子：$y=(A,B,C)$，$P(y)=P(A)P(B|A)P(C|AB)$，Advantage $=+2$：

$$L=2\log P(A)+2\log P(B|A)+2\log P(C|AB)$$

三个 token 各获得 $+2$ 的强化信号。这不是额外假设，是由 $P(\text{sequence})=\prod_t P(\text{token}_t|\text{prefix})$ 自动推出来的。

所以：

$$\boxed{\text{sequence-level log-prob objective}=\text{token-level objectives 的和}}$$

### 但要区分两个问题

| 问题 | 答案 |
|---|---|
| **数学上能这么做吗？** | 能，因为 $\log P(\text{seq})=\sum_t\log P(\text{token})$ |
| **这样的 credit assignment 好吗？** | **很粗糙** |

一条 100-token reasoning 最后答对，可能只有第 80~100 token 真正关键，但 vanilla GRPO 让**前面所有 sampled token 都共享这个正 advantage**（所有 token 都"沾光"）。反过来最后答错，即使中间有很多正确推理，所有 token 也一起被惩罚。

$$\boxed{\text{GRPO 并没有解决 credit assignment}}$$

它只是说：我没有 token-level reward，那就把 trajectory-level signal 分配给整条 trajectory。这正是 reasoning RL / long-horizon agent 的核心难题。

### 结构总结

$$\text{GRPO 的 reward/advantage 是 sequence-level，但 PPO-style ratio/loss 仍是 token-level 计算}$$

对第 $i$ 条 response：$A_{i,t}=A_i$ 对所有 $t$ 相同，但每个 token 的 ratio 不同：

$$r_{i,t}=\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{old}(y_{i,t}|x,y_{i,<t})}$$

$$L_{i,t}=\min\big(r_{i,t}A_i,\ \mathrm{clip}(r_{i,t})A_i\big)$$

所以每个 token 最终的 policy gradient 仍然不同。

## 5. 为什么必须同 prompt 分组

> 不能拿整个 batch 的 reward 一起做 normalization。

因为我们要构造的是"**这个回答相对这个问题的正常水平**有多好"。

两个 prompt，$x_1$ 很简单、$x_2$ 很难，各采 4 条：

$$x_1:R=[1,1,1,0] \qquad x_2:R=[1,0,0,0]$$

**整个 batch 一起算**（$\mu=4/8=0.5$）：$x_2$ 那条答对的 $A=+0.5$，$x_1$ 那条答错的 $A=-0.5$。看起来还行，但 batch mean 混进了**题目难度差异**。

**按 prompt 分组**：

- $x_1$（$\bar R=0.75$）：$A=[+0.25,+0.25,+0.25,-0.75]$
  → 这题本来几乎都会，答错一次是非常差的表现 ✅
- $x_2$（$\bar R=0.25$）：$A=[+0.75,-0.25,-0.25,-0.25]$
  → 这题通常答不出来，这次居然成功了，非常值得强化 ✅

本质上，分组是在减掉一个 **prompt-specific baseline**：

$$\boxed{A_i=R_i-\underbrace{\mathbb E_{y\sim\pi(\cdot|x)}[R(x,y)]}_{\text{prompt-specific baseline}}}\ \approx\ R_i-\bar R_{group}$$

这个 $b(x)$ 扮演的正是 PPO 里 $V(s)$ 的角色（"在这个状态下正常能有多好"），只是把它从**学出来的网络**换成了**同 prompt 多次 rollout 的经验均值**。

### 天然限制

$$R=[1,1,1,1] \Rightarrow A_i=0 \quad\text{全会，没信号}$$
$$R=[0,0,0,0] \Rightarrow A_i=0 \quad\text{全不会，没信号}$$

真正有用的是 $\boxed{0<\text{success rate}<1}$。这正是 [DAPO 的 dynamic sampling](08-dapo.md#2-dynamic-sampling) 要解决的问题。

## 6. GRPO 完整流程与 loss

$$\text{同 prompt 多 rollout} \rightarrow \text{reward} \rightarrow \text{group normalize} \rightarrow A_i \rightarrow \text{PPO-style loss}$$

（对照 PPO：$\text{rollout}\rightarrow V_{old}\rightarrow \text{GAE}\rightarrow A_t\rightarrow \text{PPO loss}$）

$$L_{GRPO}=\mathbb E_{i,t}\Big[\min\big(r_{i,t}A_i,\ \mathrm{clip}(r_{i,t})A_i\big)\Big]-\beta D_{KL}(\pi_\theta\|\pi_{ref})$$

里面有**两种"不要走太远"，时间尺度不同**：

| 机制 | 比较对象 | 限制什么 |
|---|---|---|
| PPO clip（$r=\pi_\theta/\pi_{old}$） | rollout 时的快照 | **这一批 rollout 上，单次更新别太猛** |
| reference KL | RL 开始前的模型 | **整个训练过程中，不要累计偏离原模型太远** |

训练到第 1000 step 时，$\pi_{old}$ 可能已经是一个很强的 RL model，$\pi_\theta$ 只需离它很近；但这个 RL model 可能已离最初的 $\pi_{ref}$ 很远，所以 reference KL 仍提供长期约束。

> **现代趋势**：很多 reasoning RL 直接取 $\beta=0$（不加 reference KL），因为 rule-based reward 很干净、强 KL 限制探索、reasoning model 希望允许明显偏离 SFT policy。DeepSeek-R1 / DAPO-style setting 常把 KL 弱化甚至去掉。所以现代 GRPO 核心往往就是：**group advantage + PPO ratio + clip**。

## 7. 收益与代价

| 收益 | 代价 |
|---|---|
| 不需要 critic model | advantage 很粗，是 sequence-level 的 |
| 少一套 forward/backward | credit assignment 更差 |
| 少一份 optimizer state | 必须同 prompt 多采样 |
| 对大模型 RL 显存/工程友好 | 整组全对或全错时几乎没信号 |

$$\boxed{\text{牺牲更细粒度的 value estimation，换更简单、更便宜的大模型 RL}}$$

> ⚠️ 实践提醒：verl 里看到 `adv_estimator=grpo` **不代表**严格是原版 DeepSeekMath GRPO，因为你可能同时开了 asymmetric clip、dynamic sampling、token-level loss、overlong filtering，实际已经是 **DAPO-style GRPO**。

## 自测（口述版）

1. 一分钟讲清 GRPO 的原理和训练方式。
2. GRPO 的 advantage 怎么算？为什么可以不用 critic？
3. 为什么除标准差而不是方差？如果 reward 整体放大 10 倍会怎样？
4. 一条 response 的 sequence-level advantage 是怎么作用到每个 token 上的？推导一遍。
5. 为什么必须同 prompt 分组？举一个难易两道题的例子。
6. GRPO 相对 PPO 的收益和代价各是什么？
7. GRPO 解决 credit assignment 了吗？
8. $\pi_{old}$ 和 $\pi_{ref}$ 在 GRPO 里分别起什么作用？两种"不要走太远"的时间尺度有什么不同？
