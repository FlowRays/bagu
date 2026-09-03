# PPO 工程细节：完整 loss、训练循环、超参数

> 很多人知道 PPO 公式却不知道 PPO 实际怎么跑。这一篇补上时间顺序、两种 batch、超参数、监控指标，以及到 LLM 的映射。

## 1. 完整 loss

$$\boxed{L_{total}=L_{policy}+c_v L_{value}-c_e H}$$

（若写成"需要最大化"的形式则是 $L=L^{CLIP}-c_1L^{VF}+c_2S[\pi]$。）

**Policy loss**（代码里带负号做 gradient descent）：

$$L_{policy}=-\mathbb E\Big[\min\big(r_t\hat A_t,\ \mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t\big)\Big]$$

**Value loss**：

$$L_{value}=\mathbb E\big[(V_\phi(s_t)-\hat R_t)^2\big]$$

**Entropy bonus**：

$$H(\pi(\cdot|s))=-\sum_a\pi(a|s)\log\pi(a|s)$$

entropy 高 = policy 分散、还有探索性；低 = policy 很尖锐（如 $[0.99,0.005,0.005]$），可能过早 collapse。代码最小化 loss，所以写 $-c_eH$，这样会倾向让 $H\uparrow$。

三件事同时发生：

$$\boxed{\text{Policy loss：学会选更好的 action}}$$
$$\boxed{\text{Value loss：学会判断 state 值多少}}$$
$$\boxed{\text{Entropy：别太早失去探索}}$$

## 2. 训练循环（按时间顺序）

```text
1. 用当前 policy π_old rollout，收集 trajectory
2. 保存 log π_old(a_t|s_t) 和 V_old(s_t)      ← 关键：冻结快照
3. 计算 rewards
4. 算 TD residual δ_t（用 V_old）
5. 算 GAE advantage Â_t
6. 算 return target R̂_t = Â_t + V_old(s_t)
7. 冻结 Â, R̂                                 ← 关键：监督信号先固定
8. 对这批数据做多轮 minibatch SGD（ppo_epochs）
9. 更新 actor + critic
10. 用新 policy 再 rollout，进入下一轮
```

最核心的时间结构一句话：

$$\boxed{\text{先采样，再冻结监督信号，再训练}}$$

训练时这批数据已经变成 $(s_t,a_t,\log\pi_{old},V_{old},\hat A_t,\hat R_t)$，**这些后面全部固定不变**；只有 $\pi_\theta, V_\phi$ 每个 optimizer step 都在变。

## 卡点 11：为什么用 exp(对数差) 算 ratio

数学上完全等价：

$$\frac{\pi_\theta}{\pi_{old}}=\exp(\log\pi_\theta-\log\pi_{old})$$

但**数值上后者稳定得多**。LLM 里概率可能极小，比如 $\pi(a|s)=10^{-20}$ 甚至更小，直接存概率容易 underflow；而 log-prob 是 $\log 10^{-20}\approx-46$，很好表示。

所以工程里全程存 $\log\pi$，需要 ratio 时再：

$$\boxed{r=\exp(\log\pi_\theta-\log\pi_{old})}$$

这就是 verl 里常见的：

```python
old_log_prob            # rollout 时保存
log_prob                # 当前 policy 重新 forward
ratio = torch.exp(log_prob - old_log_prob)
```

## 3. 两种 batch 不要混

| | 含义 | 例子 |
|---|---|---|
| **Rollout batch** | 环境采样得到的完整数据量 $N$ | 65536 transitions |
| **SGD minibatch** | 优化时切成的小块 $M$ | 1024 → 每 epoch 64 个 optimizer step |

**PPO epochs**：同一批 rollout 数据被反复训练几遍，**不是** rollout 几遍。

```text
rollout batch = 65536
epoch 1: 打乱 → minibatch SGD
epoch 2: 打乱 → minibatch SGD
epoch 3: 打乱 → minibatch SGD
```

epoch 越多数据利用率越高，但 $\pi_\theta$ 离 $\pi_{old}$ 越远，clip 比例越来越高。**PPO 本质是在数据利用率和 on-policy 稳定性之间取平衡。** `ppo_epochs=1` 就是非常保守地每批 rollout 只优化一遍。

## 4. 超参数表

| 超参数 | 作用 | 常见值 |
|---|---|---|
| $\gamma$ | reward discount | 0.99 |
| $\lambda$ | GAE bias-variance tradeoff | 0.95 |
| $\epsilon$ / `clip_range` | PPO ratio clipping | 0.2（常见 0.1~0.3） |
| learning rate | 更新幅度 | 经典 RL $3\times10^{-4}$；**LLM RL $10^{-6}\sim10^{-5}$** |
| PPO epochs | 同批 rollout 重用几遍 | 3~10，经典常用 4 |
| minibatch size | 每次 SGD 样本数 | 64/128/256 |
| $c_v$ | value loss 权重 | 0.5 |
| $c_e$ | entropy bonus 权重 | 0~0.01（LLM reasoning RL 常直接 0） |
| max grad norm | gradient clipping | 0.5 |

$\gamma=0.99$ 时 100 步后的 reward 权重约 $0.99^{100}\approx0.366$；长 horizon 有时用 0.995、0.999。

$\epsilon$ 小 = 更保守；大 = 单批 rollout 可以更新得更猛。

⚠️ **经典 PPO 默认值和大模型 RL 不是一个尺度**，尤其 learning rate。对 LLM/VLM 的 PPO/GRPO，真正关键的是：rollout batch size、mini/micro-batch、ppo epochs、clip range、KL、learning rate。

## 5. Advantage normalization

实际 PPO 常做：

$$A_t\leftarrow\frac{A_t-\mu_A}{\sigma_A+\epsilon}$$

因为 advantage 的量级可能变化极大（某个 batch $A\in[-1000,1000]$，另一个 $A\in[-0.1,0.1]$），梯度尺度会完全不同。归一化后 $\mathbb E[A]\approx0,\ \mathrm{std}(A)\approx1$，训练稳很多。

## 6. 三个必看监控指标

$$\boxed{KL,\quad clipfrac,\quad entropy}$$

**approx KL**：用 $\log r_t=\log\pi_\theta-\log\pi_{old}$ 近似计算。真正关心的是 policy 到底变了多少。如果 KL 突然很大（如 0.5）说明 policy 已经跑飞，有些实现会 early stop PPO epochs / 自适应 lr / 加 KL penalty。

**clipfrac**：落到 clipping region 的样本比例。

$$\text{clipfrac}=\frac{\#\{t:\ r_t\ \text{进入 clip 区域}\}}{N}$$

clipfrac=0.8 表示 80% 的 token 都撞 clip 了，说明当前 policy 已离 rollout policy 很远。可能原因：lr 太大 / PPO epochs 太多 / minibatch 太小 / advantage 太极端 / rollout-train policy mismatch。

**entropy**：监控是否过早 collapse。

## 7. 映射到 LLM

传统 RL 与 LLM 的对应：

$$s_t=(x,y_{<t}) \quad\text{（prompt + 已生成 token）}, \qquad a_t=y_t \quad\text{（下一个 token）}$$

$$\pi_\theta(a_t|s_t)=P_\theta(y_t|x,y_{<t})$$

一个 response $y=(y_1,\dots,y_T)$ 就相当于一条 trajectory $\tau$，但 **reward 通常只在最后得到** $R(y)$（例如数学题答对 = 1、答错 = 0）。ratio 完全同构：

$$r_t=\frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{old}(y_t|x,y_{<t})}=\exp(\log p_\theta(y_t)-\log p_{old}(y_t))$$

### old policy ≠ reference policy（LLM RL 高频混淆）

| | $\pi_{old}$ | $\pi_{ref}$ |
|---|---|---|
| 是什么 | 产生这批 rollout 的 actor 快照 | SFT model / RL 开始前固定的模型 |
| 用途 | 算 importance ratio $r_t$ | KL regularization，防止跑离原始 LM 太远 |
| 更新频率 | 每轮 rollout 都变 | 长期不动 |
| 时间尺度 | 短期训练基准 | 长期行为锚点 |

$$\boxed{\text{old policy} \ne \text{reference policy}}$$

详见 [07-kl.md](07-kl.md)。

### 典型 RLHF PPO 架构

```text
Actor ──rollout──> responses
                     │
   Reward Model ─────┼──> reward
   Critic ───────────┼──> value
                     ▼
                    GAE ──> advantage ──> PPO loss ──> update actor
   Reference Model（固定）──> KL penalty
```

actor / critic / reference / reward 可能是四个不同的模型 —— 这正是 GRPO 想砍掉 critic 的动机（32B actor 再配一个 32B critic，显存、forward、backward、optimizer state 都很贵）。

## 8. PPO vs Q-learning

| | PPO | DQN |
|---|---|---|
| 学什么 | 直接参数化 policy $\pi_\theta(a|s)$ | 学 $Q(s,a)$，policy 由 $\arg\max_a Q$ 间接产生 |
| 类别 | policy-based / actor-critic | value-based |

## 自测

**1.** 默写完整 PPO loss 三项，说明各自作用和符号方向。

> **答：** $$L=\underbrace{-\mathbb E\big[\min(r_tA_t,\ \text{clip}(r_t,1-\epsilon,1+\epsilon)A_t)\big]}_{\text{policy loss，要最小化所以取负}}+\underbrace{c_1\,\mathbb E\big[(V_\phi(s_t)-\hat R_t)^2\big]}_{\text{value loss，正号}}-\underbrace{c_2\,\mathbb E[\mathcal H(\pi_\theta)]}_{\text{entropy bonus，负号=鼓励探索}}$$
>  policy 项要最大化 surrogate 所以前面加负号；value 项是回归误差直接最小化；entropy 项要**最大化**熵所以减去它。

**2.** 按时间顺序说出一次 PPO iteration 的步骤，指出哪些量在什么时候被冻结。

> **答：** ① 用 $\pi_{\theta}$ rollout 采数据，**同时记录 $\log\pi_{\text{old}}$ 和 $V_{\text{old}}$**（此刻冻结）；
> ② 算 reward、用 $V_{\text{old}}$ 算 GAE 得到 $\hat A$，再算 $\hat R=\hat A+V_{\text{old}}$（$\hat A,\hat R$ 此后是常量，stop-gradient）；
> ③ 把这批数据切成 minibatch，跑 `ppo_epochs` 轮：每次算当前 $\log\pi_\theta$、$r=\exp(\log\pi_\theta-\log\pi_{\text{old}})$、loss、backward、step；
> ④ 丢弃这批数据，回到 ①。

**3.** 为什么用 `exp(log_prob - old_log_prob)` 而不是直接相除？

> **答：** 概率本身可能极小（长序列上是很多个小数连乘），直接相除会**下溢**成 0/0。
> 模型输出的本来就是 log prob，用 $\exp(\log\pi_\theta-\log\pi_{\text{old}})$ 在对数域做减法再取指数，数值稳定得多，也少一次 log 运算。

**4.** rollout batch 和 SGD minibatch 的区别？`ppo_epochs=4` 是什么意思？

> **答：** **rollout batch** 是一次采样收集到的全部数据（决定 $\pi_{\text{old}}$ 是谁）；**SGD minibatch** 是从中切出来做一次参数更新的小块。
> `ppo_epochs=4` 表示同一批 rollout 数据要被完整遍历 4 遍。正因为要重复利用，$\pi_\theta$ 才会逐渐偏离 $\pi_{\text{old}}$，才需要 ratio 修正和 clip 约束。

**5.** clipfrac 很高说明什么？可能的原因有哪些？

> **答：** clipfrac = 被 clip 掉的 token 比例。很高说明 **$\pi_\theta$ 已经离 $\pi_{\text{old}}$ 太远**，大量样本不再提供梯度，数据利用率低。
> 可能原因：lr 太大、`ppo_epochs` 太多、$\epsilon$ 太小、advantage 尺度过大（没归一化）、batch 太小导致噪声大。

**6.** $\pi_{old}$ 和 $\pi_{ref}$ 有什么区别？

> **答：** $\pi_{\text{old}}$ 是**产生当前这批 rollout 的 policy 快照**，每轮 rollout 都更新，用于 importance ratio，是**短期训练基准**。
> $\pi_{\text{ref}}$ 是**训练开始前冻结的模型**（通常是 SFT model），长期不动，用于 KL penalty，是**长期行为锚点**。

**7.** LLM 里 $s_t$、$a_t$ 分别是什么？reward 在什么时候给？

> **答：** $s_t=(x,y_{<t})$ 是 prompt 加上已生成的前缀；$a_t=y_t$ 是下一个 token。所以一条 response 就是一条 trajectory。
> reward 通常是 **sequence-level 的**：只在最后一个 token 给（RM 打分或规则判对错），中间 token 的即时 reward 为 0（若有 reference KL 惩罚则逐 token 折进去）。这正是 credit assignment 困难的根源。

