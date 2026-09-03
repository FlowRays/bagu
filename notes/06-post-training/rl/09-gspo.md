# GSPO：Group Sequence Policy Optimization

> Qwen 团队 2025 年提出，用于 Qwen3 系列的 RL scaling。
> ⚠️ **本篇内容较薄**，只来自对话开头的总览，没有像 PPO/GRPO 那样逐层追问过。如果需要深挖到这里，得再补一轮。

## 1. 核心质疑

GSPO 认为 GRPO 有一个更根本的问题：

> **reward 明明是 sequence-level 的，但 importance sampling 却是在 token-level 做。**

GRPO 里每个 token 各自算 ratio 再各自 clip：

$$r_{i,t}=\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{old}(y_{i,t}|x,y_{i,<t})}$$

但一个 LLM rollout 的"真实 action"更自然地是整条 sequence $y_i=(y_{i,1},\dots,y_{i,T})$，其概率是：

$$\pi(y_i|x)=\prod_t\pi(y_{i,t}|x,y_{i,<t})$$

## 2. Sequence-level importance ratio

为防止长度造成数值爆炸，通常写成长度归一化后的形式：

$$\boxed{r_i^{seq}=\exp\Big[\frac{1}{|y_i|}\sum_t\log\frac{\pi_\theta(y_{i,t}|\cdot)}{\pi_{old}(y_{i,t}|\cdot)}\Big]}$$

可以理解成**每个 token ratio 的几何平均**。然后 **clip 也在 sequence level 做**。

## 3. GRPO vs GSPO

| | 粒度 | 问的问题 |
|---|---|---|
| GRPO | $r_{i,1},r_{i,2},\dots,r_{i,T}$ | 每个 token：**你变太多了吗？** |
| GSPO | $r_i^{seq}$ | 整条 trajectory：**这整个回答相对 old policy 变化太大了吗？** |

$$\boxed{\text{GRPO：token-level importance weighting}}$$
$$\boxed{\text{GSPO：sequence-level importance weighting}}$$

Qwen 团队称这样训练更稳定，**对 MoE RL 尤其明显**。

## 4. 为什么这个想法很自然

对 Atari，每个 timestep $a_t$ 确实是环境中的一个 action。但 LLM 的 $\text{token}_t$ 真的应该被视作独立 RL action 吗？

- 从 MDP 表述上：可以。
- 但 reward $R(y)$ **往往只在整个 response 完成后产生**（数学答案对/错）。

所以真正获得 reward 的对象更像是 $\boxed{\text{整个 reasoning trajectory}}$。GSPO 的哲学就是：

> 那么 policy correction / clipping 也应该更贴近 trajectory-level。

这与 long-horizon agent 遇到的问题是直接相连的。

## 5. 四者的最终对照

$$\boxed{PPO\rightarrow GRPO}\ \text{主要改：advantage 怎么估计}$$

$$\boxed{GRPO\rightarrow DAPO}\ \text{主要改：怎么让 reasoning RL 稳定、高效地 scale}$$

$$\boxed{GRPO/DAPO\rightarrow GSPO}\ \text{触碰更底层的问题：}$$

> **LLM RL 的 policy ratio，到底应该把 token 当 action，还是把整个 sequence 当 action？**

## 待补（下次跟 GPT 深挖）

- [ ] GSPO 的完整 loss 形式（clip 在 sequence level 具体怎么写）
- [ ] 为什么对 MoE RL 收益特别明显（router 波动导致 token-level ratio 噪声？）
- [ ] 长度归一化的几何平均，对长短 response 的实际影响
- [ ] 与 DAPO token-level loss 聚合的关系（一个改 ratio 粒度、一个改 loss 聚合粒度，是否冲突）

## 自测

**1.** GSPO 质疑 GRPO 的什么？

> **答：** 质疑 **token-level importance ratio 本身是否合理**。
> GRPO 的 advantage 是 sequence-level 的（整条 response 一个 $A$），但 ratio 和 clip 却是逐 token 做的。GSPO 认为既然奖励和 advantage 都定义在序列上，**重要性比也应该定义在序列上**，token-level ratio 会引入不必要的方差。

**2.** sequence-level ratio 怎么写？为什么要做长度归一化？

> **答：** $$s_i(\theta)=\left(\frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)}\right)^{1/|y_i|}=\exp\Big(\frac1{|y_i|}\sum_t\big[\log\pi_\theta(y_{i,t})-\log\pi_{\text{old}}(y_{i,t})\big]\Big)$$
> 即**几何平均**。做长度归一化是因为整条序列的 log prob 之和随长度线性增长，不归一化的话长序列的 ratio 会指数级偏离 1，clip 区间对不同长度的序列含义完全不同，没法用统一的 $\epsilon$。

**3.** GRPO 和 GSPO 的 clip 分别在什么粒度上做？

> **答：** **GRPO 在 token 粒度**：每个 token 各自算 $r_t$、各自 clip，一条序列里可能一部分 token 被 clip、一部分没有。
> **GSPO 在 sequence 粒度**：整条序列一个 $s_i$、一起 clip，要么整条被截断要么整条不被截断。

**4.** 为什么说"把整条 sequence 当 action"更自然？

> **答：** 因为在 LLM RL 里，**reward 是对整条 response 给的**，advantage 也是 sequence-level 的。既然优化目标的基本单位是序列，那么做 importance sampling 修正和 proximal 约束时，基本单位也应该是序列，这样 ratio 的分子分母才和 reward 的定义域一致。
> token-level ratio 相当于把一个序列级的决策硬拆成 $|y|$ 个独立决策，引入了额外的方差。

