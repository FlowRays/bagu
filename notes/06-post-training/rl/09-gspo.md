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

1. GSPO 质疑 GRPO 的什么？
2. sequence-level ratio 怎么写？为什么要做长度归一化？
3. GRPO 和 GSPO 的 clip 分别在什么粒度上做？
4. 为什么说"把整条 sequence 当 action"更自然？
