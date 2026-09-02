# Importance Sampling 与 ratio：为什么变成 rA

> 承接 [01](01-from-J-to-loss.md)。这一篇解决两个卡点：$A\log\pi \to Ar$ 到底是什么关系，以及为什么不是 $rA\log\pi$。

## 1. PPO 想解决的问题：采样效率

vanilla on-policy PG 的节奏是：

$$\pi_\theta \rightarrow \text{rollout} \rightarrow \text{更新一次} \rightarrow \text{数据扔掉} \rightarrow \text{重新 rollout}$$

rollout 很贵（LLM 尤其），一批数据只更新一次太浪费。PPO 想：

$$\pi_{old} \rightarrow \text{一次 rollout 一大批} \rightarrow \text{同一批数据多更新几次}$$

但第一次 SGD 之后 $\pi_\theta \ne \pi_{old}$，而手里的数据仍然是 $a\sim\pi_{old}$ 采出来的。

> **注意问题的准确表述**：不是"训练目标 $J$ 变了"。$J$ 一直没变，我们始终想最大化期望回报。
> 变的是 **"我们用来估计 $\nabla J$ 的样本分布错位了"**。

## 2. 用一个具体例子看清"错位"

old policy：$P(L)=0.5,\ P(R)=0.5$，rollout 1000 次 ≈ 500L + 500R。

训练后 new policy：$P(L)=0.9,\ P(R)=0.1$。如果真的用新 policy 重采 1000 次，应该是 900L + 100R。

但手里的数据还是 500L/500R，已经不是当前 policy 的分布了。

## 3. Importance Sampling：把旧样本重新加权

基本恒等式：

$$\boxed{\mathbb E_{x\sim p}[f(x)]=\mathbb E_{x\sim q}\Big[\frac{p(x)}{q(x)}f(x)\Big]}$$

取 $p=\pi_\theta$（想估计的）、$q=\pi_{old}$（实际采样的），定义：

$$\boxed{r=\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}}$$

回到刚才的例子：

- $L$：$r_L=0.9/0.5=1.8$，意思是"旧数据里 L 出现得太少了"，每个 L 样本算 1.8 份 → $500\times1.8=900$
- $R$：$r_R=0.1/0.5=0.2$，意思是"旧数据里 R 出现得太多了"，每个 R 样本算 0.2 份 → $500\times0.2=100$

正好恢复 900L/100R。所以 $r$ 一点都不神秘：

> **它就是把 old-policy 的样本重新加权，让它们看起来像是 current-policy 采的。**

$r$ 的读法：$r=1.5$ 表示当前 policy 给这个 action 的概率是原来的 1.5 倍；$r=1$ 表示没变。**刚 rollout 完那一刻 $\theta=\theta_{old}$，所以所有 $r=1$**，随着 SGD 才逐渐偏离 1。

## 卡点 3：A log pi 换成 A r 不是代数替换

> **常见卡点**："为什么要把原先的 $A\log\pi$ 变成 $A r$？"

**不是**这样（❌）：

$$A\log\pi \overset{\text{某种代数}}{=} Ar$$

它们是**两个不同阶段、为不同采样条件构造的 surrogate objective**：

| 条件 | 想要的梯度 | 构造的 surrogate |
|---|---|---|
| on-policy（数据来自 $\pi_\theta$） | $A\nabla\log\pi$ | $A\log\pi$ |
| off-policy（数据来自 $\pi_{old}$） | $rA\nabla\log\pi$ | $rA$ |

两者的真正联系是：**在 $\pi_\theta=\pi_{old}$（即 $r=1$）的起点，它们给出相同的梯度。**

因为（$\pi_{old}$ 对当前 $\theta$ 是常数）：

$$\nabla r=\frac{1}{\pi_{old}}\nabla\pi_\theta=\frac{\pi_\theta}{\pi_{old}}\nabla\log\pi_\theta$$

$$\boxed{\nabla r=r\nabla\log\pi_\theta}$$

所以 $\nabla(rA)=Ar\nabla\log\pi$，当 $r=1$ 时就退化成 $A\nabla\log\pi$，和原始 policy gradient 完全一样。

## 卡点 4（最关键）：为什么 surrogate 里没有 log pi

> **常见卡点**："为什么 $L=\mathbb E[A\log\pi] \to L=\mathbb E_{old}[rA]$，而不是 $L=\mathbb E_{old}[rA\log\pi]$？"

**根因**：PPO 不是"把原来的 surrogate $\mathbb E[A\log\pi]$ 用 importance sampling 改写"。如果真那样机械改写，确实会得到 $\mathbb E_{old}[rA\log\pi]$。

PPO 要匹配的是**原始 policy gradient 的梯度**，不是保留原 surrogate 的代数形式。

### 正确的推导顺序

第一步，importance sampling 直接作用在**梯度的期望**上：

$$\nabla J=\mathbb E_{\pi_\theta}[A\nabla\log\pi_\theta] \;=\; \mathbb E_{\pi_{old}}\big[\,r\,A\nabla\log\pi_\theta\,\big]$$

第二步，现在要找一个新 surrogate $L$，使得 $\nabla L$ 等于 $rA\nabla\log\pi_\theta$。

利用 $\nabla r=r\nabla\log\pi_\theta$，直接取 $L=\mathbb E_{old}[rA]$ 就有：

$$\nabla L=\mathbb E_{old}[A\nabla r]=\mathbb E_{old}[rA\nabla\log\pi_\theta]$$

✅ **正好对上。**

### 反过来验证 $rA\log\pi$ 为什么不行

$$\nabla(rA\log\pi)=A\big[(\nabla r)\log\pi+r\nabla\log\pi\big]$$

代入 $\nabla r=r\nabla\log\pi$：

$$=Ar\nabla\log\pi\,(\log\pi+1)$$

**多出来一个 $(\log\pi+1)$ 因子**，已经不是我们想要的 $rA\nabla\log\pi$ 了。

### 一句话记住

> **不是在给 $A\log\pi$ 乘 $r$，而是在重新找一个能产生正确梯度的 surrogate。**

这就是 objective 和 gradient 分不清的最后一个关键点：**先确定想要什么梯度，再反推 surrogate，而不是对 surrogate 做代数变形。**

## 4. 到这里的 PPO（还没 clip）

$$\boxed{L_{PPO}^{unclipped}=\mathbb E_{\pi_{old}}[rA]}$$

直觉检查：$A>0$ 时最大化 $rA$ 就要抬高 $r$，也就是提高 $\pi_\theta(a|s)$；$A<0$ 时要压低 $r$。仍然是"好动作提概率、坏动作降概率"。

到这里 PPO 还只是"能用旧数据训练"，还没有任何 "Proximal"（近端）成分。下一步：[为什么 $rA$ 还不够，需要 clip](03-clip-and-min.md)。

## 自测

1. PPO 遇到的问题是 "$J$ 变了" 还是别的？准确说是什么变了？
2. 用 500L/500R 的例子解释 $r$ 在做什么。
3. 推导 $\nabla r=r\nabla\log\pi$。
4. 为什么 surrogate 是 $rA$ 而不是 $rA\log\pi$？把 $\nabla(rA\log\pi)$ 算出来。
5. 为什么 $r$ 在 rollout 刚结束时等于 1？
