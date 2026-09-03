# 第三个维度：KL 到底用几个 token 来估

> 前面两个维度定了 **prefix 从哪来** 和 **对齐哪个方向**。还剩最后一个工程维度：
> 在一个 prefix 上，这个 divergence 到底拿多少个 token 来算。
>
> 这是很多人（包括很多八股材料）漏掉的一层，但它直接决定显存和梯度方差。

## 1. 三档

```text
                    OPD
                     │
          student-generated prefix
                     │
                s_t=(x,y_<t)
                     │
       ┌─────────────┼──────────────┐
       │             │              │
 sampled-token      top-k      full-vocabulary
       │             │              │
  采 1 个 token   Student Top-K    全词表
       │             │              │
  MC estimate     subset KL      exact KL
       │             │              │
 最便宜/高方差       折中        最贵/最 dense
```

| 方法 | 每个 prefix 上看什么 | 是否精确 | 监督密度 |
|---|---|---|---|
| Sampled-token | student 实际采样的 **1 个 token** | KL 的**单样本无偏估计** | 最稀疏 |
| Top-k | student 概率最高的 **k 个 token** | 近似 | 中等 |
| Full-vocabulary | 整个词表 | **精确 token-level KL** | 最密 |

## 2. Sampled-token：为什么它是无偏的

真正的 reverse KL 是

$$D_{KL}(p_t\|q_t)=\sum_{v\in\mathcal V}p_t(v)\log\frac{p_t(v)}{q_t(v)}$$

从 student 采一个 token $\hat y_t\sim p_t$，只算

$$\ell_t^{\text{sample}}=\log p_t(\hat y_t)-\log q_t(\hat y_t)$$

取期望：

$$\mathbb E_{\hat y_t\sim p_t}\big[\log p_t(\hat y_t)-\log q_t(\hat y_t)\big]=\sum_v p_t(v)\log\frac{p_t(v)}{q_t(v)}=D_{KL}(p_t\|q_t)$$

$$\boxed{\text{unbiased single-sample estimator}}$$

### 它的问题：一次只看到一个 token

$$p_S=\{A:0.4,\ B:0.3,\ C:0.2,\ D:0.1\},\qquad p_T=\{A:0.1,\ B:0.6,\ C:0.2,\ D:0.1\}$$

teacher 真正想说的是：**A 太高了，B 太低了，把概率从 A 搬到 B。**

但如果这次刚好采到 $\hat y_t=A$，估计量只看到 $\log0.4-\log0.1$，于是这一步只知道"A 该压"，**完全没看到 B 该升**。要等下次采到 B 才知道。

$$\boxed{\text{期望正确，但 variance 大、supervision 稀疏}}$$

味道和 REINFORCE / Monte Carlo 完全一样。

## 3. Full-vocabulary：贵在哪里

直接算完整分布：

```text
A: student 0.4 vs teacher 0.1 → 降
B: student 0.3 vs teacher 0.6 → 升
C: student 0.2 vs teacher 0.2 → 差不多
D: student 0.1 vs teacher 0.1 → 差不多
```

一次 forward 就拿到 **distribution-level dense supervision**，梯度方差低很多。

代价是显存：teacher 和 student 每个位置都要 $z\in\mathbb R^{V}$。

$$\text{memory}\sim O(B\cdot T\cdot V)$$

粗算一下，$T=2048$、$V=150\text{K}$、bf16：

$$2048\times150000\times2\ \text{bytes}\approx614\ \text{MB}$$

teacher + student 光 logits 就 $\approx1.2$ GB / sample；$T=16\text{K}$ 时约 4.8 GB。$B=128,\ T=4096$ 时 logits 元素数约 $7.9\times10^{10}$。

真实训练当然会 TP / 分块计算 / 及时释放 / activation checkpointing，不会天真地全堆着，但已经能看出为什么它贵。

需要区分的是：

$$\boxed{\text{主要痛点是显存和带宽，而不只是 FLOPs}}$$

因为正常 LM 训练算 CE 本来就要过 $h_tW_{\text{vocab}}$ 产生完整 logits。full-vocab 蒸馏多出来的是：student 一次 forward + **teacher 又一次 forward** + 两边完整 logits + 对整个 $V$ 做 softmax 和 KL + 为 backward 保留中间量。

## 4. Top-k：折中

取 student 自己的 top-k 集合

$$S_t=\operatorname{TopK}(p_t,k)$$

例如 student 是 `A 0.40, B 0.30, C 0.20, D 0.05, E 0.03, …`，$k=3$ 则 $S_t=\{A,B,C\}$。

**关键操作是重新归一化**（两边都要）：

$$\bar p(v)=\frac{p(v)}{\sum_{u\in S_t}p(u)},\qquad \bar q(v)=\frac{q(v)}{\sum_{u\in S_t}q(u)}$$

然后算 $D_{KL}(\bar p_t^{S_t}\|\bar q_t^{S_t})$。复杂度从 $O(TV)$ 降到 $O(TK)$。

它的含义非常清楚：

> **我不要求 student 学 teacher 的整个 vocabulary distribution，只要求 student 在"自己目前真正在考虑的几个候选"之间，学会像 teacher 一样排序和分配概率。**

也就是只训练 student 的 **decision boundary**，不管剩下十几万个 `banana / Tokyo / running / ###`。

### 和 Rethink OPD 正好接上

[Rethink OPD](07-rethink-and-mopd.md#3-opd-的学习到底发生在哪里) 的核心发现是：OPD 的有效监督实际上集中发生在 **student / teacher 的 high-probability token overlap** 上，而且这个很小的共享集合能覆盖 97%–99% 的 probability mass。

Top-k OPD 干脆直接把训练限制在这个 high-probability region 上 —— 所以 top-k 不是"抠显存的将就做法"，它和 OPD 真正在学的东西是对齐的。

## 5. Full-vocab 的稳定性问题：per-entry KL clipping

OPSD 论文的主实验做的是 full-vocab forward KL，**没有用 top-k 近似**，而是遍历整个词表做 full softmax。它解决稳定性用的是另一个东西：

$$\boxed{\sum_{v\in\mathcal V}\min(\ell_{t,v},\ \tau)}$$

即限制**每个位置、每个 token 的 divergence 贡献**不超过 $\tau$。

动机：`wait / therefore / however / think` 这类 **style token 的 KL 特别大**，会把真正的数学 reasoning token 的梯度淹掉。

$$\boxed{\text{clipping}\ne\text{top-k}}$$

clipping 还是遍历整个 vocab，只是限制单项贡献；top-k 是根本不看其余 token。

> 论文附录里的 `Top-k = -1` 是**生成时 sampling 的参数**，不是说 KL 用不用 top-k，别看混了。

### 如果自己要做 top-k 近似

teacher top-k 的概率和通常 $\sum_{v\in K}p_T(v)<1$，剩下的 probability mass 要处理，常见做法：renormalize top-k / 加一个 "other" bucket / 取 teacher 和 student top-k 的 union / sparse KD。

## 卡点：loss 的估计量 ≠ 梯度的估计量

这两件事非常容易混：

**(a) loss value 的 MC estimator**

$$\ell_t^{\text{sample}}=\log p_t(\hat y_t)-\log q_t(\hat y_t)$$

**(b) 怎么得到正确的 policy gradient**

$$A_t=\operatorname{sg}\big[\log q_t(\hat y_t)-\log p_t(\hat y_t)\big],\qquad L_{\text{PG}}=-A_t\log p_\theta(\hat y_t)$$

看到论文写 $\mathbb E_{\hat y\sim p_\theta}[\log p_\theta(\hat y)-\log q(\hat y)]$，**不能理解成"采一个 token 然后普通 CE backprop 就完了"** —— 因为 $\hat y_t\sim p_\theta$ 这个 sampling 操作本身不可导，$\theta$ 同时出现在期望的分布里和被积函数里。必须走 [04](04-reverse-kl-as-pg.md#2-把梯度真正推一遍) 那条 score-function 路线，$A_t$ 上要 stop-gradient。

这也正是 $A_t=\log\pi_T-\log\pi_S$ 这个 advantage formulation 存在的意义。

## 面试一句话

> **OPD 首先是 on-policy state distribution：prefix 由 student rollout 产生；然后在这些 student-visited prefix 上最小化 student 到 teacher 的 KL。具体 KL 可以用 sampled-token Monte-Carlo estimator、student top-k 的 renormalized subset KL，或者 full-vocabulary 精确 KL。sampled-token 最便宜但监督稀疏、方差高；full-vocab 最 dense 但有 $O(BT|\mathcal V|)$ 的 memory cost；top-k 只对 student high-probability region 做归一化后的 KL，是两者的折中。**

## 自测（口述版）

**1.** 证明 sampled-token estimator 是 KL 的无偏估计。

> **答：** 从 student 采一个 token $\hat y_t\sim p_t$，只算 $\ell_t=\log p_t(\hat y_t)-\log q_t(\hat y_t)$。取期望：
> $$\mathbb E_{\hat y_t\sim p_t}\big[\log p_t(\hat y_t)-\log q_t(\hat y_t)\big]=\sum_v p_t(v)\log\frac{p_t(v)}{q_t(v)}=D_{KL}(p_t\|q_t)$$
> 所以它是 KL 的**单样本无偏估计**。

**2.** 用 $p_S=\{A:0.4,B:0.3\}$、$p_T=\{A:0.1,B:0.6\}$ 说明 sampled-token 的稀疏性问题。

> **答：** teacher 真正想说的是「**A 太高了，B 太低了，把概率从 A 搬到 B**」。
> 但如果这一次恰好采到 $\hat y_t=A$，估计量只看到 $\log0.4-\log0.1$，于是这一步只知道「A 该压」，**完全没看到 B 该升**；要等下次采到 B 才知道。
> 所以它期望正确但**方差大、监督稀疏** —— 和 REINFORCE 是同一个味道。

**3.** 估算 $T=2048,V=150\text{K}$、bf16 时 full-vocab logits 的显存，说清楚痛点是 FLOPs 还是显存/带宽。

> **答：** $2048\times150000\times2\ \text{B}\approx614$ MB 一份，teacher + student 约 **1.2 GB/sample**；$T=16$K 时约 4.8 GB。
> **痛点主要是显存和带宽**，不是 FLOPs：正常 LM 训练算 CE 本来就要过 $h_tW_{\text{vocab}}$ 产生完整 logits。full-vocab 蒸馏多出来的是 teacher 的额外一次 forward、两边完整 logits、整个 $V$ 上的 softmax+KL、以及为 backward 保留的中间量。

**4.** Top-k KL 为什么必须重新归一化？归一化到什么？

> **答：** 因为截断后 $\sum_{v\in S_t}p(v)<1$，不再是一个概率分布，直接算 KL 没有意义。
> 两边都要归一化到 **top-k 子集内部的条件分布**：$\bar p(v)=\frac{p(v)}{\sum_{u\in S_t}p(u)}$、$\bar q$ 同理，再算 $D_{KL}(\bar p\|\bar q)$。
> 含义：不要求 student 学 teacher 的整个词表分布，只要求它**在「自己真正在考虑的几个候选」之间学会像 teacher 一样排序和分配概率**。

**5.** per-entry KL clipping 和 top-k 有什么本质区别？OPSD 为什么需要它？

> **答：** **clipping 是 $\sum_v\min(\ell_{t,v},\tau)$，仍然遍历整个 vocab**，只限制单项的贡献上限；**top-k 则根本不看其余 token**。
> OPSD 需要它是因为 `wait / therefore / however / think` 这类 **style token 的 KL 特别大**，会把真正的数学 reasoning token 的梯度淹掉。
> 注意论文附录里的 `Top-k = -1` 是**生成时 sampling 的参数**，不是说 KL 用了 top-k。

**6.** "loss 的 MC estimator" 和 "policy gradient 的估计量" 差在哪？为什么不能直接对采样 token 做 CE backprop？

> **答：** 前者 $\ell_t=\log p_t(\hat y_t)-\log q_t(\hat y_t)$ 是 KL **数值**的无偏估计；后者是 $A_t=\operatorname{sg}[\log q_t-\log p_t]$、$L_{\text{PG}}=-A_t\log p_\theta(\hat y_t)$，是为了得到**正确的梯度**。
> 不能直接对采样 token 做普通 CE backprop，因为 $\hat y_t\sim p_\theta$ 这个 **sampling 操作本身不可导**，而且 $\theta$ 同时出现在期望的分布里和被积函数里。必须走 score-function（log-derivative）路线，且 $A_t$ 上要 stop-gradient。

