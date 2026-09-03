# 熵、交叉熵、KL、JSD：蒸馏的信息论地基

> 这四个量在蒸馏里天天出现，但很容易只会背公式不会用。本篇的目标是**把它们串成一条链**，
> 并且让"SFT 是 forward KL""soft CE 就是 forward KL"这两句话变成能当场推出来的结论。
>
> forward / reverse KL 的**记忆方法**（看谁在期望下面）在 [RL/07 KL 散度](../rl/07-kl.md) 里，本篇不重复。

## 1. 熵：用 $q$ 自己编码自己

$$\boxed{H(q)=-\sum_v q(v)\log q(v)}$$

含义：数据真的来自 $q$，用**最优编码**（为 $q$ 量身定做的编码）平均需要多少信息量。

- 分布越集中（one-hot），$H\to0$：完全确定，不需要信息
- 分布越均匀，$H$ 越大：最不确定

LLM 里的 entropy 就是这个东西按 next-token 分布逐位置算出来的，RL 训练监控的 `entropy` 指标即由此而来（entropy 塌缩 = 模型变得过度确定）。

## 2. 交叉熵：数据来自 $q$，却用 $p$ 去编码

$$\boxed{H(q,p)=-\sum_v q(v)\log p(v)}$$

含义：真实分布是 $q$，但你手上只有模型 $p$，只能用 $p$ 的编码去编 $q$ 的样本，平均要付出多少信息量。

$p=q$ 时代价最小，此时 $H(q,p)=H(q)$。

## 3. 交叉熵 = 熵 + KL（最该背的一条恒等式）

$$\boxed{H(q,p)=H(q)+D_{KL}(q\|p)}$$

一行推导：

$$H(q,p)=-\sum_v q\log p=-\sum_v q\log q+\sum_v q\log\frac{q}{p}=H(q)+D_{KL}(q\|p)$$

**训练时 $q$（label / teacher）是固定的，所以 $H(q)$ 是常数**，于是：

$$\boxed{\min_p H(q,p)\iff\min_p D_{KL}(q\|p)}$$

这就是那句八股的来源：

> **交叉熵本质上是在做 forward KL。**

注意方向：$D_{KL}(q\|p)$，**数据/teacher 在前，模型在后**。所以任何"最小化交叉熵"的训练，天然是 mode-covering 的那一侧。

## 4. 三种交叉熵：hard / soft / 采样

设 student 输出 $p=[0.4,0.3,0.3]$。

### (a) Hard CE（one-hot label）

label 是 $q=[1,0,0]$，于是

$$L=-\sum_v q(v)\log p(v)=-\log p(A)$$

这就是 SFT 每天见到的 $-\log\pi_\theta(y_t^*\mid x,y^*_{<t})$。

由于 one-hot 分布的熵 $H(q)=0$，此时恒等式退化成一个**严格相等**：

$$\boxed{\text{CE}=D_{KL}(\delta_{y^*}\|\pi_\theta)}$$

### (b) Soft CE（teacher 的完整分布当 label）

teacher 给 $q=[0.7,0.2,0.1]$：

$$\boxed{L_{\text{softCE}}=-\sum_v \pi_T(v|s)\log\pi_S(v|s)}$$

它和 $D_{KL}(\pi_T\|\pi_S)$ 只差一个与 student 无关的常数 $-H(\pi_T)$，所以**优化上完全等价**。

"soft" 的含义：label 不再是"A 是唯一正确答案"，而是"A 很好，B 也可以，C 有一点可能"。

### (c) 从 teacher 采样再做 hard CE

按 $q$ 采样：70% 采到 A、20% B、10% C。单次 loss 是 $-\log p_A$ 或 $-\log p_B$……，但期望是

$$\mathbb E_{a\sim q}[-\log p(a)]=-\sum_v q(v)\log p(v)=H(q,p)$$

$$\boxed{\text{teacher 采样的 hard CE}\ \xrightarrow{\ \text{期望}\ }\ \text{soft CE}}$$

所以三者的关系是：

| 形式 | label | 方差 | 需要 teacher 什么 |
|---|---|---|---|
| hard CE | one-hot GT | — | 只要一条数据 |
| soft CE | teacher 完整分布 | **最低** | 全词表 logits（贵） |
| 采样 CE | teacher 采出的 token | 高（单样本 MC） | 只要 teacher 能生成 |

**soft CE 是把所有可能的 teacher sample 一次性平均掉；hard SFT 是从这个分布里抽一个样本。**

## 5. 一个统一的梯度：logits 梯度等于 p 减 q

对 softmax + 交叉熵，无论 $q$ 是 one-hot 还是 soft，**对 logits 的梯度形式完全一样**：

$$\boxed{\frac{\partial L}{\partial z}=p-q}$$

这一条把 hard 和 soft 统一了，非常好用。

**Hard**（$p=[0.4,0.3,0.3]$，$q=[1,0,0]$）：

$$p-q=[-0.6,\ 0.3,\ 0.3]$$

A 的 logit 上升，B/C 下降 → **把所有概率往唯一 GT token 上压**。

**Soft**（$q=[0.7,0.2,0.1]$）：

$$p-q=[-0.3,\ 0.1,\ 0.2]$$

A 上升，B 略降，C 降更多 → 目标是 $p\to q$，**而不是** $p\to[1,0,0]$。

一句话：hard CE 让分布坍缩到一个点，soft CE 让分布去拟合 teacher 的形状。

## 6. JSD：把两个 KL 的爆炸都拆掉

先构造 teacher / student 的 mixture：

$$m=\beta\,p_T+(1-\beta)\,p_S$$

然后

$$\boxed{\mathrm{JSD}_\beta(p_T,p_S)=\beta\,D_{KL}(p_T\|m)+(1-\beta)\,D_{KL}(p_S\|m)}$$

$\beta=\tfrac12$ 是经典对称版本，$m=\tfrac{p_T+p_S}{2}$。

### 为什么要绕一个 $m$

两个 KL 都有"分母趋近 0 就爆炸"的问题：

$$D_{KL}(T\|S):\ p_T(v)>0,\ p_S(v)\to0\ \Rightarrow\ \log\frac{p_T}{p_S}\to\infty$$
$$D_{KL}(S\|T):\ p_S(v)>0,\ p_T(v)\to0\ \Rightarrow\ \log\frac{p_S}{p_T}\to\infty$$

而混合之后，只要**有一边**有概率，$m$ 就不会是 0：

$$p_T(v)>0,\ p_S(v)=0\ \Rightarrow\ m(v)=\tfrac12 p_T(v)>0\ \Rightarrow\ \frac{p_T(v)}{m(v)}=2$$

于是 JSD 有两条 KL 没有的好性质：

$$\boxed{\mathrm{JSD}(T,S)=\mathrm{JSD}(S,T)}\qquad\boxed{0\le \mathrm{JSD}\le\ln 2\ (\beta=0.5,\ \text{自然对数})}$$

**即使两个分布 support 完全不重叠，JSD 仍然有限。**

### 三者的定位

$$\boxed{\text{Forward KL：偏 teacher，cover}}\quad\boxed{\text{Reverse KL：偏 student，seek}}\quad\boxed{\text{JSD：折中，更温和}}$$

$\beta$ 就是这个折中的旋钮：$\beta\to1$ 时 $m\to p_T$，行为偏向 forward 一侧。

## 7. 一张总结表

| 量 | 公式 | 一句话 |
|---|---|---|
| 熵 | $H(q)=-\sum q\log q$ | 用自己编码自己 |
| 交叉熵 | $H(q,p)=-\sum q\log p$ | 数据来自 $q$，却用 $p$ 编码 |
| KL | $D_{KL}(q\|p)=\sum q\log\frac{q}{p}$ | 多付出的那部分代价 |
| 恒等式 | $H(q,p)=H(q)+D_{KL}(q\|p)$ | **最小化 CE ≡ 最小化 forward KL** |
| JSD | $\beta KL(p_T\|m)+(1-\beta)KL(p_S\|m)$ | 对称、有界、温和 |

## 自测（口述版）

**1.** 写出熵、交叉熵、KL 三个定义，并当场推出 $H(q,p)=H(q)+D_{KL}(q\|p)$。

> **答：** $H(q)=-\sum_v q\log q$；$H(q,p)=-\sum_v q\log p$；$D_{KL}(q\|p)=\sum_v q\log\frac qp$。
> 推导：$H(q,p)=-\sum q\log p=-\sum q\log q+\sum q\log\frac qp=H(q)+D_{KL}(q\|p)$。
> 语义：熵是用 $q$ 自己的最优编码编 $q$；交叉熵是数据来自 $q$ 却用 $p$ 的编码；KL 是多付出的那部分代价。

**2.** 为什么说"最小化交叉熵就是在做 forward KL"？方向是哪个在前？

> **答：** 训练时 $q$（label / teacher）是固定的，所以 $H(q)$ 是常数，于是 $\min_p H(q,p)\iff\min_p D_{KL}(q\|p)$。
> 方向是 **$D_{KL}(q\|p)$：数据/teacher 在前，模型在后**，这正是 forward KL，所以任何「最小化交叉熵」的训练天然是 mode-covering 那一侧。

**3.** one-hot label 时，CE 和 KL 是"等价"还是"相等"？为什么？

> **答：** 是**相等**，不只是等价。因为 one-hot 分布的熵 $H(\delta_{y^*})=0$，恒等式退化成 $\text{CE}=0+D_{KL}(\delta_{y^*}\|\pi_\theta)$，两者数值完全一样，不差常数。

**4.** 写出 softmax+CE 对 logits 的梯度，并用 $q=[1,0,0]$ 和 $q=[0.7,0.2,0.1]$ 各算一次，说明行为差别。

> **答：** $\boxed{\frac{\partial L}{\partial z}=p-q}$，**hard 和 soft 通用**。设 $p=[0.4,0.3,0.3]$：
> **hard** $q=[1,0,0]$：$p-q=[-0.6,\,0.3,\,0.3]$ → A 的 logit 上升、B/C 下降，**把所有概率往唯一 GT token 上压**。
> **soft** $q=[0.7,0.2,0.1]$：$p-q=[-0.3,\,0.1,\,0.2]$ → A 上升、B 略降、C 降更多，目标是 $p\to q$（**拟合 teacher 的形状**）而不是 $p\to[1,0,0]$。

**5.** 从 teacher 采样做 hard CE，和直接用 teacher 分布做 soft CE，是什么关系？谁方差小？

> **答：** **期望相等**：$\mathbb E_{a\sim q}[-\log p(a)]=-\sum_v q(v)\log p(v)=H(q,p)$，所以采样版是 soft CE 的**无偏 Monte-Carlo 估计**。
> **soft CE 方差更小**（把所有可能的 teacher sample 一次性平均掉了），但需要 teacher 的完整分布；采样版只需要 teacher 能生成，更便宜。

**6.** 写出 JSD 的定义，解释为什么要引入 mixture $m$，以及它有界这件事从哪来。

> **答：** $m=\beta p_T+(1-\beta)p_S$，$\mathrm{JSD}_\beta=\beta D_{KL}(p_T\|m)+(1-\beta)D_{KL}(p_S\|m)$。
> 引入 $m$ 是因为两个 KL 都有「分母趋 0 就爆炸」的问题；混合后只要**有一边**有概率，$m$ 就不为 0：$p_S(v)=0$ 时 $\frac{p_T(v)}{m(v)}=2$，是有限的。
> 有界正是从这里来：$\beta=0.5$、自然对数时 $0\le\mathrm{JSD}\le\ln2$，而且对称。**即使两个分布 support 完全不重叠也有限。**

