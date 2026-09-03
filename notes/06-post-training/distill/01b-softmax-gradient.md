# 前置：softmax 求导的三个式子（从零推）

> SFT 的 $\pi-\text{onehot}$、蒸馏的全部梯度、attention 的反向传播，**都建立在这三个式子上**。
> 这一篇假设你只记得两件事：**链式法则**、**$e^x$ 求导还是 $e^x$**。其余从头讲。

## 0. 符号表：先把字面意思对上

| 符号 | 读法 | 翻译成代码 |
|---|---|---|
| $\mathcal V$ | 词表 | 所有 token 的集合，大小 $V$ |
| $z$ | logits | 网络最后一层输出的 $V$ 个实数，`z[0..V-1]` |
| $z_v$ | 第 $v$ 个 logit | `z[v]` |
| $\pi(v)$ | 概率 | softmax 之后的第 $v$ 个概率，`p[v]` |
| $\sum_v a_v$ | 对 $v$ 求和 | `sum(a[v] for v in range(V))` |
| $\sum_u$ | 对 $u$ 求和 | $u$ 只是**另一个循环变量名**，和 $v$ 没关系 |
| $\delta_{vk}$ | Kronecker delta | `int(v == k)` |
| $\dfrac{\partial f}{\partial z_k}$ | 对 $z_k$ 的偏导 | 只把 `z[k]` 当变量、其他 `z[j]` 当常数，然后求导 |
| $\nabla_z f$ | 梯度 | 把 $\dfrac{\partial f}{\partial z_0},\dots,\dfrac{\partial f}{\partial z_{V-1}}$ 排成一个长度 $V$ 的向量 |
| $\mathbb E_{v\sim\pi}[f(v)]$ | 期望 | $\sum_v \pi(v)f(v)$，**就是权重加起来等于 1 的加权平均** |

最后一行特别重要：以后看到 $\mathbb E$ 不要紧张，**它就是个加权求和**，权重是概率。

## 1. 为什么会出现两个下标 v 和 k

softmax 的输入是 $V$ 个数，输出也是 $V$ 个数：

```text
z[0..V-1]  --softmax-->  p[0..V-1]
```

所以"softmax 的导数"**不是一个数**，是一张 $V\times V$ 的表（Jacobian 矩阵）：

$$J[v][k]\ :=\ \frac{\partial\,p[v]}{\partial\,z[k]}$$

读作：**把第 $k$ 个 logit 往上推一点点，第 $v$ 个概率会变多少。**

- $k$ = 你**推**的那个（输入侧）
- $v$ = 你**看**的那个（输出侧）

**先建立直觉，不用算**：把 `z[k]` 推高，`p[k]` 肯定变大；但所有概率必须加起来等于 1，所以其他 `p[v]` 只能一起变小。

$$\boxed{\text{这张表：对角线为正，非对角线为负，且每一列加起来等于 }0}$$

后面算出来的式子必须符合这个，不符合就是算错了。

## 2. delta 符号就是 `v == k`

$$\delta_{vk}=\begin{cases}1,& v=k\\[2pt] 0,& v\ne k\end{cases}$$

不是什么高深东西，**就是把一个 if 写进公式里**。它存在的唯一理由：想用**一个式子**同时覆盖"对角线"和"非对角线"两种情况，不想写成分段函数。

所以以后看到 $\delta_{vk}-\pi(k)$，直接翻译成：

```python
int(v == k) - p[k]    # v == k 时:  1 - p[k]  ← 正数
                      # v != k 时:  0 - p[k]  ← 负数
```

正好就是第 1 节的直觉。**看到 delta 就想"这是在区分 v 和 k 是不是同一个"。**

## 3. 第一个式子：log-softmax 的导数

### 3.1 为什么先推 log 而不是先推 softmax

因为 log 之后**除法变成减法**，会好算非常多。而且实践中大家本来就全程待在 log 空间（`log_softmax`），推完 log 版再乘一下就能得到 softmax 版。

### 3.2 化简

softmax 的定义：

$$\pi(v)=\frac{e^{z_v}}{\sum_u e^{z_u}}$$

分母那个 $\sum_u e^{z_u}$ 是**把所有 logit 取 exp 再加起来**的一个标量，给它起个名字 $S$：

$$S:=\sum_u e^{z_u}=e^{z_0}+e^{z_1}+\dots+e^{z_{V-1}}$$

两边取对数，用 $\log\frac{a}{b}=\log a-\log b$ 和 $\log e^x=x$：

$$\log\pi(v)=\log\frac{e^{z_v}}{S}=\underbrace{\log e^{z_v}}_{=\,z_v}-\log S=\boxed{z_v-\log S}$$

**这一步是全篇最关键的化简**：一个又是指数又是分数的东西，变成了「一个 logit 减一个公共项」。

### 3.3 求导：拆成两块

$$\frac{\partial\log\pi(v)}{\partial z_k}=\underbrace{\frac{\partial z_v}{\partial z_k}}_{\text{第 1 块}}-\underbrace{\frac{\partial \log S}{\partial z_k}}_{\text{第 2 块}}$$

**第 1 块**：$z_v$ 会不会随 $z_k$ 变？只有 $v$ 和 $k$ 是同一个下标时才会，这时导数是 1；否则 $z_v$ 对 $z_k$ 来说是个常数，导数是 0。

$$\frac{\partial z_v}{\partial z_k}=\begin{cases}1,&v=k\\0,&v\ne k\end{cases}=\ \boxed{\delta_{vk}}$$

$$\boxed{\text{delta 就是从这里冒出来的 —— 它不是被"引入"的，是求导自然的结果}}$$

**第 2 块**：链式法则，外层是 $\log$，内层是 $S$：

$$\frac{\partial\log S}{\partial z_k}=\frac{1}{S}\cdot\frac{\partial S}{\partial z_k}$$

再算里面的 $\dfrac{\partial S}{\partial z_k}$。把 $S$ 展开：

$$\frac{\partial S}{\partial z_k}=\frac{\partial}{\partial z_k}\Big(e^{z_0}+e^{z_1}+\dots+e^{z_k}+\dots+e^{z_{V-1}}\Big)$$

对 $z_k$ 求偏导时，**除了 $e^{z_k}$ 那一项，其他每一项都是常数，导数全是 0**；只有第 $k$ 项活下来，而 $e^x$ 求导还是 $e^x$：

$$\frac{\partial S}{\partial z_k}=e^{z_k}$$

代回去：

$$\frac{\partial\log S}{\partial z_k}=\frac{e^{z_k}}{S}=\boxed{\pi(k)}$$

$$\boxed{\text{softmax 自己又长出来了 —— 这是这套推导最漂亮的地方}}$$

### 3.4 合起来

$$\boxed{\ \frac{\partial\log\pi(v)}{\partial z_k}=\delta_{vk}-\pi(k)\ }\qquad\text{(式 1)}$$

**读法**：推高第 $k$ 个 logit，第 $v$ 个 token 的对数概率变化 = 「$v$ 是不是就是 $k$」减去「$k$ 的当前概率」。

**注意一个结构事实**（后面反复用到）：$v$ **只出现在 delta 里**，$-\pi(k)$ 这半边跟 $v$ 完全无关。

## 4. 第二个式子：softmax 本身的导数

不用重推，从式 1 直接换算。链式法则的一个常用变形：

$$\frac{\partial \log f}{\partial t}=\frac{1}{f}\cdot\frac{\partial f}{\partial t}\qquad\Longrightarrow\qquad \frac{\partial f}{\partial t}=f\cdot\frac{\partial\log f}{\partial t}$$

（这就是 log-derivative trick 的全部内容，后面 policy gradient 用的也是它。）

令 $f=\pi(v)$、$t=z_k$：

$$\boxed{\ \frac{\partial\pi(v)}{\partial z_k}=\pi(v)\big(\delta_{vk}-\pi(k)\big)\ }\qquad\text{(式 2)}$$

**式 2 就是式 1 乘个 $\pi(v)$，没有新东西。**

回头验第 1 节的直觉：

- $v=k$：$\pi(k)(1-\pi(k))>0$ ✓ 对角线为正
- $v\ne k$：$-\pi(v)\pi(k)<0$ ✓ 非对角线为负

## 5. 第三个式子：score 期望为零

$$\sum_v\pi(v)\big(\delta_{vk}-\pi(k)\big)\ \overset{?}{=}\ 0\qquad\text{(式 3)}$$

**这是三个里最重要的一个**，后面所有"某一项精确消掉"都是它。逐步展开：

**第一步，拆成两个求和：**

$$\sum_v\pi(v)\big(\delta_{vk}-\pi(k)\big)=\underbrace{\sum_v\pi(v)\delta_{vk}}_{(A)}-\underbrace{\sum_v\pi(v)\pi(k)}_{(B)}$$

**第二步，算 (A)。** 把求和号完全写开：

$$(A)=\pi(0)\delta_{0k}+\pi(1)\delta_{1k}+\dots+\pi(k)\delta_{kk}+\dots+\pi(V-1)\delta_{(V-1)k}$$

$\delta_{vk}$ 只在 $v=k$ 时是 1，其余全是 0，所以**整个求和只有第 $k$ 项活下来**：

$$(A)=\pi(k)$$

$$\boxed{\text{记住这个套路：}\sum_v(\cdots)\delta_{vk}\text{ 就是把求和号删掉、把 }v\text{ 换成 }k}$$

**第三步，算 (B)。** $\pi(k)$ 里没有 $v$，对这个求和来说是个常数，提到求和号外面：

$$(B)=\pi(k)\sum_v\pi(v)=\pi(k)\cdot 1=\pi(k)$$

用到了 **概率加起来等于 1**。

**第四步：**

$$(A)-(B)=\pi(k)-\pi(k)=0\qquad\blacksquare$$

### 5.1 这个 0 的两种读法

**读法一（概率视角）**：$\sum_v\pi(v)f(v)$ 就是期望，所以式 3 说的是

$$\mathbb E_{v\sim\pi}\Big[\frac{\partial\log\pi(v)}{\partial z_k}\Big]=0$$

「$\nabla\log\pi$ 这个量，在 $\pi$ 自己的分布下平均是 0」。它在 RL 里叫 **score function**，这条性质是 baseline 可以随便加而不引入偏差的原因。

**读法二（更好记）**：用式 2 反着看，

$$\sum_v\pi(v)\frac{\partial\log\pi(v)}{\partial z_k}=\sum_v\frac{\partial\pi(v)}{\partial z_k}=\frac{\partial}{\partial z_k}\underbrace{\sum_v\pi(v)}_{\equiv\,1}=\frac{\partial}{\partial z_k}1=0$$

$$\boxed{\text{概率永远加起来等于 1，所以「所有概率的总变化量」必然是 0 —— 你没法让它们一起变大}}$$

这也正是第 1 节说的「Jacobian 每一列加起来等于 0」。

## 6. 立刻兑现：推出 SFT 的梯度

SFT 的 loss 就是标准答案 token $y^*$ 的负对数概率：

$$L=-\log\pi(y^*)$$

直接套式 1（取 $v=y^*$）：

$$\frac{\partial L}{\partial z_k}=-\big(\delta_{y^*k}-\pi(k)\big)=\pi(k)-\delta_{y^*k}$$

写成向量：

$$\boxed{\nabla_z L=\pi-e_{y^*}}$$

$e_{y^*}$ 是标准答案的 one-hot 向量。这就是那句人人都背过的 **"预测分布减 one-hot"** —— 它不是记住的结论，是式 1 一行代出来的。

**直觉**：正确 token 那一维梯度是 $\pi(y^*)-1<0$，梯度下降会把它的 logit 推高；其他每一维梯度是 $\pi(k)>0$，logit 被压低。压低的力度正比于模型当前给它的概率 —— **模型越自信地答错，被压得越狠**。

## 7. 三个式子放一起

$$\frac{\partial\log\pi(v)}{\partial z_k}=\delta_{vk}-\pi(k)\qquad\text{(式 1，地基)}$$

$$\frac{\partial\pi(v)}{\partial z_k}=\pi(v)\big(\delta_{vk}-\pi(k)\big)\qquad\text{(式 2 = 式 1}\times\pi(v)\text{)}$$

$$\sum_v\pi(v)\big(\delta_{vk}-\pi(k)\big)=0\qquad\text{(式 3，消项全靠它)}$$

三个套路记住就够用了：

1. 看到 $\delta_{vk}$ → 就是 `v == k`
2. 看到 $\sum_v(\cdots)\delta_{vk}$ → 删掉求和号，把 $v$ 换成 $k$
3. 看到 $\sum_v\pi(v)\times(\text{跟 }v\text{ 无关的东西})$ → 概率和为 1，直接提出来

接着看 [reverse KL 就是 policy gradient 的第 7 节](04-reverse-kl-as-pg.md#7-sft-与-opd-的梯度为什么一个要-pg一个不要)，那里就是把这三个式子反复用在 SFT / forward KL / reverse KL 上。

## 自测（口述版）

**1.** ⭐ softmax 的"导数"为什么是一张 $V\times V$ 的表？$v$ 和 $k$ 分别是什么？

> **答：** softmax 输入 $V$ 个 logit、输出 $V$ 个概率，每个输出都受每个输入影响，所以是 Jacobian 矩阵。$k$ 是**被推的输入**，$v$ 是**被看的输出**。$J[v][k]$ = 推高第 $k$ 个 logit 时第 $v$ 个概率的变化率。对角线为正、非对角线为负、每列和为 0（因为概率总和恒为 1）。

**2.** ⭐ $\delta_{vk}$ 是什么？它在推导里是从哪一步冒出来的？

> **答：** 就是 `int(v == k)`，把 if 写进公式，好处是一个式子同时覆盖对角线和非对角线。它来自 $\dfrac{\partial z_v}{\partial z_k}$ —— $z_v$ 只有在 $v=k$ 时才随 $z_k$ 变，否则是常数。**不是人为引入的记号，是求导的自然结果。**

**3.** ⭐⭐ 推导 $\dfrac{\partial\log\pi(v)}{\partial z_k}=\delta_{vk}-\pi(k)$。

> **答：** 先化简 $\log\pi(v)=z_v-\log S$，$S=\sum_u e^{z_u}$（除法变减法，这是关键）。求导拆两块：① $\partial z_v/\partial z_k=\delta_{vk}$；② $\partial\log S/\partial z_k=\frac1S\cdot\frac{\partial S}{\partial z_k}$，而 $S$ 展开后只有 $e^{z_k}$ 一项含 $z_k$，导数为 $e^{z_k}$，所以这块 $=e^{z_k}/S=\pi(k)$，**softmax 自己长了出来**。相减即得。

**4.** ⭐ 由式 1 怎么一步得到 $\dfrac{\partial\pi(v)}{\partial z_k}$？

> **答：** 用 $\dfrac{\partial f}{\partial t}=f\cdot\dfrac{\partial\log f}{\partial t}$（log-derivative trick），取 $f=\pi(v)$，得 $\pi(v)(\delta_{vk}-\pi(k))$。**式 2 就是式 1 乘 $\pi(v)$。** 验证：$v=k$ 时 $\pi(k)(1-\pi(k))>0$，$v\ne k$ 时 $-\pi(v)\pi(k)<0$，符合直觉。

**5.** ⭐⭐⭐ 证明 $\sum_v\pi(v)(\delta_{vk}-\pi(k))=0$，并给出它的直观含义。

> **答：** 拆成 $\sum_v\pi(v)\delta_{vk}-\sum_v\pi(v)\pi(k)$。前者只有 $v=k$ 一项活下来 $=\pi(k)$；后者把与 $v$ 无关的 $\pi(k)$ 提出，$=\pi(k)\sum_v\pi(v)=\pi(k)$。相减为 0。
> **含义**：$\mathbb E_{v\sim\pi}[\nabla\log\pi(v)]=0$（score function 期望为零，baseline 不引入偏差的根据）。更好记的读法：$\sum_v\nabla\pi(v)=\nabla\sum_v\pi(v)=\nabla 1=0$ —— **概率恒和为 1，不可能让所有概率一起变大**。

**6.** ⭐⭐ 用式 1 推出 SFT 的梯度，并解释它的物理含义。

> **答：** $L=-\log\pi(y^*)$，套式 1 得 $\frac{\partial L}{\partial z_k}=\pi(k)-\delta_{y^*k}$，即 $\nabla_z L=\pi-e_{y^*}$（预测分布减 one-hot）。正确 token 那维为 $\pi(y^*)-1<0$，logit 被推高；其余维为 $\pi(k)>0$，logit 被压低，**压低力度正比于模型当前给它的概率 —— 越自信地答错，被压得越狠**。
