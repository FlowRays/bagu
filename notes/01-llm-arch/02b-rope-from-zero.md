# RoPE 从零：旋转、相对位置、2D RoPE 与 MRoPE

> **数学前置**。[02 位置编码](02-position-encoding.md) 直接从 $R_m^\top R_n=R_{n-m}$ 开始，如果那一步看不懂，先看这篇。
> 这篇只做一件事：**用两个 token、两个二维向量、具体数字，手算一遍 RoPE**，然后再往上加维度。

## 1. Transformer 本身不知道顺序

```text
我 爱 猫
```

embedding 之后是 $x_0,x_1,x_2$。self-attention 只看到这三个向量，**并不知道谁在前谁在后**。所以必须补位置信息。

最朴素的做法是加一个位置向量：$x_i+PE(i)$。

## 2. RoPE 的想法不一样

RoPE = **Rotary Position Embedding**。它不是 $x_i+PE(i)$，而是：

> 根据 token 的位置，把 Q 和 K 的一部分维度**旋转一个角度**。

$$q_i\rightarrow R(i)q_i,\qquad k_j\rightarrow R(j)k_j$$

$$\boxed{\text{RoPE 不改原始 embedding，它在 attention 之前改 Q 和 K}}$$

## 3. 什么叫"旋转一个向量"

$q=\begin{bmatrix}1\\0\end{bmatrix}$ 就是平面上朝右的一根箭头。旋转 $90^\circ$ 变成朝上：$\begin{bmatrix}0\\1\end{bmatrix}$。

数学上旋转 $\theta$ 就是乘一个矩阵：

$$R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix},\qquad q'=R(\theta)q$$

**RoPE 做的事情本质上就是这个。**

## 4. token 的位置决定转多少

极度简化，假设每前进一个位置转 $30^\circ$，即 $\theta_p=p\times30^\circ$：

```text
token 0 → 转  0°
token 1 → 转 30°
token 2 → 转 60°
token 3 → 转 90°
```

位置就这样被"编码进" Q/K 里了。

## 5. ⭐ 手算一遍

两个 token：**A 在位置 $i=1$，B 在位置 $j=3$**。假设 Q/K 都只有 2 维，且初始方向完全一样：

$$q=\begin{bmatrix}1\\0\end{bmatrix},\qquad k=\begin{bmatrix}1\\0\end{bmatrix}$$

### 没有 RoPE 时

$$q^\top k=[1,0]\begin{bmatrix}1\\0\end{bmatrix}=1$$

结果里**完全没出现 $i=1$ 和 $j=3$** —— 光看这个点积，你看不出两个 token 相距 2 个位置。

### 加上 RoPE

A 在位置 1，转 $30^\circ$：

$$q'=\begin{bmatrix}\cos30^\circ\\\sin30^\circ\end{bmatrix}=\begin{bmatrix}\frac{\sqrt3}{2}\\\frac12\end{bmatrix}\approx\begin{bmatrix}0.866\\0.5\end{bmatrix}$$

B 在位置 3，转 $90^\circ$：

$$k'=\begin{bmatrix}\cos90^\circ\\\sin90^\circ\end{bmatrix}=\begin{bmatrix}0\\1\end{bmatrix}$$

现在算 attention：

$$q'^\top k'=[0.866,\ 0.5]\begin{bmatrix}0\\1\end{bmatrix}=\boxed{0.5}$$

**为什么正好是 0.5？** 因为两个向量的夹角是 $90^\circ-30^\circ=60^\circ$，而 $\cos60^\circ=0.5$。也就是

$$q'^\top k'=\cos(\theta_j-\theta_i)=\cos\big((j-i)\omega\big)$$

这里 $j-i=3-1=2$。

### 换一对位置验证

$i=5,\ j=7$：$\theta_i=150^\circ$、$\theta_j=210^\circ$，角度差仍然是 $60^\circ$。

$$(1,3)\ \text{和}\ (5,7)\ \text{绝对位置完全不同，但}\ 3-1=7-5=2$$

**RoPE 让它们拥有相同的"相对距离模式"。**

## 6. 这就是 RoPE 的全部核心

给 A 的 Q 转 $i\omega$，给 B 的 K 转 $j\omega$，点积真正关心的是

$$j\omega-i\omega=(j-i)\omega$$

$$\boxed{\text{attention 自动看到相对位置 }j-i\text{，而不是绝对位置 }i,j}$$

写成矩阵形式就是 [02](02-position-encoding.md#关键性质内积只依赖相对位置) 里那个式子：旋转矩阵满足 $R(i)^\top R(j)=R(j-i)$，所以

$$(R(i)q_i)^\top(R(j)k_j)=q_i^\top R(j-i)k_j$$

这正是语言里最有用的东西 —— 模型通常更关心"**某个词在我前面几格**"，而不是"它是整句话的第 382 个 token"。

## 7. 真实模型：两两分组，每组转速不同

真实的 $q\in\mathbb R^{128}$。RoPE 把维度**两两配对**，每对看成一个二维平面各自旋转：

```text
(q0,q1)     → 一个平面
(q2,q3)     → 一个平面
...
(q126,q127) → 一个平面
```

$d_h=128$ 就是 64 个二维平面。关键是：**每一对用的旋转速度不同**。

$$\theta_{p,m}=p\cdot\omega_m$$

$p$ 是 token 位置，$m$ 是第几组，$\omega_m$ 是这组的频率。

### 为什么要不同频率：钟表类比

```text
秒针：转得快    →  只看它，转一圈就重复了，分不清 1 分钟和 2 分钟
分针：中等
时针：转得慢    →  三根一起看，就能表达很大的时间范围
```

RoPE 同理：**高频维度对短距离敏感，低频维度表达长距离**。大量不同频率组合起来才能编码丰富的位置关系。

### 真实公式

$$\omega_m=\frac1{\theta^{2m/d}},\qquad \theta=10000\ \text{（经典实现）}$$

所以 $\omega_0>\omega_1>\omega_2>\cdots$ 越来越小。第 $p$ 个 token 的第 $m$ 组转角 $\phi_{p,m}=p\,\omega_m$，然后

$$q'_{2m}=q_{2m}\cos\phi-q_{2m+1}\sin\phi$$
$$q'_{2m+1}=q_{2m}\sin\phi+q_{2m+1}\cos\phi$$

**就是刚才那个二维旋转复制了 64 遍。** K 完全一样。

## 8. 图片为什么不能直接用 1D RoPE

一张图切成 $2\times3$ 个 patch：

```text
A B C
D E F
```

强行 flatten：

```text
A B C D E F
0 1 2 3 4 5
```

问题来了 —— 按序列看 $C\to D$ 是 $3-2=1$，**好像是邻居**。但实际上：

```text
      C          C 在右上
    ↙
  D              D 在左下
```

它们根本不是水平相邻。**1D 位置表达不了这个。**

所以图片 patch 的位置应该是二维的：

$$A=(0,0),\quad B=(0,1),\quad C=(0,2),\quad D=(1,0)$$

## 9. 2D RoPE：把 channel 分成两半

极度简化地看，把 $q$ 切成两段，一段用 $h$ 转、一段用 $w$ 转：

$$q=[q^{(h)},\ q^{(w)}]\quad\Longrightarrow\quad q^{(h)\prime}=R(h)q^{(h)},\quad q^{(w)\prime}=R(w)q^{(w)}$$

于是 attention 能同时感受到 $\Delta h$ 和 $\Delta w$。用 $2\times2$ 的图验证：

| | 关系 | $\Delta h$ | $\Delta w$ |
|---|---|---:|---:|
| A→B | 同一行，右边一格 | 0 | 1 |
| A→C | 同一列，下面一格 | 1 | 0 |

两种空间关系被清清楚楚地分开了。1D RoPE 只有一个 $\Delta p$，做不到。

## 10. MRoPE：再加一维时间

视频比图片多一个时间轴，所以 patch 的位置是 $(t,h,w)$：

```text
frame 0        frame 1
A B            E F
C D            G H
```

$$A=(0,0,0),\quad B=(0,0,1),\quad C=(0,1,0),\quad E=(1,0,0)$$

| | 含义 | $(\Delta t,\Delta h,\Delta w)$ |
|---|---|---|
| A→B | 同一帧，右边一个 patch | $(0,0,1)$ |
| A→C | 同一帧，下面一个 patch | $(0,1,0)$ |
| A→E | **下一帧，同一个空间位置** | $(1,0,0)$ |

做法和 2D RoPE 一样，把 head_dim 分给 $t/h/w$ 三段，分别按各自的坐标旋转：

$$q=[q_t,q_h,q_w]\ \Longrightarrow\ q'=[R(t)q_t,\ R(h)q_h,\ R(w)q_w]$$

K 同理。于是 $q_i'^\top k_j'$ 里同时包含了 $\Delta t,\Delta h,\Delta w$。

$$\boxed{\text{MRoPE}=\text{Temporal}+\text{Height}+\text{Width RoPE}}$$

### ⭐ 文本 token 怎么办

这是 MRoPE 很巧的一点：**令 $t=h=w=p$**。

```text
token0 → (0,0,0)
token1 → (1,1,1)
token2 → (2,2,2)
```

三个分量取同一个值，行为就退化成普通的 1D RoPE。因此**文本、图片、视频可以塞进同一个 Transformer**。

## 11. 一张脑图

```text
Transformer attention 本身不知道位置
        ↓
RoPE：位置 p → Q/K 旋转 pω
        ↓
q_i' · k_j' 里自然出现 (j−i)ω
        ↓
相对位置
```

$$\boxed{\text{RoPE}:p}\quad\longrightarrow\quad\boxed{\text{2D RoPE}:(h,w)}\quad\longrightarrow\quad\boxed{\text{MRoPE}:(t,h,w)}$$

```text
Text   : position
Image  : height + width
Video  : time + height + width
```

最省事的记法 —— **MRoPE 就是把一个旋钮变成三个旋钮**：

```text
普通 RoPE                MRoPE
position = 17            time   = 3  ─→ 一部分维度旋转
       ↓                 height = 5  ─→ 一部分维度旋转
     旋转                 width  = 8  ─→ 一部分维度旋转
```

所以 **MRoPE 不是一种全新的位置编码**，它还是 RoPE 的旋转机制，只是把"一个位置坐标"扩展成了"多个位置坐标"。

⚠️ 注意区分层级：[Qwen2.5-VL 的 ViT 内部用 2D RoPE，LLM 内部才是 MRoPE](../07-vlm/05-qwen25-vl.md#4-vit-内部是-2d-rope不是-mrope)，是两个不同模块。

## 面试版

> RoPE 按 token 的位置把 Q/K 的每一对相邻维度在二维平面上旋转 $p\omega_m$。因为旋转矩阵满足 $R(i)^\top R(j)=R(j-i)$，两个向量做点积时绝对位置抵消、只剩 $j-i$，所以用绝对位置的操作得到了相对位置的效果。不同维度对用不同频率（$\omega_m=\theta^{-2m/d}$，$\theta=10000$），像秒针分针时针一样，高频管短距离、低频管长距离。图片不能直接 flatten 成 1D，因为一行末尾和下一行开头在序列上相邻、空间上并不相邻，所以 2D RoPE 把 channel 分成两段分别按 $(h,w)$ 旋转；视频再加一维时间就是 MRoPE 的 $(t,h,w)$。文本 token 令 $t=h=w=p$ 即可退化成 1D RoPE，于是三种模态能共用同一个 Transformer。

## 自测

**1.** RoPE 和"加一个位置向量"有什么本质不同？改的是谁？

> **答：** 朴素做法是 $x_i+PE(i)$，加在 embedding 上。RoPE **不改原始 embedding**，而是在 attention 之前**按位置旋转 Q 和 K**：$q_i\to R(i)q_i$、$k_j\to R(j)k_j$。好处之一是不占 residual stream 的容量。

**2.** ⭐⭐ 手算：$q=k=[1,0]^\top$，每前进一个位置转 $30^\circ$，A 在位置 1、B 在位置 3，算 $q'^\top k'$。

> **答：** $q'=[\cos30^\circ,\sin30^\circ]^\top\approx[0.866,0.5]^\top$，$k'=[\cos90^\circ,\sin90^\circ]^\top=[0,1]^\top$。
> $q'^\top k'=0.5$。因为夹角是 $90^\circ-30^\circ=60^\circ$，$\cos60^\circ=0.5$，即 $\cos((j-i)\omega)$，$j-i=2$。
> 没有 RoPE 时 $q^\top k=1$，**里面完全没有位置信息**。

**3.** ⭐ 为什么 $(1,3)$ 和 $(5,7)$ 的 attention 行为类似？

> **答：** $\theta_i=150^\circ$、$\theta_j=210^\circ$，角度差还是 $60^\circ$。绝对位置完全不同，但 $3-1=7-5=2$，**RoPE 只让点积依赖差值**。这正符合语言的需求 —— 关心"在我前面几格"，不关心"是第 382 个 token"。

**4.** ⭐ RoPE 怎么处理 128 维？为什么要不同频率？

> **答：** 维度**两两配对**成 64 个二维平面，各自旋转；每对的转速不同，$\theta_{p,m}=p\,\omega_m$，$\omega_m=\theta^{-2m/d}$（$\theta=10000$），所以 $\omega$ 越往后越小。
> 不同频率的理由用**钟表**记：只有秒针的话转一圈就重复了，分不清 1 分钟和 2 分钟；秒针+分针+时针才能覆盖大范围。**高频维度对短距离敏感，低频维度表达长距离。**

**5.** ⭐⭐ 图片为什么不能直接 flatten 用 1D RoPE？

> **答：** $2\times3$ 的图 flatten 成 `A B C D E F` 后，$C\to D$ 的序列距离是 1，**看起来像邻居**；但 C 在右上、D 在左下，空间上根本不相邻。1D 位置只有一个 $\Delta p$，表达不了这个。所以要给每个 patch 二维坐标 $(h,w)$。

**6.** 2D RoPE 怎么做？用 $2\times2$ 的图说明它能区分什么。

> **答：** 把 Q/K 的 channel 分成两段，一段按 $h$ 旋转、一段按 $w$ 旋转。于是 A→B 是 $(\Delta h,\Delta w)=(0,1)$「同一行右边一格」，A→C 是 $(1,0)$「同一列下面一格」，两种空间关系被清楚区分。

**7.** ⭐⭐ MRoPE 是什么？文本 token 的 position id 怎么填？

> **答：** 把位置从 $p$ 扩展成 $(t,h,w)$，head_dim 分成三段分别按 temporal / height / width 旋转，于是 $q_i'^\top k_j'$ 里同时含 $\Delta t,\Delta h,\Delta w$。
> **文本 token 令 $t=h=w=p$**（`token1→(1,1,1)`），行为退化成 1D RoPE，因此文本/图片/视频能共用同一个 Transformer。
> $\boxed{\text{MRoPE 不是新的位置编码，就是把一个旋钮变成三个旋钮}}$。

**8.** 视频里 A→E（下一帧的同一个空间位置）的位置差是多少？

> **答：** $(\Delta t,\Delta h,\Delta w)=(1,0,0)$ —— 时间前进一帧，空间不动。对比 A→B 是 $(0,0,1)$、A→C 是 $(0,1,0)$。
