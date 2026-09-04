# 位置编码：从绝对位置到 RoPE 与外推

> 高频考点：RoPE 怎么让 attention 依赖相对位置，NTK 和 YaRN 各改了什么。
> 看不懂"旋转"和 $R_m^\top R_n=R_{n-m}$ 就先看 [02b RoPE 从零](02b-rope-from-zero.md)（两个 token 手算一遍）。

## 1. 为什么需要位置编码

self-attention 本身是**置换等变**的：把输入 token 打乱，输出也只是跟着打乱，模型看不出顺序。

$$\text{Attn}(PX)=P\,\text{Attn}(X)$$

所以必须显式注入位置信息。

## 2. 几种做法

| 方法 | 怎么做 | 问题 |
|---|---|---|
| 可学习绝对 PE | 每个位置一个可学习向量，加到 embedding 上 | 超过训练长度就没有对应向量，**完全无法外推** |
| 正弦绝对 PE | 用不同频率的 sin/cos | 能算出任意位置，但外推效果一般 |
| **ALiBi** | 不加 PE，直接在 attention score 上按距离减一个线性偏置 $-m\cdot\lvert i-j\rvert$ | 简单、外推好；但是硬编码的衰减，表达能力受限 |
| **RoPE** | 按位置**旋转** $q,k$ | 现在的主流 |
| NoPE | 什么都不加 | causal mask 本身泄露了位置信息，小模型上居然能work |

## 3. RoPE

核心思想：把 $q,k$ 的相邻两维看成复平面上的一个点，按位置 $m$ 旋转 $m\theta$ 角。

对第 $i$ 组（维度 $2i,2i+1$）：

$$\theta_i=\text{base}^{-2i/d},\qquad \text{base}=10000$$

$$\begin{pmatrix}q'_{2i}\\ q'_{2i+1}\end{pmatrix}=\begin{pmatrix}\cos m\theta_i & -\sin m\theta_i\\ \sin m\theta_i & \cos m\theta_i\end{pmatrix}\begin{pmatrix}q_{2i}\\ q_{2i+1}\end{pmatrix}$$

### 关键性质：内积只依赖相对位置

记旋转矩阵为 $R_m$，则

$$\langle R_m q,\ R_n k\rangle = q^\top R_m^\top R_n k = q^\top R_{n-m}\,k$$

$$\boxed{\text{attention score 只和 }n-m\text{ 有关，与绝对位置无关}}$$

这就是 RoPE 最漂亮的地方：**用绝对位置的操作，得到相对位置的效果**。

而且它是**乘在 $q,k$ 上**而不是加在 embedding 上，所以不占 residual stream 的容量。

### 频率的含义

$\theta_i=\text{base}^{-2i/d}$ 随 $i$ 增大而减小：

- **低维（$i$ 小）→ 高频**：转得快，刻画近距离的精细位置差别
- **高维（$i$ 大）→ 低频**：转得慢，刻画远距离的粗糙位置

一个周期对应的距离是 $2\pi/\theta_i$。最低频那一维的周期决定了模型能"分辨"的最大距离尺度。

### Partial RoPE

只对一部分维度做旋转，剩下的维度不带位置信息。好处是保留一些「位置无关」的通道，有些模型（如部分 MLA 实现）会这么做 —— MLA 里 RoPE 必须作用在一个单独的小分支上，因为低秩压缩后的 latent 无法直接施加旋转。

## 4. 长度外推：NTK 与 YaRN

**问题**：训练时最长见过 4K，推理要 32K。位置 $m$ 超出训练范围后，$m\theta_i$ 落到了训练时从没出现过的角度区间，模型直接崩。

### 直接位置插值（PI）

把位置压缩回训练范围：$m\to m\cdot\frac{L_{\text{train}}}{L_{\text{target}}}$。

等价于所有频率一起被拉伸。问题是**高频维度被压得太狠**，近距离的分辨能力被破坏了。

### NTK-aware（调 base）

不动位置，改 **base**：

$$\text{base}:10000\to 10000\cdot s^{d/(d-2)}$$

效果是**低频维度被拉伸得多，高频维度几乎不动**：

$$\boxed{\text{NTK：靠调大 base，让低频多拉伸、高频少变化}}$$

这样保住了近距离分辨率，同时扩展了远距离覆盖。

### YaRN

在 NTK 的基础上做得更细，**按频率分段处理**：

| 频段 | 判据 | 处理 |
|---|---|---|
| 低频（波长 > 上下文长度） | 一个周期都转不完 | **强拉伸**（做插值） |
| 高频（波长很短） | 训练时已经转过很多圈 | **尽量不动**（保分辨率） |
| 中频 | 之间 | **平滑过渡**（斜坡插值） |

另外 YaRN 还加了 **attention scaling**：把 $1/\sqrt{d}$ 改成 $t/\sqrt{d}$，补偿上下文变长后 attention 熵的变化。

$$\boxed{\text{PI：一刀切拉伸}\ \to\ \text{NTK：调 base，按频率自然区分}\ \to\ \text{YaRN：显式分段 + attention scaling}}$$

## 5. 一个趋势：Kimi K3 已经没有 RoPE

最新的一些模型开始尝试去掉 RoPE。原因是当模型足够大、数据足够多时，causal mask 本身携带的位置信息 + 特定的 attention 结构（如线性注意力、滑窗）可能已经足够，而 RoPE 反而成为长度外推的约束。

这块还在演进，面试提一句趋势即可，不要说死。

## 6. 多模态的 M-RoPE

VLM 里图像是二维的，一维位置编码不够用。**M-RoPE / 2D RoPE** 把 RoPE 的维度分成几组，分别编码 $(t, h, w)$：

- 文本 token：三个分量取同一个值（`token1→(1,1,1)`），退化成 1D RoPE
- 图像 token：按它在图里的行列填 $h,w$
- 视频：再加上帧号 $t$

DeepSeek-V4-Flash-Vision 的 ViT 内部用的就是 2D RoPE（见 [visual token 压缩](../07-vlm/03-visual-token-compression.md#6-案例deepseek-v4-flash-vision-exp)）。

⚠️ 别混层级：**Qwen2.5-VL 的 ViT 内部是 2D RoPE，LLM 内部才是 MRoPE**，两个不同模块（见 [05 Qwen2.5-VL](../07-vlm/05-qwen25-vl.md#4-vit-内部是-2d-rope不是-mrope)）。逐维推导和"为什么 flatten 不行"见 [02b](02b-rope-from-zero.md#8-图片为什么不能直接用-1d-rope)。

## 自测（口述版）

**1.** 为什么 self-attention 必须加位置编码？写出置换等变的式子。

> **答：** $\text{Attn}(PX)=P\,\text{Attn}(X)$：把输入 token 打乱，输出也只是跟着打乱，模型看不出顺序。所以必须显式注入位置信息。

**2.** 推导 RoPE 的核心性质：为什么 $\langle R_mq, R_nk\rangle$ 只依赖 $n-m$？

> **答：** 旋转矩阵满足 $R_m^\top R_n=R_{n-m}$，于是
> $$\langle R_mq,R_nk\rangle=q^\top R_m^\top R_n k=q^\top R_{n-m}k$$
> 只和 $n-m$ 有关，与绝对位置无关。这就是「**用绝对位置的操作得到相对位置的效果**」。而且它乘在 $q,k$ 上而不是加在 embedding 上，不占 residual stream 的容量。

**3.** RoPE 的 $\theta_i$ 随维度怎么变？低维和高维分别刻画什么？

> **答：** $\theta_i=\text{base}^{-2i/d}$，随 $i$ 增大而**减小**。
> **低维（$i$ 小）是高频**，转得快，刻画近距离的精细位置差别；**高维（$i$ 大）是低频**，转得慢，刻画远距离的粗糙位置。一个周期对应的距离是 $2\pi/\theta_i$。

**4.** ALiBi 和 RoPE 的区别？各自的优缺点。

> **答：** ALiBi 不加位置编码，直接在 attention score 上按距离减一个线性偏置 $-m\lvert i-j\rvert$：简单、外推好，但衰减是**硬编码**的，表达能力受限。
> RoPE 是旋转 $q,k$：表达能力更强、是现在的主流，但原生外推能力差，需要 NTK/YaRN 这类手段。

**5.** 直接位置插值的问题是什么？NTK 怎么解决？

> **答：** PI 把位置压缩回训练范围（$m\to m\cdot L_{\text{train}}/L_{\text{target}}$），等价于**所有频率一起被拉伸**，**高频维度被压得太狠**，近距离的分辨能力被破坏。
> NTK 不动位置，改 **base**（$10000\to10000\cdot s^{d/(d-2)}$），效果是低频维度拉伸得多、高频几乎不动，保住近距离分辨率的同时扩展远距离覆盖。

**6.** YaRN 相比 NTK 多做了哪两件事？三个频段分别怎么处理？

> **答：** ① **按频率显式分段**：低频（波长 > 上下文长度，一个周期都转不完）**强拉伸**做插值；高频（训练时已转过很多圈）**尽量不动**保分辨率；中频**平滑过渡**（斜坡插值）。
> ② 加 **attention scaling**：把 $1/\sqrt d$ 改成 $t/\sqrt d$，补偿上下文变长后 attention 熵的变化。

**7.** M-RoPE 怎么处理文本 / 图像 / 视频的位置？

> **答：** 把 RoPE 的维度分成几组，分别编码 $(t,h,w)$。文本 token 三个分量取同一个值，退化成 1D RoPE；图像 token 按它在图里的行列填 $h,w$；视频再加上帧号 $t$。

