# Patch Embedding：从卷积讲起，到一张图变成 N 个 token

> **数学前置**。看不懂"Conv3D 把图片切成 patch"就先看这篇。
> 结论先放这：**ViT 的 patch embedding 根本不是什么复杂卷积，它就是「不重叠切块 + 一个共享的 Linear」。**

## 1. Conv 到底在干什么

一张灰度图 $X\in\mathbb R^{5\times5}$，一个 $3\times3$ 的 kernel：

$$K=\begin{bmatrix}k_{11}&k_{12}&k_{13}\\k_{21}&k_{22}&k_{23}\\k_{31}&k_{32}&k_{33}\end{bmatrix}$$

它在图上滑动，每到一个位置，取当前 $3\times3$ 区域做

$$y=\sum_{i,j}X_{ij}K_{ij}$$

也就是**逐元素乘，然后全加起来**，得到一个数。比如

$$\begin{bmatrix}1&2&3\\4&5&6\\7&8&9\end{bmatrix}\ \text{配}\ \begin{bmatrix}1&0&-1\\1&0&-1\\1&0&-1\end{bmatrix}\ \Rightarrow\ 1-3+4-6+7-9=-6$$

然后 kernel 向右滑，再算下一个数。

$$\boxed{\text{Conv}=\text{对一个局部区域做相同的线性变换}}$$

## 2. 和 Linear 的区别只有两点

Linear 是 $y=Wx$，输入 flatten 成 $x\in\mathbb R^{HWC}$，**每个输出都连着整张图**。

Conv 是 $y_{i,j}=W\cdot x_{\text{局部区域}}$，只看附近一小块，而且这个 $W$ 在所有位置**共享**。

| | Linear | Conv |
|---|---|---|
| 看多大 | 整张图 | 一个局部窗口 |
| 权重 | 每个位置各一套 | 全图共享一套 |

## 3. RGB 图片：kernel 其实是三维的

真实图片是 $X\in\mathbb R^{3\times H\times W}$（R/G/B 三个 channel）。所以当我们说"$3\times3$ 的 kernel"时，它真正的形状是

$$\boxed{\underbrace{3}_{C}\times\underbrace{3}_{H_k}\times\underbrace{3}_{W_k}}$$

**一个 kernel 同时读取 RGB 三个通道。**

## 4. out_channels 是什么：不是一个 kernel，是很多个

```python
Conv2d(in_channels=3, out_channels=64, kernel_size=3)
```

这里**有 64 个 kernel**，每个 $W_k\in\mathbb R^{3\times3\times3}$。64 个 kernel 产生 64 张 feature map：

$$[3,H,W]\ \longrightarrow\ [64,H',W']$$

## 5. ⭐ 一个 Conv kernel 本质就是一个 Linear

这是理解 patch embedding 最关键的一步。

kernel 是 $14\times14$、输入 RGB，那么一个 kernel 有 $3\times14\times14=588$ 个参数。
现在取图中一个 $14\times14$ 的 patch，它也有 $3\times14\times14=588$ 个数字。

两边都 flatten：

$$x_{\text{patch}}\in\mathbb R^{588},\qquad w\in\mathbb R^{588}$$

卷积在这个位置做的事情就是

$$y=w^\top x_{\text{patch}}$$

$$\boxed{\text{这就是 Linear}}$$

如果 `out_channels=768`，就是 768 个不同的 $w_i\in\mathbb R^{588}$，每个给出 $y_i=w_i^\top x$，合起来

$$\boxed{\mathbb R^{3\times14\times14}\ \longrightarrow\ \mathbb R^{768}}$$

**这已经就是 patch embedding 了。**

## 6. 所以经典 ViT 的 patch embedding 极其简单

$224\times224$ 图片、patch $16\times16$ → 切成 $14\times14=196$ 个 patch。每个 patch $16\times16\times3$ flatten 成 768 维，再做 Linear：

$$\boxed{\text{Patchify}\rightarrow\text{Flatten}\rightarrow\text{Linear}}$$

工程上直接写成一行就完事：

```python
Conv2d(in_channels=3, out_channels=d, kernel_size=16, stride=16)
```

**完全等价。**

## 7. 为什么 stride 也要等于 patch size

$kernel=14$、$stride=14$ 时，kernel 第一次看 pixel 0~13，下一次直接跳到 14~27：

```text
┌────────┬────────┬────────┐
│ patch1 │ patch2 │ patch3 │
├────────┼────────┼────────┤
│ patch4 │ patch5 │ patch6 │
└────────┴────────┴────────┘
```

patch **不重叠**。所以

$$\boxed{kernel\_size=stride=patch\_size\ \Longrightarrow\ \text{就是在切 patch}}$$

### ⚠️ 但要说准确：不是"因为 stride=kernel，卷积才是线性的"

卷积**无论 stride 多少**，对每个局部窗口都是 $y=Wx+b$，一直都是线性变换。

`stride = kernel` 真正带来的是 $\boxed{\text{窗口不重叠}}$ ——所以它才**恰好**等价于 ViT 那种"先切不重叠 patch，再逐 patch 做同一个 Linear"。

如果 $K=14,S=7$，仍然可以理解成"extract patch + Linear"，只是这些 patch 会互相重叠：

```text
patch 1: pixel 0~13
patch 2: pixel 7~20
patch 3: pixel 14~27
```

那就不是经典 ViT 的 non-overlapping patch embedding 了。

$$\boxed{K=S=P\ \Rightarrow\ \text{Conv PatchEmbed}\equiv\text{不重叠切 patch}+\text{每个 patch 用同一个 Linear}}$$

## 8. Conv3D 只是多滑一个维度

Qwen2.5-VL 要同时处理图片和视频，视频多一个时间维：$X\in\mathbb R^{C\times T\times H\times W}$。

| | kernel 形状 |
|---|---|
| Conv2D | $C\times H_k\times W_k$ |
| Conv3D | $\boxed{C\times T_k\times H_k\times W_k}$ |

Qwen2.5-VL 用 $T_k=2$、$H_k=W_k=14$：

```python
nn.Conv3d(in_channels=3, out_channels=1280,
          kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False)
```

每个 kernel 看的是一个 **tubelet**：连续 2 帧里的一个 $14\times14$ 区域。

```text
Frame t              Frame t+1
┌───────────┐        ┌───────────┐
│ 14×14 RGB │        │ 14×14 RGB │
└───────────┘        └───────────┘
            ↓ 合起来
        2 × 14 × 14 × 3  =  1176 个数
```

所以本质还是

$$\boxed{\text{Linear}(1176,\ 1280)}$$

$$\boxed{\text{Conv3D PatchEmbed}\equiv\text{Patchify}+\text{Flatten}+\text{Linear}}$$

记法：看到 **Conv2D PatchEmbed** 就读成"每个图片 patch → 一个 token"；看到 **Conv3D PatchEmbed** 就读成"每个 video tubelet → 一个 token"。

## 9. ⭐ channel 和 token 维度别搞混

`Conv3D: 3 → 1280` 得到 1280 个 out channel，但**在一个 patch 位置上，每个 channel 只有 1 个标量**。

单独拿一个 tubelet：

$$[3,2,14,14]\ \xrightarrow{\ \text{Conv3D}\ }\ [1280,1,1,1]$$

1280 个 channel 合起来才组成这个 patch 的 $\boxed{1280\text{ 维 visual token}}$。

如果输入是整段视频 $[3,8,448,448]$，输出是 $[1280,4,32,32]$，这时两个视角完全等价：

| 视角 | 读法 |
|---|---|
| **channel 视角** | 1280 张 $4\times32\times32$ 的时空 feature map，第 $c$ 张是"第 $c$ 种 learned feature 在所有位置的激活" |
| **token 视角** | 固定位置 $(t,h,w)$，把 1280 个 channel 的值取出来 $Y[:,t,h,w]\in\mathbb R^{1280}$，就是那个位置的 token |

$$[1280,4,32,32]\ \longleftrightarrow\ [4\times32\times32,\ 1280]=[4096,1280]$$

ViT 之后一律用 token 视角：$\boxed{N\times1280}$，其中 1280 就是 ViT 的 `hidden_size`。

## 10. 完整走一遍：一张 448×448 图片进 ViT 之前

Qwen2.5-VL-7B：`patch_size=14`、`temporal_patch_size=2`、$d_{vit}=1280$。

**① 原始图片** $I\in\mathbb R^{3\times448\times448}$ —— 此时只有 pixel，还没有 token。

**② resize / normalize**：dynamic resolution，不强行变成固定 $224\times224$，只要求 $H',W'$ 能被 $14\times2=28$ 整除（14 是 patch size，2 是后面的 spatial merge）。$448$ 本来就满足。

**③ 补时间维**：Conv3D 要 $T=2$，但图片只有一帧，所以 processor 把最后一帧重复：

$$[3,448,448]\ \longrightarrow\ \boxed{[3,2,448,448]}$$

概念上就是 $I\rightarrow[I,I]$。这样 image 和 video 可以共用同一套 patch embedding，不需要两份代码。

**④ 切 patch**：kernel = stride = $(2,14,14)$，所以不重叠。

$$T'=\tfrac22=1,\quad H'=\tfrac{448}{14}=32,\quad W'=\tfrac{448}{14}=32$$

patch grid $=1\times32\times32$，总数 $N=\boxed{1024}$。

**⑤ 每个 patch 有多少数**：$[3,2,14,14]\Rightarrow 3\times2\times14\times14=1176$。整张图看成 $[1024,1176]$。

**⑥ 共享 Linear**：$W\in\mathbb R^{1280\times1176}$，1024 个 patch 全用同一个 $W$。

$$[1024,1176]\ \longrightarrow\ \boxed{[1024,1280]}$$

这就是真正送进 ViT 第一个 block 的 hidden states。

```text
[3, 448, 448]
      │ resize / normalize
      ▼
[3, 448, 448]
      │ 复制一帧补 temporal dim
      ▼
[3, 2, 448, 448]
      │ 不重叠切 tubelet, kernel = stride = (2,14,14)
      ▼
grid = [1, 32, 32]  →  1024 个 patch
      │ 每个 patch [3,2,14,14] flatten
      ▼
[1024, 1176]
      │ 共享 Linear 1176 → 1280
      ▼
[1024, 1280]  ──→  ViT Block 0
```

对单张图片，$T=2$ 恒成立，所以直接记：

$$\boxed{N=\frac{H}{14}\cdot\frac{W}{14}}$$

⚠️ **这里的 1024 是 ViT 内部的 token 数，不是最终送给 LLM 的 visual token 数。** 后面还有 $2\times2$ PatchMerger 把它压到 256，见 [Qwen2.5-VL](05-qwen25-vl.md#5-patchmerger既是压缩也是-projector)。

## 面试版

> ViT 的 patch embedding 就是"不重叠切 patch + 一个共享 Linear"。工程上用 `Conv2d(3, d, kernel=P, stride=P)` 实现，因为 stride 等于 kernel 时窗口不重叠，每个输出位置正好对应一个独立 patch，卷积退化成对每个 patch 做同一个 $y=Wx$。Qwen2.5-VL 为了图片和视频共用一套代码用的是 Conv3D，kernel = stride = $(2,14,14)$，一个 tubelet 是 $2\times14\times14\times3=1176$ 个数，经 $\text{Linear}(1176,1280)$ 变成一个 1280 维 token；图片会把最后一帧复制一次来凑 $T=2$。

## 自测

**1.** 一句话说清 conv 和 linear 的区别。

> **答：** Linear 每个输出连着整个输入；Conv 只看一个**局部窗口**，而且这套权重在所有位置**共享**。除此之外二者都是 $y=Wx+b$。

**2.** ⭐ `Conv2d(3, 1280, kernel_size=14, stride=14)` 里有几个 kernel？每个多少参数？整体等价于什么？

> **答：** **1280 个** kernel，每个形状 $3\times14\times14$，即 $588$ 个参数。1280 个合起来就是 $W\in\mathbb R^{1280\times588}$，整体等价于 $\text{Linear}(588,1280)$ 作用在每个不重叠的 $14\times14$ RGB patch 上。

**3.** ⭐⭐ 为什么 `Conv(K=P, S=P)` 等价于 ViT 的 patch embedding？说准确一点。

> **答：** 因为 **stride 等于 kernel 时窗口不重叠**，每个输出位置正好对应输入里一个独立 patch，于是每个 patch 都在做同一个 $y_i=Wx_i+b$，$W\in\mathbb R^{d_{out}\times CK_HK_W}$。
> ⚠️ 但**不是"因为 stride=kernel 卷积才是线性的"** —— 卷积对每个局部窗口一直都是线性变换。stride=kernel 带来的是**不重叠**，所以才恰好等价于"切不重叠 patch + 共享 Linear"。$K=14,S=7$ 也是"extract patch + Linear"，只是 patch 会重叠。

**4.** Qwen2.5-VL 为什么用 Conv3D 而不是 Conv2D？图片怎么办？

> **答：** 为了**图片和视频共用一套 patch embedding**。Conv3D kernel 多一个时间维 $C\times T_k\times H_k\times W_k$，$(2,14,14)$ 表示一个 tubelet = 连续 2 帧里的一个 $14\times14$ 区域。图片没有时间维，processor 会把最后一帧重复补足 $T=2$（概念上 $I\to[I,I]$），这样不需要两套代码。

**5.** ⭐ `out_channels=1280`，每个 channel 内部的维度是多少？

> **答：** 对**一个 patch 位置**，每个 channel 只有 **1 个标量**；1280 个 channel 合起来才是这个 patch 的 1280 维 token（$[3,2,14,14]\to[1280,1,1,1]$）。
> 对**整段视频** $[3,8,448,448]$，输出 $[1280,4,32,32]$，此时第 $c$ 个 channel 是一张 $4\times32\times32$ 的时空 feature map。两个视角等价：$[1280,4,32,32]\leftrightarrow[4096,1280]$。

**6.** ⭐⭐ 一张 $448\times448$ 的图进 Qwen2.5-VL，到 ViT 第一层之前 tensor 怎么变？

> **答：** $[3,448,448]$ → resize/normalize（要求边长能被 $14\times2=28$ 整除）→ 复制一帧 $[3,2,448,448]$ → kernel=stride=$(2,14,14)$ 不重叠切块，grid $=1\times32\times32$ 即 **1024 个 patch** → 每个 patch $3\times2\times14\times14=1176$ 个数，整体 $[1024,1176]$ → 共享 $\text{Linear}(1176,1280)$ → $\boxed{[1024,1280]}$。
> 注意 1024 是 **ViT 内部** token 数，不是给 LLM 的 visual token 数（后面还有 $2\times2$ merge 压到 256）。

**7.** 为什么 preprocessing 要求边长能被 28 整除？

> **答：** $28=14\times2$，其中 **14 是 patch size**（切 ViT patch 用），**2 是 spatial merge size**（后面 PatchMerger 把 $2\times2$ 个 ViT token 合成一个）。两级都要能整除，patch grid 才能正好按 $2\times2$ 分组。
