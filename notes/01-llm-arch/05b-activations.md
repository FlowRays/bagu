# 激活函数：从正态分布、CDF 到 GLU 家族

> **数学前置**。[05 FFN 与 MoE](05-ffn-moe.md) 直接写 $\text{GELU}(x)=x\Phi(x)$，如果 $\Phi$ 是什么已经忘了，先看这篇。
> 核心主张：**这些公式不用当数学考试去背**，记住四件事就够 —— ①图像大概什么样 ②输入很大/很小时会怎样 ③直觉作用 ④代码里能认出来。

$$\boxed{\text{ReLU}\rightarrow\text{GELU / SiLU}\rightarrow\text{GLU}\rightarrow\text{SwiGLU}}$$

前两个是普通激活函数，后两个更准确地说是**带门控的 FFN 结构**。

## 1. 先补标准正态分布

$X\sim\mathcal N(0,1)$，就是最经典的钟形曲线，均值 0、方差 1：

```text
           /\
         /    \
       /        \
-----/------------\-----
    -3 -2 -1  0  1  2  3
```

密度函数 $f(x)=\frac1{\sqrt{2\pi}}e^{-x^2/2}$ —— **完全不用硬背**。真正要记的只有三点：

- 0 附近概率最大
- 越远离 0 概率越小
- 左右完全对称

## 2. CDF $\Phi(x)$ 是什么

$$\Phi(x)=P(X\le x),\qquad X\sim\mathcal N(0,1)$$

从负无穷一路累加到 $x$ 的概率。所以值一定在 $(0,1)$，图像不是钟形，而是 **S 形**：

```text
 1 |                    ______
   |                ___/
   |             __/
0.5|----------__/
   |        __/
   |     __/
 0 |____/
    ------------------------> x
     -3 -2 -1  0  1  2  3
```

$$x\to-\infty:\Phi\to0,\qquad \Phi(0)=0.5,\qquad x\to+\infty:\Phi\to1$$

$$\boxed{\Phi(x)\approx\text{一个平滑版的开关}}$$

## 3. sigmoid：长得几乎一样

$$\sigma(x)=\frac1{1+e^{-x}}$$

也是 S 形，也是把实数压到 $(0,1)$：

$$\sigma(0)=0.5,\qquad x\gg0\Rightarrow\sigma\approx1,\qquad x\ll0\Rightarrow\sigma\approx0$$

$$\boxed{\Phi(x)\approx\sigma(x)}$$

严格说两者不同，但**直觉上都可以看成"平滑门"**。常被解释成概率或门控强度。

## 4. ⭐ ReLU / GELU / SiLU：一个统一模板

### ReLU：硬门

$$\text{ReLU}(x)=\max(0,x)$$

```text
y ^        /
  |      /
  |    /
  |  /
  |/____________> x
  0
```

> **负数砍掉，正数原样通过。**

简单便宜，但负半轴梯度直接为 0，可能出现 dead neuron。现在主流 LLM 一般不直接用。

### GELU 和 SiLU：软门

$$\text{GELU}(x)=x\,\Phi(x),\qquad \text{SiLU}(x)=x\,\sigma(x)$$

**结构一模一样**，只是门函数不同：

$$\boxed{\text{输出}=\text{输入}\times\text{一个 }0\text{ 到 }1\text{ 的平滑门}}$$

| | 门是什么 | 常见于 |
|---|---|---|
| GELU | $\Phi(x)$，标准正态 CDF | BERT、GPT-2 那一代 |
| SiLU（= Swish，$\beta=1$） | $\sigma(x)$，sigmoid | 现代 LLM，SwiGLU 的一半 |

拆开看就很直观：

- $x$ 是大正数 → 门 $\approx1$ → 输出 $\approx x$
- $x$ 是大负数 → 门 $\approx0$ → 输出 $\approx0$
- $x$ 在 0 附近 → **平滑地放一点过去，不是直接砍掉**

```text
y ^          /
  |        /
  |      /
  |    /
  |  _/
  |_/
  |________________> x
```

和 ReLU 的区别：ReLU 左边严格贴着 0，GELU/SiLU 左边会**稍微往下鼓一点**、过渡更平滑。

$$\boxed{\text{GELU / SiLU 都可以理解成"更顺滑的 ReLU"}}$$

> GELU 看起来更难记，只是因为写着一个陌生的 $\Phi$。换个角度：**SiLU $=x\cdot\sigma(x)$、GELU $=x\cdot\Phi(x)$，结构完全相同**，就不难了。

## 5. GLU：从"激活函数"升级成"门控结构"

普通 Transformer FFN 只有一条路：

$$\text{FFN}(x)=W_2\,\phi(W_1x),\qquad 4096\to16384\to4096$$

GLU（**Gated Linear Unit**）的想法是：**同时算两条分支，一条提供内容，另一条决定门开多大。**

$$\text{GLU}(x)=(W_1x)\odot\sigma(W_2x)$$

$\odot$ 是逐元素乘。记 $a=W_1x$（内容）、$g=\sigma(W_2x)\in(0,1)$（门）：

$$\boxed{\text{output}=\text{content}\times\text{gate}}$$

具体感受一下：

$$a=[2,\ -1,\ 4],\quad g=[0.9,\ 0.1,\ 0.5]\ \Longrightarrow\ a\odot g=[1.8,\ -0.1,\ 2]$$

网络因此能学到：**哪些 feature 放过去，哪些压制。**

$$\text{普通 FFN}:\ \text{feature extraction}\qquad\text{GLU}:\ \text{feature extraction}+\text{feature selection}$$

## 6. GLU 是一个家族：换门函数就换名字

把 $(W_1x)\odot\phi(W_2x)$ 里的 $\phi$ 换掉：

| 名称 | 门函数 |
|---|---|
| GLU | $\sigma$ |
| ReGLU | ReLU |
| GEGLU | GELU |
| **SwiGLU** | **SiLU** |

$$\boxed{\text{SwiGLU}=\text{GLU}+\text{SiLU}}$$

现代 LLM 的实现：

$$\boxed{\text{FFN}(x)=W_{down}\big[\text{SiLU}(W_{gate}x)\odot(W_{up}x)\big]}$$

**三个 Linear**（$W_{gate},W_{up},W_{down}$），传统 FFN 只有两个。代码里看到

```python
down_proj(silu(gate_proj(x)) * up_proj(x))
```

就要立刻认出是 **SwiGLU**。⚠️ 注意**只有 gate 分支过 SiLU，up 分支不过**。

## 7. 为什么要两条独立的 projection

容易糊涂的一点：为什么不能直接 $\text{SiLU}(a)\odot a$（只用一个 $W$）？

理论上可以，但**表达能力弱**。用独立的 $W_{up}$ 和 $W_{gate}$ 意味着：

> **一组特征产生内容，另一组完全独立的特征决定门控。**

比如"检测到代码语境"这个特征（来自 gate 分支）可以去控制"打开某些 coding feature"（来自 up 分支）。这种 **multiplicative interaction** 本身就增加表达能力。

### SwiGLU 为什么效果好（面试三条就够）

1. **平滑非线性**：SiLU 比 ReLU 平滑，负半轴仍保留信息
2. **动态 gating**：能根据当前输入决定哪些 feature 往下传
3. **更强表达能力**：两个独立 projection + 乘性交互

参数量的账（为什么 $d_{ff}$ 是 $\frac83d$ 而不是 $4d$）见 [05](05-ffn-moe.md#为什么-d_ff-变成-83-d)。

## 8. 记忆清单

**不要从公式背起，从"作用模板"记：**

| 模板 | 形式 | 口诀 |
|---|---|---|
| 硬门 | $\max(0,x)$ | 负数砍掉，正数通过 |
| 软门 | $x\times(\text{0 到 1 的函数})$ | 输入乘上一个平滑门 |
| 门控结构 | $\text{content}\odot\text{gate}$ | 一条分支给内容，一条决定门，逐元素相乘 |

**六句最小必背：**

1. **标准正态分布**：以 0 为中心、左右对称的钟形曲线
2. **CDF $\Phi(x)$**：累计概率，S 形，值在 0 到 1
3. **sigmoid**：也是 S 形，把实数压到 0 到 1
4. **ReLU** $=\max(0,x)$：硬截断
5. **GELU** $=x\Phi(x)$：平滑版 ReLU
6. **SiLU** $=x\sigma(x)$：也是平滑激活，现代 LLM 常见

**图像只记趋势，不用记精确画法：**

- ReLU：左边严格 0，右边直线
- GELU / SiLU：左边接近 0 但不突然切断，0 附近平滑过渡，右边接近 $y=x$
- sigmoid / $\Phi$：S 形，左 0、中 0.5、右 1

## 面试版

**什么是 GELU？**

> 公式是 $x\Phi(x)$，$\Phi$ 是标准正态分布的 CDF。直觉上是给输入乘了一个平滑门，所以可以看成比 ReLU 更平滑的激活：大的正值基本保留，负值被平滑抑制而不是像 ReLU 那样硬截断。

**什么是 SiLU？**

> $x\sigma(x)$。和 GELU 结构一样，都是输入乘一个 0 到 1 的门，只是门换成 sigmoid。它平滑，所以现代模型常和 GLU 结合成 SwiGLU。

**什么是 SwiGLU？**

> GLU 家族的一员：$W_{down}[\text{SiLU}(W_{gate}x)\odot W_{up}x]$，两条独立线性分支、一条过 SiLU 当门、逐元素相乘。相比普通 FFN 多了动态 gating 和乘性交互；因为有三个矩阵，$d_{ff}$ 要缩到 $\frac83d$ 才和普通 FFN 参数量持平。

## 自测

**1.** 标准正态分布和 $\Phi(x)$ 分别是什么？图像什么样？

> **答：** $\mathcal N(0,1)$ 是均值 0、方差 1 的**钟形曲线**，0 附近概率最大、左右对称。$\Phi(x)=P(X\le x)$ 是它的 **CDF**（累计分布函数），图像是 **S 形**，$\Phi(-\infty)=0$、$\Phi(0)=0.5$、$\Phi(+\infty)=1$。密度公式不用背。

**2.** ⭐ GELU 和 SiLU 的关系是什么？为什么说 GELU 不难记？

> **答：** 结构**完全一样**：$\text{GELU}=x\cdot\Phi(x)$、$\text{SiLU}=x\cdot\sigma(x)$，都是 $\boxed{\text{输入}\times\text{一个 0 到 1 的平滑门}}$，只是门函数一个用正态 CDF、一个用 sigmoid。而且 $\Phi(x)\approx\sigma(x)$，直觉上可以互相替代。

**3.** ⭐ ReLU / GELU / SiLU 的图像趋势各是什么？

> **答：** ReLU 左边**严格 0**、右边直线；GELU/SiLU 左边接近 0 但**不突然切断**（会稍微往下鼓一点）、0 附近平滑过渡、右边接近 $y=x$。所以 GELU/SiLU 都是"更顺滑的 ReLU"。ReLU 的问题是负半轴梯度为 0，可能 dead neuron。

**4.** ⭐⭐ GLU 的核心思想是什么？写出公式。

> **答：** 不再只有一条 FFN 分支，而是**同时算两条：一条提供内容，另一条控制门开多大**。
> $\text{GLU}(x)=(W_1x)\odot\sigma(W_2x)$，即 $\boxed{\text{output}=\text{content}\times\text{gate}}$。
> 相比普通 FFN 只做 feature extraction，GLU 多了 **feature selection**。

**5.** GLU 家族有哪些？彼此差在哪？

> **答：** 只差**门函数**：GLU 用 $\sigma$、ReGLU 用 ReLU、GEGLU 用 GELU、**SwiGLU 用 SiLU**。所以 $\boxed{\text{SwiGLU}=\text{GLU}+\text{SiLU}}$。

**6.** ⭐⭐ 为什么 SwiGLU 要两条独立的 projection？只用一个 $W$ 行不行？

> **答：** 理论上 $\text{SiLU}(a)\odot a$ 也能算，但**表达能力弱**。独立的 $W_{gate}$ 和 $W_{up}$ 意味着"**一组特征产生内容，另一组完全独立的特征决定门控**"——比如 gate 分支检测到"代码语境"，去打开 up 分支里的某些 coding feature。这种乘性交互本身就增加表达能力。

**7.** 看到 `down_proj(silu(gate_proj(x)) * up_proj(x))` 是什么？哪个分支过激活？

> **答：** **SwiGLU FFN**。$W_{down}[\text{SiLU}(W_{gate}x)\odot(W_{up}x)]$，**只有 gate 分支过 SiLU**，up 分支不过。三个 Linear（传统 FFN 只有两个），所以 $d_{ff}$ 要缩到 $\frac83d$。
