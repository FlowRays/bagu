# Diffusion

> 世界模型和 diffusion policy 的共同地基。目标是把**前向加噪 → 反向去噪 → 训练目标为什么塌成一个 MSE** 这条链推顺。

## 1. 前向过程：加噪有闭式解

定义一条马尔可夫链，逐步把数据加噪成高斯：

$$q(x_t\mid x_{t-1})=\mathcal N\big(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big)$$

记 $\alpha_t=1-\beta_t$、$\bar\alpha_t=\prod_{s\le t}\alpha_s$，则**可以一步跳到任意 $t$**：

$$\boxed{q(x_t\mid x_0)=\mathcal N\big(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\big)\ \Longleftrightarrow\ x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon}$$

**这个闭式解是训练能高效进行的关键**：不需要真的一步步加噪，随机采一个 $t$ 直接算出 $x_t$ 即可。

$\bar\alpha_t$ 从 1 单调降到约 0：$t=0$ 时是原图，$t=T$ 时是纯噪声。

## 2. 反向过程：学去噪

真实的反向 $q(x_{t-1}\mid x_t)$ 不可解，但当 $\beta_t$ 足够小时它近似高斯，于是用网络参数化：

$$p_\theta(x_{t-1}\mid x_t)=\mathcal N\big(x_{t-1};\ \mu_\theta(x_t,t),\ \Sigma_t\big)$$

推导（对 ELBO 做变分）后可以证明最优均值有闭式形式，且**只依赖于"$x_t$ 里混了哪个噪声"**。所以与其预测 $\mu$，不如直接预测噪声 $\epsilon$：

$$\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big)$$

## 3. 训练目标塌成一个 MSE

完整的变分下界是一堆 KL 项，但代入上面的参数化并**丢掉与 $\theta$ 无关的加权系数**后，DDPM 的简化目标就是：

$$\boxed{\mathcal L_{\text{simple}}=\mathbb E_{x_0,\ \epsilon\sim\mathcal N(0,I),\ t\sim\mathcal U[1,T]}\Big[\big\|\epsilon-\epsilon_\theta\big(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\ t\big)\big\|^2\Big]$$

训练循环只有四行：

```text
1) 取一个 x0
2) 随机采 t ~ U[1,T]，采 ε ~ N(0,I)
3) 合成 x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε
4) 最小化 ‖ε − ε_θ(x_t, t)‖²
```

$$\boxed{\text{训练目标就是一个 MSE，这是 diffusion 比 GAN 稳定得多的根本原因}}$$

**去掉的加权系数是什么**：完整 ELBO 里每个 $t$ 有一个权重，简化目标相当于给所有 $t$ 同等权重。实践发现这样效果更好（等价于更重视中等噪声水平），但也意味着它**不再是严格的似然下界**。

## 4. 和 score matching 的等价

对高斯有 $\nabla_{x_t}\log q(x_t\mid x_0)=-\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}=-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$，所以

$$\boxed{s_\theta(x_t,t)\ :=\ \nabla_{x_t}\log p(x_t)\ =\ -\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}}$$

**预测噪声 = 预测 score（对数密度的梯度），只差一个已知的缩放。** 这个等价把 DDPM 和 score-based SDE 两套理论统一了起来，也是后来能用各种 ODE/SDE 求解器加速采样的基础。

## 5. DDIM：把采样变成确定性 ODE

DDPM 采样要走 $T=1000$ 步，太慢。DDIM 观察到：**训练目标只约束了边缘分布 $q(x_t|x_0)$，没有约束联合的马尔可夫结构**，所以可以构造一族非马尔可夫的反向过程，共享同一个训练好的 $\epsilon_\theta$。

$$x_{t-1}=\sqrt{\bar\alpha_{t-1}}\underbrace{\Big(\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta}{\sqrt{\bar\alpha_t}}\Big)}_{\text{预测的 }\hat x_0}+\underbrace{\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta}_{\text{指向 }x_t\text{ 的方向}}+\sigma_t\epsilon$$

- $\sigma_t=0$ → **完全确定性**，同一个初始噪声总是给出同一张图（可以做插值和编辑）
- 可以**跳步**：只在 $T$ 里挑 20–50 个时间点，质量掉得很少

$$\boxed{\text{DDIM 不用重新训练，只是换了个采样器}}$$

## 6. Classifier-free guidance

想让生成服从条件 $c$（文本、观测、目标）。CFG 的做法是**训练时随机丢掉条件**（比如 10% 概率把 $c$ 置空），让同一个网络既会条件生成又会无条件生成，采样时做线性外推：

$$\boxed{\tilde\epsilon_\theta(x_t,c)=\epsilon_\theta(x_t,\varnothing)+w\cdot\big[\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing)\big]}$$

- $w=0$：无条件；$w=1$：普通条件生成；$w>1$：**放大条件的影响**
- 代价：每步要跑**两次**网络（条件 + 无条件），采样成本翻倍
- $w$ 太大会导致多样性下降、过饱和

直觉：$\epsilon_c-\epsilon_\varnothing$ 是"条件指向的方向"，沿这个方向多走一点，就更贴合条件。

## 7. 预测什么：$\epsilon$ / $x_0$ / $v$

| 参数化 | 预测目标 | 特点 |
|---|---|---|
| $\epsilon$-pred | 噪声 | 最常用；但 $t\to0$（噪声很小）时信噪比失衡 |
| $x_0$-pred | 原始数据 | $t$ 大时不稳 |
| **$v$-pred** | $v=\sqrt{\bar\alpha_t}\,\epsilon-\sqrt{1-\bar\alpha_t}\,x_0$ | 在全时间范围上都良态，蒸馏和高分辨率常用 |

三者可以互相换算，**训练好之后是等价的**，区别在训练时的数值条件。相关的还有 **SNR 加权**（按信噪比给不同 $t$ 加权，如 Min-SNR）。

## 8. 在具身里怎么用

| 用途 | 怎么用 |
|---|---|
| **Diffusion Policy** | 把「未来 $H$ 步动作序列」当作要去噪的样本 $x_0$，条件是观测。见 [action head](../05-action/02-action-head.md) |
| **世界模型 / 视频生成** | 生成未来帧，条件是当前帧 + 动作。见 [世界模型](../03-world-model/00-map.md) |
| **数据增强** | 生成新场景、新视角 |

在 diffusion policy 里，**多模态**正是要的性质：同一个观测下"从左绕"和"从右绕"是分布的两个峰，diffusion 能各自采到，而 MSE 回归只会给出中间那个会撞上去的平均值。

## 自测

**1.** 写出前向过程的闭式解，说明它为什么关键。

> **答：** $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$，其中 $\bar\alpha_t=\prod_{s\le t}(1-\beta_s)$。
> 关键在于**训练时不需要真的一步步加噪**：随机采一个 $t$，一步就能合成出 $x_t$，训练才能高效并行。

**2.** 写出 DDPM 的简化训练目标和四行训练循环。

> **答：** $\mathcal L=\mathbb E_{x_0,\epsilon,t}\big[\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\ t)\|^2\big]$。
> 循环：① 取 $x_0$；② 采 $t\sim\mathcal U[1,T]$、$\epsilon\sim\mathcal N(0,I)$；③ 合成 $x_t$；④ 最小化 $\|\epsilon-\epsilon_\theta(x_t,t)\|^2$。
> **训练目标就是一个 MSE**，这是它比 GAN 稳定得多的根本原因。

**3.** 简化目标丢掉了什么？有什么后果？

> **答：** 丢掉了完整 ELBO 里每个 $t$ 的加权系数，相当于给所有 $t$ 同等权重。实践上效果更好（等价于更重视中等噪声水平），但**它不再是严格的似然下界**。

**4.** 推导预测噪声和预测 score 的等价关系。

> **答：** 对高斯 $q(x_t|x_0)$ 有 $\nabla_{x_t}\log q=-\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}=-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$，所以
> $$s_\theta(x_t,t)=-\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}$$
> **预测噪声就是预测 score，只差一个已知缩放。** 这把 DDPM 和 score-based SDE 统一了，也是各种 ODE 求解器能加速采样的基础。

**5.** DDIM 为什么不用重新训练就能跳步？$\sigma_t=0$ 意味着什么？

> **答：** 因为训练目标只约束了**边缘分布** $q(x_t|x_0)$，没约束联合的马尔可夫结构。所以可以构造一族非马尔可夫的反向过程，共享同一个 $\epsilon_\theta$，只是换了采样器。
> $\sigma_t=0$ 时采样**完全确定性**：同一个初始噪声总给出同一个结果，因此可以做 latent 插值和图像编辑。

**6.** 写出 CFG 的公式，说明代价和 $w$ 过大的问题。

> **答：** $\tilde\epsilon=\epsilon_\theta(x_t,\varnothing)+w\big[\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing)\big]$。训练时随机丢条件（如 10%），让一个网络兼具条件和无条件能力。
> 代价：**每步要跑两次网络**，采样成本翻倍。$w$ 太大会多样性下降、过饱和。
> 直觉：$\epsilon_c-\epsilon_\varnothing$ 是条件指向的方向，沿它多走一点就更贴合条件。

**7.** $\epsilon$-pred / $x_0$-pred / $v$-pred 各有什么特点？

> **答：** $\epsilon$-pred 最常用，但 $t\to0$ 时信噪比失衡；$x_0$-pred 在 $t$ 大时不稳；**$v$-pred**（$v=\sqrt{\bar\alpha_t}\epsilon-\sqrt{1-\bar\alpha_t}x_0$）在全时间范围都良态，蒸馏和高分辨率常用。
> 三者可互相换算，训练好后等价，区别只在训练时的数值条件。

**8.** diffusion policy 里「多模态」为什么是优点而不是问题？

> **答：** 同一个观测下，「从左绕」和「从右绕」是动作分布的两个峰。**MSE 回归会输出两者的平均**（正中间，直接撞上障碍物）；diffusion 建模的是整个分布，能各自采到其中一个模式。
> 所以多模态正是要的性质，也是必须用生成式建模而非回归的根本理由。
