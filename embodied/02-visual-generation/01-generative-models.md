# 生成模型谱系

> 这一册是**世界模型**和 **diffusion / flow action head** 的共同地基。
> 先把五条路线的取舍摆清楚，再深入 diffusion 和 flow matching。

## 1. 五条路线在解决同一个问题

都想学 $p_\theta(x)\approx p_{\text{data}}(x)$ 并能从中采样，区别在于**怎么绕开配分函数**。

| 路线 | 怎么建模 | 采样 | 似然 | 主要毛病 |
|---|---|---|---|---|
| **VAE** | 引入 latent $z$，优化 ELBO 下界 | 一步，快 | 有下界 | 输出模糊 |
| **GAN** | 不建密度，直接学采样器 | 一步，快 | 没有 | 训练不稳、mode collapse |
| **Autoregressive** | $p(x)=\prod_i p(x_i\mid x_{<i})$ | **逐元素，慢** | **精确** | 采样慢、需要定序 |
| **Diffusion** | 学一条从噪声到数据的去噪路径 | 多步，慢 | 有下界 | 采样步数多 |
| **Flow matching** | 学一个把噪声搬到数据的速度场 | **少步甚至一步** | 有 | 较新，生态还在追 |

$$\boxed{\text{质量、多样性、采样速度、训练稳定性 —— 四者不可兼得，五条路线是不同的折中}}$$

## 2. VAE

$$\log p(x)\ \ge\ \underbrace{\mathbb E_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{重建}}-\underbrace{D_{KL}\big(q_\phi(z|x)\,\|\,p(z)\big)}_{\text{正则}}$$

**重参数化**：$z=\mu+\sigma\odot\epsilon,\ \epsilon\sim\mathcal N(0,I)$，把采样从计算图里挪出去，梯度才能回传到 $\mu,\sigma$。

**为什么输出糊**：解码器用的是逐像素高斯似然（等价于 MSE），而 MSE 的最优解是**条件均值** —— 多个合理答案的平均自然就是模糊的。这和 [SFT 用 MSE 回归动作会糊掉](../05-action/02-action-head.md#2-连续回归为什么不够) 是同一个道理。

**posterior collapse**：解码器太强时会忽略 $z$，KL 项把 $q(z|x)$ 压成先验，latent 失效。缓解手段有 KL annealing、free bits、弱化解码器。

在具身里 VAE 主要不是当生成器用，而是当**压缩器**（latent diffusion 的第一级、video VAE）。

## 3. GAN

生成器和判别器对抗。采样一步、图像锐利，但训练不稳、mode collapse（只生成少数几种），而且**没有似然**难以评估。

现在在具身里基本被 diffusion 取代，但它的一个思想仍然重要：**不要求密度可计算，只要求能采样**。

## 4. Autoregressive + 离散化

$$p(x)=\prod_i p(x_i\mid x_{<i})$$

关键前置是**把连续信号离散成 token**：

- **VQ-VAE**：encoder 输出连续向量 → 在 codebook 里找最近邻 → 用 index 当 token
    - 梯度问题：最近邻查找不可导，用 **straight-through**（$z_q=z_e+\text{sg}[z_q-z_e]$）
    - 三个 loss：重建 + codebook + commitment（见 [手撕 RQ-VAE](../../notes/code/07-handwrite/03-llm.md#rq-vae-loss)）
    - codebook collapse：大量码字never被用到
- **VQ-GAN**：加上感知损失和判别器，重建更锐利
- **RQ / 残差量化**：多级量化，逐级编码残差，用更少码字达到更高保真

离散化之后，图像/视频/动作就都能**当语言模型来做** —— 这正是 RT-2、OpenVLA 把动作 tokenize 的思路来源（见 [action head](../05-action/02-action-head.md)）。

$$\boxed{\text{离散化是把「生成问题」翻译成「语言建模问题」的桥}}$$

## 5. 为什么具身最后选了 diffusion 和 flow

三个理由：

1. **动作分布是多模态的**。绕过障碍物可以从左也可以从右，MSE 回归会输出两者的平均（撞上去）。生成式建模才能表达"要么左要么右"。
2. **训练稳定**。相比 GAN，diffusion 的目标就是一个 MSE，没有对抗，几乎不会崩。
3. **可条件化**。观测、语言、本体状态都能自然地作为条件注入。

缺点是采样要多步 —— 这正是 **flow matching / rectified flow** 在 VLA 里迅速取代 diffusion 的原因（见 [flow matching](03-flow-matching.md)）。

## 自测

**1.** 五条生成路线各自怎么建模、采样快慢、有没有精确似然？

> **答：** VAE 优化 ELBO 下界，一步采样，似然有下界，输出糊；GAN 不建密度直接学采样器，一步，无似然，训练不稳；AR 用链式分解，**逐元素采样很慢**但**似然精确**；Diffusion 学去噪路径，多步采样，似然有下界；Flow matching 学速度场，**少步甚至一步**。

**2.** VAE 的输出为什么糊？这和动作回归有什么共同点？

> **答：** 解码器用逐像素高斯似然（等价 MSE），而 **MSE 的最优解是条件均值**，多个合理答案取平均自然模糊。
> 和用 MSE 回归动作是同一个问题：多模态分布下回归出来的是"平均动作"，可能哪个模式都不是（比如绕障碍时输出正中间，直接撞上）。

**3.** 重参数化在解决什么？

> **答：** 采样 $z\sim q(z|x)$ 这个操作不可导。写成 $z=\mu+\sigma\odot\epsilon$、$\epsilon\sim\mathcal N(0,I)$ 之后，随机性被挪到与参数无关的 $\epsilon$ 上，梯度就能回传到 $\mu,\sigma$ 了。

**4.** VQ-VAE 的梯度怎么过最近邻查找？三个 loss 是什么？

> **答：** 用 **straight-through**：$z_q=z_e+\text{sg}[z_q-z_e]$，前向取量化值、反向把梯度直接拷给 $z_e$。
> 三个 loss：**重建** + **codebook loss**（拉码字靠近编码器输出，$\|\text{sg}[z_e]-e\|^2$）+ **commitment loss**（拉编码器靠近码字，$\beta\|z_e-\text{sg}[e]\|^2$），两个 loss 的 stop-gradient 加在相反一侧。

**5.** 离散化对具身意味着什么？

> **答：** 把连续信号变成 token 之后，图像/视频/动作就能**当语言模型来做**，直接复用 LLM 的整套架构、训练和推理基建。RT-2、OpenVLA 把动作分 bin 变 token 接到 VLM 上，就是这个思路。

**6.** 为什么具身最后选了 diffusion / flow 而不是 GAN 或直接回归？

> **答：** ① **动作分布多模态**，回归会输出平均值（可能哪个模式都不是），生成式建模才能表达"要么左要么右"；② 训练稳定，目标就是一个 MSE，没有对抗；③ 观测、语言、本体状态都能自然条件化。
> 代价是采样多步，所以 VLA 正在从 diffusion 转向 **flow matching / rectified flow**。
