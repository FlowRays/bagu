# AE → VAE → VQ-VAE → Latent Diffusion：从零走一遍

> [01](01-generative-models.md) 是五条路线的结论版（ELBO 公式、posterior collapse、straight-through、VQ 的三个 loss 都在那里）。
> 这一篇是**前置走查**：按 tensor shape 把 AE / VAE 跑一遍，讲清楚每个设计为什么存在，最后接到 latent diffusion 和世界模型。

一条主线：

$$\boxed{\text{AE：学压缩}\ \to\ \text{VAE：学规整的概率 latent}\ \to\ \text{VQ-VAE：latent 离散成 token}\ \to\ \text{Latent Diffusion：在连续 latent 里生成}}$$

## 1. AE：只做压缩和重建

Autoencoder 就一件事：

$$x\xrightarrow{\text{Encoder}}z\xrightarrow{\text{Decoder}}\hat x,\qquad \hat x\approx x$$

以图像为例：

$$x\in\mathbb R^{3\times256\times256}\ \longrightarrow\ z\in\mathbb R^{4\times32\times32}\ \longrightarrow\ \hat x\in\mathbb R^{3\times256\times256}$$

空间尺寸缩小 8 倍。训练就是重建 loss：

$$\mathcal L=\|x-\hat x\|^2\qquad\text{（图像里也常用 L1、perceptual loss）}$$

### AE 的问题：latent 有空洞

AE 只要求"编进去的点能解回来"，没人管 latent 空间整体长什么样：

```text
latent space:

 猫 z1 ●
                         ● z2 狗
            ?   ← 这片区域 decoder 从没见过
```

随便采一个 $z\sim\mathcal N(0,I)$ 送进 decoder，大概率落在空洞里，出来是垃圾。

$$\boxed{\text{AE 适合压缩 / 表示，天然不适合随机生成}}$$

## 2. VAE：encoder 输出一个分布

VAE 最关键的改动不在 decoder，而在 encoder：**不输出一个点，输出一个高斯分布**。

$$x\ \to\ (\mu,\sigma)\ \to\ q(z|x)=\mathcal N(\mu,\sigma^2)\ \to\ z\sim q(z|x)\ \to\ \hat x$$

直觉：每张图不再对应一个孤零零的点，而是 latent 里**一小片概率区域**；再要求所有区域整体靠近 $\mathcal N(0,I)$。这样生成时直接 $z\sim\mathcal N(0,I)$ 采样，落点周围都是训练时见过的区域。

```text
AE:   A → ●          B → ●            （点，中间空洞）
VAE:  A → ○○○        B → ○○○          （团，且都挤在原点附近）
```

## 3. 一次完整 forward，按 shape 走

### Encoder：输出通道翻倍后劈两半

```text
x        3 × 256 × 256
  → 128 × 128 × 128
  → 256 × 64 × 64
  → 512 × 32 × 32
  → 8 × 32 × 32          ← encoder 最后一层
       ↓ split(dim=channel)
  μ       4 × 32 × 32
  logvar  4 × 32 × 32
```

### 为什么预测 logvar 而不是 σ

方差必须 $>0$，但网络输出是任意实数。预测 $\log\sigma^2\in(-\infty,+\infty)$，再恢复：

$$\sigma^2=e^{\text{logvar}}>0,\qquad \sigma=\sqrt{e^{\log\sigma^2}}=e^{\frac12\log\sigma^2}$$

这就是代码里 `std = torch.exp(0.5 * logvar)` 那个 **0.5 的来源**。

### 采样：重参数化

直接从 $\mathcal N(\mu,\sigma^2)$ 采样不可导。改写成

$$\epsilon\sim\mathcal N(0,I),\qquad \boxed{z=\mu+\sigma\odot\epsilon}$$

随机性挪到与参数无关的 $\epsilon$ 上，$z$ 对 $\mu,\sigma$ 是普通可微运算，梯度 $\mathcal L\to z\to\mu,\sigma$ 能回传。

**手算一维**：$\mu=2,\ \sigma=0.5$，采到 $\epsilon=-0.8$：

$$z=2+0.5\times(-0.8)=1.6$$

$z$ 不是永远等于 2，而是在 2 附近波动。

**为什么 $\mu+\sigma\epsilon$ 真的服从 $\mathcal N(\mu,\sigma^2)$**：

$$\mathbb E[z]=\mu+\sigma\,\mathbb E[\epsilon]=\mu$$

$$\mathrm{Var}(z)=\mathrm{Var}(\sigma\epsilon)=\sigma^2\,\mathrm{Var}(\epsilon)=\sigma^2$$

高斯的线性变换还是高斯，所以 $z\sim\mathcal N(\mu,\sigma^2)$。

### Decoder：反向把 shape 放大

```text
z      4 × 32 × 32
  → 512 × 32 × 32
  → 256 × 64 × 64
  → 128 × 128 × 128
  → 3 × 256 × 256      = x̂
```

整次 forward：

$$\boxed{x\ \to\ (\mu,\log\sigma^2)\ \to\ z\ \to\ \hat x}$$

## 4. Loss：重建 + KL

$$\boxed{\mathcal L=\mathcal L_{\text{recon}}+\beta\,D_{KL}\big(q(z|x)\,\|\,\mathcal N(0,I)\big)}$$

- **重建项**：$\hat x\approx x$，保证 latent 里的信息够用。
- **KL 项**：encoder 给出的分布不要离标准正态太远，保证 latent 空间规整。

两个高斯的 KL 有闭式解（每一维，多维求和）：

$$\boxed{D_{KL}\big(\mathcal N(\mu,\sigma^2)\,\|\,\mathcal N(0,1)\big)=\tfrac12\left(\mu^2+\sigma^2-\log\sigma^2-1\right)}$$

代码里常见的写法是同一个公式换了个符号：

```python
kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
```

### KL 在推什么

代入 $\mu=0,\sigma=1$：$\tfrac12(0+1-0-1)=0$，最小。所以 KL 一直在把 **$\mu\to0$、$\sigma\to1$**。

### 两项在拉扯

如果完全听 KL 的，所有图片都是 $\mathcal N(0,I)$，$z$ 里没有任何图片信息，decoder 没法重建。重建项说"不同图片的 latent 必须有区别"，KL 说"你们都别离原点太远"。VAE 就是在找这个平衡。

⚠️ **KL 不是要求 $q(z|x)=\mathcal N(0,I)$**，只是一个惩罚。结果是各类图片都围着原点、但彼此仍有区别：

```text
猫  μ = (-0.7, 0.4)
狗  μ = ( 0.8, 0.2)
车  μ = ( 0.2,-0.9)
```

### β-VAE

$\beta=1$ 是标准 VAE。$\beta>1$ 更强调规整：重建变差，但 latent 可能更**解耦（disentangled）**，理想情况下某一维只管颜色、某一维只管旋转角度。

### 完整代码骨架

```python
h = encoder(x)                          # [B, 8, 32, 32]
mu, logvar = h.chunk(2, dim=1)          # 各 [B, 4, 32, 32]
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + std * eps                      # 重参数化
x_hat = decoder(z)                      # [B, 3, 256, 256]

recon_loss = F.mse_loss(x_hat, x, reduction="sum")
kl_loss = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum()
loss = recon_loss + beta * kl_loss
```

## 5. 变分推断：「Variational」这个词

隐变量模型 $z\to x$。看到 $x$ 想反推 latent，要的是后验：

$$p(z|x)=\frac{p(x|z)\,p(z)}{p(x)},\qquad p(x)=\int p(x|z)\,p(z)\,dz$$

**分母这个积分极难算**，所以真后验 $p(z|x)$ 算不出来。

变分推断的做法：自己造一个简单、可计算的分布 $q_\phi(z|x)$（比如高斯，$\mu,\sigma$ 由网络输出），通过优化让它逼近 $p(z|x)$。

$$\boxed{\text{变分推断 = 用一个可学习的简单分布 }q_\phi\text{ 去近似难算的真后验 }p(z|x)\text{，把求分布变成优化问题}}$$

- **推断**：由观测 $x$ 推背后的 $z$。VAE 的 encoder $q_\phi(z|x)$ 就是一个 **inference network**。
- **变分**：在一族候选分布（随 $\phi$ 变化）里找最接近目标的那个。不需要学变分法。

所以名字拆开就是：

```text
Decoder  p_θ(x|z)   负责生成
Encoder  q_φ(z|x)   负责近似后验（inference）
VAE = 用神经网络做变分推断的 Autoencoder
```

## 6. ELBO：只讲到这一层

生成模型最终想最大化 $\log p(x)$（模型觉得真实数据出现的概率高），但直接算不动。于是优化一个可算的下界：

$$\text{ELBO}\le\log p(x),\qquad \text{ELBO}=\underbrace{\mathbb E_{q}[\log p(x|z)]}_{\text{重建项}}-D_{KL}\big(q(z|x)\,\|\,p(z)\big)$$

把 ELBO 推高，就是在努力推高 $\log p(x)$；取负号变成最小化，就是上面的"重建 + KL"。**VAE 的 loss 不是拍脑袋把两项加起来，而是从概率目标里推出来的。** 完整不等式见 [01](01-generative-models.md)。

对应关系：

- $\log p(x|z)$ ↔ decoder 重建得好不好。解码器假设高斯似然时就是 MSE。
- $p(z)$ ↔ 事先规定的 prior，通常 $\mathcal N(0,I)$。

### prior 为什么必须有

生成时**没有输入图片**，$z$ 只能从 $p(z)$ 采。如果训练时 encoder 产生的 latent 在 $(100,53,-72)$ 附近，生成时却从 $\mathcal N(0,1)$ 采，decoder 根本没见过，直接崩。

$$\boxed{\text{KL 的实际作用：让训练时 encoder 产生的 latent 和生成时采的 latent 接得上}}$$

### 为什么 VAE 输出糊

重建要求"这张图对应一个精确的 z"，KL 要求"周围一小片 z 采出来也都得像这张图"，decoder 于是学成平滑映射；再加上逐像素高斯似然等价 MSE、最优解是条件均值 —— 多个合理细节平均一下就糊了。结构对，细节软。

## 7. VQ-VAE：把 latent 离散成 token

动机：Transformer 最自然处理离散 token。能不能把图片也变成 token ID？

准备一个 codebook $\{e_1,\dots,e_K\}$（比如 $K=8192$，每个 $e_k\in\mathbb R^d$），encoder 先输出连续向量 $z_e$，然后**找最近的码字**：

$$k^*=\arg\min_k\|z_e-e_k\|^2,\qquad z_q=e_{k^*}$$

**手算**：$z_e=[0.9,1.2]$，codebook 是 $e_1=[-1,-1]$、$e_2=[1,1]$、$e_3=[2,-3]$。

$$\|z_e-e_1\|^2=1.9^2+2.2^2=8.45,\quad \|z_e-e_2\|^2=0.1^2+0.2^2=0.05,\quad \|z_e-e_3\|^2=1.1^2+4.2^2=18.85$$

选 $e_2$，这个位置的 token 就是 **2**。

整张图 $256\times256$ 经 encoder 到 $32\times32$，每个位置一个码字 ID：

```text
27   27   103  103
27   51   51   103
82   82   94   94
...                    共 32×32 = 1024 个离散 visual token
```

于是可以像 LLM 一样建模 $p(t_1,\dots,t_{1024})=\prod_i p(t_i|t_{<i})$，用交叉熵做 next-token prediction（早期 DALL·E 这条路线）。

$$\boxed{\text{VQ-VAE}\approx\text{learned image tokenizer}}$$

最近邻查找不可导怎么办、codebook / commitment loss 怎么写，见 [01 第 4 节](01-generative-models.md)。

### VQ-VAE 的问题

- **量化误差**：$z_e=[0.93,1.17]$ 被硬拉成 $[1,1]$，信息必然丢。压缩率越高，人脸、文字、纹理、运动这种细节越容易坏。
- **codebook 大小两难**：$K=256$ 表达力不够；$K=65536$ 难训，大量码字没人用 —— **codebook collapse / dead codes**。

### VQGAN

$$\boxed{\text{VQGAN}\approx\text{VQ-VAE}+\text{perceptual loss}+\text{GAN loss}}$$

纯像素重建还是糊，于是加感知损失（比较高层视觉特征，而不是逐像素）和判别器（让细节更锐利）。后面很多 image tokenizer 都受它影响。

## 8. VQ token 和 VLM 的 visual token 不是一回事

| | VQ-VAE token | VLM visual token（Qwen-VL / LLaVA / InternVL） |
|---|---|---|
| 是什么 | **离散 ID**，属于有限 codebook $\{1..K\}$ | **连续向量** $z_i\in\mathbb R^d$，没有 ID |
| 例子 | `[137, 51, 802, 17, ...]` | `[[0.13,-0.44,...], [1.02,0.37,...], ...]` |
| 目标 | 可重建、可生成 | 理解 |
| 需要 decoder 回到像素吗 | 需要 | 不需要 |

VLM 只要"看懂"，`image → ViT → 连续 embedding → LLM` 就够了，**不需要一个可逆的 tokenizer**。图像生成最终必须 latent → image，所以才要 VAE / VQ-VAE 这种带 decoder 的东西。见 [VLM 架构](../../notes/07-vlm/00-map.md)。

$$\boxed{\text{VQ token 是词表里的离散符号；VLM visual token 通常只是连续向量序列}}$$

## 9. 为什么生成主流回到了「连续 VAE latent + diffusion」

两条路线都是"先压缩再生成"：

```text
路线 A:  x → VQ-VAE → [t_1..t_N] → AR Transformer 逐 token 生成
路线 B:  x → VAE → z ∈ R^{C×H'×W'} → diffusion 去噪生成 z → VAE decoder → x
```

路线 A 很有吸引力：一切都统一成 next-token prediction，理论上文本、图像、音频、视频、动作 token 可以放进一个词表。但它输在三点：

**① 量化误差**：连续 VAE latent 没有硬量化这一层。

**② 串行维度不同**：

$$\boxed{\text{AR：串行长度}\approx\text{token 数}\qquad\text{Diffusion：串行长度}\approx\text{去噪步数}}$$

一张图 $64\times64=4096$ 个 token，AR 要串行 4096 次；视频 32 帧 × 每帧 $32\times32$ 就是 $32\times32\times32=32768$ 个 token。diffusion 每一步是对**整个 latent tensor** 一次 forward，20–50 步，每步高度并行。

**③ 高斯噪声天然属于连续空间**：$z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\,\epsilon$ 在 $\mathbb R^d$ 里很自然；token ID 137 加 $0.3\epsilon$ 毫无意义。

Latent Diffusion 把两边优势拼起来：VAE 把 $3\times1024\times1024$ 压到小得多的连续 latent（比如 $16\times128\times128$），diffusion 在上面建模复杂分布 —— 高质量、省算力、连续空间好优化。细节见 [Diffusion](02-diffusion.md)。

### DiT 不是自回归

DiT（Diffusion Transformer）只是把 diffusion 的去噪网络从 U-Net 换成 Transformer：

$$z_t\ \to\ \text{Transformer}\ \to\ \hat\epsilon\ /\ v\ /\ x_0$$

它仍然是 diffusion，每一步并行处理整个 latent。换 Transformer 是因为它 scale 起来性能持续涨，这在 LLM 上已经被证明过。演进：`VAE + U-Net`（早期 Stable Diffusion）→ `VAE + DiT`。

现代图像 / 视频生成可以拆成三块：

$$\boxed{\text{Autoencoder（pixel}\leftrightarrow\text{latent）}+\text{生成 backbone（DiT 生成 latent）}+\text{条件编码器（text}\to\text{condition）}}$$

## 10. 现代 VAE 更像 learned codec

经典 VAE 自己就是生成器（$z\sim\mathcal N(0,I)\to$ decoder）。Stable Diffusion / 视频模型里的 VAE 职责变了：

$$\boxed{\text{VAE 负责压缩 / 解压（定义 latent space），diffusion 负责生成 latent（学 latent 分布）}}$$

**那还要 KL 吗？** prior 约束的需求确实弱了很多，因为不再从 $\mathcal N(0,I)$ 直接解码。但 diffusion 仍然希望 latent 数值范围稳定、空间连续、不同样本尺度相近、适合加高斯噪声 —— 一个样本在 $[-0.1,0.2]$、另一个在 $[-1000,3000]$ 是训不动的。所以仍然用小权重 KL / latent scaling 让它 well-behaved。

**重建 loss 也变了**：除 L1/L2 外加 perceptual loss（边缘错位 1 像素时 MSE 很大但人看不出差别）和 adversarial loss（细节更锐）。

**视频更明显**：Video VAE 同时压时间和空间，概念上 $16\times720\times1280$ 可以压到 $4\times90\times160$ 这种量级的 latent grid，得到 $z\in\mathbb R^{C\times T'\times H'\times W'}$，Video DiT 在上面生成。此时 VAE 就是一个**神经网络视频 codec**。

## 11. 三类生成，以及离散路线没有死

| 类型 | 做法 | 定位 |
|---|---|---|
| Pixel generation | 直接生成 $x$ | 成本最高 |
| **Continuous latent** | $x\leftrightarrow z\in\mathbb R^d$，diffusion / flow 生成 $z$ | 现代图像 / 视频生成主流 |
| **Discrete latent** | $x\leftrightarrow[t_1..t_N]$，AR Transformer 生成 | 统一性好，适合多模态 / 世界模型研究 |

离散 token 在**世界模型**和统一多模态里仍很有吸引力：把观测和动作都 token 化，

```text
visual tokens | action token | visual tokens | action token | ...
```

next-token prediction 就直接是在学 $p(z_{t+1}\mid z_t,a_t)$。连续版本则是 $z_t,a_t\to z_{t+1}$ 的 latent dynamics（Dreamer / JEPA 的大方向）。见 [世界模型](../07-world-model/01-world-model-map.md)。

两个别混：

- VAE 和 diffusion 不是竞争关系：**VAE 定义 latent space，diffusion 学 latent distribution**。
- VQ-VAE 和 Transformer 不是一个东西：**VQ-VAE 是 tokenizer / codec，Transformer 是生成模型 / 世界模型**。

## 12. latent 到底学到了什么

$$\boxed{\text{latent = 对当前训练目标有用的压缩表示}}$$

AE 从没被告诉"这是猫"，只被要求 $D(E(x))\approx x$。它学到什么，完全取决于**什么信息对重建最有用**。

- **为什么会出现"语义"**：自然图像统计规律很强（两只眼睛总一起出现、耳朵形状固定）。逐像素硬记压缩效率太差，学"这是一个猫脸结构"更省 —— **为了压缩 / 预测，模型被迫学到结构**，这就是 representation learning 的核心。
- **不可逐维解释**：不要期待 $z_1$=颜色、$z_2$=猫还是狗。一个概念通常分散在很多维上。
- **spatial latent 保留位置**：$4\times32\times32$ 是低分辨率 feature map，左上角 latent 主要描述原图左上区域（卷积 / attention 让感受野更大，不是严格一一对应）。这正是 latent diffusion 适合图像的原因。
- **channel 不是 RGB**：4 个 channel 是学出来的特征维，可能偏低频结构、颜色、边缘、纹理，但通常是混合的。
- **为什么压缩还能重建**：自然数据高度冗余。一片蓝天没必要逐像素存，"这里是一大片浅蓝"就够。真实图片只占 $\mathbb R^{3\times256\times256}$ 里极小的一块结构化区域 —— **data manifold**，latent 就是描述它的低维坐标（随机采 RGB 数字几乎全是噪声）。
- **和 PCA 的关系**：PCA 也是 $x\to z\to\hat x$，只是线性的。$\text{PCA}\approx\text{线性 AE}$，深度 AE 能学非线性 manifold。

## 13. 转折点：重建好的表示 ≠ 适合预测未来的表示

世界模型不想在几万个像素上预测 $o_t,a_t\to o_{t+1}$，而是想在紧凑状态上做

$$\boxed{z_t,\ a_t\ \to\ z_{t+1}}$$

问题是 AE 学出来的 $z_t$ **只对 $z_t\to o_t$ 负责**。以一个 2D 横版游戏为例：

- **背景纹理**占大量像素，对重建很重要，对控制几乎没用；
- **角色的水平速度**在单帧画面上几乎看不出来（视觉差异极细微），却决定了下一帧在哪。

$$\boxed{\text{good reconstruction representation}\neq\text{good dynamics representation}}$$

这正是从 AE / VAE 的表示学习走向 [JEPA](../10-jepa/01-jepa.md)（在表示空间预测、不重建像素）、Dreamer / WAM 的 [latent dynamics](../07-world-model/01-world-model-map.md) 时最重要的一步。

## 自测

**1.** AE 为什么适合压缩却不适合随机生成？

> **答：** AE 只要求 $D(E(x))\approx x$，没人约束 latent 空间的整体形状。训练样本编码成一个个孤立的点，点之间是 decoder 从没见过的空洞。随手采 $z\sim\mathcal N(0,I)$ 大概率落在空洞里，解出来是垃圾。

**2.** VAE 的 encoder 相比 AE 改了什么？直觉上带来什么？

> **答：** 不输出一个点 $z$，而输出高斯分布的参数 $\mu,\log\sigma^2$，定义 $q(z|x)=\mathcal N(\mu,\sigma^2)$ 再从中采样。
> 每张图对应 latent 里一小片概率区域，再用 KL 把所有区域拉到 $\mathcal N(0,I)$ 附近，于是生成时从 $\mathcal N(0,I)$ 采样也落在 decoder 见过的区域。

**3.** 为什么预测 logvar？`std = torch.exp(0.5 * logvar)` 的 0.5 从哪来？

> **答：** 方差必须为正，网络输出是任意实数；预测 $\log\sigma^2$ 再取指数就保证 $\sigma^2>0$。
> $\sigma=\sqrt{\sigma^2}=\sqrt{e^{\log\sigma^2}}=e^{\frac12\log\sigma^2}$，0.5 就是开根号。

**4.** 重参数化怎么写？证明 $z=\mu+\sigma\epsilon$ 服从 $\mathcal N(\mu,\sigma^2)$。$\mu=2,\sigma=0.5,\epsilon=-0.8$ 时 $z$ 是多少？

> **答：** $\epsilon\sim\mathcal N(0,I)$，$z=\mu+\sigma\odot\epsilon$，随机性挪到与参数无关的 $\epsilon$，梯度能回传到 $\mu,\sigma$。
> $\mathbb E[z]=\mu+\sigma\mathbb E[\epsilon]=\mu$；$\mathrm{Var}(z)=\sigma^2\mathrm{Var}(\epsilon)=\sigma^2$；高斯线性变换仍是高斯。
> $z=2+0.5\times(-0.8)=1.6$。

**5.** 写出一次 VAE forward 的 shape 链（输入 $3\times256\times256$，latent $4\times32\times32$）。

> **答：** $3\times256\times256\to\dots\to512\times32\times32\to8\times32\times32$，按 channel 劈成 $\mu$ 和 logvar 各 $4\times32\times32$；$z=\mu+\sigma\epsilon$ 为 $4\times32\times32$；decoder $\to512\times32\times32\to256\times64\times64\to128\times128\times128\to3\times256\times256$。

**6.** 写出高斯 KL 的闭式解。它把 $\mu,\sigma$ 往哪推？既然推向 $\mathcal N(0,I)$，不同图片的 latent 为什么不会都一样？

> **答：** $D_{KL}=\tfrac12(\mu^2+\sigma^2-\log\sigma^2-1)$，代码写成 `-0.5*(1+logvar-mu**2-logvar.exp()).sum()`。$\mu=0,\sigma=1$ 时为 0，所以推 $\mu\to0,\sigma\to1$。
> KL 只是惩罚，不是硬约束。重建项要求不同图片的 latent 必须可区分，两者拉扯，结果是各类都在原点附近但彼此不同。完全听 KL 的话 $z$ 不含信息，decoder 无法重建。

**7.** β-VAE 的 β 调大会怎样？

> **答：** 更强调 latent 规整：重建质量下降，但 latent 可能更解耦（某一维只管颜色、某一维只管角度）。$\beta=1$ 是标准 VAE。

**8.** 变分推断在解决什么？"推断"和"变分"各指什么？

> **答：** 真后验 $p(z|x)=p(x|z)p(z)/p(x)$ 的分母 $p(x)=\int p(x|z)p(z)dz$ 极难算。变分推断用一个简单可学习的 $q_\phi(z|x)$ 去近似 $p(z|x)$，把求分布变成优化问题。
> **推断**：由观测 $x$ 反推 latent $z$，VAE encoder 就是 inference network。**变分**：在一族随参数变化的候选分布里找最接近目标的那个。

**9.** ELBO 和 $\log p(x)$ 什么关系？prior $p(z)$ 为什么必须有？

> **答：** $\text{ELBO}\le\log p(x)$，$\text{ELBO}=\mathbb E_q[\log p(x|z)]-D_{KL}(q(z|x)\|p(z))$，推高它就是在推高 $\log p(x)$，取负就是"重建 + KL"。
> 生成时没有输入图片，$z$ 只能从 prior 采；KL 保证训练时 encoder 产生的 latent 和生成时采的 latent 在同一片区域，否则 decoder 面对没见过的 $z$ 会崩。

**10.** VQ-VAE 怎么把一张图变成 token？$z_e=[0.9,1.2]$，codebook 为 $[-1,-1],[1,1],[2,-3]$，选哪个？VQ-VAE 有什么问题？

> **答：** encoder 输出每个空间位置的连续向量，在 codebook 里找 $\arg\min_k\|z_e-e_k\|^2$，用下标当 token，整张图变成 token grid（$32\times32=1024$ 个），可以做 next-token prediction。
> 距离平方分别是 8.45、0.05、18.85，选 $e_2$，token = 2。
> 问题：**量化误差**（细节丢失）；codebook 太小表达不够、太大难训且出现 **dead codes**。VQGAN 加 perceptual + GAN loss 让重建更锐。

**11.** ⭐ VQ-VAE 的 visual token 和 Qwen-VL 这类 VLM 的 visual token 有什么区别？VLM 为什么不需要 VQ-VAE？

> **答：** VQ token 是有限 codebook 里的**离散 ID**；VLM 的 visual token 是 ViT 输出的**连续 embedding**，没有 ID，只是被当作序列喂进 Transformer。
> VLM 只需要理解图片，不需要从 visual token 还原像素，所以不需要可逆 tokenizer；生成模型必须 latent → image，才需要带 decoder 的 VAE / VQ-VAE。

**12.** ⭐ 为什么现代图像 / 视频生成主流是连续 VAE latent + diffusion，而不是 VQ + 自回归？

> **答：** ① 连续 latent 没有量化误差；② **AR 的串行长度≈token 数**（一张图 4096、32 帧视频 32768 次），**diffusion 的串行长度≈去噪步数**（20–50），每步对整个 latent 并行；③ 加高斯噪声天然属于连续空间，token ID 加噪声没有意义。
> Latent diffusion 让 VAE 负责压缩、diffusion 负责在紧凑连续空间里建模分布。DiT 只是把去噪网络换成 Transformer，仍然是 diffusion，不是自回归。

**13.** latent diffusion 里 VAE 的角色和经典 VAE 有什么不同？为什么还保留 KL？

> **答：** 经典 VAE 自己是生成器；现代 VAE 更像 **learned codec**，只负责 pixel ↔ latent，生成交给 diffusion。
> 对 prior 的需求弱了，但 diffusion 仍需要 latent 数值范围稳定、空间连续、尺度一致、适合加噪，所以保留小权重 KL / scaling；重建端还加 perceptual 和 GAN loss 让细节更锐。视频 VAE 同时压时间和空间维。

**14.** ⭐ AE 学到的 latent 为什么不一定适合做世界模型？

> **答：** AE 的 latent 只对"重建当前画面"负责。对重建重要的（大面积背景纹理）可能对控制没用；对预测未来关键的（角色速度，单帧上视觉差异极小）可能在重建目标里无足轻重。
> $\text{good reconstruction representation}\neq\text{good dynamics representation}$，所以世界模型要学 $z_t,a_t\to z_{t+1}$ 意义下的表示，JEPA 干脆不重建像素、改在表示空间预测。
