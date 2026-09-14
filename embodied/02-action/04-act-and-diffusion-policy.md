# ACT 与 Diffusion Policy：连续 action chunk 从哪来

> 第一代 VLA（[OpenVLA](../01-vla/02-gen1-ar-vla.md)）是"强 VLM + 很原始的动作生成"。同一时期机器人学习里还有另一条线：**没有语言大模型，但动作生成做得很好**。π0 / G0 就是把这两条线拼起来（[03](../01-vla/03-pi0-pi05-g0.md)）。
> 前置：[chunking 与实时性](03-chunking-and-realtime.md)（chunk、temporal ensembling 的基本概念）、[Diffusion](../09-visual-generation/02-diffusion.md)、[AE / VAE 从零](../09-visual-generation/01b-ae-vae-from-zero.md)。

```text
连续 BC（MSE，单步）
  ↓ 误差累积
ACT：一次预测 k 步 + CVAE + temporal ensembling
  ↓ MSE 仍然会平均多个模式
Diffusion Policy：学整个 p(A|o)，从噪声去噪出 chunk
  ↓ 没有语义；去噪步数多
π0：VLM + flow matching action expert
```

## 1. 连续 BC 和 compounding error

$$\hat a_t=\pi_\theta(o_t),\qquad \mathcal L_{\text{BC}}=\|\hat a_t-a_t\|^2$$

推理逐步闭环：$o_0\to\hat a_0\to o_1\to\hat a_1\to\cdots$。问题在于：

$$\boxed{\text{小动作误差}\to\text{状态分布偏移}\ (d_\pi\neq d_{\text{expert}})\to\text{更大的预测误差}\to\text{更大的偏移}}$$

训练数据只覆盖专家轨迹附近；夹爪偏了 5 mm，模型就进入没见过的状态，下一步错得更多，形成正反馈。高频控制尤其严重：10 s 任务、50 Hz 就是 **500 次**连续决策。

## 2. ACT：Action Chunking with Transformers

来自 ALOHA 那篇工作，目标是**高精度双臂模仿**，不是通用 VLA。

| | ACT |
|---|---|
| 输入 | 多视角 RGB + 当前双臂关节位置 $q_t$ |
| 输出 | 未来 $k$ 步**双臂目标关节位置** $\hat A_t\in\mathbb R^{k\times D}$ |
| 频率 | 数据和遥操作都 50 Hz，底层 PID 跟踪关节目标 |
| 语言 / 预训练 VLM | 没有 |

### 为什么 chunk 有用（两个理由）

1. **有效决策 horizon 降 $k$ 倍**：$T=500$、$k=100$ 时决策点从 500 变成 5 左右（配合 temporal ensembling 并不是真的闭眼执行）。
2. **人类演示有时间相关、非 Markov 的行为**：插电池时"停顿一下"和"准备继续插"的单帧观测几乎一样，单步策略分不清；一次预测一整段就能学到"现在开始的是一段插入动作"。

### 卡点：decoder 是 k 个 query 并行出 chunk，不是自回归

```text
query_1 → 未来第 1 步
query_2 → 未来第 2 步
...
query_k → 未来第 k 步      ← 一次 forward 同时得到 [k, D]
```

没有 OpenVLA 那种"一个 token 一个 token 解码"的延迟问题。

### 卡点：ACT 其实是一个 CVAE

同一个观测下，人的演示本来就有差异（从左一点靠近 / 从右一点靠近）。ACT 用一个隐变量 $z$ 表示这种"风格"：

```text
训练：
  未来专家动作 a_{t:t+k} + 当前关节 ──→ CVAE encoder ──→ μ, σ ──→ z
  图像 + 当前关节 + z ────────────────────────────────→ ACT decoder ──→ Â
```

$$\mathcal L=\underbrace{\mathrm{MSE}(\hat A,A)}_{\text{重建}}+\beta\,\underbrace{D_{\text{KL}}\big(q_\phi(z\mid A,q_t)\,\|\,\mathcal N(0,I)\big)}_{\text{正则}}$$

**为什么必须有 KL**：encoder 能看到答案。没有 KL 时它会把整段未来动作偷偷塞进 $z$，decoder 不看图像也能把 loss 压到很低；测试时没有未来动作，这个 $z$ 根本拿不到。KL 把 $q(z|\cdot)$ 拉向 $\mathcal N(0,I)$，限制了这条"答案通道"（VAE 的信息瓶颈，见 [AE / VAE](../09-visual-generation/01b-ae-vae-from-zero.md)）。

**推理时没有未来动作，$z$ 怎么来**：直接取 **$z=0$**（先验的均值），$(o_t,z=0)\to\hat A_t$，确定性推理。

$$\boxed{\text{ACT 用 CVAE 吸收演示差异，但默认部署固定 }z=0\text{，不是每次采样不同模式}}$$

所以别夸大成"ACT 解决了多模态"：它比纯 MSE 更有生成模型的味道，但**本质上仍是回归**。

### Temporal ensembling 和普通平滑不一样

ACT 每个时间步都重新推理一次，同一个物理时刻 $t$ 会收到多个 chunk 的预测：

$$a_t=\frac{\sum_i w_i\,\hat a_t^{(i)}}{\sum_i w_i},\qquad w_i=e^{-m\,i}$$

- **普通平滑**：把 $a_{t-1},a_t,a_{t+1}$ 这些**不同时刻**的动作平均。
- **temporal ensembling**：把**同一个时刻 $t$**、基于不同时间观测做出的预测平均。

于是 ACT 同时有两个看似矛盾的性质：**一次预测很长的 chunk**（轨迹连贯、有效 horizon 短）+ **每步都重新看观测**（闭环纠错）。权重细节见 [03](03-chunking-and-realtime.md)。

### ACT 剩下的根本问题

同一个 $o$ 下有两条合理 chunk $A^{(1)}$（左绕）、$A^{(2)}$（右绕），如果 $z=0$ 分不开它们，MSE 的解就是：

$$\hat A\approx\frac{A^{(1)}+A^{(2)}}{2}\quad\Rightarrow\quad\text{从中间撞上去}$$

## 3. 为什么 MSE 一定会取平均

固定 $o$，真实动作 $A\sim p(A|o)$，模型输出确定值 $\hat A$：

$$L(\hat A)=\mathbb E\big[(A-\hat A)^2\big],\qquad \frac{\partial L}{\partial\hat A}=\mathbb E\big[2(\hat A-A)\big]=0\ \Rightarrow\ \boxed{\hat A^*=\mathbb E[A\mid o]}$$

一半人左绕 $A_L=-1$、一半右绕 $A_R=+1$：$p(A|o)=0.5\,\delta(A+1)+0.5\,\delta(A-1)$，条件均值 $=0$，正好是直线撞障碍。

$$\boxed{\text{要学的不是一个点 }f(o)\text{，而是整个分布 }p(A\mid o)\text{，再从中采一个}}$$

可运行的双峰实验见 [手撕：diffusion / flow](../08-handwrite/01-diffusion-flow.md)。

## 4. Diffusion Policy：把"图片"换成"动作轨迹"

### 两个时间别混

$$\boxed{t=\text{机器人时间（环境步）}}\qquad\boxed{k=\text{扩散时间（去噪步 }K\to0\text{）}}$$

### 干净样本是一段 action chunk

$$A^0=[a_t,\dots,a_{t+H-1}]\in\mathbb R^{H\times D},\qquad\text{例如 }H=16,\ D=7$$

### 训练

$$A^k=\sqrt{\bar\alpha_k}\,A^0+\sqrt{1-\bar\alpha_k}\,\epsilon,\qquad \mathcal L=\big\|\epsilon-\epsilon_\theta(A^k,O_t,k)\big\|^2$$

手算：$A^0=0.8$，$\epsilon=-0.4$，$\sqrt{\bar\alpha_k}=0.8$，$\sqrt{1-\bar\alpha_k}=0.6$ → $A^k=0.64-0.24=0.40$。网络看到 $0.40$、观测和 $k$，要猜出加进去的噪声 $-0.4$。

网络输入三样：noisy 轨迹 $A^k$、观测条件 $O_t$（如最近 2 帧 RGB + 本体）、扩散步 $k$；输出 $[H,D]$ 的噪声预测。

### 卡点：这里也是 MSE，为什么不再取平均

回归直接学 $o\to A$ 一个点。diffusion 学的是**在动作空间每个 noisy 点上，该往哪个方向走**——一个向量场，可以有多个吸引子：

```text
          ← ← ←     → → →
    ↙ ↙                     ↘ ↘
     ●                       ●
   左绕                     右绕
```

初始噪声落在左边盆地就走到左绕，落在右边就走到右绕。

$$\boxed{\text{采样不是两个模式的平均，而是最终选中一个模式}}$$

MSE 平均的是**噪声的条件期望**，这正好就是指向数据分布的方向（和 score 等价，见 [Diffusion](../09-visual-generation/02-diffusion.md)），不是动作本身的平均。

### 推理

$$A^K\sim\mathcal N(0,I)\ \xrightarrow{\ \epsilon_\theta(\cdot,O_t,K)\ }\ A^{K-1}\ \to\cdots\to\ A^0=[a_t,\dots,a_{t+H-1}]$$

| | OpenVLA | Diffusion Policy |
|---|---|---|
| 为什么要多次 forward | 生成**不同的** token | **同一个完整 $[H,D]$ 张量**被反复细化 |
| 每次 forward 更新 | 1 个 token | $H\times D$ 个值一起（$16\times7=112$） |

步数：原论文训练用 **100** 个扩散步，真机推理用 DDIM 降到约 **16** 步。这就是后面换 flow matching 的动机（1–10 步）。

**工程点**：观测只当条件，不参与扩散。视觉 encoder **算一次**，特征供所有去噪步复用；否则 16 步都跑一次视觉编码器，实时性就没了。

### 三个 horizon 和 receding horizon

| 符号 | 含义 | 典型值 |
|---|---|---|
| $T_o$ | observation horizon：看过去几帧 | 2 |
| $T_p$ | prediction horizon：一次生成几步 | 16 |
| $T_a$ | action / execution horizon：执行几步后重新观测 | 8 |

```text
t=0 预测 a0 ........ a15      执行 a0..a7
t=8 重新观测，预测 a8' ....... a23'   执行 a8'..a15'
```

$$\boxed{\text{既不每步都跑昂贵的扩散，也不把 16 步全部闭眼执行}}$$

$T_a$ 太短缺少时间一致性（容易在模式间切换），太长对扰动反应慢；论文多数任务约 8 步最好。

### 两种 backbone

- **CNN 版**：把 $H$ 当时间维，用 1D temporal U-Net 对动作序列去噪，观测特征作条件（FiLM 调制）。
- **Transformer 版**：noisy action 每步一个 token，观测 embedding 通过 **cross-attention** 注入。

### 为什么比 CVAE 更自然地表达多模态

CVAE 要把复杂分布压进一个有限的 $z$，还得在推理时选 $z$；diffusion 直接在完整的 $H\times D$ 动作空间里学向量场，不用预先规定几个高斯或几个离散模式。

### ⚠️ 它还不是 VLA

没有语言大模型。贡献在**动作生成**，不在语义智能。

## 5. 三条线放在一起

| | OpenVLA | ACT | Diffusion Policy |
|---|---|---|---|
| 定位 | 通用 VLA | 高精度模仿 | 视觉运动策略 |
| 输入 | 单图 + 语言 | 多视角 + 关节 | 最近几帧 + 本体 |
| 动作 | 7 维末端增量 | 双臂关节目标 | 连续轨迹 |
| 输出长度 | 1 步 | chunk | chunk |
| 表示 | 256-bin token | 连续 | 连续 |
| 生成方式 | 自回归 token | 并行一次回归 | 迭代去噪（每步整段并行） |
| loss | CE | MSE + KL | 噪声 MSE |
| 多模态 | 分类分布 | CVAE（推理 $z=0$） | **完整分布** |
| 预训练 VLM | ✅ | ❌ | ❌ |

$$\boxed{\text{第一代 VLA 有语义没控制，ACT / DP 有控制没语义}\ \Rightarrow\ \text{VLM + 生成式 action expert}}$$

再把去噪换成少步的 flow matching，就是 [π0](../01-vla/03-pi0-pi05-g0.md)。

## 自测

**1.** compounding error 的因果链是什么？为什么高频控制更严重？

> **答：** 小动作误差 → 进入专家数据没覆盖的状态（$d_\pi\neq d_{\text{expert}}$）→ 预测误差更大 → 偏移更大，正反馈。10 s 任务在 50 Hz 下要连续决策 500 次，每步一点误差都会累积。

**2.** action chunking 缓解问题的两个理由？

> **答：** ① 有效决策 horizon 缩短约 $k$ 倍（500 步、$k=100$ → 约 5 个决策点）；② 人类演示有时间相关、非 Markov 的行为（停顿 vs 继续），单帧分不清，一次预测一整段能学到运动模式。

**3.** ACT 的 decoder 是怎么输出 chunk 的？和 OpenVLA 解码有什么不同？

> **答：** 给未来每一步一个 query，Transformer decoder **一次 forward 并行**输出 $[k,D]$ 连续值。OpenVLA 是逐 token 自回归解码离散 bin。

**4.** ⭐ ACT 为什么是 CVAE？没有 KL 会怎样？推理时 $z$ 怎么取？

> **答：** 人类演示有差异，用 $z$ 吸收"风格"。encoder 输入未来专家动作和当前关节得到 $z$，decoder 用图像、关节和 $z$ 预测 chunk，loss 为 MSE + $\beta$KL。
> 没有 KL，encoder 会把整段答案塞进 $z$，decoder 不看图也能拟合，测试时没有未来动作就崩。推理时取 **$z=0$**（先验均值），确定性输出，所以 ACT 并没有真正按模式采样。

**5.** temporal ensembling 和普通时间平滑的区别？

> **答：** 普通平滑平均**不同时刻**的动作；temporal ensembling 平均的是**同一时刻 $t$** 由不同时间的观测给出的多个预测，权重 $w_i=e^{-mi}$。它让 ACT 既能预测长 chunk，又能每步闭环纠错。

**6.** 证明 MSE 最优解是条件均值，并用左右绕障碍说明后果。

> **答：** $L=\mathbb E[(A-\hat A)^2]$，求导 $\mathbb E[2(\hat A-A)]=0$ 得 $\hat A^*=\mathbb E[A|o]$。左绕 $-1$、右绕 $+1$ 各一半时均值为 $0$，就是直线撞障碍。

**7.** Diffusion Policy 的训练输入、输出、loss 各是什么？两个"时间"分别指什么？

> **答：** 输入 noisy chunk $A^k=\sqrt{\bar\alpha_k}A^0+\sqrt{1-\bar\alpha_k}\epsilon$、观测条件 $O_t$、扩散步 $k$；输出 $[H,D]$ 噪声预测；loss $\|\epsilon-\epsilon_\theta(A^k,O_t,k)\|^2$。$t$ 是机器人环境步，$k$ 是去噪步，两者无关。

**8.** ⭐ 同样是 MSE，为什么 diffusion 不会像回归那样平均两个模式？

> **答：** 回归学的是 $o\to A$ 一个点；diffusion 在每个 noisy 点上学"往哪走"，整体是一个可以有多个吸引子的向量场。采样从随机噪声出发，落在哪个盆地就收敛到哪个模式。MSE 平均的是噪声的条件期望（等价于 score 方向），不是动作本身。

**9.** Diffusion Policy 和 OpenVLA 都要多次 forward，本质区别是什么？

> **答：** OpenVLA 每次 forward 生成一个**新的** token；Diffusion Policy 每次 forward 把**同一个完整的 $[H,D]$ 张量**整体细化一次（$16\times7$ 个值同时更新），是迭代式并行生成。

**10.** 为什么观测只做条件、视觉特征只算一次？

> **答：** 观测不参与扩散，每个去噪步用同一份条件。视觉编码器算一次、特征复用到约 16 个 DDIM 步，否则每步都要重跑视觉编码，实时推理跑不动。

**11.** $T_o,T_p,T_a$ 分别是什么？典型值？$T_a$ 怎么取舍？

> **答：** 观测 horizon（看几帧，2）、预测 horizon（一次生成几步，16）、执行 horizon（执行几步后重新观测，8），这就是 receding horizon。$T_a$ 太短缺少时间一致性、易在模式间切换；太长对扰动反应慢，论文多数任务约 8 最好。

**12.** 为什么说 OpenVLA 和 ACT / Diffusion Policy 各有一半，第二代 VLA 怎么拼？

> **答：** OpenVLA 有互联网规模语义但动作生成原始（256-bin、单步、慢）；ACT / DP 有连续、高频、chunk 化、多模态的动作生成，但没有语言大模型。第二代把预训练 VLM 当理解模块，接一个生成式 action expert 输出连续 chunk，再把 diffusion 换成少步 flow matching，就是 π0 / G0。
