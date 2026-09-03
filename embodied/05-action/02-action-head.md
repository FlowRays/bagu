# Action head：怎么把表示解码成动作

> **这一节是 VLA 的核心分歧点。** 四种做法的取舍，面试极高频。

## 1. 四种做法

| 做法 | 输出 | 多模态 | 精度 | 推理成本 | 代表 |
|---|---|---|---|---|---|
| **连续回归** | 直接回归动作向量 | ❌ | 受 MSE 限制 | 最低（1 次前向） | 早期 BC |
| **离散 tokenization** | 分 bin 后当 token 预测 | ✅ | 受 bin 数限制 | 中（自回归解码） | RT-2、OpenVLA |
| **Diffusion head** | 去噪出动作序列 | ✅ | 高 | **高**（多步） | Diffusion Policy |
| **Flow matching head** | 解 ODE 出动作序列 | ✅ | 高 | **低**（1–10 步） | π0 / π0.5 |

## 2. 连续回归为什么不够

用 MSE 回归动作：

$$\mathcal L=\|a-f_\theta(o)\|^2\ \Longrightarrow\ f_\theta^*(o)=\mathbb E[a\mid o]$$

**MSE 的最优解是条件均值。** 但动作分布经常是多模态的：

```text
        障碍物
   起点 ──┬──→ 从左绕   (合理)
          └──→ 从右绕   (合理)
    均值:    直接撞上去  (灾难)
```

$$\boxed{\text{多模态分布下，回归输出的"平均动作"可能哪个模式都不是}}$$

这和 [VAE 输出为什么糊](../02-visual-generation/01-generative-models.md#2-vae) 是同一个数学原因。

L1 loss 会给出**条件中位数**，稍好一点（中位数至少是某个真实值附近），但仍然无法表达"两个都行"。

**什么时候回归够用**：动作分布确实单峰的任务（比如跟随一条固定轨迹）。所以别一概而论说回归不行。

## 3. 离散 tokenization

把每一维动作分成 $N$ 个 bin（如 256），bin 下标当 token，接到 VLM 的词表上，用**交叉熵**训练。

```text
动作 (Δx, Δy, Δz, Δrx, Δry, Δrz, grip)
  → 每维分 256 bin
  → 7 个 token
  → 当语言 token 自回归预测
```

**优点**：
- 直接复用 VLM 的全部架构和权重，不用新设计 head
- 交叉熵天然是**分类**，能表达多模态（softmax 可以有多个峰）
- 可以直接享受 LLM 的采样技巧（temperature、top-p）

**缺点**：
- **精度受 bin 数限制**：256 bin 在整个动作范围上分辨率有限；bin 加多则每个 bin 数据变稀
- 每维独立分 bin，**丢掉了维度间的相关性**（自回归解码能部分挽回）
- 7 维动作 × chunk 长度 = 很多 token，**自回归解码慢**
- bin 边界处的量化误差在长 horizon 上累积

### FAST：把动作序列压缩成更少的 token

朴素分 bin 在 action chunk 上会产生大量 token（比如 50 步 × 7 维 = 350 个 token）。

**FAST** 的思路：动作序列在时间上是**平滑**的，高频分量很小。所以
1. 对每一维的时间序列做 **DCT**（离散余弦变换）
2. 量化系数，丢掉高频
3. 对量化后的系数序列做 **BPE** 压缩

$$\boxed{\text{用信号压缩的思路，把动作 token 数降一个量级，自回归才跑得动}}$$

这是把「动作当语言」这条路线做到实用的关键一步。

## 4. Diffusion policy

把**未来 $H$ 步的动作序列**当作要去噪的样本 $x_0\in\mathbb R^{H\times d}$，观测作为条件：

```text
训练: 采 t、采 ε → x_t = √ᾱ·A + √(1-ᾱ)·ε → 最小化 ‖ε − ε_θ(x_t, t, o)‖²
推理: 从纯噪声出发，去噪 K 步 → 得到动作序列 A
```

**优点**：多模态建模能力强、精度高（连续空间，无量化误差）、训练稳（就是 MSE）。

**缺点**：**推理要 K 步**。$K=50$、每步一次 head 前向，在 10–50 Hz 的控制频率下预算根本不够。

细节见 [Diffusion](../02-visual-generation/02-diffusion.md)。

## 5. Flow matching head（现在的主流）

同样把动作序列当作要生成的样本，但用**速度场 + ODE**：

$$\mathcal L=\mathbb E_{t,\epsilon,A}\big[\|v_\theta\big((1-t)\epsilon+tA,\ t,\ o\big)-(A-\epsilon)\|^2\big]$$

推理时从噪声出发欧拉法走几步：

```python
x = randn(H, d)                      # 从噪声出发
for i in range(K):                   # K 通常是 4~10
    t = i / K
    x = x + (1/K) * v_theta(x, t, obs)
action_chunk = x
```

$$\boxed{\text{保留了 diffusion 的多模态能力，但采样从 50 步降到 1–10 步}}$$

这就是 π0 用 flow matching action expert 的直接原因。细节见 [Flow Matching](../02-visual-generation/03-flow-matching.md)。

### Action expert 的结构

π0 的做法是：VLM backbone 处理图像和语言，另外接一个**较小的 action expert**（几百 M 参数），只有它参与那 K 步迭代。

$$\boxed{\text{大 backbone 跑一次，小 expert 跑 K 次}}$$

这是让「大模型 + 生成式 head」满足实时性的关键工程设计 —— 否则 K 步都跑整个 VLM 完全不可能。

## 6. 怎么选

```text
动作分布单峰、要求极低延迟          → 回归
想最大化复用 VLM、能接受自回归延迟   → 离散 tokenization（+ FAST 压缩）
要最高精度、离线或低频场景          → diffusion
要多模态 + 高频实时（大多数 VLA）    → flow matching + action expert
```

## 自测

**1.** 为什么 MSE 回归动作不够？写出数学原因。

> **答：** $\mathcal L=\|a-f_\theta(o)\|^2$ 的最优解是 $f^*_\theta(o)=\mathbb E[a|o]$，即**条件均值**。
> 而动作分布常是多模态的：绕障碍可以从左也可以从右，两者的**平均是直接撞上去**，哪个模式都不是。
> L1 给的是条件中位数，稍好但仍无法表达"两个都行"。**动作分布确实单峰时回归是够用的**，别一概而论。

**2.** 离散 tokenization 的优缺点？

> **答：** 每维分 $N$ 个 bin、bin 下标当 token 接到 VLM 词表上，用交叉熵训练。
> **优点**：直接复用 VLM 全部架构和权重；交叉熵是分类，softmax 能有多个峰所以**能表达多模态**；可以用 temperature / top-p 等采样技巧。
> **缺点**：精度受 bin 数限制（加多则每 bin 数据稀）；每维独立分 bin 丢掉维度相关性；token 数多导致**自回归解码慢**；量化误差在长 horizon 累积。

**3.** FAST 在解决什么？思路是什么？

> **答：** 解决**动作 token 太多**（50 步 × 7 维 = 350 个 token，自回归跑不动）。
> 思路是利用动作序列在时间上**平滑、高频分量小**这个性质：① 对每维时间序列做 **DCT**；② 量化系数、丢高频；③ 对系数序列做 **BPE** 压缩。用信号压缩把 token 数降一个量级。

**4.** diffusion policy 怎么做？它的致命缺点是什么？

> **答：** 把**未来 $H$ 步动作序列**当作要去噪的样本 $x_0\in\mathbb R^{H\times d}$，观测作条件；训练是 $\|\epsilon-\epsilon_\theta(x_t,t,o)\|^2$，推理从噪声去噪 $K$ 步。
> 优点：多模态强、连续空间无量化误差、训练稳。
> **致命缺点：推理要 $K$ 步**，$K=50$ 时在 10–50 Hz 控制频率下预算完全不够。

**5.** flow matching head 的训练目标和推理循环？

> **答：** $\mathcal L=\mathbb E_{t,\epsilon,A}\big[\|v_\theta((1-t)\epsilon+tA,\ t,\ o)-(A-\epsilon)\|^2\big]$。
> 推理：从噪声出发欧拉法走 $K$（4~10）步，`x += (1/K) * v_theta(x, t, obs)`。
> 保留 diffusion 的多模态能力，采样从 50 步降到 1–10 步。

**6.** action expert 的结构设计解决什么？

> **答：** 让「大 VLM + 生成式 head」满足实时性。VLM backbone 处理图像语言**只跑一次**，另接一个较小的 action expert（几百 M）**只有它参与那 K 步迭代**。
> 否则 K 步都跑整个 VLM，延迟完全不可接受。

**7.** 四种 head 怎么选？

> **答：** 动作单峰 + 极低延迟 → **回归**；想最大化复用 VLM、能接受自回归延迟 → **离散 tokenization（+FAST）**；要最高精度、离线或低频 → **diffusion**；要多模态 + 高频实时（大多数 VLA）→ **flow matching + action expert**。
