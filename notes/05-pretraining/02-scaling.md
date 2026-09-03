# Scaling law 与算力分配

## 1. 基本形式

模型 loss 随规模呈**幂律**下降：

$$L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}$$

- $N$：参数量，$D$：训练 token 数
- $E$：不可约误差（数据本身的熵）
- 后两项分别是参数不足和数据不足带来的损失

**关键含义**：loss 是可预测的。可以用一堆小模型拟合出曲线，**外推到大模型**，从而在真正开训之前就知道该用多大模型、多少数据。这是大模型训练能"一次成功"的基础。

## 2. Chinchilla：compute-optimal

给定算力预算 $C\approx 6ND$（训练 FLOPs 的经验公式），怎么分配 $N$ 和 $D$？

Kaplan 早期的结论是"优先加大模型"，Chinchilla 重新做实验后修正为：

$$\boxed{N\ \text{和}\ D\ \text{应该}\ \textbf{等比例}\ \text{增长，约}\ D\approx20N}$$

也就是 1B 参数配 20B token。按这个标准，GPT-3（175B / 300B token）是**严重训练不足**的。

### 但实践早就超过了 20 倍

LLaMA 系列用 **1–2 个数量级更多的 token**（7B 模型训 2T+ token，比例 300:1）。原因：

$$\boxed{\text{Chinchilla 优化的是「训练算力」，实际要优化的是「训练 + 全生命周期推理算力」}}$$

一个模型训一次、推理无数次。小模型多训数据，训练时多花一点，但每次推理都省 —— 总账更划算。所以**推理成本敏感的场景应该训"过度训练"的小模型**。

## 3. MoE 的 scaling

MoE 打破了 $N$ 的单一含义，要区分：

- $N_{\text{total}}$：决定模型容量和**显存**
- $N_{\text{active}}$：决定每 token 的 **FLOPs**

$$\boxed{\text{MoE 用「同样的 FLOPs 买到更大的容量」，代价是显存}}$$

所以 MoE 的 scaling law 要额外引入稀疏度（$N_{\text{total}}/N_{\text{active}}$）这个维度。见 [参数构成](../04-distributed-infra/03-model-param-breakdown.md)。

## 4. 涌现能力的争议

有些能力（多步推理、指令跟随）在小模型上接近随机，到某个规模突然出现，称为**涌现**。

但也有工作指出这可能是**评测指标的假象**：用 exact-match 这类不连续指标会显示"突然出现"，换成 token-level 的连续指标（如 log-likelihood）就是平滑上升的。

面试里提一句这个争议，比单纯说"涌现"更显水平。

## 5. 训练 FLOPs 的估算

$$C\approx 6ND$$

来源：前向每个参数约 2 FLOPs（一次乘一次加），反向约 2 倍于前向，合计 $2+4=6$。

**用途**：估训练时间。

$$\text{时间}=\frac{6ND}{\text{GPU 数}\times\text{单卡 FLOPS}\times\text{MFU}}$$

**MFU**（Model FLOPs Utilization）是实际达到的算力占峰值的比例，大规模训练能做到 **40%–50%** 就算不错。

例：7B 模型训 2T token，$C=6\times7\times10^9\times2\times10^{12}=8.4\times10^{22}$ FLOPs。1024 张 H100（每张 BF16 约 $1\times10^{15}$ FLOPS）、MFU 45%：

$$\frac{8.4\times10^{22}}{1024\times10^{15}\times0.45}\approx1.8\times10^{5}\ \text{s}\approx\textbf{2.1 天}$$

## 自测（口述版）

**1.** 写出 scaling law 的形式，解释三项。它最重要的实践意义是什么？

> **答：** $L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}$。$E$ 是不可约误差（数据本身的熵），第二项是参数不足的损失，第三项是数据不足的损失。
> **实践意义：loss 是可预测的。** 可以用一堆小模型拟合出曲线再外推到大模型，从而在真正开训之前就知道该用多大模型、多少数据 —— 这是大模型训练能「一次成功」的基础。

**2.** Chinchilla 的结论是什么？为什么实践中远超 20:1？

> **答：** 给定算力 $C\approx6ND$，$N$ 和 $D$ 应该**等比例增长**，约 $D\approx20N$（1B 参数配 20B token）。按此标准 GPT-3（175B/300B token）严重训练不足。
> 实践远超是因为：**Chinchilla 优化的是「训练算力」，实际要优化的是「训练 + 全生命周期推理算力」**。模型训一次、推理无数次，小模型多训数据虽然训练时多花一点，但每次推理都省，总账更划算。所以推理成本敏感的场景应该训「过度训练」的小模型（LLaMA 7B 训 2T+ token，比例 300:1）。

**3.** MoE 的 scaling 要额外考虑什么？它用什么换什么？

> **答：** 要区分 $N_{\text{total}}$（决定容量和**显存**）和 $N_{\text{active}}$（决定每 token 的 **FLOPs**），并额外引入稀疏度 $N_{\text{total}}/N_{\text{active}}$ 这个维度。
> **MoE 用「同样的 FLOPs 买到更大的容量」，代价是显存。**

**4.** 涌现能力的争议是什么？

> **答：** 有些能力（多步推理、指令跟随）在小模型上接近随机，到某个规模突然出现，称为涌现。
> 但也有工作指出这可能是**评测指标的假象**：exact-match 这类不连续指标会显示「突然出现」，换成 token-level 的连续指标（如 log-likelihood）就是平滑上升的。面试提一句这个争议比单纯说「涌现」更显水平。

**5.** 写出训练 FLOPs 的估算公式并说明 6 的来源。MFU 是什么，大规模训练的典型值？

> **答：** $C\approx6ND$。来源：前向每个参数约 2 FLOPs（一次乘一次加），反向约为前向的 2 倍，合计 $2+4=6$。
> **MFU**（Model FLOPs Utilization）= 实际达到的算力 ÷ 峰值算力。大规模训练做到 **40%–50%** 算不错，低于 30% 就该查了。

**6.** 估算 7B 模型训 2T token 在 1024 张 H100 上要多久。

> **答：** $C=6\times7\times10^9\times2\times10^{12}=8.4\times10^{22}$ FLOPs。
> 1024 张 H100（BF16 约 $1\times10^{15}$ FLOPS/张）、MFU 45%：
> $$\frac{8.4\times10^{22}}{1024\times10^{15}\times0.45}\approx1.8\times10^5\ \text{s}\approx\mathbf{2.1\ 天}$$

