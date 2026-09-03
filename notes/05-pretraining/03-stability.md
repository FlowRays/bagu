# 训练稳定性

> 大规模预训练最贵的失败是**训到一半发散**。现代 LLM 的很多结构设计（pre-norm、RMSNorm、QK-norm、去 bias）
> 治的都是稳定性而不是效果，见 [LLM 架构](../01-llm-arch/00-map.md#必须能当场答出来的十条)。

## 1. Loss spike

**现象**：训练曲线平稳下降中突然出现尖峰，有时能自己恢复，有时直接发散成 NaN。

**常见原因**：

| 原因 | 说明 |
|---|---|
| 脏数据 | 某个 batch 里有异常样本（超长重复串、乱码） |
| **attention logits 爆炸** | $q^\top k$ 尺度失控 → softmax 饱和 |
| 梯度爆炸 | 深层网络累积 |
| 数值溢出 | FP16 的动态范围不够 |
| lr 过大或 warmup 太短 | |

**应对**：

- **gradient clipping**（global norm，最基本的保险丝，见 [gradient clipping](../03-training-fundamentals/04-packing-and-grad-clip.md#二gradient-clipping)）
- **QK-norm**：直接锁住 attention logits 的尺度
- **BF16 而不是 FP16**：exponent 位宽和 FP32 一样，动态范围大得多
- **跳过异常 batch**：检测到 loss 或 grad norm 异常就跳过这一步
- **回滚**：从最近的健康 checkpoint 恢复，跳过引发 spike 的数据段

$$\boxed{\text{监控 grad norm 比监控 loss 更早发现问题}}$$

## 2. 数值精度

| 精度 | 用在哪 |
|---|---|
| **BF16** | 权重、激活、通信（现在的默认） |
| **FP32** | optimizer state（Adam 的 $m,v$）、master weight、loss 累加、norm 的统计量 |
| FP8 | 前沿的训练尝试（Hopper 之后） |

**为什么 BF16 而不是 FP16**：BF16 的 exponent 是 8 位（和 FP32 一样），动态范围大，几乎不会溢出；FP16 只有 5 位，训练大模型时经常要配 loss scaling 才能不溢出。代价是 BF16 的尾数只有 7 位，精度低，所以关键累加要在 FP32 做。

细节见 [显存账本](../03-training-fundamentals/01-memory-accounting.md#2-为什么有的-2-byte-有的-4-byte)。

## 3. 初始化与 lr schedule

**初始化**：标准做法是 $\mathcal N(0,\sigma^2)$ 且 $\sigma\propto 1/\sqrt{d}$，残差分支的输出投影再额外缩放 $1/\sqrt{2N_{\text{layer}}}$，让残差累加后方差不随深度增长。

**lr schedule**：

```text
warmup（线性升到峰值，通常总步数的 1%~3%）
  → cosine 或 WSD 衰减
```

**为什么必须 warmup**：训练最开始 Adam 的二阶动量 $v$ 估计还不准（$v\approx0$ 导致 $m/\sqrt v$ 很大），直接用大 lr 会一步毁掉初始化。

**WSD（Warmup-Stable-Decay）** 现在很流行：warmup → 长时间恒定 lr → 最后快速衰减。好处是恒定段可以随时加长（不用预先定总步数），而且衰减段正好配合 [数据退火](01-data.md#3-数据配比与课程)。

## 4. muP

**问题**：小模型上调好的超参（尤其 lr），换到大模型上不再最优，每换一次规模都要重调，极贵。

**muP（Maximal Update Parametrization）**：重新设计初始化和 lr 随宽度的缩放规则，使得**最优超参在不同宽度下保持不变**。

$$\boxed{\text{在小模型上调超参，直接迁移到大模型，省掉大规模超参搜索}}$$

这是大模型训练里非常实用的一项工程。

## 5. 该监控什么

| 指标 | 看什么 |
|---|---|
| **loss** | 平滑下降，无尖峰 |
| **grad norm** | 稳定；突然变大是发散前兆（比 loss 更早） |
| **lr** | 确认 schedule 符合预期 |
| **激活 / logits 的最大值** | 检测数值爆炸 |
| **MFU** | 掉了说明有性能问题（通信、数据加载） |
| **各 expert 的负载**（MoE） | 检测负载不均和 expert collapse |
| **token 消耗速度** | 数据管线是否成为瓶颈 |

## 自测（口述版）

**1.** loss spike 的常见原因？五种应对手段？为什么监控 grad norm 比 loss 更好？

> **答：** 原因：脏数据（超长重复串、乱码）、**attention logits 爆炸**、梯度爆炸、数值溢出、lr 过大或 warmup 太短。
> 应对：① gradient clipping（global norm，最基本的保险丝）；② QK-norm 锁住 attention logits 尺度；③ 用 BF16 而不是 FP16；④ 检测到异常就跳过该 batch；⑤ 从最近的健康 checkpoint 回滚并跳过引发 spike 的数据段。
> **grad norm 在崩之前就会先飙升**，比 loss 更早暴露问题。

**2.** 为什么用 BF16 而不是 FP16？代价是什么？哪些地方必须用 FP32？

> **答：** BF16 的 exponent 是 8 位（和 FP32 一样），**动态范围大、几乎不会溢出**；FP16 只有 5 位，训练大模型经常要配 loss scaling 才不溢出。
> 代价是 BF16 尾数只有 7 位、**精度低**。
> 必须用 FP32 的地方：optimizer state（Adam 的 $m,v$）、master weight、loss 累加、norm 的统计量。

**3.** 残差分支的初始化为什么要额外缩放 $1/\sqrt{2N_{\text{layer}}}$？

> **答：** 因为 residual stream 是逐层**累加**的：$x_L=x_0+\sum_l\text{Sublayer}_l$。若每层输出方差都是 1，累加 $N$ 层后方差会随深度线性增长。
> 把残差分支的输出投影缩放 $1/\sqrt{2N_{\text{layer}}}$，可以让累加后的方差不随深度增长，深层网络才训得稳。

**4.** 为什么必须 warmup？用 Adam 的二阶动量解释。

> **答：** 训练最开始 Adam 的二阶动量 $v$ 估计还不准（$v\approx0$），导致更新量 $m/\sqrt{v}$ 非常大。此时若直接用峰值 lr，一步就能毁掉初始化。
> warmup 用很小的 lr 走一段，等 $m,v$ 的统计量稳定下来再升到峰值。典型 warmup 是总步数的 1%–3%。

**5.** WSD schedule 是什么？两个好处？

> **答：** **Warmup-Stable-Decay**：warmup → 长时间**恒定 lr** → 最后快速衰减。
> 好处：① 恒定段可以**随时加长**，不用预先确定总步数（cosine 必须一开始就定好）；② 衰减段正好配合**数据退火**，在低 lr 时喂高质量数据，两者叠加收益最大。

**6.** muP 解决什么问题？

> **答：** 解决「小模型上调好的超参（尤其 lr）换到大模型上不再最优，每换一次规模都要重调」的问题。
> **muP（Maximal Update Parametrization）** 重新设计初始化和 lr 随宽度的缩放规则，使**最优超参在不同宽度下保持不变**，于是可以在小模型上调好直接迁移到大模型，省掉极贵的大规模超参搜索。

**7.** 大规模训练该监控哪些指标？MoE 要额外看什么？

> **答：** loss（平滑下降无尖峰）、**grad norm**（发散前兆）、lr（确认 schedule）、激活/logits 的最大值（数值爆炸）、MFU（性能问题）、token 消耗速度（数据管线瓶颈）。
> MoE 还要额外看**各 expert 的负载分布**，检测负载不均和 expert collapse。

