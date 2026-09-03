# Gradient Checkpointing（activation recomputation）

> 它**只**动一块显存：$M_A$。parameter / gradient / optimizer state 完全不变。
> 更准确的名字其实是 **activation recomputation**。

## 1. 核心想法

普通训练：**forward 时全存，backward 时直接用。**

```text
L1 → 存 a1 ；L2 → 存 a2 ；… ；L8 → 存 a8   （全部一直留着）
```

Checkpointing：**只存少量边界节点，其余丢掉；backward 需要时重新算。**

```text
x → L1 → L2 → [存 a2] → L3 → L4 → [存 a4] → L5 → L6 → [存 a6] → L7 → L8
```

backward 到 $L_3,L_4$ 时，从 $a_2$ 重新 forward 一遍得到内部 activation，backward 完立刻释放。

$$\boxed{\text{不存 activation}\iff\text{之后重新 forward}}$$

## 2. 实际实现：以 Transformer block 为单位

LLM 里最常见的**不是**教科书的"每 $\sqrt N$ 层放一个 checkpoint"，而是：

```python
from torch.utils.checkpoint import checkpoint
for layer in model.layers:
    hidden = checkpoint(layer, hidden, use_reentrant=False)
```

HuggingFace 一行：`model.gradient_checkpointing_enable()`。

$$\boxed{\text{保存 block 边界 hidden，block 内部 activation 重算}}$$

块内的 Q/K/V、MLP intermediate、softmax 中间量都不保存。

粒度是可调的：不 checkpoint / 每个 block / 几个 block 一组 / 更复杂的 recursive schedule。但普通使用者基本就是开个配置，具体策略由框架和 infra 决定。**注意它通常不是默认开启的**，因为确实牺牲吞吐。

## 3. 卡点：能不能一个都不存

理论上可以：forward 只留最初的 $x$，backward 每一层都从头重算。此时 activation 接近 $O(1)$，但 recomputation 是

$$N+(N-1)+\cdots+1=O(N^2)$$

完全不划算。所以实际做法是在这个 trade-off 上取中点：每个 block 通常**只多 forward 一次**，这才是额外计算只有约 33% 而不是 $O(N^2)$ 的原因。

也不可能真降到 0：至少要留 checkpoint 边界输入、当前正在 backward 的那段 activation、RNG state 等。

## 4. 省多少显存

只影响 $M_A$，所以**总收益取决于 activation 原本占多大比例**：

```text
model states  40 GB          model states  30 GB
activation    30 GB          activation    50 GB
total         75 GB          →  长序列 / 大 microbatch 时收益大得多
      ↓ checkpoint
activation  8~15 GB
total      53~60 GB   （总显存 ↓ 20%~30%）
```

经验区间：

$$\boxed{\text{activation memory 下降约 }40\%\!-\!70\%}\qquad\boxed{\text{总显存下降约 }20\%\!-\!40\%}$$

不要背"gradient checkpointing 省 50% 显存"。

## 5. 增加多少计算

粗估 forward $=F$，backward $\approx2F$：

$$\text{普通：}F+2F=3F\qquad\text{full checkpoint：}F+F+2F=4F$$

$$\boxed{\text{理论额外 FLOPs}\approx\frac{4F-3F}{3F}=33\%}$$

但 wall-clock 通常只慢 $10\%\sim30\%$（forward 不一定占 1/3 时间、通信可重叠、kernel 利用率、FlashAttention、checkpoint 范围不同）。

面试记两个数：

$$\boxed{\text{理论 compute }+33\%}\qquad\boxed{\text{实测 wall-clock 慢 }15\%\!-\!30\%}$$

## 6. Selective checkpointing

不是非黑即白。现代训练会挑：**哪些 activation 占显存大、但重算又便宜**（LayerNorm、激活函数、部分 MLP intermediate）去 recompute，昂贵的算子结果则保留。本质是在

$$\text{memory saving}\quad\text{vs}\quad\text{recompute FLOPs}$$

之间找最优点。

## 7. 不影响 P / G / O

- **parameter**：$W_1,\dots,W_N$ 无论如何都得在 → 不变
- **optimizer**：Adam 的 $m,v$ 每个参数一份 → 不变
- **gradient**：backward 最终还是要得到 $\nabla_\theta L$ → 不变

$$\boxed{\text{Gradient Checkpointing 只动 Activation}}$$

所以如果 `model states = 110GB, activation = 10GB`，checkpoint 救不了你，那时候要上的是 [ZeRO](../04-distributed-infra/01-ddp-and-zero.md)。

## 自测（口述版）

**1.** checkpoint 存的是什么、丢的是什么？backward 怎么恢复？

> **答：** 存的是 **block 边界的 hidden**（某个计算段的输入），丢的是 block 内部的 activation（Q/K/V、MLP intermediate、softmax 中间量）。
> backward 到该 block 时，从保存的 input 重新 forward 一次得到内部 activation，backward 完立刻释放。所以更准确的名字是 **activation recomputation**。

**2.** LLM 里最常见的 checkpoint 粒度是什么？写出 PyTorch 的两行实现。

> **答：** 以**一个 Transformer block** 为单位（不是教科书里「每 $\sqrt N$ 层放一个」）。
> ```python
> for layer in model.layers:
>     hidden = checkpoint(layer, hidden, use_reentrant=False)
> ```
> HuggingFace 一行 `model.gradient_checkpointing_enable()`。注意它**通常不是默认开启**的，因为会牺牲吞吐。

**3.** 如果什么都不存、每次从头重算，activation 和计算量各是什么量级？为什么不这么做？

> **答：** activation 可以接近 $O(1)$，但 recomputation 是 $N+(N-1)+\cdots+1=O(N^2)$ 次 forward，完全不划算。
> 实际做法是每个 block **只多 forward 一次**，这才是额外计算只有约 33% 而不是 $O(N^2)$ 的原因。也不可能真降到 0：至少要留 checkpoint 边界输入、当前正在 backward 的那段 activation、RNG state。

**4.** 它能省多少显存？为什么不能给一个固定百分比？

> **答：** 因为它**只影响 $M_A$**，总收益取决于 activation 原本占总显存多大比例。
> 经验：**activation 下降约 40%–70%，总显存下降约 20%–40%**（长 context / 大 microbatch 时更明显）。不要背「省 50% 显存」。
> 如果 model states 110 GB、activation 只有 10 GB，checkpoint 救不了，那时候要上 ZeRO。

**5.** 推导额外计算约 33%。为什么实测 wall-clock 往往低于这个数？

> **答：** 粗估 forward $=F$、backward $\approx2F$。普通训练 $F+2F=3F$；full checkpoint 是 $F+F+2F=4F$，额外 $\frac{4F-3F}{3F}=33\%$。
> 实测通常只慢 10%–30%，因为 forward 不一定占总时间 1/3、通信可以重叠、kernel 利用率、FlashAttention、checkpoint 范围不同。记两个数：**理论 +33%，实测慢 15%–30%**。

**6.** 它对 parameter / gradient / optimizer 有影响吗？为什么？

> **答：** **完全没有。** 参数无论如何都必须在；Adam 的 $m,v$ 每个参数一份；backward 最终还是要得到 $\nabla_\theta L$。
> 所以记死：**Gradient Checkpointing 只动 Activation。**

