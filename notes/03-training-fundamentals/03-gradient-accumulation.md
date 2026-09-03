# Gradient Accumulation：和大 batch 等价吗

> 和 [gradient checkpointing](02-gradient-checkpointing.md) 都降 activation，但**机制完全不同**，这是很容易被追问的点。

## 1. 做法

想要 global batch 64，但显存只放得下 8：

```python
for i in range(8):                # micro batch = 8
    loss = compute_loss(microbatch[i]) / 8
    loss.backward()               # .grad 自动累加
optimizer.step(); optimizer.zero_grad()
```

$$g=\tfrac18g_1+\cdots+\tfrac18g_8=\nabla L_{\text{64 samples}}$$

## 2. 理想情况下确实等价

只要满足：① 样本相同 ② loss normalization 一致 ③ 中途不 step ④ 最后只做一次 step ⑤ scheduler 只走一步 ⑥ **gradient clipping 放在 accumulation 之后**，那么

$$\boxed{8\times\text{microbatch }8\ \equiv\ \text{batch }64}$$

对 SGD、AdamW 都成立。**Adam 不会破坏这个等价**，因为它最终看到的也是同一个聚合梯度 $g$，然后才算 $m_t,v_t$。

## 3. 但不是 bitwise 等价

| 原因 | 说明 | LLM 里是否要紧 |
|---|---|---|
| 浮点加法顺序 | $((a+b)+c)+d\ne(a+b)+(c+d)$ | 数学等价，不保证 bitwise identical |
| Dropout | 随机 mask 顺序不同 | 现代模型多数 dropout=0，通常没问题 |
| BatchNorm | $\mu_B,\sigma_B$ 依赖 batch 内统计，$\mu_{64}\ne\mu_8$ | **真不等价**，但 Transformer 用 LayerNorm/RMSNorm，不受影响 |

$$\boxed{\text{数学等价}\ne\text{bitwise identical}}$$

## 4. 卡点：真正的坑是 token normalization

变长 SFT 里这个比"数学等价吗"重要得多。

```text
microbatch 1: 1000 valid tokens
microbatch 2: 3000 valid tokens
```

如果各自算 $L_1=\frac1{1000}\sum l_i$、$L_2=\frac1{3000}\sum l_i$ 再取 $\frac{L_1+L_2}{2}$，等于给两个 microbatch **各 50% 权重**。

但真正把 4000 token 放一个 batch 时：

$$L=\frac{\sum_{1}^{1000}l_i+\sum_{1}^{3000}l_i}{4000}$$

第二个 microbatch 应该占 **75%**，第一个 25%。

$$\boxed{\text{要按总 valid token 数 normalize，而不是对每个 microbatch 的 loss 求平均}}$$

（这和 [DAPO 的 token-level loss 聚合](../06-post-training/rl/08-dapo.md) 是同一类问题。）

## 5. 省的是哪块显存

| | 变化 |
|---|---|
| parameter | 不变 |
| optimizer | 不变 |
| **gradient** | **不变** —— 仍要为整个模型维护一份 grad buffer，只是往同一块累加 |
| **activation** | $\propto B_{\text{micro}}$，**大幅下降** |

## 6. 和 checkpointing 的本质区别

| | Checkpointing | Gradient Accumulation |
|---|---|---|
| 降低 activation | ✅ | ✅ |
| P / G / O | 不变 | 不变 |
| 方法 | **recompute** | **拆 microbatch** |
| 有效 batch size | 不变 | 可保持不变 |
| 理论 FLOPs | **增加约 33%** | **基本不增加** |
| 降低吞吐 | ✅ | ✅（microbatch 太小 → GPU 利用率下降） |

$$\boxed{\text{Checkpointing = 用 recomputation 换 activation memory}}$$
$$\boxed{\text{Accumulation = 用 parallelism 换 activation memory}}$$

同样的数据量，checkpointing 把**同一个 token 的 forward 算了两次**；accumulation 处理的样本总数不变，只是从并行改成串行。

## 7. 两者正交，可以叠加

$$M_A\approx \underbrace{B_{\text{micro}}}_{\text{accumulation 降这个}}\times\underbrace{\text{每个 token 保存的 activation}}_{\text{checkpointing 降这个}}$$

这个视角非常好用。

## 自测（口述版）

**1.** 写出 gradient accumulation 的代码骨架，说明 `/8` 放在哪、为什么。

> **答：** ```python
> for i in range(8):                # micro batch = 8
>     loss = compute_loss(microbatch[i]) / 8
>     loss.backward()               # .grad 自动累加
> optimizer.step(); optimizer.zero_grad()
> ```
> `/8` 放在 backward 之前，因为目标是 $g=\frac18g_1+\cdots+\frac18g_8=\nabla L_{\text{64 samples}}$，即整批的**平均**梯度。

**2.** 它和大 batch 数学等价吗？需要哪些条件？Adam 会破坏等价吗？

> **答：** 理想情况**等价**。条件：① 样本相同；② loss normalization 相同；③ 中途不 step；④ 最后只做一次 step；⑤ scheduler 也只走一步；⑥ **gradient clipping 在 accumulation 之后做**。
> **Adam 不会破坏等价**，因为它最终看到的也是同一个聚合梯度 $g$，然后才算 $m_t,v_t$。

**3.** 哪三种情况会导致不是 bitwise 等价？其中哪一种在 Transformer 里其实不用担心？

> **答：** ① **浮点加法顺序**：$((a+b)+c)+d\ne(a+b)+(c+d)$，数学等价但不 bitwise identical；
> ② **Dropout**：随机 mask 顺序不同（现代模型多数 dropout=0，通常不存在这个问题）；
> ③ **BatchNorm**：依赖 batch 内统计量，$\mu_{64}\ne\mu_8$，**真的不等价** —— 但 Transformer 用 LayerNorm/RMSNorm，**不受影响**。

**4.** 变长 SFT 里 token normalization 为什么会错？举 1000/3000 token 的例子。

> **答：** microbatch 1 有 1000 个 valid token、microbatch 2 有 3000 个。若分别算 $L_1=\frac1{1000}\sum l_i$、$L_2=\frac1{3000}\sum l_i$ 再取 $\frac{L_1+L_2}{2}$，等于给两者**各 50% 权重**；
> 但真正把 4000 个 token 放进一个 batch 时 $L=\frac{\sum^{1000}l_i+\sum^{3000}l_i}{4000}$，第二个应该占 **75%**、第一个 25%。
> 所以要**按总 valid token 数 normalize**，而不是对每个 microbatch 的 loss 求平均。

**5.** 它降的是哪一项显存？gradient buffer 会变小吗？

> **答：** 只降 **activation**（$\propto B_{\text{micro}}$）。parameter、optimizer 不变；**gradient 也不变** —— 仍要为整个模型维护一份 grad buffer，只是不断往同一块累加。

**6.** 它和 checkpointing 在「用什么换显存」上的区别是什么？

> **答：** **Checkpointing = 用 recomputation 换 activation memory**：同一个 token 的 forward 被算了两次，理论 FLOPs 真的增加约 33%。
> **Accumulation = 用 parallelism 换 activation memory**：处理的样本总数不变，只是从并行改成串行，理论 FLOPs 基本不增加，但 microbatch 太小会降低 GPU 利用率。
> 两者正交可叠加：$M_A\approx B_{\text{micro}}\times(\text{每 token 保存的 activation})$，accumulation 降左边，checkpointing 降右边。

