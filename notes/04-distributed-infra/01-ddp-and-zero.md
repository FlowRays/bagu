# DDP → ZeRO-1/2/3 → FSDP

> 前面 [显存账本](../03-training-fundamentals/01-memory-accounting.md) 里 $P+G+O=12N$ 这一大坨一直没动过。
> checkpointing / accumulation 只能动 $A$。ZeRO 才是专门来切这三座山的。

## 1. DDP：复制模型，切数据

每张 GPU 一份**完整**模型，处理不同数据；backward 后 **All-Reduce** 梯度：

$$g=\tfrac1D\sum_i g_i$$

因为初始参数、梯度、optimizer state 都一样，更新后各卡参数始终一致。

**但每张卡仍然要放：**

$$M_{\text{DDP/device}}=M_P+M_G+M_O+M_A$$

7B 按 12 B/param 就是 **84 GB**，8×80G 也放不下。

$$\boxed{\text{DDP 增加吞吐，不解决 model-state 冗余}}$$

## 2. ZeRO = Zero Redundancy Optimizer

名字就说明一切：消灭 data parallel 里每张卡重复保存的状态。切的顺序是 $O\to G\to P$，因为通常 $O>P\approx G$，先切 optimizer 最划算、通信也最简单。

| 方法 | Parameter | Gradient | Optimizer |
|---|---:|---:|---:|
| DDP | $P$ | $G$ | $O$ |
| ZeRO-1 | $P$ | $G$ | $O/D$ |
| ZeRO-2 | $P$ | $G/D$ | $O/D$ |
| ZeRO-3 | $P/D$ | $G/D$ | $O/D$ |

$$\boxed{\text{ZeRO-1 切 O，ZeRO-2 再切 G，ZeRO-3 再切 P}}$$

### 7B + 8 GPU（$P=14$, $G=14$, $O=56$ GB）

| | 每卡 model states |
|---|---:|
| DDP | $14+14+56=$ **84 GB** |
| ZeRO-1 | $14+14+56/8=$ **35 GB** |
| ZeRO-2 | $14+14/8+56/8=$ **22.75 GB** |
| ZeRO-3 | $84/8=$ **10.5 GB** |

### 各 stage 怎么工作

- **ZeRO-1**：每张卡只持有 $1/D$ 的 optimizer state，因此只负责更新对应的参数分片，更新完再同步让所有卡重新拿到完整参数。forward 时每卡**仍有完整 parameter**。
- **ZeRO-2**：backward 用 **Reduce-Scatter** 代替 All-Reduce，每卡只留下自己负责的 gradient shard —— 正好和它持有的 optimizer shard 对上，`G_i + O_i → 更新 P_i`。
- **ZeRO-3**：连 $P$ 也切了。计算某层前 **All-Gather** 该层参数，算完立刻释放，下一层再 gather。

$$\boxed{\text{ZeRO-3 的核心：需要哪一层，就临时聚合哪一层参数}}$$

## 3. 通信量：为什么 ZeRO-2 是 sweet spot

设一份 parameter/gradient tensor 大小为 $M$。高效 All-Reduce $=$ Reduce-Scatter $+$ All-Gather：

$$C_{\text{DDP}}\approx M+M=2M$$

- **ZeRO-1**：调度优化后仍约 $2M$
- **ZeRO-2**：Reduce-Scatter 梯度 $M$ + step 后 All-Gather 参数 $M$ $=2M$
- **ZeRO-3**：forward All-Gather $M$ + backward All-Gather $M$ + gradient Reduce-Scatter $M$ $=3M$

| | DDP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---:|---:|---:|---:|
| 通信 volume | $2M$ | $\sim2M$ | $2M$ | **$3M$** |
| 相对 DDP | 1× | ~1× | ~1× | **1.5×** |
| Transformer FLOPs | 1× | ≈1× | ≈1× | ≈1× |

$$\boxed{\text{ZeRO-2：显存省很多，通信 volume 却没增加}}$$

### 卡点：ZeRO 换的是通信，不是计算

$$\boxed{\text{ZeRO 主要用通信换显存，基本不增加模型的数学计算量}}$$

它不像 gradient checkpointing 那样把 forward 重算一遍。

### 通信 +50% ≠ 训练时间 +50%

$$T=T_{\text{compute}}+T_{\text{exposed communication}}+\cdots$$

大量通信可以和计算 **overlap**（backward layer 30 的同时 reduce-scatter layer 31 的梯度；计算 layer $i$ 的同时 prefetch layer $i+1$ 的参数）。

经验范围（**不是公式**）：ZeRO-1 约 0–5%、ZeRO-2 约 0–10%、ZeRO-3/FULL_SHARD 约 5–25%；多机 + 网络慢 + 小模型 + overlap 差时可能 20–50%+。

面试稳妥说法：

> ZeRO-1/2 理论通信 volume 和标准 DP 相同，性能开销通常较小；ZeRO-3 因 forward 和 backward 都要 parameter all-gather，总通信从约 $2M$ 增到 $3M$，理论 +50%，但 wall-clock slowdown 取决于通信能否和计算 overlap，不能直接等同于 50%。

## 4. 卡点：activation 不会被 ZeRO 除以 D

ZeRO 只处理 $P,G,O$。activation 取决于 $B_{\text{local}},L,H$，每张 DP rank forward 自己的数据就必须产生自己的 activation：

$$M_{\text{ZeRO3/device}}\approx\frac{P+G+O}{D}+A_{\text{local}}+M_{\text{temp}}$$

$$\boxed{M_A\ \text{不会因为 ZeRO-3 变成 }M_A/D}$$

这就是为什么"开了 ZeRO-3 长 context 还是 OOM" —— 还得配 gradient checkpointing、减 microbatch、FlashAttention。

## 5. 一般开到几

**没有"多卡就自动 ZeRO-X"这回事。** PyTorch DDP 就是 DDP；DeepSpeed 要显式配 `zero_optimization.stage`。

| 情况 | 选择 |
|---|---|
| DDP 能放下 | **DDP** |
| DDP 放不下，但完整 parameter 能放下 | **ZeRO-2** |
| 完整 parameter / states 已成瓶颈 | **ZeRO-3 / FSDP full-shard** |

跳过 ZeRO-1 是因为它和 ZeRO-2 通信同阶，而 ZeRO-2 还多切了 gradient。

如果直接用 PyTorch FSDP，默认 `FULL_SHARD` 本身就是 ZeRO-3-like。

## 6. FSDP1 / FSDP2 是什么

$$\boxed{\text{FSDP1/FSDP2 是 PyTorch 两代实现，不是 stage 1 / stage 2}}$$

不要和 ZeRO-1/ZeRO-2 混。

- **FSDP1**：`FullyShardedDataParallel(model)`，默认 `FULL_SHARD` 时 $P/G/O$ 全 shard，算法上 $\approx$ ZeRO-3。PyTorch 官方说 FSDP 受 ZeRO Stage 3 启发。它的 `ShardingStrategy.SHARD_GRAD_OP` 则被官方描述为 **ZeRO-2-style**，所以"FSDP = ZeRO-3"要限定成 `FULL_SHARD`。
- **FSDP2**：仍是 fully sharded data parallel，主要是**底层表示和 API 重做**：FSDP1 把一组参数 flatten 成一个大 `FlatParameter` 再 shard；FSDP2 用 **DTensor + per-parameter sharding**（每个参数自己在 dim-0 上切），更容易和其他并行组合、更好管理状态。PyTorch 已建议从 FSDP1 迁移到 FSDP2。

生命周期都一样：

```text
pre-forward: All-Gather 本 block 参数 → forward → reshard/free
backward:    All-Gather 参数 → backward → Reduce-Scatter 梯度 → 只留 grad shard
```

$$\boxed{\text{ZeRO-3}\approx\text{FSDP1 FULL\_SHARD}\approx\text{FSDP2 fully\_shard}}$$

思想和主要通信模式一致，但**实现不同，不能说完全等价**。

## 7. 一句话总结

$$\boxed{\text{DDP：复制状态，提高吞吐}}\qquad\boxed{\text{ZeRO：仍是 data parallel，但把冗余状态 shard}}$$

ZeRO **不是**和 data parallel 对立的另一种 parallelism，它就是**更省显存的 data parallel**。

## 自测（口述版）

**1.** DDP 每张卡放什么？它省 model states 吗？7B 每卡多少 GB？

> **答：** 每张卡放**一份完整模型**（$P+G+O$）+ 自己那份 activation，处理不同数据，backward 后 All-Reduce 梯度。
> **不省 model states**。7B 按 12 B/param 就是 **84 GB/卡**，8×80G 也放不下。**DDP 增加吞吐，不解决 model-state 冗余。**

**2.** ZeRO 三个 stage 分别切什么？为什么是 $O\to G\to P$ 这个顺序？

> **答：** ZeRO-1 切 optimizer state，ZeRO-2 再切 gradient，ZeRO-3 再切 parameter。
> 顺序是因为通常 $O>P\approx G$（Adam 的 $m,v$ 是 FP32，8 B/param），**先切最大的最划算**，而且 optimizer 的切分通信复杂度最低。

**3.** 7B + 8 卡，四种方案每卡 model states 各多少？当场算。

> **答：** $P=14$、$G=14$、$O=56$ GB：
> DDP $=14+14+56=$ **84 GB**；ZeRO-1 $=14+14+56/8=$ **35 GB**；ZeRO-2 $=14+14/8+56/8=$ **22.75 GB**；ZeRO-3 $=84/8=$ **10.5 GB**。

**4.** ZeRO-2 的 backward 用什么 collective 代替 All-Reduce？为什么它和 optimizer shard 天然对上？

> **答：** 用 **Reduce-Scatter**：每张卡只留下自己负责的那一片聚合后的 gradient。
> 因为它正好也持有对应的 optimizer shard，于是 `G_i + O_i → 更新 P_i`，分工天然一致，不需要额外通信去凑。

**5.** 推导 DDP / ZeRO-2 / ZeRO-3 的通信量分别是 $2M$ / $2M$ / $3M$。

> **答：** 高效 All-Reduce $=$ Reduce-Scatter $+$ All-Gather，所以 $C_{\text{DDP}}\approx M+M=2M$。
> ZeRO-2：backward 的 Reduce-Scatter 梯度 $M$ + step 后 All-Gather 参数 $M$ $=2M$。
> ZeRO-3：forward All-Gather 参数 $M$ + backward All-Gather 参数 $M$ + gradient Reduce-Scatter $M$ $=3M$，相对 DDP **+50%**。
> 所以 **ZeRO-2 是 sweet spot：显存省很多，通信 volume 却没增加。**

**6.** 「ZeRO 用通信和计算换显存」这句话错在哪？

> **答：** 错在「计算」。**ZeRO 主要用通信换显存，基本不增加模型的数学计算量** —— 它不像 gradient checkpointing 那样把 forward 重算一遍，矩阵乘 FLOPs 基本不变，增加的是 collective communication 和参数生命周期管理。

**7.** 通信 +50% 等于训练慢 50% 吗？为什么？

> **答：** 不等于。$T=T_{\text{compute}}+T_{\text{exposed communication}}+\cdots$，大量通信可以和计算 **overlap**（backward layer 30 的同时 reduce-scatter layer 31 的梯度；算 layer $i$ 的同时 prefetch layer $i+1$ 的参数）。
> 经验：ZeRO-1 约 0–5%、ZeRO-2 约 0–10%、ZeRO-3 约 5–25%；多机 + 网络慢 + 小模型 + overlap 差时可能 20–50%+。

**8.** activation 会被 ZeRO-3 除以 D 吗？为什么？

> **答：** **不会。** ZeRO 只处理 $P,G,O$；activation 取决于 $B_{\text{local}},L,H$，每张 DP rank forward 自己的那份数据就必须产生自己的 activation。
> $M_{\text{ZeRO3/device}}\approx\frac{P+G+O}{D}+A_{\text{local}}+M_{\text{temp}}$。这就是「开了 ZeRO-3 长 context 还是 OOM」的原因，还得配 gradient checkpointing、减 microbatch、FlashAttention。

**9.** FSDP1 和 FSDP2 的区别是什么？FSDP 一定等于 ZeRO-3 吗？

> **答：** **FSDP1/FSDP2 是 PyTorch 的两代实现，不是 stage 1 / stage 2**，别和 ZeRO-1/2 混。
> FSDP1 把一组参数 flatten 成一个大 `FlatParameter` 再 shard；FSDP2 用 **DTensor + per-parameter sharding**（每个参数自己在 dim-0 上切），更容易和其他并行组合、更好管理状态。PyTorch 已建议迁移到 FSDP2。
> **不一定等于 ZeRO-3**：只有 `FULL_SHARD` 才约等于 ZeRO-3；`SHARD_GRAD_OP` 被官方描述为 **ZeRO-2-style**。而且思想一致不代表实现等价。

