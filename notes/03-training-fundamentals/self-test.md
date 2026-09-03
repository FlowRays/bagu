# 显存 / 训练工程 / 分布式 自测题库（关掉笔记用）

> 覆盖 [显存账本](01-memory-accounting.md)、[checkpointing](02-gradient-checkpointing.md)、[accumulation](03-gradient-accumulation.md)、[packing & clipping](04-packing-and-grad-clip.md)、
> [DDP/ZeRO](../04-distributed-infra/01-ddp-and-zero.md)、[并行全图](../04-distributed-infra/02-parallelism-map.md)、[参数构成](../04-distributed-infra/03-model-param-breakdown.md)、[算法显存对比](../06-post-training/memory-sft-opd-rl.md)。
>
> 侧边栏顶部有 **「答案」开关**。标 ⭐ 的是高频考点。

## A. 显存账本

**1.** ⭐ 写出 SFT 显存的两大类拆分，各自和什么有关。

> **答：** $M=\underbrace{M_P+M_G+M_O}_{\text{model states}}+\underbrace{M_A+M_{\text{logits}}+M_{\text{temp}}}_{\text{runtime}}$。
> model states 只和**参数量 $N$** 有关；runtime 只和 $B,L,H,N_{\text{layer}},V$ 有关。这个分离是所有显存直觉的地基。

**2.** FP32 / FP16 / BF16 各几 byte？BF16 的优势和代价？

> **答：** 4 / 2 / 2 byte（$8\text{ bits}=1\text{ Byte}$）。BF16 的 exponent 和 FP32 一样是 8 位（FP16 只有 5），**动态范围大、不容易 overflow**，所以现在 LLM 训练更常用；代价是 mantissa 只有 7 位（FP16 有 10），精度低。

**3.** ⭐⭐ 12 还是 16 bytes/param？7B 各多少 GB？

> **答：** BF16 param 2 + BF16 grad 2 + Adam $m$ 4 + Adam $v$ 4 = **12 B/param**，7B → **84 GB**；再加 FP32 master weight 4 = **16 B/param**，7B → **112 GB**。
> 正确说法：**经典 mixed-precision Adam 估算通常是 12–16 Byte/parameter，具体取决于 gradient precision 和是否维护 FP32 master weights。** 别说成"一定 16"。
> 顺带记死：**BF16 裸权重 1B ≈ 2GB**（7B≈14GB，32B≈64GB，70B≈140GB）。

**4.** Adam 为什么占 8 B/param？

> **答：** 它为每个参数维护一阶动量 $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$ 和二阶动量 $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$，通常用 **FP32** 保数值稳定，$4+4=8$。

**5.** ⭐ 为什么 backward 必须保存 forward 的 activation？

> **答：** $y=Wx$ 的 backward 是 $\frac{\partial L}{\partial W}=\frac{\partial L}{\partial y}x^\top$，**需要 forward 的输入 $x$**。所以 autograd 建 computation graph 保留中间量。Transformer 一层要存 input hidden、normalized hidden、Q/K/V、attention output、MLP intermediate（$H\to3H\sim4H$，比 hidden 还大）。

**6.** activation 正比于什么？估算一份 BF16 hidden。

> **答：** $M_A\propto B\times L\times H\times N_{\text{layer}}$。$B=4,L=8192,H=4096$ 时一份 $[B,L,H]$ BF16 是 $4\times8192\times4096\times2\approx268$ MB —— **仅仅一份 tensor**，再乘每层若干份、乘几十层，很快几十 GB。

**7.** ⭐ FlashAttention 把什么从 $O(L^2)$ 降到 $O(L)$？什么没变？

> **答：** naive attention 的 $QK^\top\in\mathbb R^{B\times h\times L\times L}$ 显存是 $O(L^2)$。FlashAttention 分块计算，**不把完整 $L\times L$ 矩阵写进 HBM**，把 attention 显存降到接近 $O(L)$。**计算复杂度仍然是 $O(L^2)$**，省的是显存和 IO。

**8.** logits 为什么可能有 10 GB？

> **答：** $Z\in\mathbb R^{B\times L\times V}$，$B=4,L=8192,V=150\text{K}$、BF16 → $4\times8192\times150000\times2\approx9.8$ GB。框架用 fused cross entropy / chunked loss / 及时释放来避免长期保存完整 logits。

**9.** batch 1→8（seq 不变），哪些显存变、哪些不变？

> **答：** model states（P/G/O）**完全不变**，activation 大幅增加。反过来 7B→14B 时 model states 约 ×2，activation 也因 $H$、$N_{\text{layer}}$ 变大而增加。**参数量和 batch/seq 影响的是不同东西。**

## B. Gradient Checkpointing

**10.** ⭐ checkpoint 存什么、丢什么？backward 怎么恢复？

> **答：** 存 **block 边界 hidden**，丢 block 内部 activation（Q/K/V、MLP intermediate、softmax 中间量）。backward 到该 block 时从保存的 input 重新 forward 一次得到内部 activation，backward 完立刻释放。所以更准确的名字是 **activation recomputation**。

**11.** LLM 里最常见的粒度？写出实现。

> **答：** 以**一个 Transformer block** 为单位（不是教科书的每 $\sqrt N$ 层）。
> ```python
> for layer in model.layers:
>     hidden = checkpoint(layer, hidden, use_reentrant=False)
> ```
> HuggingFace 一行 `model.gradient_checkpointing_enable()`。粒度可调（不 checkpoint / 每 block / 几个 block 一组 / recursive schedule），但普通使用者基本就是开配置。**注意它通常不是默认开启的**，因为牺牲吞吐。

**12.** ⭐ 如果什么都不存、每次从头重算会怎样？

> **答：** activation 可以接近 $O(1)$，但 recomputation 是 $N+(N-1)+\cdots+1=O(N^2)$，完全不划算。实际做法是每个 block **只多 forward 一次**，这才是额外计算约 33% 而不是 $O(N^2)$ 的原因。也不可能真降到 0：至少要留 checkpoint 边界输入、当前正在 backward 的那段 activation、RNG state。

**13.** ⭐ 能省多少显存？为什么不能给固定百分比？

> **答：** 它只影响 $M_A$，所以总收益取决于 activation 原本占多大比例。经验：**activation ↓ 约 40–70%，总显存 ↓ 约 20–40%**（长 context / 大 microbatch 时更明显）。不要背"省 50% 显存"。
> 如果 `model states = 110GB, activation = 10GB`，checkpoint 救不了，那时候要上 ZeRO。

**14.** ⭐ 推导额外计算约 33%。实测为什么更低？

> **答：** forward $=F$、backward $\approx2F$。普通 $F+2F=3F$；full checkpoint $F+F+2F=4F$，$\frac{4F-3F}{3F}=33\%$。
> 实测 wall-clock 通常只慢 10–30%：forward 不一定占 1/3 时间、通信可重叠、kernel 利用率、FlashAttention、checkpoint 范围不同。记两个数：**理论 compute +33%，实测慢 15–30%**。

**15.** 它对 P/G/O 有影响吗？

> **答：** 完全没有。参数必须都在；Adam 的 $m,v$ 每个参数一份；backward 最终还是要得到 $\nabla_\theta L$。**Gradient Checkpointing 只动 Activation。**

## C. Gradient Accumulation

**16.** ⭐ 它和大 batch 数学等价吗？需要什么条件？Adam 会破坏吗？

> **答：** 理想情况**等价**。条件：样本相同、loss normalization 一致、中途不 step、最后只做一次 step、scheduler 只走一步、**gradient clipping 放在 accumulation 之后**。
> **Adam 不破坏等价**，因为它最终看到的也是同一个聚合梯度 $g$，然后才算 $m_t,v_t$。

**17.** 哪三种情况不是 bitwise 等价？哪个在 Transformer 里不用担心？

> **答：** ① 浮点加法顺序（$((a+b)+c)+d\ne(a+b)+(c+d)$）—— 数学等价但不 bitwise identical；② Dropout 随机 mask 顺序不同 —— 现代模型多数 dropout=0；③ **BatchNorm** 依赖 batch 内统计 $\mu_{64}\ne\mu_8$，**真不等价** —— 但 Transformer 用 LayerNorm/RMSNorm，**不受影响**。

**18.** ⭐⭐ 变长 SFT 里 token normalization 为什么会错？

> **答：** microbatch 1 有 1000 valid token、microbatch 2 有 3000。分别算 $L_1=\frac1{1000}\sum l_i$、$L_2=\frac1{3000}\sum l_i$ 再取 $\frac{L_1+L_2}{2}$，等于给两者**各 50% 权重**；但真正放一个 batch 时 $L=\frac{\sum^{1000}l_i+\sum^{3000}l_i}{4000}$，第二个应占 **75%**。
> **要按总 valid token 数 normalize，而不是对每个 microbatch 的 loss 求平均。** 这比"数学等价吗"更值得注意。

**19.** 它降的是哪一项？gradient buffer 会变小吗？

> **答：** 只降 activation（$\propto B_{\text{micro}}$）。parameter / optimizer 不变；**gradient 也不变** —— 仍要为整个模型维护一份 grad buffer，只是往同一块累加。

**20.** ⭐ 它和 checkpointing 在"用什么换显存"上的区别？

> **答：** **Checkpointing = 用 recomputation 换 activation memory**（同一个 token 的 forward 算了两次，理论 FLOPs +33%）；**Accumulation = 用 parallelism 换 activation memory**（处理的样本总数不变，只是从并行改成串行，理论 FLOPs 基本不增加，但 microbatch 太小会降 GPU 利用率）。
> 两者正交可叠加：$M_A\approx B_{\text{micro}}\times(\text{每 token 保存的 activation})$，accumulation 降左边，checkpointing 降右边。

## D. Packing 与 Gradient Clipping

**21.** ⭐ packing 需要哪些配套？少了会怎样？

> **答：** ① block-diagonal causal attention mask / `cu_seqlens`；② **`position_ids` 每条 sequence 从 0 重新开始**；③ loss mask 不算跨样本交界处的 next-token prediction；④ loss normalization 一致。
> 理想实现下 $L_{\text{packed}}=L_A+L_B+L_C$，梯度也相同，**packing 只改变怎么把数据塞进 GPU，不改变训练目标**。但如果 position 没 reset、或不同 sequence 之间能互相 attention，就**不等价**。

**22.** ⭐ 写出 global norm clipping，用 grad=(3,4)、$c=1$ 算一遍。

> **答：** $g'=g\cdot\min\big(1,\frac{c}{\|g\|_2}\big)$，$\|g\|_2=\sqrt{\sum g_i^2}$。
> $\|g\|=\sqrt{3^2+4^2}=5$，$c=1$ → 全部 ×1/5 → $(0.6,0.8)$，新 norm $=\sqrt{0.36+0.64}=1$。
> **所有参数乘同一个系数，所以 $\frac{g'}{\|g'\|}=\frac{g}{\|g\|}$，方向完全不变，只缩短长度。** $\|g\|\le c$ 时完全不处理 —— 它是保险丝，不是每步主动缩小。

**23.** norm clipping 和 value clipping 的区别？

> **答：** "每个 gradient 超过 1 就设成 1"是 **value clipping**；LLM 训练里更常见的是 **global norm clipping**（按整体 L2 norm 等比例缩放）。

**24.** ⭐ clipping 会导致梯度消失吗？

> **答：** **不会。** 它只在 $\|g\|>c$ 时触发，缩放后 norm 恰好等于 $c$（比如 1.0），离浮点下溢很远。真正的 vanishing gradient 是深层链式相乘的指数衰减 $\frac{\partial L}{\partial x_1}=\frac{\partial L}{\partial x_n}\prod\frac{\partial x_{i+1}}{\partial x_i}$，是另一个问题。
> 但**阈值设太小确实拖慢训练**：正常 $\|g\|\approx10$ 而设 $c=0.01$ 等于每步缩小约 1000 倍，有效 step size 长期过小。这叫收敛变慢，不叫 gradient vanishing。

**25.** FSDP / ZeRO 下 global grad norm 怎么算？

> **答：** 梯度被 shard 到不同 GPU，**不能每张卡各算各的**。必须跨卡聚合各 shard 的平方和得到真正的 global grad norm，再统一缩放。

## E. DDP / ZeRO / FSDP

**26.** ⭐ DDP 省 model states 吗？

> **答：** **不省。** 每张卡一份完整模型，backward 后 All-Reduce 梯度 $g=\frac1D\sum g_i$。每卡仍要 $M_P+M_G+M_O+M_A$，7B 按 12 B/param 就是 84 GB。**DDP 增加吞吐，不解决 model-state 冗余。**

**27.** ⭐⭐ ZeRO 三个 stage 切什么？为什么是这个顺序？7B+8卡各多少？

> **答：** ZeRO-1 切 O，ZeRO-2 再切 G，ZeRO-3 再切 P。顺序是因为通常 $O>P\approx G$，先切 optimizer 最划算、通信也最简单。
> $P=14,G=14,O=56$ GB：DDP $=84$；ZeRO-1 $=14+14+7=35$；ZeRO-2 $=14+1.75+7=22.75$；ZeRO-3 $=84/8=10.5$ GB。

**28.** ZeRO-2 的 backward 用什么 collective？为什么和 optimizer shard 天然对上？

> **答：** **Reduce-Scatter** 代替 All-Reduce，每卡只留下自己负责的 gradient shard，正好和它持有的 optimizer shard 对应：`G_i + O_i → 更新 P_i`。

**29.** ⭐⭐ 推导三者的通信量。

> **答：** 高效 All-Reduce = Reduce-Scatter + All-Gather，所以 $C_{\text{DDP}}\approx2M$。
> ZeRO-1 调度优化后仍约 $2M$；ZeRO-2 = Reduce-Scatter 梯度 $M$ + step 后 All-Gather 参数 $M$ = $2M$；**ZeRO-3 = forward All-Gather $M$ + backward All-Gather $M$ + gradient Reduce-Scatter $M$ = $3M$**，即相对 DDP **+50%**。
> 所以 **ZeRO-2 是 sweet spot：显存省很多，通信 volume 却没增加。**

**30.** ⭐ "ZeRO 用通信和计算换显存"错在哪？

> **答：** **ZeRO 主要用通信换显存，基本不增加模型的数学计算量。** 它不像 gradient checkpointing 那样把 forward 重算一遍，矩阵乘 FLOPs 基本不变，增加的是 collective communication 和参数生命周期管理。

**31.** 通信 +50% 等于训练慢 50% 吗？

> **答：** 不等于。$T=T_{\text{compute}}+T_{\text{exposed communication}}+\cdots$，大量通信可以和计算 **overlap**（backward layer 30 的同时 reduce-scatter layer 31 的梯度；算 layer $i$ 的同时 prefetch layer $i+1$ 参数）。经验：ZeRO-1 约 0–5%、ZeRO-2 约 0–10%、ZeRO-3 约 5–25%；多机+网络慢+小模型+overlap 差时可能 20–50%+。

**32.** ⭐ activation 会被 ZeRO-3 除以 D 吗？

> **答：** **不会。** ZeRO 只处理 $P,G,O$；activation 取决于 $B_{\text{local}},L,H$，每张 DP rank forward 自己的数据就必须产生自己的 activation。$M_{\text{ZeRO3/device}}\approx\frac{P+G+O}{D}+A_{\text{local}}+M_{\text{temp}}$。这就是"开了 ZeRO-3 长 context 还是 OOM"的原因，还得配 checkpointing / 减 microbatch / FlashAttention。

**33.** 一般开到几？

> **答：** **没有"多卡就自动 ZeRO-X"这回事** —— PyTorch DDP 就是 DDP，DeepSpeed 要显式配 `zero_optimization.stage`。
> DDP 能放下 → DDP；放不下但完整 parameter 能放下 → **ZeRO-2**；parameter/states 已成瓶颈 → ZeRO-3 / FSDP full-shard。跳过 ZeRO-1 是因为它和 ZeRO-2 通信同阶但少切一项。

**34.** ⭐ FSDP1 / FSDP2 是什么？FSDP 一定等于 ZeRO-3 吗？

> **答：** **FSDP1/FSDP2 是 PyTorch 两代实现，不是 stage 1 / stage 2**，别和 ZeRO-1/2 混。
> FSDP1 默认 `FULL_SHARD` 时 $P/G/O$ 全 shard，算法上 ≈ ZeRO-3；但它的 `ShardingStrategy.SHARD_GRAD_OP` 被官方描述为 **ZeRO-2-style**，所以"FSDP = ZeRO-3"要限定成 `FULL_SHARD`。
> FSDP2 仍是 fully sharded data parallel，主要是底层表示重做：FSDP1 把一组参数 flatten 成一个大 `FlatParameter` 再 shard，FSDP2 用 **DTensor + per-parameter sharding**，更容易和其他并行组合。PyTorch 已建议迁移到 FSDP2。
> 三者思想和主要通信模式一致，但**实现不同，不能说完全等价**。

## F. 并行全图

**35.** ⭐⭐ 五个维度对应哪五种并行？

> **答：** $B\xrightarrow{DP}$、$L\xrightarrow{SP/CP}$、$H\xrightarrow{TP}$、$N_{\text{layer}}\xrightarrow{PP}$、$N_{\text{expert}}\xrightarrow{EP}$。

**36.** ⭐⭐ TP 和 ZeRO-3 的本质区别？

> **答：** ZeRO-3 参数是 shard 的，但**计算前要 all-gather 成完整参数**，然后每卡做完整 layer 计算；TP 参数 shard 后**永远不需要恢复完整 $W$**，$W_i$ 直接参与本地计算。
> **TP：参数分片 = 计算分片。** 所以 TP 不只省显存，还把 FLOPs 也分了（每卡 $\sim P/T$ 参数、$\sim1/T$ FLOPs、部分 activation 也被切）。

**37.** Transformer 怎么做 TP？哪一步需要 All-Reduce？

> **答：** Attention 天然按 head 切（TP=4 就是每卡 8 个 head）。MLP 第一层 $H\to4H$ 用 column parallel（$Y=X[W_1,W_2]=[XW_1,XW_2]$），第二层 $4H\to H$ 用 row parallel，$Z=\sum_i Y_iW_{2,i}$ 需要 **All-Reduce**。
> TP 的代价是**高频通信** —— 每个 Transformer block 都可能有 collective，所以最适合放在同一节点的 NVLink/NVSwitch 上。

**38.** PP 的 bubble fraction？为什么 microbatch 要远多于 stage？

> **答：** $\sim\frac{P-1}{M+P-1}$。$M\gg P$ 时 bubble 很小，$M\approx P$ 时非常明显。
> PP 的 microbatch 和 gradient accumulation 常一起出现但目的不同：**accumulation 是为了累积梯度后再 step，pipeline microbatch 是为了填满流水线**。PP stage 之间传的是 **activation**，不是参数。

**39.** ⭐ SP 和 CP 都切 $L$，区别是什么？

> **答：** **SP 切普通 activation，CP 切 attention context。**
> SP 是 TP 的配套：TP 之后 LayerNorm / Dropout / residual 的 activation 仍是每个 TP rank 各存一份完整 $[B,L,H]$，SP 沿 $L$ 把这些分掉，所以一般**复用 TP group**。
> CP 解决超长 context（128K）下 attention 本身放不下的问题，难点是 $Q_0$ 仍要 attend 所有位置的 $K,V$，所以卡之间必须**交换 K/V**（ring attention）。

**40.** ⭐ MoE 一层的四步？为什么是 All-to-All？

> **答：** ① router 算 $p(e|x)$ 取 Top-k；② **dispatch：All-to-All** 把 token 发到对应 expert 所在 GPU；③ expert FFN 本地计算；④ **combine：All-to-All** 送回，按 routing weight 加权。
> 因为不同 expert 在不同 GPU 上，token 要**点对点重分布**，不是所有卡算同一件事再规约，所以是 All-to-All 而非 All-Reduce。
> EP 独有问题是**负载不均衡**：router 可能把 600 个 token 都发给一个 expert → straggler，所以要 load balancing loss 或 aux-loss-free balancing。

**41.** ⭐⭐ 这些并行会影响训练效果吗？

> **答：** 原则上**不会** —— DP/TP/PP/SP/CP/ZeRO/FSDP/EP 本身都是数学等价的分片（$Y=[XW_1,XW_2]$ 和 $Y=XW$ 一样；PP 只是换执行位置）。现实里因为浮点归约顺序不同**不保证 bitwise identical**，但通常不影响最终质量。
> **真正会改变训练的**是：global batch 被顺手改了（→ lr / warmup / steps 要跟着调）、precision、sequence packing 实现、**MoE token dropping（超 expert capacity）**、router auxiliary loss、gradient clipping 方式。

**42.** ⭐ 8 卡训 7B 怎么排 DP/TP？为什么不 TP=8？

> **答：** **TP=1 + FSDP/ZeRO-3 over 8 DP ranks**（裸 DDP 不行，84 GB 放不下）。
> 原则：**模型放得下的前提下，TP 尽可能小、DP 尽可能大** —— DP 每 step 才同步一次梯度，TP 每层都要通信。多机时 TP 限制在 node 内走 NVLink，DP 跨 node。
> 选择顺序：DDP → ZeRO-2 → FSDP/ZeRO-3 → TP → PP。

## G. 参数构成与算法显存

**43.** ⭐ 大 MoE 的参数都在哪？写出 MoE 参数量公式。

> **答：** $P_{\text{MoE}}\gg P_{\text{attn}}\gg P_{\text{emb/head}}\gtrsim P_{\text{vision}}$。典型比例：MoE 97–99%、attention 1–2%、emb+head 0.1–0.4%。
> 一个 GLU expert $\approx3d_{\text{model}}d_{\text{ffn}}$，整体 $P_{\text{MoE}}\approx3dd_{\text{ffn}}\times N_{\text{expert}}\times N_{\text{MoE layer}}$。
> DeepSeek-V4-Flash 手算：$3\times4096\times2048=25.17$M，$\times257\times43\approx278.1$B，即 284B 的 97.8%。

**44.** Embedding + LM head 为什么乘 2？

> **答：** `tie_word_embeddings=false` 时输入 embedding 和输出 LM head 是两份独立矩阵，$P=2Vd$。tie 的话就只有一份。

**45.** ⭐⭐ 从 2.8T total 推出 104B activated。

> **答：** Kimi-K3 每 token 只选 $16/896$ 个 routed expert：$2.723\text{T}\times\frac{16}{896}\approx48.6$B，再加 shared experts ~12.2B、latent projections ~4.7B、attention ~36.2B、dense FFN ~0.7B $\approx103$B，接近官方 104B。同理 V4-Pro 1.6T→49B、V4-Flash 284B→13B。

**46.** ⭐ 大 MoE 的显存由 total 还是 activated 决定？

> **答：** **权重 / optimizer 显存由 total params 决定，每 token 的 FFN FLOPs 由 activated params 决定。** 2.8T MoE 的训练不是"104B 模型的显存"。这正是 EP 重要的原因：约 98% 参数是 experts，EP 就是切这 98%；TP 主要处理那 1–2% 的 dense attention / shared path。

**47.** ⭐ SFT / OPD / GRPO / PPO 的显存大小关系？各多了什么？

> **答：** $\text{SFT}<\text{OPD}<\text{GRPO}<\text{PPO}$。
> SFT = $P_S+G_S+O_S+A_S$；OPD 额外一个 **frozen teacher**（只有 $P_T$，没有 $G_T$、$O_T$）+ on-policy 时的 student rollout；GRPO = actor 训练 + frozen ref + **rollout engine（KV cache）**；PPO 再加一个**训练态 critic**（完整 $P+G+O+A$）和 frozen reward model。

**48.** ⭐ teacher 便宜在哪？要 KV cache 吗？

> **答：** teacher 不训练，只有 $P_T$（7B ≈14 GB），**没有** 14 GB gradient 和 56 GB Adam；32B teacher BF16 权重 64 GB，若训练则要约 384 GB。forward 在 `no_grad()` 下不建 backward graph，用完即释放，所以 $A_T^{\text{inference}}\ll A_S^{\text{training}}$。
> **不需要 KV cache** —— teacher 是对 student 已生成好的 trajectory 做一次 teacher-forcing forward 拿 logprob，不是 autoregressive 生成。

**49.** ⭐⭐ supervision tensor 的 $[B,L]$ vs $[B,L,V]$ 差多少？

> **答：** $B=8,L=8192$：sampled-token logprob（FP32）$8\times8192\times4\approx0.26$ MB；full-vocab（$V=150$K、BF16）$8\times8192\times150000\times2\approx19.7$ GB。完全不是一个数量级 —— 这是蒸馏显存里最关键的区别，也是 reverse-KL OPD 工程友好的重要原因。
> 实际做 forward KL 时也不会傻存完整 logits：top-K 蒸馏、chunked loss、从 teacher 采样近似。

**50.** ⭐⭐ 为什么不能把 rollout / teacher / training 显存直接相加？

> **答：** 要区分**逻辑组件**和**物理峰值**。on-policy OPD 一个 step 是分阶段的（Phase 1 rollout → Phase 2 teacher scoring → Phase 3 student training），如果 infra 做得好（engine sleep、free KV cache、offload/gather weights、分时复用 GPU），
> $M^{\text{peak}}=\max(M_{\text{rollout}},M_{\text{teacher}},M_{\text{training}})$ 而不是三者相加。
> 反过来，如果部署成三个独立 engine 占不同 GPU group，三者相加说的是**集群总资源**，不是单卡峰值。

**51.** ⭐ $\pi_{\text{old}}$ 要常驻一份权重吗？

> **答：** **不一定。** rollout 时直接把 $\log\pi_{\text{old}}(a_t|s_t)$ 存下来，只有 $[B,L]$；训练时 $r_t=\exp(\log\pi_\theta-\log\pi_{\text{old}})$。逻辑上有 old policy，物理上不一定有第二份 actor 权重。别把 GRPO 说成"actor + old actor + ref 三个完整模型"。

**52.** ⭐ 一句话总结 SFT/OPD/RL 的显存差异。

> **答：** SFT 的显存核心是单模型训练状态（参数、梯度、optimizer state、activation）；OPD 在此基础上额外常驻一个 frozen teacher，但 teacher 没有梯度和 optimizer；LLM RL 还必须承担 rollout 的 inference/KV cache，其中 GRPO 通常是 actor + ref + rollout，PPO 还要一个训练的 critic 和 frozen reward model，因此显存通常 **PPO > GRPO > OPD > SFT**。
> **SFT/OPD 的瓶颈更偏 training memory；RL 同时有 training memory + rollout memory** —— 这就是 verl 里 FSDP2、vLLM、engine sleep、offload、resharding 同时出现的原因。
