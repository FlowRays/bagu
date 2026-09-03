# 推理与部署自测题库（关掉笔记用）

> 侧边栏顶部有 **「答案」开关**。标 ⭐ 的是高频考点。

## A. 基础（[01](01-inference-basics.md)）

**1.** ⭐⭐ prefill 和 decode 各是什么 bound？为什么？

> **答：** prefill 一次并行处理 $L$ 个 token，矩阵乘是真正的 GEMM，**compute-bound**；decode 每次只处理 1 个 token，退化成 GEMV，但仍要把**整个模型权重**从 HBM 读一遍，**memory-bound**。
> 这是整块内容的主线，所有优化都在打这两个之一。

**2.** ⭐⭐ 用算术强度解释 decode 的下限。

> **答：** 算术强度 $=\frac{\text{FLOPs}}{\text{读取字节}}\approx\frac{2Nd\cdot1}{2N}=O(1)$，是常数级，说明受带宽而非算力限制。
> 7B BF16 权重 14 GB，A100 带宽约 2 TB/s，光读一遍权重就要 $14/2000\approx7$ ms，这是单条 decode 的速度下限，**和算力几乎无关**。

**3.** ⭐ 为什么 batch 起来几乎免费？

> **答：** batch=1 和 batch=32 读权重的次数一样（都是一遍），但产出的 token 数是 32 倍。这正是 **continuous batching** 的理论基础。

**4.** 没有 KV cache 会怎样？

> **答：** 每步都要重新前向整个前缀，第 $t$ 步代价 $O(t^2)$，总计 $O(N^3)$，完全不可用。有了之后新瓶颈变成**显存**：$M_{KV}=2BLN_{\text{layer}}h_{kv}d_h\cdot\text{bytes}$，32K 上下文可达 17 GB，比 7B 权重还大。

**5.** ⭐ 写出四个指标，各由什么决定。

> **答：** **TTFT**（首 token 延迟）由 prefill 决定，正比于 prompt 长度；**TPOT**（每 token 间隔）由 decode 决定，正比于权重大小/带宽；**Latency** $=\text{TTFT}+\text{TPOT}\times(N_{\text{out}}-1)$；**Throughput** = 并发数 × 单条速度。

**6.** ⭐ 延迟和吞吐为什么矛盾？

> **答：** 增大 batch 会让吞吐升（权重只读一次摊给更多请求），但单条 TPOT 降（每步算得更多、KV 读得更多）。**对话产品优化 TTFT/TPOT，离线批处理优化吞吐**，面试先反问场景。

**7.** ⭐ top-k 和 top-p 的区别？为什么 top-p 更好？

> **答：** top-k 固定**候选个数**，top-p（nucleus）固定**累计概率质量**、候选数自适应。
> 模型很确定时 top-k 仍强行留 $k$ 个候选可能引入噪声，top-p 会自动收缩到 1–2 个；分布平坦时 top-p 又能放开。min-p 用相对阈值 $p_i\ge p_{\min}p_{\max}$，更稳。

**8.** 采样各步顺序？不同任务的参数取向？

> **答：** `repetition penalty → temperature → top-k → top-p → 采样`。
> 推理/数学任务用小 $T$ 甚至 greedy 求确定性；创作用较大 $T$ + top-p；RL rollout 通常 $T\ge1$ 保证探索多样性。

**9.** temperature 的作用？$T\to0$ 会怎样？

> **答：** $p=\text{softmax}(z/T)$。$T<1$ 分布更尖锐更保守，$T>1$ 更平更随机，$T\to0$ 退化成 greedy。

## B. 服务化优化（[02](02-serving-optimization.md)）

**10.** ⭐⭐ PagedAttention 解决什么？三种浪费是什么？

> **答：** 解决 KV cache 的显存碎片。传统方案按 `max_len` 预分配连续显存，三种浪费：**内部碎片**（预留用不完）、**外部碎片**（剩余显存零散凑不出连续块）、**无法共享**（相同前缀各存一份）。传统方案有效利用率只有 20%–40%。
> 做法借鉴操作系统虚拟内存：把 KV 切成固定大小 block（如 16 token），用 **block table** 做逻辑到物理的映射，物理块无需连续。

**11.** ⭐ PagedAttention 怎么支持前缀共享？

> **答：** 不同请求的 block table 可以指向**同一个物理块**，靠**引用计数 + copy-on-write** 保证安全。这也让 beam search 的分支共享变得廉价。代价是 attention kernel 必须能按 block table 取 KV。

**12.** ⭐⭐ continuous batching 和静态 batching 的区别？

> **答：** 静态 batching 以 **request** 为调度粒度，要等一整批全部生成完，短请求被最长的那条拖着空等。continuous batching 以 **iteration** 为粒度，每完成一步 decode 就踢掉结束的、补进新的，batch 槽位始终填满。
> 它建立在「decode 是 memory-bound、batch 增大几乎免费」这条性质上。也叫 in-flight / rolling batching。

**13.** ⭐ prefix caching 改善哪个指标？什么场景收益最大？

> **答：** 改善 **TTFT**，因为跳过的是 compute-bound 的 prefill，省的是实打实的算力。
> 做法是按 block 做哈希，前缀相同直接复用。**依赖 PagedAttention 的 block 机制**。
> 收益最大的场景：同一 system prompt 的批量请求、few-shot、多轮对话历史、agent 的工具定义（每轮都带长历史）。

**14.** ⭐⭐ chunked prefill 解决什么？为什么混着跑更高效？

> **答：** 解决长 prompt 的 prefill 占满算力、把正在 decode 的请求全卡住导致 TPOT 毛刺。做法是把长 prefill 切块，每步只做一块并**和 decode 请求混在同一个 batch**。
> 混着跑更高效是因为 prefill 的 GEMM 是 compute-bound、decode 的 GEMV 是 memory-bound，两者**互补**，硬件利用率更高。
> 本质是用 TTFT 的少量牺牲换 TPOT 稳定和整体吞吐。

**15.** KV 量化为什么比权重量化容易？

> **答：** KV 对量化相对不敏感，比权重更耐受，FP8 KV 现在很常见。注意 **K 通常比 V 更敏感**（要参与 softmax），有些方案对两者用不同精度。

**16.** FlashAttention 和 FlashInfer 的定位区别？

> **答：** FlashAttention 面向训练和 prefill，分块 + online softmax，不落地 $L\times L$ 矩阵。FlashInfer 面向**推理**，支持 paged KV、变长、各种 mask 形态、GQA/MLA 特化。
> decode 阶段 $q$ 只有 1 个 token，形状和训练时完全不同，需要单独优化的 kernel。

**17.** ⭐ 给出「优化 ↔ 瓶颈」对应表。

> **答：** PagedAttention→KV 碎片→并发/吞吐；continuous batching→GPU 空转→吞吐；prefix caching→重复 prefill→**TTFT**；chunked prefill→prefill 阻塞 decode→**TPOT 稳定性**；KV 量化/offload→KV 显存→并发；GQA/MLA→KV 大小→并发；投机解码→decode 串行性→**TPOT**；权重量化→权重读取量→**TPOT**。

## C. 投机解码与量化（[03](03-speculative-and-quant.md)）

**18.** ⭐⭐ 投机解码为什么能加速？

> **答：** decode 是 memory-bound，一次前向读完整个模型只产出 1 个 token，算力大量闲置。投机解码让 target 模型**一次前向验证 $k$ 个 token**，把闲置算力用上，摊薄了读权重的成本。
> 流程：draft 模型猜 $k$ 个 → target 一次前向算出 $k+1$ 个位置的分布 → 逐个验证接受前缀 → 第一个被拒的位置重采样、后面丢弃 → 至少前进 1 个 token。

**19.** ⭐⭐ 为什么说投机解码无损？

> **答：** 用 speculative sampling 的接受-拒绝规则：以 $\min\big(1,\frac{p(x)}{q(x)}\big)$ 接受 draft 的 $x$（$q$ 是 draft 分布、$p$ 是 target 分布）；拒绝时从**残差分布** $\text{norm}(\max(0,p-q))$ 重采样。
> 可以证明最终采样分布**严格等于 $p$**。它不是近似，不掉点。

**20.** 加速比取决于什么？$k$ 为什么不能太大？

> **答：** 期望前进 token 数 $=\frac{1-\alpha^{k+1}}{1-\alpha}$，$\alpha$ 是接受率。draft 越接近 target 收益越大，所以 draft 要**同源**（同系列、同 tokenizer）。
> $k$ 大了后面的 token 接受率**指数衰减**，而 draft 成本线性增长，所以存在最优 $k$。

**21.** ⭐ 什么时候投机解码不划算？

> **答：** **batch 很大时**。此时 decode 已接近 compute-bound（算力被填满），没有闲置算力可用，反而因为多算了被拒的 token 而变慢。所以它适合小 batch / 低延迟场景。

**22.** 投机解码有哪些变体？

> **答：** 标准（独立小模型）、**自投机 MTP**（用模型自带的多 token 预测头，不需要额外模型）、Medusa（主干挂多个头）、Lookahead / n-gram（从已生成文本找候选，零训练）、EAGLE（特征层面 draft，接受率更高）。

**23.** ⭐⭐ 量化在推理侧省的是算力还是带宽？

> **答：** **带宽**。$\text{TPOT}\propto\frac{\text{权重字节数}}{\text{带宽}}$，BF16→INT4 理论上 decode 快 4 倍。
> 所以 **decode 场景通常 W4A16 就够**（只量化权重）；prefill 和大 batch 是 compute-bound，才需要 W8A8 连激活一起量化来用低精度 tensor core。

**24.** 量化的三个维度？

> **答：** 量化什么（权重 / 激活 / KV）、何时量化（**PTQ** 训练后几小时 / **QAT** 训练中很贵）、粒度（per-tensor / per-channel / **per-group** 如每 128 个数一组，最常用）。写法如 `W4A16`、`W8A8`、`FP8`。

**25.** ⭐⭐ GPTQ 的核心思路？

> **答：** 逐层做，把量化看成优化问题：在校准数据上让**该层输出**的误差最小而非权重本身误差最小，$\min_{\hat W}\|WX-\hat WX\|^2$。逐列量化，每量化一列就把误差**补偿到还没量化的列**上（基于 Hessian 近似）。需要少量校准数据。

**26.** ⭐⭐ AWQ 的核心思路？

> **答：** 观察到**激活值大的通道对应的权重更重要**，约 1% 的显著通道决定大部分误差。但不做混合精度（硬件不友好），而是量化前给显著通道**乘放大系数** $s$、激活侧除回来（$W\to \hat{(Ws)}$，$X\to X/s$）。放大后这些通道的相对量化误差变小，等价于把误差从重要通道转移到不重要通道。
> 一句话：**GPTQ 用 Hessian 做误差补偿，AWQ 按激活幅度保护显著通道**。

**27.** FP8 相比 INT8 的优势？

> **答：** **动态范围大**（E4M3 / E5M2），不需要复杂 scale 校准，对异常值更宽容。Hopper 之后硬件原生支持，训练和推理都在往 FP8 走，KV cache 用 FP8 也常见。

**28.** ⭐ 激活量化难在哪？怎么应对？

> **答：** LLM 激活里存在**极少数维度幅值极大**（outlier channels），量程被撑开后其余数值有效位数被压没。
> 应对：① per-group / per-channel 代替 per-tensor；② **SmoothQuant** 把激活的难度按通道迁移一部分到权重（$X/s$、$Ws$）；③ 保留少数 outlier 通道用高精度（LLM.int8()）。

**29.** ⭐ 投机解码和量化对照。

> **答：** 投机解码省 decode 的**串行轮数**、**无损**、适合小 batch 低延迟；权重量化省**读取字节数**、有损但可控、适合所有 decode 场景；KV 量化省 KV 显存、有损、适合长上下文高并发；激活量化让计算走低精度 tensor core、较难、适合 prefill 和大 batch。

**30.** ⭐⭐ 一分钟答「怎么优化推理」。

> **答：** 先定位瓶颈：prefill 是 compute-bound 决定 TTFT，decode 是 memory-bound 决定 TPOT。
> 压 TTFT：prefix caching 复用相同前缀的 KV、chunked prefill 避免长 prompt 阻塞、必要时上 TP。
> 压 TPOT：权重量化减少读取字节、投机解码把多步合并成一次验证（无损）、结构上换 GQA/MLA 减小 KV cache。
> 提吞吐：continuous batching 保持 batch 满、PagedAttention 提高显存利用率从而提高并发。
> 最后强调延迟和吞吐是矛盾的，对话产品和离线批处理取舍完全不同。
