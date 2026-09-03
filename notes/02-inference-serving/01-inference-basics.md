# 推理基础：prefill / decode 与四个指标

> **最核心的一句话**：prefill 是 compute-bound，decode 是 memory-bound。
> 几乎所有推理优化都能归到「这两个阶段的哪一个」。

## 1. 两个阶段

```text
prompt (L 个 token)
      ↓
   ┌────────── Prefill ──────────┐
   │ 一次性并行处理全部 L 个 token │  →  写入 KV cache，产出第 1 个 token
   └─────────────────────────────┘
      ↓
   ┌────────── Decode ───────────┐
   │ 每次只处理 1 个 token         │  →  读全部 KV cache，产出下一个 token
   │ 循环 N 次                    │
   └─────────────────────────────┘
```

| | Prefill | Decode |
|---|---|---|
| 一次处理几个 token | $L$ 个（并行） | **1 个** |
| 矩阵乘形状 | $(L,d)\times(d,d)$，真正的 GEMM | $(1,d)\times(d,d)$，退化成 **GEMV** |
| 瓶颈 | **算力**（compute-bound） | **显存带宽**（memory-bound） |
| 决定什么指标 | TTFT | TPOT |

### 卡点：decode 为什么是 memory-bound

decode 每步要把**整个模型的权重**从 HBM 读进来，但只算 1 个 token 的矩阵向量乘。

$$\text{算术强度}=\frac{\text{FLOPs}}{\text{读取字节数}}\approx\frac{2Nd\cdot 1}{2N}=O(1)$$

7B 模型 BF16 权重 14 GB，A100 的 HBM 带宽约 2 TB/s，那么**光是把权重读一遍**就要

$$\frac{14\ \text{GB}}{2\ \text{TB/s}}=7\ \text{ms}$$

这就是单条序列 decode 的速度下限，和算力几乎无关。

$$\boxed{\text{decode 慢不是因为算不过来，是因为搬不过来}}$$

**推论**：batch 起来几乎是免费的。batch=1 和 batch=32 读权重的次数一样，所以 batch 越大吞吐越高，这正是 continuous batching 的理论基础。

## 2. KV cache

decode 第 $t$ 步需要 $q_t$ 对 $k_{1..t},v_{1..t}$ 做 attention。前面的 $k,v$ 不随步数变化，所以缓存起来，避免每步重算整个前缀。

$$M_{KV}=2\times B\times L\times N_{\text{layer}}\times h_{kv}\times d_h\times\text{bytes}$$

**没有 KV cache 会怎样**：每步都要重新前向整个前缀，第 $t$ 步的代价是 $O(t^2)$，总代价 $O(N^3)$，完全不可用。

**有了 KV cache 之后**，显存成了新瓶颈。$L=32$K、32 层、32 头、$d_h=128$、BF16 时约 **17 GB**，比 7B 权重还大（见 [attention](../01-llm-arch/04-attention.md#3-kv-cache-是这一切的动机)）。

所以推理服务的核心矛盾变成：

$$\boxed{\text{显存} = \text{权重（固定）} + \text{KV cache（正比于并发} \times \text{长度）}}$$

能并发多少条请求，直接由剩余显存除以单条 KV cache 决定。

## 3. 四个指标

| 指标 | 全称 | 含义 | 由谁决定 |
|---|---|---|---|
| **TTFT** | Time To First Token | 从请求到吐出第一个 token | **prefill** 耗时（正比于 prompt 长度） |
| **TPOT** | Time Per Output Token | 后续每个 token 的间隔 | **decode** 耗时（正比于权重大小 / 带宽） |
| **Latency** | 端到端延迟 | $\text{TTFT}+\text{TPOT}\times(N_{\text{out}}-1)$ | 两者 |
| **Throughput** | 吞吐 | 全系统每秒总 token 数 | 并发数 × 单条速度 |

### 卡点：延迟和吞吐是矛盾的

增大 batch：

- 吞吐 ↑（权重只读一次，摊给更多请求）
- 单条 TPOT ↑（每步要算更多、KV cache 读得更多）

$$\boxed{\text{对话产品优化 TTFT/TPOT；离线批处理优化 throughput}}$$

面试问「怎么优化推理」时先反问是哪个场景，会显得很专业。

## 4. 采样

logits 出来后怎么选下一个 token。

| 方法 | 做法 | 效果 |
|---|---|---|
| greedy | 取 argmax | 确定性，容易重复、无聊 |
| **temperature** $T$ | $p=\text{softmax}(z/T)$ | $T<1$ 更尖锐更保守；$T>1$ 更平更随机；$T\to0$ 退化成 greedy |
| **top-k** | 只在概率最高的 $k$ 个里采样 | 固定候选数，尾部长时可能仍包含垃圾 |
| **top-p**（nucleus） | 按概率降序累加到 $\ge p$ 为止，在这个集合里采样 | **候选数自适应**：分布尖锐时候选少，平坦时候选多 |
| **min-p** | 只保留 $p_i\ge p_{\text{min}}\cdot p_{\max}$ 的 token | 用相对阈值，比 top-p 更稳 |
| **repetition penalty** | 对已出现的 token 的 logit 除以（或减去）一个惩罚 | 抑制复读 |

**顺序**：一般是 `repetition penalty → temperature → top-k → top-p → 采样`。

$$\boxed{\text{top-k 固定候选个数，top-p 固定累计概率质量}}$$

**为什么 top-p 通常更好**：模型很确定时（比如接一个固定搭配），top-k 仍会强行留 $k$ 个候选，可能引入噪声；top-p 会自动收缩到 1–2 个。

**推理任务常用 $T$ 小甚至 greedy**（要确定性和正确率），**创作任务用较大 $T$ + top-p**。RL rollout 时通常要 $T\ge1$ 保证探索多样性。

## 5. 一次请求的完整时间线

```text
请求到达
  → 排队（等调度）
  → prefill：一次前向 L 个 token，写 KV cache     ← TTFT 主要来源
  → 吐出第 1 个 token
  → decode 循环：读全部权重 + 读 KV cache → 1 个 token   ← 每次 TPOT
  → 遇到 EOS 或达到 max_tokens
  → 释放 KV cache 块
```

优化点分别落在：排队（调度）、prefill（chunked prefill、prefix caching）、decode（batching、量化、投机解码）、显存（PagedAttention）。见 [服务化优化](02-serving-optimization.md)。

## 自测（口述版）

**1.** prefill 和 decode 各是 compute-bound 还是 memory-bound？为什么？

> **答：** **prefill 是 compute-bound**：一次并行处理 $L$ 个 token，矩阵乘是真正的 GEMM，算力吃满。
> **decode 是 memory-bound**：每次只处理 1 个 token，退化成 GEMV，但仍要把**整个模型权重**从 HBM 读一遍。
> 这是整块内容的主线，所有优化都在打这两个之一。

**2.** 用算术强度解释 decode 为什么受带宽限制。7B BF16 在 2 TB/s 带宽上单步下限是多少？

> **答：** 算术强度 $=\frac{\text{FLOPs}}{\text{读取字节}}\approx\frac{2Nd\cdot1}{2N}=O(1)$，是常数级，说明受带宽而不是算力限制。
> 7B BF16 权重 14 GB，$\frac{14\ \text{GB}}{2\ \text{TB/s}}=7$ ms —— **光把权重读一遍就要 7 ms**，这是单条序列 decode 的速度下限，和算力几乎无关。

**3.** 为什么 batch 起来几乎免费？这是什么优化的理论基础？

> **答：** batch=1 和 batch=32 读权重的次数一样（都是一遍），但产出的 token 数是 32 倍。所以 batch 越大吞吐越高。
> 这正是 **continuous batching** 的理论基础。

**4.** 没有 KV cache 的总复杂度是多少？有了之后新瓶颈是什么？

> **答：** 没有 cache 时每步都要重新前向整个前缀，第 $t$ 步代价 $O(t^2)$，总代价 $O(N^3)$，完全不可用。
> 有了之后新瓶颈变成**显存**：$M_{KV}=2BLN_{\text{layer}}h_{kv}d_h\cdot\text{bytes}$，32K 上下文可达 17 GB，比 7B 权重还大。能并发多少条请求 = 剩余显存 ÷ 单条 KV cache。

**5.** TTFT 和 TPOT 分别由哪个阶段决定？延迟和吞吐为什么矛盾？

> **答：** **TTFT 由 prefill 决定**（正比于 prompt 长度），**TPOT 由 decode 决定**（正比于权重大小/带宽）。端到端延迟 $=\text{TTFT}+\text{TPOT}\times(N_{\text{out}}-1)$。
> 矛盾在于：增大 batch → 吞吐↑（权重只读一次摊给更多请求），但单条 TPOT↑（每步算得更多、KV 读得更多）。**对话产品优化 TTFT/TPOT，离线批处理优化吞吐**，面试先反问场景。

**6.** top-k 和 top-p 的区别？为什么 top-p 通常更好？

> **答：** top-k 固定**候选个数**；top-p（nucleus）按概率降序累加到 $\ge p$，固定**累计概率质量**、候选数自适应。
> top-p 更好是因为：模型很确定时 top-k 仍强行留 $k$ 个候选可能引入噪声，top-p 会自动收缩到 1–2 个；分布平坦时 top-p 又能放开。min-p 用相对阈值 $p_i\ge p_{\min}p_{\max}$，比 top-p 更稳。

**7.** 采样各步的顺序是什么？推理任务和创作任务的参数取向有何不同？

> **答：** 顺序：`repetition penalty → temperature → top-k → top-p → 采样`。
> 推理/数学任务用小 $T$ 甚至 greedy 求确定性和正确率；创作任务用较大 $T$ + top-p；RL rollout 通常 $T\ge1$ 保证探索多样性。

