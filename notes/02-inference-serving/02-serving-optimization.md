# 服务化优化：PagedAttention、continuous batching、prefix caching

> 这几项是 vLLM / SGLang 的核心，也是推理岗最高频的追问。
> 每一项都对应 [推理基础](01-inference-basics.md) 里的一个具体瓶颈。

## 1. PagedAttention：解决 KV cache 的碎片

**问题**：传统实现给每条请求预分配一块**连续**显存，按 `max_len` 开。

```text
请求 A: max_len=2048，实际只生成了 100 个 token
        [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   浪费 95%
```

三种浪费：

| 类型 | 说明 |
|---|---|
| **内部碎片** | 按 max_len 预留，实际用不到 |
| **外部碎片** | 剩余显存零散，凑不出一块连续的 |
| **无法共享** | 多条请求的相同前缀各存一份 |

vLLM 报告传统方案的 KV 显存有效利用率只有 20%–40%。

**做法**：借操作系统的虚拟内存思想。

$$\boxed{\text{把 KV cache 切成固定大小的 block（如 16 个 token），用 block table 做逻辑到物理的映射}}$$

```text
逻辑序列:  [tok0..15][tok16..31][tok32..47]
block table:   ↓         ↓          ↓
物理块:      #7        #3         #12       ← 物理上不连续
```

收益：

- 按需分配，几乎没有内部碎片（最多浪费一个 block）
- 物理块不需要连续，外部碎片消失
- **不同请求可以指向同一个物理块** → 前缀共享、beam search 的分支共享，靠引用计数 + copy-on-write

代价是 attention kernel 要能按 block table 取 KV（这就是 `PagedAttention` kernel 的由来）。

$$\boxed{\text{PagedAttention 提高的是显存利用率，从而提高并发数，间接提高吞吐}}$$

## 2. Continuous batching：解决 GPU 空转

**问题**：静态 batching 要等一整批全部生成完才能换下一批。

```text
static batching:
  req A ████████████████████  (生成 200 token)
  req B ███░░░░░░░░░░░░░░░░░  (生成 30 token 就结束了，但要空等)
  req C ██░░░░░░░░░░░░░░░░░░
        ← 大量 GPU 时间浪费在等最长的那条 →
```

**做法**：以 **iteration 为粒度**调度，而不是以 request 为粒度。每完成一步 decode，就检查：谁结束了就踢出去，队列里有新请求就立刻加进来。

```text
continuous batching:
  req A ████████████████████
  req B ███→ D 进来 ████████
  req C ██→ E 进来 █████████
        ← batch 槽位始终填满 →
```

$$\boxed{\text{decode 是 memory-bound，batch 增大几乎免费，所以「保持 batch 满」就是保持高吞吐}}$$

这条直接建立在 [decode 是 memory-bound](01-inference-basics.md#卡点decode-为什么是-memory-bound) 之上。也叫 in-flight batching / rolling batching。

## 3. Prefix caching：复用相同前缀的 KV

**观察**：很多请求共享前缀 —— 同一个 system prompt、few-shot 例子、多轮对话的历史、agent 的工具定义。

**做法**：把 KV cache 按 block 做哈希，前缀相同的 block 直接复用，跳过这部分 prefill。

```text
请求 1: [system prompt 500 tok][用户问题 A]
请求 2: [system prompt 500 tok][用户问题 B]
                ↑ 这 500 个 token 的 KV 只算一次
```

收益：**TTFT 大幅下降**（prefill 是 compute-bound，省下的是实打实的算力）。agent 场景收益尤其大，因为每轮都带着长长的历史。

实现依赖 PagedAttention 的 block 机制：相同前缀 → 相同 block hash → 指向同一物理块 + 引用计数。

$$\boxed{\text{PagedAttention 是地基，prefix caching 是它最大的红利}}$$

## 4. Chunked prefill：平衡两个阶段

**问题**：prefill 和 decode 抢同一个 GPU。一个 8K 的长 prompt 做 prefill 时会占满算力，此时所有正在 decode 的请求都被卡住，TPOT 出现毛刺。

**做法**：把长 prefill 切成小块，每一步只做一块，**和 decode 请求混在同一个 batch 里**跑。

```text
不分块:  [────── prefill 8K ──────] 期间 decode 全部停摆
分块:    [prefill 1K + decode×N][prefill 1K + decode×N]...
```

好处：

- decode 不再被长 prompt 饿死，TPOT 更平稳
- prefill 的 GEMM 和 decode 的 GEMV 混在一起，正好把 compute-bound 和 memory-bound 的工作**互补**起来，硬件利用率更高

$$\boxed{\text{chunked prefill 用 TTFT 的少量牺牲换 TPOT 的稳定和整体吞吐}}$$

## 5. KV cache 量化与 offload

**量化**：把 KV 从 BF16 降到 FP8 / INT8 / INT4。

$$M_{KV}\ \propto\ \text{bytes}\ \Rightarrow\ \text{FP8 直接省一半}$$

KV 对量化相对不敏感（比权重更耐受），FP8 KV 现在很常见。注意 K 和 V 的敏感度不同，K 通常更敏感（因为要参与 softmax），有些方案对两者用不同精度。

**offload**：把暂时用不到的 KV 换到 CPU 内存甚至 NVMe。适合多轮对话中长时间不活跃的会话。代价是换回来要走 PCIe，延迟高，所以要配合预取。

## 6. FlashAttention 与 FlashInfer

| | 定位 |
|---|---|
| **FlashAttention** | 训练和 prefill 阶段的融合 attention kernel，分块 + online softmax，不落地 $L\times L$ 矩阵 |
| **FlashInfer** | 专为**推理**设计的 attention kernel 库，支持 paged KV、变长、各种 mask 形态、GQA/MLA 的特化 |

decode 阶段 $q$ 只有 1 个 token，形状和训练时完全不同，需要单独优化的 kernel，这就是 FlashInfer 这类库存在的原因。

## 7. 把优化和瓶颈对上号

| 优化 | 治的是什么 | 直接改善 |
|---|---|---|
| PagedAttention | KV 显存碎片 | 并发数 → 吞吐 |
| Continuous batching | GPU 空转等待 | 吞吐 |
| Prefix caching | 重复 prefill | **TTFT** |
| Chunked prefill | prefill 阻塞 decode | **TPOT 稳定性** |
| KV 量化 | KV 显存占用 | 并发数 |
| KV offload | 显存容量上限 | 可容纳的会话数 |
| GQA / MLA | KV cache 大小（模型结构层面） | 并发数 |
| 投机解码 | decode 的串行性 | **TPOT** |
| 权重量化 | 权重读取字节数 | **TPOT** |

$$\boxed{\text{回答「怎么优化推理」时，先说瓶颈在 prefill 还是 decode，再对号入座}}$$

## 自测（口述版）

**1.** PagedAttention 解决什么问题？三种浪费分别是什么？借鉴了什么思想？

> **答：** 解决 KV cache 的**显存碎片**。传统实现按 `max_len` 预分配一块连续显存，三种浪费：**内部碎片**（预留用不完）、**外部碎片**（剩余显存零散凑不出连续块）、**无法共享**（相同前缀各存一份）。传统方案有效利用率只有 20%–40%。
> 借鉴**操作系统虚拟内存**：把 KV 切成固定大小 block（如 16 token），用 **block table** 做逻辑到物理的映射，物理块不需要连续。

**2.** PagedAttention 怎么支持前缀共享？靠什么机制保证安全？

> **答：** 不同请求的 block table 可以指向**同一个物理块**，靠**引用计数 + copy-on-write** 保证安全。这也让 beam search 的分支共享变得廉价。代价是 attention kernel 必须能按 block table 取 KV。

**3.** continuous batching 和静态 batching 的区别？调度粒度是什么？它建立在哪条性质上？

> **答：** 静态 batching 以 **request** 为调度粒度，要等一整批全部生成完，短请求被最长的那条拖着空等。
> continuous batching 以 **iteration** 为粒度：每完成一步 decode 就踢掉结束的、补进新的，batch 槽位始终填满。
> 它建立在「**decode 是 memory-bound、batch 增大几乎免费**」这条性质上。也叫 in-flight / rolling batching。

**4.** prefix caching 改善哪个指标？为什么它依赖 PagedAttention？什么场景收益最大？

> **答：** 改善 **TTFT**，因为跳过的是 compute-bound 的 prefill，省的是实打实的算力。
> 依赖 PagedAttention 是因为它需要按 block 做哈希、让相同前缀指向同一物理块 + 引用计数。
> 收益最大的场景：同一 system prompt 的批量请求、few-shot、多轮对话历史、**agent 的工具定义**（每轮都带长历史）。

**5.** chunked prefill 解决什么？为什么把 prefill 和 decode 混在一个 batch 里反而更高效？

> **答：** 解决长 prompt 的 prefill 占满算力、把正在 decode 的请求全卡住导致 TPOT 毛刺。做法是把长 prefill 切块，每步只做一块并**和 decode 请求混在同一个 batch**。
> 更高效是因为 prefill 的 GEMM 是 compute-bound、decode 的 GEMV 是 memory-bound，两者**互补**，硬件利用率更高。本质是用 TTFT 的少量牺牲换 TPOT 稳定和整体吞吐。

**6.** KV 量化为什么比权重量化更容易做？K 和 V 哪个更敏感？

> **答：** KV 对量化相对不敏感，比权重更耐受，所以 FP8 KV 现在很常见。
> **K 通常比 V 更敏感**，因为它要参与 softmax，误差会被指数放大；有些方案对两者用不同精度。

**7.** FlashAttention 和 FlashInfer 的定位区别？为什么 decode 需要单独的 kernel？

> **答：** FlashAttention 面向**训练和 prefill**：分块 + online softmax，不落地 $L\times L$ 矩阵。
> FlashInfer 面向**推理**：支持 paged KV、变长、各种 mask 形态、GQA/MLA 特化。
> decode 阶段 $q$ 只有 1 个 token，矩阵形状和训练时完全不同（GEMV 而非 GEMM），需要单独优化的 kernel。

**8.** 给出一张「优化 ↔ 瓶颈」的对应表。

> **答：** PagedAttention→KV 显存碎片→并发/吞吐；continuous batching→GPU 空转→吞吐；prefix caching→重复 prefill→**TTFT**；chunked prefill→prefill 阻塞 decode→**TPOT 稳定性**；KV 量化/offload→KV 显存→并发；GQA/MLA→KV cache 大小（结构层面）→并发；投机解码→decode 串行性→**TPOT**；权重量化→权重读取字节数→**TPOT**。

