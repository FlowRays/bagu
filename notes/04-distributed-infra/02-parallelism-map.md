# 并行方式全图：DP / TP / PP / SP / CP / EP

> 最值得记的一句话：**每种并行都是在切一个不同的维度。**

$$\boxed{B\xrightarrow{DP}\quad L\xrightarrow{SP/CP}\quad H\xrightarrow{TP}\quad N_{\text{layer}}\xrightarrow{PP}\quad N_{\text{expert}}\xrightarrow{EP}}$$

对着 $X\in\mathbb R^{B\times L\times H}$、$N_{\text{layer}}$ 层、$N_{\text{expert}}$ 个 expert 看这张图就全清楚了。

## 1. Tensor Parallel：把一个矩阵乘拆到多张卡

### Column parallel

$$Y=X[W_1,W_2]=[XW_1,\ XW_2]$$

GPU0 算 $XW_1$，GPU1 算 $XW_2$，各存一半 $W$、各算一半输出 channel。

### 卡点：TP 和 ZeRO-3 的本质区别

- **ZeRO-3**：参数是 shard 的，但**计算前要 all-gather 成完整参数**，然后每张卡做完整 layer 计算。
- **TP**：参数 shard 后**永远不需要恢复完整 $W$**，$W_i$ 直接参与本地计算。

$$\boxed{\text{TP：参数分片 = 计算分片}}$$

所以 TP 不只省显存，还真的把 FLOPs 也分了：每卡 $\sim P/T$ 参数、$\sim1/T$ 矩阵乘 FLOPs、部分 activation 也被切。

### Transformer 里怎么拆

- **Attention**：天然有多个 head，TP=4 就是每卡负责 8 个 head，非常自然。
- **MLP**：第一层 $H\to4H$ 用 column parallel，第二层 $4H\to H$ 用 row parallel，$Z=\sum_i Y_iW_{2,i}$ 需要 **All-Reduce**。

### 代价：高频通信

ZeRO 的通信围绕 layer 参数 gather / 梯度 reduce-scatter；**TP 是在 forward 的矩阵乘中间就要通信**，每个 Transformer block 都可能有 collective。

$$\boxed{\text{TP 最适合放在同一节点的 NVLink/NVSwitch 上}}$$

## 2. Pipeline Parallel：按层切

32 层、PP=4：GPU0 拿 layer 0–7，GPU1 拿 8–15…。stage 之间传的**不是参数，是 activation**（边界 hidden states）。

### Pipeline bubble

直接跑一个 batch 的话只有一张卡在忙。必须把 batch 切成 microbatch 流水起来：

```text
time →
GPU0: M1 M2 M3 M4 M5 …
GPU1:    M1 M2 M3 M4 …
GPU2:       M1 M2 M3 …
GPU3:          M1 M2 …
```

$$\text{bubble fraction}\sim\frac{P-1}{M+P-1}$$

所以需要 $M\gg P$。PP 的 microbatch 和 gradient accumulation 常常一起出现，但目的不同：**accumulation 是为了累积梯度后再 step，pipeline microbatch 是为了填满流水线**。

## 3. Sequence Parallel：TP 的配套

TP 之后，LayerNorm / Dropout / residual / element-wise op 的 activation 仍是每个 TP rank 各存一份完整 $[B,L,H]$（replicated）。

SP 沿 $L$ 把这些也分掉：$L=8192$、SP=4 时每卡只处理 2048 个 token 的 LayerNorm 等。

$$\boxed{\text{SP 通常是 TP 的配套 activation optimization}}$$

所以 SP 一般**复用 TP group**（`tensor_model_parallel_size=8` + `sequence_parallel=true`），而不是独立的一维 world size。

## 4. Context Parallel：解决超长 context

也切 $L$，但解决的问题更大：$L=128\text{K}$ 时 attention 本身放不下。

难点在于 GPU0 的 $Q_0$ 仍要 attend 到**所有**位置的 $K,V$，所以卡之间必须交换 K/V（ring attention）：

```text
GPU0: Q0×(K0,V0) → 收到 K1,V1 → Q0×(K1,V1) → 收到 K2,V2 → …
```

| | SP | CP |
|---|---|---|
| 切什么 | sequence $L$ | sequence $L$ |
| 目的 | 减少 TP 下 replicated activation | 超长 context 的 attention |
| attention 本身是否分布式 | 通常不是重点 | **是** |
| 绑定谁 | TP group | 更常见是独立维度 |
| 通信重点 | gather/scatter activation | **交换 K/V** |

$$\boxed{SP=\text{切普通 activation}}\qquad\boxed{CP=\text{切 attention context}}$$

## 5. Expert Parallel：MoE 专用

MoE 的 router 对每个 token 选 top-k expert：$y=w_3E_3(x)+w_7E_7(x)$。64 experts / top-2 时一个 token 只激活 $2/64$。

$$\boxed{\text{总参数量很大，但每 token 激活参数少}}$$

问题是每卡都复制 50B 的 expert 参数受不了，所以**把不同 expert 放不同 GPU**：

```text
GPU0: Expert 0,1   GPU1: Expert 2,3   GPU2: Expert 4,5   GPU3: Expert 6,7
```

一层 MoE 的四步：

1. **router**：算 $p(e|x)$，Top-k
2. **dispatch**：**All-to-All**，把 token 发到对应 expert 所在的 GPU
3. **expert FFN**：各 GPU 算自己收到的 token
4. **combine**：**All-to-All** 送回，按 routing weight 加权

$$\boxed{\text{TP 的典型通信是 All-Reduce/AG/RS；EP 的典型通信是 All-to-All}}$$

### EP 独有的问题：负载不均衡

router 可能把 600 个 token 都发给 E2，其他 expert 只有几十个 → **straggler**，吞吐大幅下降。所以 MoE 要加 **load balancing loss**，或用 aux-loss-free balancing。这已经属于 MoE 的算法设计，不只是并行 infra。

## 6. 这些并行会影响训练效果吗

原则上：

$$\boxed{\text{并行本身是数学等价的分片，不改变训练目标}}$$

- DP：$\frac18\sum_r g_r$ 还是同一个 gradient
- TP：$Y=[XW_1,XW_2]$ 和 $Y=XW$ 完全一样
- PP：$f_4(f_3(f_2(f_1(x))))$ 只是换了执行位置
- SP/CP/ZeRO/FSDP：都只是分片 + 通信恢复

但现实里**不保证 bitwise identical**（All-Reduce 的浮点归约顺序不同）。通常不影响最终质量。

### 真正会改变训练的是这些

| 类别 | 例子 |
|---|---|
| 数学等价的并行 | DP / TP / PP / SP / CP / ZeRO / FSDP / EP 本身 |
| **会改变训练** | **global batch 被顺手改了**、precision、sequence packing 实现、**MoE token dropping（超 expert capacity）**、router auxiliary loss、gradient clipping 方式 |

$$\boxed{\text{DP 本身不改效果，但它顺手把 global batch 改了，lr / warmup / steps 都要跟着调}}$$

$$\boxed{\text{EP 本身不改效果，但 MoE 的 routing / capacity 策略会}}$$

## 7. 8 卡怎么排 DP 和 TP

$DP\times TP=8$，核心原则：

$$\boxed{\text{在模型放得下的前提下，TP 尽可能小，DP 尽可能大}}$$

因为 DP 每个 step 才同步一次梯度，TP 每层都要通信。

| 情况 | 配置 |
|---|---|
| 7B dense，8×80G | **TP=1 + FSDP/ZeRO-3 over 8 DP ranks**（裸 DDP 不行，84 GB 放不下） |
| 单层/working set 较重 | TP=2, DP=4 |
| 更大模型 | TP=4, DP=2 |
| 巨大 dense model | TP=8, DP=1（吞吐扩展性最差） |

多机时通常 **TP 限制在 node 内**（走 NVLink），DP 跨 node（低频通信更适合慢网络）：

```text
Node 0..7 各 8 GPU = 一个 TP=8 副本  →  8 个副本之间 DP=8
```

一般选择顺序（不是绝对）：

$$\text{DDP}\rightarrow\text{ZeRO-2}\rightarrow\text{FSDP/ZeRO-3}\rightarrow\text{TP}\rightarrow\text{PP}$$

大规模时几维一起用：$\text{world size}=DP\times TP\times PP$（如 $16\times8\times8=1024$），而 DP 这一维内部还可以再叠 ZeRO/FSDP，TP+FSDP 组合现在很常见。

## 8. 总表

| 并行 | 切什么 | 主要省什么 | 主要通信 | 典型目的 |
|---|---|---|---|---|
| DP | Batch | 本身不省 P/G/O | All-Reduce grad | 吞吐 |
| ZeRO/FSDP | P/G/O | model states | AG / RS | 显存 |
| TP | Hidden/head | 参数+计算+部分 activation | AR/AG/RS | 单层太大 |
| PP | Layers | 模型层 | Send/Recv activation | 模型太深 |
| SP | Sequence | replicated activation | AG/RS | TP 配套 |
| CP | Context | 长上下文 attention | 交换 K/V | 32K–1M context |
| EP | Experts | expert 参数 | **All-to-All** | MoE |

## 自测（口述版）

1. 写出五个维度对应的五种并行。
2. TP 的 column parallel 怎么拆？TP 和 ZeRO-3 在"要不要恢复完整参数"上的区别是什么？
3. Transformer 的 attention 和 MLP 分别怎么做 TP？哪一步需要 All-Reduce？
4. PP 的 bubble fraction 公式？为什么 microbatch 要远多于 stage 数？
5. SP 和 CP 都切 $L$，区别是什么？SP 为什么常复用 TP group？
6. MoE 一层的四个步骤？为什么是 All-to-All 而不是 All-Reduce？EP 独有的问题是什么？
7. 这些并行会影响训练效果吗？哪些东西才真的会？
8. 8 卡训 7B，你怎么排 DP/TP？为什么不 TP=8？

> 带答案的题库在 [显存 / 训练工程 / 分布式 自测](../03-training-fundamentals/self-test.md)。
