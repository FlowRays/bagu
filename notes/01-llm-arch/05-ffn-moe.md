# FFN 与 MoE

> Attention 负责 token **之间**交换信息，FFN 负责在**每个 token 内部**做非线性变换。
> 手写实现见 [手撕：大模型](../code/07-handwrite/03-llm.md)。

## 1. FFN 在做什么

$$\text{FFN}(x)=W_2\,\sigma(W_1x+b_1)+b_2,\qquad d\to d_{ff}\to d$$

它对每个 token **独立**作用（position-wise），不看别的 token。

分工：

$$\boxed{\text{Attention：从别的 token 拿信息}\qquad\text{FFN：把拿到的信息在本 token 内部加工}}$$

一个常见解释是 FFN 相当于一组 key-value 记忆：$W_1$ 的每一行是一个"模式检测器"，$W_2$ 的每一列是对应的"输出内容"。Dense 模型里绝大部分参数都在 FFN。

典型 $d_{ff}=4d$。

## 2. 激活函数：GELU → SwiGLU

**GELU**：$x\cdot\Phi(x)$，比 ReLU 平滑，早期 Transformer 常用。

**SwiGLU**（LLaMA 系）引入门控，把一层拆成三个矩阵：

$$\text{SwiGLU}(x)=\big(\text{SiLU}(xW_{\text{gate}})\odot(xW_{\text{up}})\big)W_{\text{down}}$$

其中 $\text{SiLU}(z)=z\cdot\sigma(z)$。

**注意只有 gate 分支过 SiLU，up 分支不过。**

### 为什么 $d_{ff}$ 变成 $\frac83 d$

普通 FFN 两个矩阵：$2\cdot d\cdot d_{ff}=8d^2$（$d_{ff}=4d$）。
SwiGLU 三个矩阵：$3\cdot d\cdot d_{ff}$。

要保持参数量一致：

$$3\,d\,d_{ff}=8d^2\ \Rightarrow\ d_{ff}=\tfrac83 d$$

所以 LLaMA 的 $d_{ff}$ 是 $\frac83d$ 再向上取整到硬件友好的数。

$$\boxed{\text{SwiGLU 多一个矩阵，所以要缩 }d_{ff}\text{ 才公平比较}}$$

## 3. MoE：总参数大，激活参数小

把一个大 FFN 换成 $N$ 个小 expert，每个 token 只走其中 $k$ 个：

$$y=\sum_{j\in\text{TopK}}p_j\,E_j(x),\qquad p=\text{softmax}(\text{router}(x))$$

$$\boxed{P_{\text{total}}\gg P_{\text{active}}}$$

Kimi-K3：2.8T 总参数，每 token 只激活 104B（见 [参数构成](../04-distributed-infra/03-model-param-breakdown.md)）。

### 组件

| 组件 | 说明 |
|---|---|
| **router** | 一个小线性层 $x\to\mathbb R^N$，算每个 expert 的分数 |
| **top-k routing** | 每个 token 选分数最高的 $k$ 个（典型 $k=2\sim8$） |
| **routed expert** | 被 router 动态选中的 expert |
| **shared expert** | **每个 token 都走**，用来学通用能力，让 routed expert 专注差异化（DeepSeek 系） |

一个 GLU expert 的参数量：$3\,d\,d_{ff}$，整体 $P_{\text{MoE}}\approx 3dd_{ff}\cdot N\cdot N_{\text{MoE layer}}$。

### 卡点：为什么显存不按 activated 算

**权重和 optimizer state 由 total params 决定，每 token 的 FLOPs 由 activated params 决定。**

2.8T 的 MoE 不是"104B 模型的显存"，那 2.8T 参数和对应的 Adam 状态都得存下来。所以 MoE 换的是"同样算力下更大的容量"，不是"省显存"。

## 4. 负载均衡

**问题**：router 会自发地把大量 token 送给少数几个 expert（富者愈富），导致：

- 某些 expert 过载成为 straggler，拖慢整个 step
- 其他 expert 训练不足，容量浪费
- 极端情况 **expert collapse**：只有几个 expert 真正被用

### auxiliary load balancing loss

经典做法（Switch Transformer）：

$$\mathcal L_{\text{aux}}=\alpha\cdot N\sum_{j=1}^{N} f_j\,P_j$$

- $f_j$：实际被分到 expert $j$ 的 token **比例**
- $P_j$：router 给 expert $j$ 的**平均概率**

两者都均匀时该项最小。乘 $N$ 是为了让最优值与 $N$ 无关。

**问题**：它是一个和主任务无关的额外 loss，$\alpha$ 太大伤效果，太小不起作用。

### aux-loss-free balancing

DeepSeek-V3 提出的做法：给每个 expert 的 router 分数加一个**可学习的 bias** $b_j$，**只用于 top-k 选择，不进入最终的加权**：

$$\text{选 expert}:\ \text{TopK}(s_j+b_j),\qquad \text{加权用}:\ s_j$$

训练中动态调整：某个 expert 过载就调低它的 $b_j$，欠载就调高。

$$\boxed{\text{把负载均衡从"额外 loss"变成"路由时的偏置调节"，不污染主目标}}$$

### expert capacity 与 token dropping

工程上每个 expert 有容量上限（capacity factor）。超出的 token 会被**丢弃**（直接走 residual 跳过 FFN）。

$$\boxed{\text{EP 本身数学等价，但 token dropping 会真的改变训练}}$$

这是 [并行是否影响效果](../04-distributed-infra/02-parallelism-map.md#6-这些并行会影响训练效果吗) 里提到的少数几个「真的会改变训练」的因素之一。

## 5. Expert Parallel 与 All-to-All

不同 expert 放在不同 GPU 上，一层 MoE 的四步：

```text
router 打分 → All-to-All 把 token 发到对应 GPU → 各自算 expert FFN → All-to-All 送回来加权
```

$$\boxed{\text{TP 的典型通信是 All-Reduce；EP 的典型通信是 All-to-All}}$$

因为约 98% 的参数都是 expert，EP 才是大 MoE 最关键的并行维度。详见 [并行全图](../04-distributed-infra/02-parallelism-map.md#5-expert-parallelmoe-专用)。

## 6. 面试常见追问

**Q：MoE 为什么能又大又快？**
每个 token 只经过 $k/N$ 的 expert，FLOPs 接近一个小 dense 模型，但模型总容量是大模型。本质是用**稀疏激活**换容量。

**Q：MoE 的缺点？**
① 显存按 total 算，非常吃显存；② 负载不均导致实际吞吐远低于理论；③ All-to-All 通信重，对网络要求高；④ 训练不稳定，需要 balancing 机制；⑤ 小 batch 推理时 expert 利用率低。

**Q：为什么要 shared expert？**
让通用能力集中在 shared expert 上，routed expert 就能更专业化，减少不同 expert 重复学同样的基础能力。

## 自测（口述版）

**1.** Attention 和 FFN 的分工是什么？为什么说 FFN 是 position-wise 的？

> **答：** Attention 负责 token **之间**交换信息，FFN 负责在**每个 token 内部**把拿到的信息做非线性加工。
> FFN 对每个位置独立作用、不看别的 token，所以叫 position-wise。Dense 模型里绝大部分参数都在 FFN。

**2.** 写出 SwiGLU 的公式，指出哪个分支过激活。推导为什么 $d_{ff}=\frac83d$。

> **答：** $\text{SwiGLU}(x)=\big(\text{SiLU}(xW_{\text{gate}})\odot(xW_{\text{up}})\big)W_{\text{down}}$，**只有 gate 分支过 SiLU**，up 分支不过。
> 推导：普通 FFN 两个矩阵 $2dd_{ff}=8d^2$（$d_{ff}=4d$）；SwiGLU 三个矩阵 $3dd_{ff}$。令两者相等：$3dd_{ff}=8d^2\Rightarrow d_{ff}=\frac83d$。

**3.** 写出 MoE 的前向公式。routed expert 和 shared expert 的区别？

> **答：** $y=\sum_{j\in\text{TopK}}p_j E_j(x)$，其中 $p=\text{softmax}(\text{router}(x))$。
> **routed expert** 由 router 动态选中，每个 token 只走 $k$ 个；**shared expert 每个 token 都走**，用来承载通用能力，让 routed expert 专注差异化，避免每个 expert 都重复学基础能力。

**4.** 写出 auxiliary load balancing loss，解释 $f_j$ 和 $P_j$。它的缺点是什么？

> **答：** $\mathcal L_{\text{aux}}=\alpha\,N\sum_{j=1}^N f_j P_j$。$f_j$ 是实际被分到 expert $j$ 的 token **比例**，$P_j$ 是 router 给它的**平均概率**；两者都均匀时该项最小，乘 $N$ 是为了让最优值与 $N$ 无关。
> 缺点：它是一个和主任务无关的额外 loss，$\alpha$ 太大伤效果、太小不起作用。

**5.** aux-loss-free balancing 怎么做？bias 加在哪一步、不加在哪一步？

> **答：** 给每个 expert 的 router 分数加一个可学习 bias $b_j$，**只用于 top-k 选择**（$\text{TopK}(s_j+b_j)$），**不进入最终的加权**（加权仍用 $s_j$）。训练中动态调整：过载就调低 $b_j$、欠载就调高。
> 好处是把负载均衡从「额外 loss」变成「路由时的偏置调节」，不污染主目标。

**6.** MoE 的显存按 total 还是 activated 算？FLOPs 呢？

> **答：** **权重和 optimizer state 由 total params 决定；每 token 的 FFN FLOPs 由 activated params 决定。**
> 所以 2.8T 的 MoE 训练不是「104B 模型的显存」，那 2.8T 参数和对应的 Adam 状态都得存下来。MoE 换的是「同样算力下更大的容量」，不是省显存。

**7.** expert capacity 超了会怎样？这为什么会影响训练效果？

> **答：** 超出容量的 token 会被**丢弃**（直接走 residual 跳过 FFN）。
> 重要性在于：EP 本身是数学等价的并行方式，但 **token dropping 会真的改变训练**，是少数几个真正会影响效果的因素之一。

**8.** EP 的典型通信是什么？为什么不是 All-Reduce？

> **答：** 是 **All-to-All**（dispatch 一次、combine 一次）。
> 因为不同 expert 在不同 GPU 上，token 需要按 routing 结果**点对点重新分布**，而不是「所有卡算同一件事再规约」，所以是 All-to-All 而非 All-Reduce。

