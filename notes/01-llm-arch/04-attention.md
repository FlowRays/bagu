# Attention 家族：MHA → MQA → GQA → MLA

> **最高频的考点**，手撕出现次数第一名就是 MHA。
> 手写实现（NumPy + PyTorch 两版）见 [手撕：大模型](../code/07-handwrite/03-llm.md)。

## 1. 基本形式

$$\text{Attn}(Q,K,V)=\text{softmax}\Big(\frac{QK^\top}{\sqrt{d_h}}\Big)V$$

**为什么除 $\sqrt{d_h}$**：$q,k$ 各维独立、方差为 1 时，$q^\top k$ 的方差是 $d_h$。不缩放的话 $d_h$ 一大，logits 的尺度就大，softmax 饱和到接近 one-hot，梯度趋近 0。除以 $\sqrt{d_h}$ 把方差拉回 1。

**为什么是 $\sqrt{d_h}$ 而不是 $d_h$**：因为要抵消的是标准差不是方差。

### Causal mask

$$s_{ij}=\begin{cases}\dfrac{q_i^\top k_j}{\sqrt{d_h}} & j\le i\\[4pt] -\infty & j>i\end{cases}$$

**必须在 softmax 之前填 $-\infty$**，不能 softmax 之后乘 0：后者的分母里仍然包含了被屏蔽位置，权重就不对了。

工程写法见 [PyTorch 速成](../code/01-python-pytorch/02-torch-for-cpp.md#四causal-mask-和-masked_fill)。

## 2. 多头：为什么要拆头

单头 attention 只能学一种「关注模式」。拆成 $h$ 个头，每个头在 $d_h=d/h$ 的子空间里独立算，可以同时关注语法、指代、位置等不同关系，最后拼起来过 $W_O$ 融合。

$$\text{MHA}(X)=\big[\text{head}_1;\dots;\text{head}_h\big]W_O$$

注意总计算量和单头 $d$ 维基本一样（$h\cdot d_h=d$），拆头几乎是免费的表达力提升。

## 3. KV cache 是这一切的动机

自回归解码时，第 $t$ 步要算 $q_t$ 对 $k_{1..t}, v_{1..t}$ 的 attention。前面的 $k,v$ 每步都一样，所以缓存下来：

$$M_{KV}=2\times B\times L\times N_{\text{layer}}\times h_{kv}\times d_h\times \text{bytes}$$

（2 是 K 和 V 两份。）

**算一笔账**：$L=32\text{K}$、$N_{\text{layer}}=32$、$h=32$、$d_h=128$、BF16、$B=1$：

$$2\times32768\times32\times32\times128\times2\ \text{B}\approx 17.2\ \text{GB}$$

**一条 32K 的序列，KV cache 就 17 GB**，比 7B 模型的权重还大。这就是为什么要有 MQA/GQA/MLA。

## 4. MQA / GQA：砍 KV 头数

$$\boxed{\text{MHA：}h_{kv}=h\qquad \text{GQA：}1<h_{kv}<h\qquad \text{MQA：}h_{kv}=1}$$

Q 仍然是 $h$ 个头，但 K/V 只投影出 $h_{kv}$ 组，每组被 $g=h/h_{kv}$ 个 query head 共享。

| | $h_{kv}$ | KV cache | 质量 |
|---|---:|---:|---|
| MHA | $h$ | $1\times$ | 基准 |
| GQA | $h/g$ | $1/g$ | 几乎不掉 |
| MQA | 1 | $1/h$ | 明显下降 |

GQA 是现在的默认选择（LLaMA-2 70B 之后、Qwen、Mistral 都是），典型 $h=32,h_{kv}=8$，KV cache 直接省 4 倍。

**实现要点**：$W_K,W_V$ 的输出维是 $h_{kv}\cdot d_h$，**比 $W_Q$ 窄**。算 attention 前把 K/V 沿 head 维广播回 $h$ 组，必须用 `repeat_interleave`（AABBCC）不是 `repeat`（ABCABC），用错了 head 和 KV 组就对不上。

## 5. MLA：换个方向压

GQA 是**减少 KV 头数**，MLA（DeepSeek）是**把 KV 压到低维 latent**：

$$c_t = x_t W^{DKV}\in\mathbb R^{d_c},\qquad K=cW^{UK},\quad V=cW^{UV}$$

$$\boxed{\text{KV cache 只存 }c\text{，维度 }d_c\ll 2\,h\,d_h}$$

推理时甚至可以把 $W^{UK}$ 吸收进 $W_Q$（因为 $q^\top(cW^{UK})^\top=(qW^{UK\top})c^\top$），连升维都省了。

**代价**：RoPE 不能直接作用在压缩后的 latent 上（旋转和低秩分解不交换），所以 MLA 需要**额外拆出一个带 RoPE 的小分支**（decoupled RoPE），这是它实现上最绕的地方。

| | 压缩方向 | KV cache |
|---|---|---|
| GQA | 减少头数 | $2\,h_{kv}d_h$ |
| MLA | 降低维度 | $d_c$（+ RoPE 分支） |

DeepSeek 报告 MLA 的 KV cache 比 MHA 小一个数量级，同时质量不降反升。

## 6. Cross attention

Q 来自一个序列，K/V 来自另一个。用在 encoder-decoder、以及一些 VLM 的融合结构里。

现代 decoder-only LLM + VLM 主流是把 visual token 和 text token **拼起来做统一 self-attention**（single-stream），而不是 cross attention，见 [VLM 架构](../07-vlm/01-architecture.md#7-统一-self-attention四个象限)。

## 7. Attention sink

**现象**：训练好的 LLM 里，几乎所有 head 都会分配相当一部分注意力给序列**最开头的几个 token**，哪怕那几个 token 语义上毫无关系（比如 `<bos>`）。

**解释**：softmax 强制权重和为 1。当某个 query 其实"什么都不想关注"时，它需要一个地方倾倒这些概率质量，最靠前、所有 query 都能看到的位置就成了默认的垃圾桶。

**后果**：做滑动窗口注意力时，如果把开头几个 token 也滑出去，模型会直接崩。**StreamingLLM** 的做法就是永久保留最前面的几个 sink token + 一个滑动窗口。

$$\boxed{\text{attention sink 是 softmax 归一化的副产物，不是 bug，但必须在长文本策略里显式照顾}}$$

一些新模型直接给 softmax 加一个可学习的 sink logit（等价于允许"不关注任何 token"），从根上解决。

## 8. 复杂度与 FlashAttention

| | 计算 | 显存（naive） | 显存（Flash） |
|---|---|---|---|
| attention | $O(L^2 d)$ | $O(L^2)$ | $O(L)$ |

FlashAttention 通过分块计算 + online softmax，**不把完整的 $L\times L$ 矩阵写进 HBM**。

$$\boxed{\text{计算复杂度仍是 }O(L^2)\text{，省的是显存和 HBM IO}}$$

它是 IO-aware 优化，不是算法复杂度优化。详见 [显存账本](../03-training-fundamentals/01-memory-accounting.md#attention-的平方项与-flashattention)。

## 9. 线性 / 稀疏注意力（了解即可）

前沿模型开始混用这些来打破 $O(L^2)$：

| 方向 | 代表 | 思路 |
|---|---|---|
| 线性注意力 | Gated DeltaNet、KDA（Kimi） | 用状态矩阵递推代替全局 softmax，$O(L)$ |
| 稀疏注意力 | DSA（DeepSeek）、NSA | 只算一部分 query-key 对，用 indexer 选 |
| 滑动窗口 | Mistral、Gemma | 每个 token 只看最近 $w$ 个 |

现在常见的是**混合**：大部分层用线性/稀疏，少数层保留 full attention 保证长程能力。Kimi K3 是 69 层 KDA + 24 层 Gated MLA（见 [参数构成](../04-distributed-infra/03-model-param-breakdown.md#3-为什么-activated-参数小这么多)）。

## 自测

1. 为什么除 $\sqrt{d_h}$ 而不是 $d_h$？不除会怎样？
2. causal mask 为什么必须在 softmax 之前？之后乘 0 错在哪？
3. 算一遍 32K 上下文、32 层、32 头、$d_h=128$、BF16 的 KV cache 有多大。
4. MHA / GQA / MQA 的区别？GQA 的 $W_K$ 和 $W_Q$ 形状有什么不同？广播用什么函数？
5. MLA 压的是什么？为什么它和 RoPE 有冲突，怎么解决？
6. attention sink 是什么现象？怎么解释？对滑动窗口有什么影响？
7. FlashAttention 把什么从 $O(L^2)$ 降到 $O(L)$？什么没变？
