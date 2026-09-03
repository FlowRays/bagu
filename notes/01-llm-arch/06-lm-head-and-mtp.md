# LM head、weight tying 与 MTP

## 1. 从 hidden 到 token

$$h_t\in\mathbb R^{d}\ \xrightarrow{\ W_{\text{vocab}}\in\mathbb R^{d\times V}\ }\ z_t\in\mathbb R^{V}\ \xrightarrow{\text{softmax}}\ p_t$$

训练时对所有位置并行算；推理时只需要最后一个位置的 logits。

**这一步是训练显存的大头**：$[B,L,V]$ 的 logits，$B=4,L=8192,V=150\text{K}$、BF16 下约 9.8 GB。所以框架都用 fused cross entropy / chunked loss，算一块释放一块，不长期保存完整 logits（见 [显存账本](../03-training-fundamentals/01-memory-accounting.md#7-容易忽略的大头logits)）。

## 2. Weight tying

让输入 embedding 和输出 LM head **共享同一个矩阵**：

$$W_{\text{vocab}}=E^\top$$

| | 好处 | 代价 |
|---|---|---|
| tie | 省 $V\times d$ 参数（$V=150$K、$d=4096$ 时约 0.6B）；小模型上还能提效果（正则作用） | 输入和输出空间被强制绑定，大模型上限制表达能力 |

**经验规律**：小模型倾向 tie，大模型倾向不 tie。DeepSeek-V4 系列 `tie_word_embeddings=false`，所以 embedding + head 是两份，参数量要乘 2。

算参数量时这是个很容易错的点：

$$P_{\text{emb+head}}=\begin{cases}Vd & \text{tie}\\ 2Vd & \text{不 tie}\end{cases}$$

## 3. MTP：Multi-Token Prediction

标准 LM 每个位置只预测下一个 token。MTP 让模型在同一个位置**同时预测未来 $n$ 个 token**：

```text
标准:  h_t → 预测 x_{t+1}
MTP :  h_t → 预测 x_{t+1}, x_{t+2}, ..., x_{t+n}
```

实现上通常是主干输出后接 $n$ 个轻量的预测头（DeepSeek-V3 是串行的 MTP module，每个模块自己带一层 Transformer）。

### 两个作用

**训练时**：额外的监督信号，强迫表示编码更长程的信息，实测能提效果。这是主要动机。

**推理时**：MTP 头可以作为**自投机解码**的 draft —— 一次前向猜出未来几个 token，再用主模型验证，接受了就白赚，不接受就回退。省掉了额外的 draft 模型（见 [推理优化](../02-inference-serving/03-speculative-and-quant.md)）。

$$\boxed{\text{MTP：训练当辅助目标，推理当自带的 draft model}}$$

GLM-5.3 的 config 里 `num_nextn_predict_layers=1` 就是一个 MTP layer，算参数量时不能漏。

## 4. 采样

logits 出来之后怎么变成 token，见 [推理与采样](../02-inference-serving/01-inference-basics.md#4-采样)。

## 自测（口述版）

**1.** LM head 这一步为什么是训练显存的大头？框架怎么处理？

> **答：** 输出 logits 是 $[B,L,V]$，$B=4,L=8192,V=150\text{K}$、BF16 下约 9.8 GB。
> 框架用 **fused cross entropy / chunked loss**，算一块释放一块，不长期保存完整 logits。

**2.** weight tying 是什么？好处和代价？大小模型的经验规律？

> **答：** 让输入 embedding 和输出 LM head 共享同一个矩阵：$W_{\text{vocab}}=E^\top$。
> 好处：省 $V\times d$ 参数（$V=150$K、$d=4096$ 时约 0.6B），小模型上还有正则作用能提效果。代价：输入和输出空间被强制绑定，大模型上限制表达能力。
> 经验规律：**小模型倾向 tie，大模型倾向不 tie**。

**3.** 算 embedding+head 参数量时，tie 和不 tie 差多少？

> **答：** tie 是 $Vd$，不 tie 是 $2Vd$，**差一倍**。DeepSeek-V4 系列 `tie_word_embeddings=false`，算参数量时必须乘 2，这是很容易错的点。

**4.** MTP 是什么？训练时和推理时各起什么作用？

> **答：** Multi-Token Prediction：让模型在同一个位置**同时预测未来 $n$ 个 token**，实现上通常是主干后接 $n$ 个轻量预测头。
> **训练时**是额外的监督信号，强迫表示编码更长程的信息，实测能提效果（主要动机）。
> **推理时**这些头可以当**自投机解码的 draft**，一次前向猜出未来几个 token 再用主模型验证。

**5.** 为什么说 MTP 让模型自带 draft model？

> **答：** 投机解码需要一个又小又快、分布又接近 target 的 draft。MTP 头就长在主干上，天然同源、分布接近，而且不需要额外训练和部署一个独立的小模型，一次前向就顺带产出了候选。

