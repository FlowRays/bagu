# Normalization 与 residual stream

> 高频考点：LayerNorm vs RMSNorm vs BatchNorm，pre-norm 为什么赢了，QK-norm 解决什么。
> 手写实现见 [手撕：深度学习](../code/07-handwrite/02-dl.md)。

## 1. 三种 Norm 的对比

$$\text{LayerNorm}:\quad y=\gamma\cdot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,\qquad \mu,\sigma^2\ \text{沿}\ \textbf{特征维}\ \text{求}$$

$$\text{RMSNorm}:\quad y=\gamma\cdot\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}\qquad\textbf{不减均值、没有}\ \beta$$

$$\text{BatchNorm}:\quad \mu,\sigma^2\ \text{沿}\ \textbf{batch 维}\ \text{求}$$

| | 统计量沿哪一维 | 参数 | 推理时 |
|---|---|---|---|
| LayerNorm | 特征维（每个 token 自己） | $\gamma,\beta$ | 和训练一样 |
| RMSNorm | 特征维 | 只有 $\gamma$ | 和训练一样 |
| BatchNorm | **batch 维** | $\gamma,\beta$ | 要用**滑动平均**的统计量 |

### 卡点：LLM 为什么不用 BatchNorm

三条，按重要性：

1. **推理时 batch 和 seq 长度一直在变。** BatchNorm 依赖 batch 内统计量，batch=1 时方差没有意义。训练/推理行为不一致是致命的。
2. **语义不对。** BatchNorm 是「跨样本比较同一个 feature」，LayerNorm 是「在一个样本内部比较所有 feature」。语言模型每个 token 是独立的语义单元，后者才合理。
3. **变长序列 + padding** 会污染 batch 统计量。

$$\boxed{\text{BatchNorm 跨样本；LayerNorm 样本内。LLM 只能用后者。}}$$

### RMSNorm 为什么够用

去掉了减均值和 $\beta$，省一次求均值、一个参数、一次减法。实践发现效果基本不掉，所以 LLaMA 之后基本都是 RMSNorm。

直觉：Transformer 的 residual stream 本身有 LayerNorm 反复作用，再减一次均值的边际收益很小；真正起作用的是**把尺度控制住**。

## 2. Pre-norm vs Post-norm

$$\text{Post-norm}:\quad x\leftarrow \text{LN}\big(x+\text{Sublayer}(x)\big)$$
$$\text{Pre-norm}:\quad x\leftarrow x+\text{Sublayer}\big(\text{LN}(x)\big)$$

原始 Transformer 是 post-norm，现代 LLM 几乎全是 **pre-norm**。

### 为什么 pre-norm 更好训

pre-norm 下展开整个网络：

$$x_L=x_0+\sum_{l=1}^{L}\text{Sublayer}_l\big(\text{LN}(x_l)\big)$$

**从输入到输出存在一条完全没有被 norm 阻断的恒等路径**（residual stream）。梯度可以直接流回浅层，不会因为逐层 norm 而被反复缩放。

post-norm 则是每层输出都被 LN 重新标定，深层网络里梯度尺度容易失控，需要很小心的 warmup 才能训起来。

| | Pre-norm | Post-norm |
|---|---|---|
| 训练稳定性 | **好**，几乎不需要精细 warmup | 差，深了容易崩 |
| 最终效果 | 略差一点点（有研究认为表达能力稍弱） | 略好 |
| 现代 LLM | **主流** | 少见 |

折中方案有 **sandwich norm**（sublayer 前后各来一次）等。

## 3. Residual stream 的视角

把 pre-norm 的展开式再看一遍：

$$x_L=x_0+\sum_l \text{Sublayer}_l(\cdot)$$

可以理解成：**有一条贯穿全网络的"主干道"（residual stream），每一层只是往上面"加"自己的贡献**，而不是"替换"。

这个视角很有用：

- 解释了为什么可以做 **layer pruning / early exit**：删掉几层只是少加了几项
- 解释了 [DeepStack](../07-vlm/02-vision-encoder.md#deepstack不只用最后一层) 为什么能把视觉特征直接加进 LLM 的前几层 hidden state
- 解释了 LoRA 为什么有效：在主干道上加一个低秩的增量

## 4. QK-norm

**问题**：训练到后期，$q$、$k$ 的范数可能变得很大，导致

$$\frac{q^\top k}{\sqrt{d_h}}$$

数值过大 → softmax 饱和 → 梯度消失，或者直接出现 loss spike。

**做法**：在算 attention 之前分别对 $q$、$k$ 做一次 RMSNorm（**按 head 维**）：

$$\tilde q=\text{RMSNorm}(q),\quad \tilde k=\text{RMSNorm}(k),\quad s=\frac{\tilde q^\top\tilde k}{\sqrt{d_h}}$$

效果是把 attention logits 的尺度锁住，训练稳定性明显提升。现在很多模型（Gemma 2、Qwen3 等）都加了这个。

$$\boxed{\text{QK-norm 治的是 attention logits 爆炸导致的训练不稳，不是为了效果}}$$

## 5. No bias

现代 LLM 的线性层普遍**去掉 bias**（LLaMA 系全去，Qwen 保留了 QKV 的 bias）。

理由：

- 参数量和显存都省一点（虽然占比很小）
- 有研究发现去掉 bias 对效果没有损失，甚至训练更稳
- 少一个需要同步的张量，分布式实现更简单

## 自测

1. 写出 LayerNorm 和 RMSNorm 的公式，指出两处差别。
2. LLM 为什么不用 BatchNorm？说出三条。
3. 写出 pre-norm 和 post-norm 的公式。为什么 pre-norm 更好训？用展开式解释。
4. residual stream 是什么视角？它能解释哪几件事？
5. QK-norm 解决什么问题？加在哪一步？
6. 为什么现代 LLM 普遍去掉 bias？
