# SFT 工程：loss mask、packing、多轮

> 理论上 SFT 就是一句 $-\log\pi_\theta(y|x)$（见 [SFT 与传统 KD](../distill/02-sft-and-kd.md)），
> 但实际做错的地方几乎全在**哪些 token 算 loss、怎么拼 batch、多轮怎么处理**这三件事上。

## 1. Loss mask：只在 assistant 上算

一条 SFT 样本渲染成 token 序列后：

```text
<|im_start|>system\n You are helpful. <|im_end|>      ← mask
<|im_start|>user\n 帮我写个快排 <|im_end|>              ← mask
<|im_start|>assistant\n def quicksort(...) <|im_end|>  ← 算 loss
```

$$\mathcal L_{\text{SFT}}=-\sum_{t\in\text{assistant}}\log\pi_\theta(y_t\mid x,y_{<t})$$

实现上给非目标位置填 `ignore_index = -100`，`F.cross_entropy` 会自动跳过。

### 卡点：shift 一位

位置 $t$ 的 logits 预测的是 $t+1$ 的 token，所以：

```python
shift_logits = logits[:-1]     # 最后一位没有 target
shift_labels = labels[1:]      # 真值往左挪一格
loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
```

**这是 SFT 实现最容易错的一步**，错了 loss 看起来正常但模型学的是错位的映射。手写实现见 [手撕：SFT 损失](../../code/07-handwrite/03-llm.md#sft-损失函数shift-right)。

### 要不要训练 prompt

| 做法 | 场景 |
|---|---|
| 只算 assistant（默认） | 绝大多数情况 |
| prompt 也算 loss | 数据极少时当正则；或者希望模型学会"补全用户可能怎么问" |

主流是**只算 assistant**。prompt 算 loss 会让模型把学习容量浪费在拟合输入分布上。

### 特殊 token 的边界

`<|im_end|>` **要算 loss**，否则模型不知道什么时候停，会一直生成。这是"模型不会停"这类 bug 的常见原因。

## 2. 多轮对话怎么算

一条多轮样本有多个 assistant 段：

```text
user A → assistant A → user B → assistant B → user C → assistant C
```

两种做法：

| 做法 | 说明 | 代价 |
|---|---|---|
| **拆成多条样本** | 每条只训一个 assistant 段，前面全当 context | 前缀被重复前向 $n$ 次，浪费 |
| **一条样本多段 loss** | 一次前向，所有 assistant 段一起算 loss | **推荐**，省 $n$ 倍计算 |

第二种需要 loss mask 支持"多段"，而不是简单的"最后一段"。

$$\boxed{\text{多轮不要拆样本，用多段 mask 一次算完}}$$

**注意**：causal mask 保证了第 $i$ 轮的 assistant 看不到第 $i+1$ 轮的内容，所以一次算完在数学上和拆开算是等价的。

## 3. Packing

短样本 pad 到 max_len 会浪费大量算力。把多条拼进一个序列：

```text
不 packing:  [A 500 real + 3596 pad][B 800 real + 3296 pad]
packing:     [A 500][B 800][C 1200][D ...] ──────> 4096
```

必须配套三件事，否则**不等价**：

1. **block-diagonal causal mask**（或 `cu_seqlens` 走 FlashAttention 变长接口）—— B 不能看见 A
2. **position_ids 每条重新从 0 开始**
3. **loss 不跨样本边界**算 next-token prediction

详见 [Sequence Packing](../../03-training-fundamentals/04-packing-and-grad-clip.md#一sequence-packing)。

### 卡点：packing 之后的 loss normalization

这是变长 SFT 最隐蔽的坑。如果按「每条样本的平均 loss 再平均」，长短样本会被强行拉平；正确做法是按**总有效 token 数**归一化：

$$\mathcal L=\frac{\sum_{i}\sum_{t\in\text{assistant}_i}\ell_{i,t}}{\sum_i |\text{assistant}_i|}$$

否则 microbatch 之间权重不对，详见 [梯度累积](../../03-training-fundamentals/03-gradient-accumulation.md#4-卡点真正的坑是-token-normalization)。

## 4. 超参与经验

| 项 | 典型值 | 说明 |
|---|---|---|
| epoch | **1–3** | SFT 极易过拟合，3 轮以上通常掉点 |
| lr | 1e-5 ~ 2e-5（全参） | 比 pretrain 小 1–2 个量级 |
| lr schedule | cosine + warmup 3% | |
| batch | 128–512 条（按 token 数更准） | |
| max_len | 覆盖 99% 数据即可 | 太长浪费，太短截断丢信息 |

**为什么 lr 要小**：SFT 是在一个已经很好的分布上做微调，大 lr 会破坏 pretrain 学到的能力（灾难性遗忘）。

## 5. LoRA vs 全参

| | 全参 SFT | LoRA |
|---|---|---|
| 显存 | $P+G+O+A$，12–16 B/param | 只有 LoRA 参数有 $G,O$，base 只占 $P$ |
| 效果 | 上限更高 | 接近，但改变"怎么看/怎么想"这类深层能力时弱一些 |
| 多任务 | 每个任务一份完整权重 | 一个 base + 多个小 adapter，**可热插拔** |
| 推理 | 直接用 | 可 merge 回 base，零额外延迟 |

LoRA 的关键点：$W$ 冻结，只训 $A,B$，**$B$ 初始化为 0** 所以起点严格等于原模型；缩放 $\alpha/r$。实现见 [手撕：LoRA](../../code/07-handwrite/03-llm.md#lora-低秩适配)。

显存对比见 [显存账本](../../03-training-fundamentals/01-memory-accounting.md#9-各优化技术分别在治哪一块)。

## 6. 灾难性遗忘

SFT 之后通用能力下降是常见问题。缓解手段：

- **数据混合**：SFT 数据里掺一定比例的通用数据 / pretrain 数据（replay）
- **小 lr + 少 epoch**
- **LoRA**：参数改动受限，天然不容易遗忘
- **正则**：加对 base model 的 KL 惩罚（这就已经接近 [OPD](../distill/03-opd.md) 的思路了）

## 自测（口述版）

**1.** 写出 SFT 的 loss，loss mask 覆盖哪些位置？

> **答：** $$\mathcal L_{\text{SFT}}=-\sum_{t\in\text{assistant}}\log\pi_\theta(y_t\mid x,y_{<t})$$
> system 段和 user 段全部 mask（实现上填 `ignore_index=-100`，`F.cross_entropy` 自动跳过），**只在 assistant 段算 loss**。

**2.** shift 一位是怎么做的？写出三行代码。错了会怎样？

> **答：** 位置 $t$ 的 logits 预测的是 $t+1$ 的 token，所以要错开一格：
> ```python
> shift_logits = logits[:-1]     # 最后一位没有 target
> shift_labels = labels[1:]      # 真值往左挪一格
> loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
> ```
> 错了的话 **loss 看起来正常但模型学的是错位的映射**，训练曲线下降、生成却乱套。这是 SFT 实现最容易错的一步。

**3.** `<|im_end|>` 要不要算 loss？不算会出什么问题？

> **答：** **要算。** 不算的话模型不知道什么时候该停，会一直生成下去。这是「模型不会停 / 停不下来」这类 bug 最常见的原因。

**4.** 多轮对话两种做法各是什么？为什么推荐一次算完？数学上等价吗？

> **答：** ① **拆成多条样本**，每条只训一个 assistant 段、前面全当 context —— 前缀被重复前向 $n$ 次，浪费；
> ② **一条样本多段 loss**，一次前向把所有 assistant 段一起算 —— **推荐**，省 $n$ 倍计算，但需要 loss mask 支持「多段」而不是「最后一段」。
> **数学上等价**：causal mask 保证第 $i$ 轮的 assistant 看不到第 $i+1$ 轮的内容，所以一次算完和拆开算结果相同。

**5.** packing 必须配套哪三件事？少了会怎样？

> **答：** ① **block-diagonal causal mask**（或 `cu_seqlens` 走 FlashAttention 变长接口）；② **`position_ids` 每条重新从 0 开始**；③ **loss 不跨样本边界**算 next-token prediction。
> 少了任何一条就**不等价**：不同样本会互相 attention 造成信息串流，或位置编码错乱，训练效果下降。

**6.** packing 之后 loss 怎么归一化？按每条样本平均错在哪？

> **答：** 按**总有效 token 数**归一化：
> $$\mathcal L=\frac{\sum_i\sum_{t\in\text{assistant}_i}\ell_{i,t}}{\sum_i|\text{assistant}_i|}$$
> 按「每条样本的平均 loss 再平均」会把长短样本**强行拉平**（1000 token 的样本和 3000 token 的样本各占 50% 权重），microbatch 之间权重不对，梯度就偏了。

**7.** SFT 的 lr 为什么比 pretrain 小 1–2 个量级？epoch 为什么只有 1–3？

> **答：** **lr 小**：SFT 是在一个已经很好的分布上做微调，大 lr 会破坏 pretrain 学到的能力（灾难性遗忘）。典型 1e-5 ~ 2e-5（全参）。
> **epoch 少**：SFT 数据量相对小且高度一致，极易过拟合，3 轮以上通常掉点。这也和 LIMA 的结论一致 —— SFT 主要教「表达方式」而不是灌知识。

**8.** LoRA 和全参 SFT 在显存、效果、多任务、推理上各有什么差别？$B$ 为什么初始化为 0？

> **答：** **显存**：全参是 $P+G+O+A$（12–16 B/param）；LoRA 只有 adapter 有 $G,O$，base 只占 $P$，省很多。
> **效果**：全参上限更高；LoRA 接近，但改变「怎么看 / 怎么想」这类深层能力时弱一些。
> **多任务**：全参每个任务一份完整权重；LoRA 一个 base + 多个小 adapter，**可热插拔**。
> **推理**：LoRA 可 merge 回 base，零额外延迟。
> $B=0$ 使起点 $x W+(xAB)\cdot s=xW$ **严格等于原模型**，训练从原模型平滑出发（$A$、$B$ 不能都非零，否则起点就偏了）。

**9.** 灾难性遗忘的四种缓解手段？

> **答：** ① **数据混合**：SFT 数据里掺一定比例通用数据 / pretrain 数据（replay）；② **小 lr + 少 epoch**；③ **LoRA**：参数改动受限，天然不易遗忘；④ **加对 base model 的 KL 惩罚**（这已经接近 OPD 的思路了）。

