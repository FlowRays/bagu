# Sequence Packing 与 Gradient Clipping

> 两个高频工程八股，都很小但很容易问细节。

## 一、Sequence Packing

**把多个短样本拼进同一条长 sequence，减少 padding 浪费。**

max length = 4096，样本 A/B/C 分别 500/800/1200 token：

```text
不 packing:  A[500 real + 3596 pad]   B[800 + 3296 pad]   C[1200 + 2896 pad]
packing:     [A 500][B 800][C 1200][D …] ──────────────> 4096
```

绝大部分计算本来浪费在 padding 上。

### 关键：不同样本之间不能互相 attention

必须是 **block-diagonal causal attention**：

```text
        A1 A2 A3 | B1 B2
       ----------+------
   A1   ✓  ✓  ✓  | ×  ×
   A2   ✓  ✓  ✓  | ×  ×
   A3   ✓  ✓  ✓  | ×  ×
       ----------+------
   B1   ×  ×  ×  | ✓  ✓
   B2   ×  ×  ×  | ✓  ✓
```

配套需要：

- `attention mask` / `cu_seqlens`（FlashAttention 的变长接口）
- **`position_ids` 每条 sequence 重新从 0 开始**
- loss mask：不计算跨样本交界处的 next-token prediction
- loss normalization 保持一致

### 是等价的吗

理想实现下**是**：

$$L_{\text{packed}}=L_A+L_B+L_C,\qquad \nabla L_{\text{packed}}=\nabla L_A+\nabla L_B+\nabla L_C$$

$$\boxed{\text{packing 只改变"怎么把数据塞进 GPU"，不改变训练目标}}$$

**坑**：如果 position 没 reset，或者不同 sequence 之间能互相 attention，就**不等价**了。

一句话：**packing 是空间利用率优化，提高 token utilization 和吞吐。** SFT / pretraining 中样本长度差异大时尤其有用。

## 二、Gradient Clipping

**防止偶发的巨大梯度把一次 optimizer step 搞崩。**

正常 $\|g\|\approx1$，某个 batch 突然 $\|g\|=1000$，那么 $\theta\leftarrow\theta-\eta g$ 会导致 loss spike / NaN / 发散。

### Global norm clipping（LLM 默认）

$$\|g\|_2=\sqrt{\sum_i g_i^2},\qquad \boxed{g'=g\cdot\min\Big(1,\frac{c}{\|g\|_2}\Big)}$$

$c$ 就是 `max_grad_norm`（常见 1.0）。例：

```text
before: layer1 grad = 3, layer2 grad = 4  →  global norm = 5
c = 1  →  ×1/5  →  layer1: 0.6, layer2: 0.8  →  新 norm = 1
```

**所有参数乘同一个系数**，所以

$$\boxed{\frac{g'}{\|g'\|}=\frac{g}{\|g\|}\quad\text{方向完全不变，只缩短长度}}$$

$\|g\|\le c$ 时完全不处理 —— 它是个**保险丝**，不是每步都主动缩小。

### 卡点：norm clipping ≠ value clipping

"每个 gradient 超过 1 就设成 1"那叫 **value clipping**，LLM 训练里更常见的是 **norm clipping**。

### 卡点：会不会导致梯度消失

**不会。** clipping 只在 $\|g\|>c$ 时触发，而且缩放后 norm 恰好等于 $c$（比如 1.0），离浮点下溢还非常远。

真正的 **vanishing gradient** 指的是深层链式相乘导致的指数衰减：

$$\frac{\partial L}{\partial x_1}=\frac{\partial L}{\partial x_n}\prod_{i=1}^{n-1}\frac{\partial x_{i+1}}{\partial x_i}$$

两个是不同问题。

但**阈值设太小确实会拖慢训练**：正常 $\|g\|\approx10$ 而你设 $c=0.01$，等于每步缩小约 1000 倍，大量 step 都被 clip，有效 step size 长期过小。这叫"收敛变慢"，不叫 gradient vanishing。

### FSDP / ZeRO 下怎么算 global norm

梯度被 shard 到不同 GPU，**不能每张卡各算各的 norm**。必须跨卡聚合各 shard 的平方和得到真正的 global grad norm，再统一缩放。这一点面试挺喜欢问。

## 三、一起记

| 技术 | 解决什么 | 本质 |
|---|---|---|
| Sequence Packing | padding 太多、GPU 浪费 | 提高有效 token 比例 |
| Gradient Clipping | 梯度爆炸、loss spike | 限制一次参数更新的**幅度**，不改方向 |

## 自测（口述版）

**1.** packing 需要哪三样配套？少了 position reset 会怎样？

> **答：** ① **block-diagonal causal attention mask**（或 `cu_seqlens` 走 FlashAttention 变长接口），保证 B 看不见 A；② **`position_ids` 每条 sequence 重新从 0 开始**；③ **loss mask** 不计算跨样本交界处的 next-token prediction；（外加 loss normalization 保持一致）。
> 少了 position reset，第二条样本的位置编码会从上一条的末尾接着数，位置信息错乱，训练效果会掉。

**2.** 证明 packing 在理想实现下和不 packing 等价。

> **答：** 配套做对之后，B 完全看不到 A、A 也不受 B 影响，每条样本的条件概率与单独前向时完全一致，于是
> $$L_{\text{packed}}=L_A+L_B+L_C,\qquad \nabla L_{\text{packed}}=\nabla L_A+\nabla L_B+\nabla L_C$$
> 和正常 batch 一样。所以 **packing 只改变「怎么把数据塞进 GPU」，不改变训练目标**。

**3.** 写出 global norm clipping 的公式，用 grad=(3,4)、$c=1$ 算一遍。

> **答：** $g'=g\cdot\min\big(1,\frac{c}{\|g\|_2}\big)$，$\|g\|_2=\sqrt{\sum_i g_i^2}$。
> $\|g\|=\sqrt{3^2+4^2}=5>1$，所以全部乘 $1/5$ → $(0.6,\,0.8)$，新 norm $=\sqrt{0.36+0.64}=1$。
> **所有参数乘同一个系数**，所以 $\frac{g'}{\|g'\|}=\frac{g}{\|g\|}$：方向完全不变，只缩短长度。

**4.** norm clipping 和 value clipping 的区别？LLM 用哪个？

> **答：** value clipping 是「每个 gradient 分量超过阈值就设成阈值」，会改变梯度方向；
> norm clipping 是按整体 L2 norm 等比例缩放，方向不变。
> **LLM 训练用的是 global norm clipping。**

**5.** clipping 会不会造成梯度消失？真正的 vanishing gradient 是什么？阈值太小会怎样？

> **答：** **不会。** 它只在 $\|g\|>c$ 时触发，而且缩放后 norm 恰好等于 $c$（比如 1.0），离浮点下溢还非常远。
> 真正的 vanishing gradient 是深层链式相乘的指数衰减：$\frac{\partial L}{\partial x_1}=\frac{\partial L}{\partial x_n}\prod_i\frac{\partial x_{i+1}}{\partial x_i}$，是另一个问题。
> 但**阈值设太小确实会拖慢训练**：正常 $\|g\|\approx10$ 却设 $c=0.01$，等于每步缩小约 1000 倍，有效 step size 长期过小。这叫收敛变慢，不叫梯度消失。

**6.** FSDP 下 global grad norm 怎么算？

> **答：** 梯度被 shard 到不同 GPU 上，**不能每张卡各算各的 norm**。必须**跨卡聚合各 shard 的平方和**得到真正的 global grad norm，再统一缩放。这一点面试挺喜欢问。

