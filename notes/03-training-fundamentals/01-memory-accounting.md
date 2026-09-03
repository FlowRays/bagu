# 训练显存账本：一个参数到底占几个 byte

> **高频考点。** 目标是看到一个训练配置就能当场估显存。
> 主线：`数据类型 → P/G/O → activation → logits → 完整 SFT 账本`。

## 1. 两个大类

$$\boxed{M_{\text{SFT}}=\underbrace{M_P+M_G+M_O}_{\text{model states}}+\underbrace{M_A+M_{\text{logits}}+M_{\text{temp}}}_{\text{runtime}}}$$

- **model states** 只和**参数量 $N$** 有关
- **runtime** 只和 $B,L,H,N_{\text{layer}},V$ 有关

这个分离是所有显存直觉的地基。

## 2. 为什么有的 2 byte 有的 4 byte

$8\text{ bits}=1\text{ Byte}$，所以：

| 类型 | bits | Byte | 结构（sign / exponent / mantissa） |
|---|---:|---:|---|
| FP32 | 32 | **4** | 1 / 8 / 23 |
| FP16 | 16 | **2** | 1 / 5 / 10 |
| BF16 | 16 | **2** | 1 / **8** / 7 |

BF16 的 exponent 和 FP32 一样宽 → **动态范围大、不容易 overflow**，所以现在 LLM 训练里 BF16 比 FP16 常见。代价是 mantissa 只有 7 位，精度低。

## 3. Parameter / Gradient

权重通常 BF16：

$$M_P=2N$$

> **记死：BF16 裸权重，1B 参数 ≈ 2 GB。** 7B ≈ 14 GB，32B ≈ 64 GB，70B ≈ 140 GB。

每个可训练参数都有一个梯度，所以 $N_{\text{grad}}\approx N$。BF16 时：

$$M_G=2N$$

但 gradient 精度**不是定律**：BF16 / FP16 / FP32 都可能，取决于框架和 mixed-precision 配置。

## 4. Adam：最大的一块

AdamW 每个参数额外维护一阶、二阶动量：

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$

$$\theta\leftarrow\theta-\eta\frac{m_t}{\sqrt{v_t}+\epsilon}$$

$m,v$ 通常用 **FP32** 以保数值稳定：

$$M_O=(4+4)N=8N$$

## 5. 卡点：12 还是 16 bytes per param

| 内容 | Byte/param | 7B |
|---|---:|---:|
| BF16 parameter | 2 | 14 GB |
| BF16 gradient | 2 | 14 GB |
| Adam $m$ | 4 | 28 GB |
| Adam $v$ | 4 | 28 GB |
| **小计** | **12** | **84 GB** |
| （可选）FP32 master weight | 4 | 28 GB |
| **含 master weight** | **16** | **112 GB** |

$$\boxed{12\text{ B/param：BF16 P + BF16 G + FP32 Adam}}\qquad\boxed{16\text{ B/param：再加 FP32 master weight}}$$

面试别说"训练一定 16 byte/param"，正确说法：

> **经典 mixed-precision Adam 估算通常是 12–16 Byte/parameter，具体取决于 gradient precision 和是否维护 FP32 master weights。**

## 6. Activation：为什么必须存

$y=Wx$ 的 backward 是

$$\frac{\partial L}{\partial W}=\frac{\partial L}{\partial y}x^\top$$

**需要 forward 的输入 $x$**。所以 autograd 建 computation graph 并保留中间量，等 backward 用。

Transformer 一层里要存的东西很多：input hidden、normalized hidden、Q/K/V、attention output、MLP intermediate（$H\to3H\sim4H$，比 hidden 还大）。

$$\boxed{M_A\propto B\times L\times H\times N_{\text{layer}}}$$

一份 BF16 hidden state $[B,L,H]$：$B=4,L=8192,H=4096$ 时

$$4\times8192\times4096\times2\approx268\text{ MB}$$

**仅仅一份 tensor**，再乘每层若干份、乘 32/40/80 层，很快几十 GB。

### Attention 的 $L^2$ 项与 FlashAttention

naive attention 的 $QK^\top\in\mathbb R^{B\times h\times L\times L}$，显存 $O(L^2)$，$L$ 翻倍显存 ×4。

FlashAttention 通过分块计算**不把完整 $L\times L$ 矩阵写进 HBM**，把 attention 显存降到接近 $O(L)$。

$$\boxed{\text{计算复杂度仍然是 }O(L^2)\text{，省的是显存和 IO}}$$

## 7. 容易忽略的大头：logits

LM head 输出 $Z\in\mathbb R^{B\times L\times V}$。$B=4,L=8192,V=150\text{K}$、BF16：

$$4\times8192\times150000\times2\approx9.8\text{ GB}$$

所以框架会用 **fused cross entropy / chunked loss / 及时释放** 避免长期保存完整 logits。

## 8. 参数量和 batch/seq 影响的是不同东西

| 改动 | model states | activation |
|---|---|---|
| `batch 1 → 8`（seq 不变） | **完全不变** | 大幅增加 |
| `7B → 14B`（B、L 不变） | **约 ×2** | 也变大（$H$、$N_{\text{layer}}$ 变大） |

## 9. 各优化技术分别在治哪一块

| 技术 | 主要优化谁 |
|---|---|
| BF16 | parameter / activation |
| **Gradient Checkpointing** | **activation** |
| FlashAttention | attention activation / IO |
| **Gradient Accumulation** | **activation**（降 $B_{\text{micro}}$） |
| ZeRO-1 | **optimizer state** |
| ZeRO-2 | optimizer + **gradient** |
| ZeRO-3 / FSDP | optimizer + gradient + **parameter** |
| CPU offload | 把 states 搬到 CPU |
| LoRA | 大幅减少 trainable param / grad / optimizer |
| QLoRA | 再量化 frozen base parameter |

$$\boxed{M=P+G+O+A\ \Longrightarrow\ \text{每个技术对症一块}}$$

## 自测（口述版）

1. 写出 SFT 显存的两大类拆分，各自和什么变量有关。
2. FP32 / FP16 / BF16 各占几 byte？BF16 相比 FP16 的优势是什么，代价是什么？
3. 12 bytes/param 和 16 bytes/param 分别对应什么配置？7B 各是多少 GB？
4. 为什么 backward 必须保存 forward 的 activation？举一个最小例子。
5. activation 正比于哪几个量？估算 $B=4,L=8192,H=4096$ 时一份 BF16 hidden 的大小。
6. FlashAttention 把 attention 的什么从 $O(L^2)$ 降到 $O(L)$？什么没变？
7. logits 为什么可能有近 10 GB？框架怎么处理？
8. batch 从 1 加到 8，哪几项显存变、哪几项不变？
