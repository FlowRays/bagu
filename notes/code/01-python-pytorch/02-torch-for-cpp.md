# 给 C++ 选手的 PyTorch

> 前置：[给 C++ 选手的 NumPy](01-numpy-for-cpp.md)。torch 的张量语义几乎照搬 numpy，
> 所以这一篇只讲**多出来的三件事**：dtype/device、autograd、nn.Module，外加 API 对照。
>
> 手撕题里 35 道都给了 torch 版，见 [深度学习](../07-handwrite/02-dl.md)、[大模型](../07-handwrite/03-llm.md)。

## 一、tensor 和 ndarray 的对照

绝大部分同名同义，改动集中在几个点：

| NumPy | PyTorch | 备注 |
|---|---|---|
| `np.asarray(x)` | `torch.tensor(x)` / `torch.from_numpy(a)` | `from_numpy` **共享内存** |
| `a.astype(np.float32)` | `t.float()` / `t.to(torch.float32)` | |
| `a.reshape(...)` | `t.view(...)` / `t.reshape(...)` | **两者不等价**，见下节 |
| `a.transpose(1,0,2)` | `t.transpose(0,1)` / `t.permute(1,0,2)` | torch 的 `transpose` **只换两个轴** |
| `x[:, None, :]` | `t.unsqueeze(1)` | 也支持 `None` 写法 |
| `a.sum(axis=-1, keepdims=True)` | `t.sum(dim=-1, keepdim=True)` | **`axis`→`dim`，`keepdims`→`keepdim`（没有 s）** |
| `np.concatenate([a,b], 1)` | `torch.cat([a,b], 1)` | |
| `np.repeat(a, g, axis=0)` | `t.repeat_interleave(g, dim=0)` | `t.repeat` 是另一回事，见下 |
| `np.clip(x, lo, hi)` | `t.clamp(lo, hi)` | |
| `np.where(c, a, b)` | `torch.where(c, a, b)` | |
| `a.var(-1)` | `t.var(-1, unbiased=False)` | **torch 默认无偏（除 n-1），numpy 默认有偏（除 n）** |
| `a @ b` | `t @ t2` | 一样，高维都是批量矩阵乘 |
| `float(x)` | `t.item()` | 单元素张量取标量 |

三个最容易写错的：

$$\boxed{\texttt{keepdim}\ \text{没有 s}}\quad\boxed{\texttt{var}\ \text{的无偏默认相反}}\quad\boxed{\texttt{repeat}\ne\texttt{repeat\_interleave}}$$

```python
t = torch.tensor([1, 2, 3])
t.repeat(2)               # [1,2,3,1,2,3]     整体重复（≈ np.tile）
t.repeat_interleave(2)    # [1,1,2,2,3,3]     逐元素重复（≈ np.repeat）
```

GQA 里把 KV 广播给多个 query head，必须是 `repeat_interleave`，否则 head 和 KV 组对不上。

## 二、dtype 和 device：报错最多的地方

C++ 里类型不匹配编译期就挂了，Python 是运行时才炸，而且信息不直观。

```python
a = torch.randn(3, 4)                    # 默认 float32
b = torch.randn(3, 4, dtype=torch.float64)
a @ b        # RuntimeError: expected m1 and m2 to have the same dtype
```

**规则**：新建张量时永远跟随已有张量的 dtype 和 device。

```python
# 错：默认 float32 + CPU
self.A = nn.Parameter(torch.randn(din, r) * 0.01)

# 对：跟随 W
self.A = nn.Parameter(torch.randn(din, r, dtype=W.dtype, device=W.device) * 0.01)
```

这是 LoRA 那道题实际踩到的坑。同类的还有：

```python
mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
idx  = torch.arange(n, device=a.device)
y    = torch.zeros_like(x)          # 最省事：形状/dtype/device 全跟随
```

常见 dtype：`float32`（默认）、`float64`、`bfloat16`、`float16`、`int64`（`long`，**标签必须是这个**）、`bool`。

`F.cross_entropy` 的 target 必须是 `int64`，传 `int32` 会报错。

## 三、view / reshape / contiguous

这是 numpy 没有的一个显式概念，但根子还是 strides。

```python
t = torch.randn(2, 3, 4)
t.transpose(0, 1)          # 只换 strides，内存不连续了
t.transpose(0, 1).view(-1) # RuntimeError: view size is not compatible ...
t.transpose(0, 1).reshape(-1)              # OK，内部需要时自动复制
t.transpose(0, 1).contiguous().view(-1)    # OK，显式先复制成连续
```

| | 要求内存连续 | 会不会复制 |
|---|---|---|
| `view` | **是**，不满足直接报错 | 永不复制 |
| `reshape` | 否 | 需要时才复制 |
| `contiguous()` | — | 不连续时复制 |

**经验**：拆头做完 attention 拼回去，一律用 `reshape`。

```python
out.transpose(0, 1).reshape(L, d) @ Wo     # 别写 view，会报错
```

## 四、causal mask 和 masked_fill

你提到的这个写法就是标准做法：

```python
if mask is not None:
    scores = scores.masked_fill(mask == 0, float('-inf'))
```

`masked_fill(条件, 值)`：条件为 **True** 的位置填成给定值。所以要想清楚你的 mask 是「1 表示可见」还是「1 表示要屏蔽」。

两种常见约定：

```python
# 约定 A：mask 里 1/True 表示「可见」（多数教程、HuggingFace 的 attention_mask）
scores = scores.masked_fill(mask == 0, float('-inf'))     # 把不可见的填 -inf
scores = scores.masked_fill(~mask, float('-inf'))         # mask 是 bool 时

# 约定 B：mask 里 True 表示「要屏蔽」（PyTorch 的 attn_mask 语义）
scores = scores.masked_fill(mask, float('-inf'))
```

causal mask 的生成：

```python
L = x.size(0)
mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))  # 下三角 True = 可见
scores = scores.masked_fill(~mask, float('-inf'))
attn = F.softmax(scores, dim=-1)
```

**为什么填 `-inf` 而不是 0**：因为要在 **softmax 之前**屏蔽。`exp(-inf) = 0`，softmax 后该位置权重恰好为 0，而且分母里也不包含它。softmax 之后再乘 0 是错的，那样分母还算了被屏蔽的位置。

$$\boxed{\text{mask 一定加在 softmax 之前，填 }-\infty\text{ 不是 }0}$$

一个实际坑：如果某一行**全部**被屏蔽，softmax 会得到 `nan`（分母为 0）。变长 batch 里要保证每行至少有一个可见位置。

## 五、autograd

```python
x = torch.tensor([1.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()          # 反向传播，梯度累加到 x.grad
print(x.grad)         # tensor([2.])
```

四个必须知道的点：

1. **梯度是累加的**，不是覆盖。所以每步要 `opt.zero_grad()`（或 `x.grad = None`）。C++ 里没有对应概念，忘了就会得到「越训越离谱」。
2. **`.detach()` 就是 stop-gradient**。切断这一支的梯度回传，值不变。
   ```python
   adv = (logp_teacher - logp_student).detach()    # advantage 不回传梯度
   ```
3. **`with torch.no_grad():`** 整块不建计算图，推理和 teacher forward 都要用，能省大量显存（对应 [显存账本](../../03-training-fundamentals/01-memory-accounting.md#6-activation为什么必须存) 里说的 activation）。
4. **只有 float 张量能求导**，整数张量不行。

RQ-VAE 那道题是 `detach` 方向的典型例子：

```python
cb_loss = F.mse_loss(e, res.detach())    # 只更新码本 e，不动编码器
commit  = F.mse_loss(res, e.detach())    # 只更新编码器 res，不动码本
```

两个 loss 长得几乎一样，`detach` 加在相反一侧，含义完全不同。

## 六、nn.Module

```python
class LoRALinear(nn.Module):
    def __init__(self, W, r, alpha):
        super().__init__()                          # 必须先调
        self.register_buffer("W", W)                # 状态但不训练：不进 parameters()
        self.A = nn.Parameter(torch.randn(...))     # 训练参数：自动进 parameters()
        self.B = nn.Parameter(torch.zeros(...))
        self.scale = alpha / r                      # 普通 python 属性，不是张量
    def forward(self, x):                           # 调用 model(x) 实际走这里
        return x @ self.W + (x @ self.A @ self.B) * self.scale
```

| 放什么 | 用什么 | 会被优化器更新 | 会存进 state_dict |
|---|---|---|---|
| 要训练的权重 | `nn.Parameter` | 是 | 是 |
| 冻结的权重、running_mean 之类 | `register_buffer` | 否 | 是 |
| 超参数 | 普通属性 | 否 | 否 |

`model(x)` 而不是 `model.forward(x)`：前者会走 hook 机制，后者绕过。

常用：`model.parameters()`、`model.state_dict()`、`model.train()` / `model.eval()`（影响 dropout 和 BatchNorm）、`model.to(device)`、`.double()` / `.float()`。

## 七、F.* 速查

按手撕题里的实际使用频次。**用官方的，别自己写**，它们都做了数值稳定处理。

| API | 说明 / 坑 |
|---|---|
| `F.softmax(x, dim=-1)` | `dim` 必须显式写；内部已减最大值 |
| `F.log_softmax(x, dim=-1)` | 要 log 概率就用它，**别写 `log(softmax(x))`** |
| `F.cross_entropy(logits, target)` | **直接吃 logits**，别先 softmax；target 是类别下标且必须 `int64`；支持 `ignore_index` 做 mask |
| `F.nll_loss(logp, target)` | 吃的是 log 概率。`cross_entropy = log_softmax + nll_loss` |
| `F.binary_cross_entropy_with_logits(z, y)` | 二分类用这个，**不要用 `binary_cross_entropy`**（后者要你先 sigmoid，数值差） |
| `F.kl_div(input, target)` | **坑最多**：`input` 要 log-prob，方向是 `KL(target ‖ input)`（和数学写法反着），`reduction` 要写 `batchmean` |
| `F.mse_loss(a, b, reduction=...)` | `reduction` 可选 `mean` / `sum` / `none` |
| `F.logsigmoid(z)` | `-F.logsigmoid(z)` 就是稳定的 `-logσ(z)`，DPO 用 |
| `F.normalize(x, dim=-1)` | L2 归一化，InfoNCE 用 |
| `F.layer_norm(x, (d,), w, b)` | 第二个参数是 `normalized_shape`，传**元组**不是整数 |
| `F.batch_norm(x, mean, var, w, b, training=True)` | 要显式传 `training` |
| `F.relu` / `F.gelu` / `F.silu` | SiLU 就是 Swish，SwiGLU 用它 |
| `F.conv2d` / `F.max_pool2d` | 注意 conv 是**相关**不翻转核 |
| `F.scaled_dot_product_attention(q,k,v,is_causal=True)` | 2.0+ 的融合 attention，工程里直接用；但**面试要你手写就不能用** |

## 八、优化器

```python
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
for batch in loader:
    opt.zero_grad()          # 1) 清梯度，忘了就是累加
    loss = compute(batch)
    loss.backward()          # 2) 反向
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 3) 梯度裁剪
    opt.step()               # 4) 更新
```

这四行的顺序是死的。`clip_grad_norm_` 必须在 `backward` 之后、`step` 之前（原理见 [gradient clipping](../../03-training-fundamentals/04-packing-and-grad-clip.md#二gradient-clipping)）。

手写的 Adam / SGD 已经和 `torch.optim` 对拍一致，见 [深度学习手撕](../07-handwrite/02-dl.md)。

## 九、常见报错对照

| 报错 | 原因 |
|---|---|
| `expected m1 and m2 to have the same dtype` | float32 和 float64 混用 → 新建张量跟随 `dtype=` |
| `Expected all tensors to be on the same device` | 一半在 CPU 一半在 GPU → 加 `device=x.device` |
| `view size is not compatible ...` | `transpose` 后内存不连续 → 用 `reshape` |
| `Expected target to have dtype long` | 标签是 int32/float → `.long()` |
| `element 0 of tensors does not require grad` | 对着 `no_grad` 里算出来的东西 `backward` |
| `Trying to backward through the graph a second time` | 同一个图 backward 两次 → 加 `retain_graph=True` 或检查是不是漏了 `zero_grad` |
| loss 变 `nan` | softmax 前整行被 mask 成 `-inf`；或 `log(0)`；或学习率太大 |

## 自测

**1.** `keepdim` 和 `keepdims` 哪个是 torch 的？`var` 的默认无偏设置 torch 和 numpy 哪个不一样？

> **答：** **torch 是 `keepdim`（没有 s），numpy 是 `keepdims`**；同理 torch 用 `dim=`、numpy 用 `axis=`。
> `var`：**numpy 默认有偏**（除 $n$），**torch 默认无偏**（除 $n-1$）。要对齐得写 `t.var(-1, unbiased=False)`。
> GRPO 里做 advantage 归一化时这个差异会让两边差一个 $\sqrt{n/(n-1)}$。

**2.** `repeat` 和 `repeat_interleave` 分别产生什么？GQA 用哪个？

> **答：** `torch.tensor([1,2,3]).repeat(2)` → `[1,2,3,1,2,3]`（整体重复，≈ `np.tile`）；
> `.repeat_interleave(2)` → `[1,1,2,2,3,3]`（逐元素重复，≈ `np.repeat`）。
> **GQA 用 `repeat_interleave`**，因为相邻的 $g$ 个 query head 要共享同一组 KV。用错不报错但结果错。

**3.** `view` 和 `reshape` 的区别？什么时候 `view` 会报错？

> **答：** `view` **要求内存连续**，不满足直接报错，但永不复制；`reshape` 不要求连续，需要时会自动复制。
> 典型报错场景：`t.transpose(0,1).view(-1)` —— transpose 只换了 strides，内存已经不连续。
> 解决：用 `reshape`，或先 `contiguous().view()`。**经验：拆头做完 attention 拼回去一律用 `reshape`。**

**4.** 写出 causal mask 的完整三行（生成 mask、masked_fill、softmax）。为什么填 `-inf` 不是 0？

> **答：** ```python
> mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
> scores = scores.masked_fill(~mask, float('-inf'))
> attn = F.softmax(scores, dim=-1)
> ```
> 填 `-inf` 是因为要在 **softmax 之前**屏蔽：$e^{-\infty}=0$，该位置权重恰好为 0 **且不进入分母**。
> softmax 之后再乘 0 是错的 —— 分母里仍然算了被屏蔽的位置，剩余权重不再归一到 1。
> （`masked_fill(条件, 值)` 填的是**条件为 True** 的位置，所以要想清楚你的 mask 是「1 表示可见」还是「1 表示屏蔽」。）

**5.** 一行全被 mask 掉会发生什么？

> **答：** 该行 softmax 的分母为 0，输出 **`nan`**，然后 nan 会污染整个后续计算和梯度。
> 变长 batch 里要保证**每行至少有一个可见位置**（比如 padding 行也让它能看到自己），或者在 softmax 后把整行是 nan 的位置置零。

**6.** `detach()` 和 `no_grad()` 的区别？RQ-VAE 里两个 loss 的 detach 为什么加在相反一侧？

> **答：** `detach()` 作用在**单个张量**上，切断这一支的梯度回传，值不变；`with torch.no_grad():` 作用在**整块代码**上，里面完全不建计算图，能省大量 activation 显存。
> RQ-VAE：
> ```python
> cb_loss = F.mse_loss(e, res.detach())   # 只更新码本 e
> commit  = F.mse_loss(res, e.detach())   # 只更新编码器 res
> ```
> 两个 loss 长得几乎一样，`detach` 加在**相反**一侧，决定了梯度流向谁 —— 一个拉码本靠近残差，一个拉编码器靠近码本。

**7.** `nn.Parameter` 和 `register_buffer` 的区别？各自会不会进 `state_dict`、会不会被优化器更新？

> **答：** | | 进 `parameters()` | 被优化器更新 | 进 `state_dict` |
> |---|---|---|---|
> | `nn.Parameter` | 是 | **是** | 是 |
> | `register_buffer` | 否 | **否** | **是** |
> | 普通属性 | 否 | 否 | 否 |
> LoRA 里冻结的 $W$ 用 `register_buffer`（要存进 checkpoint 但不训练），$A,B$ 用 `nn.Parameter`，缩放系数 $\alpha/r$ 是普通属性。BatchNorm 的 running_mean/var 也是 buffer。

**8.** `F.cross_entropy` 吃的是 logits 还是概率？target 要什么 dtype？

> **答：** 吃的是 **logits**，内部自己做 log_softmax。**千万别自己先 softmax 再传进去**，等于做了两次。
> target 是类别下标且必须是 **`int64`（long）**，传 int32 会报错。
> 关系：`cross_entropy = log_softmax + nll_loss`；`F.nll_loss` 吃的才是 log 概率。

**9.** `F.kl_div` 的三个坑分别是什么？

> **答：** ① 第一个参数 `input` 要传 **log-prob**（不是 prob）；
> ② 方向是 **`KL(target ‖ input)`**，和数学写法 $D_{KL}(p\|q)$ 反着 —— 想算 $D_{KL}(p\|q)$ 要写 `F.kl_div(q.log(), p)`；
> ③ `reduction` 要用 **`batchmean`** 而不是默认的 `mean`（默认会除以元素总数而不是 batch 数，得到的不是数学定义的 KL）。

**10.** 训练四步的顺序，`clip_grad_norm_` 插在哪一步？

> **答：** ```python
> opt.zero_grad()                                    # 1) 清梯度
> loss.backward()                                    # 2) 反向
> torch.nn.utils.clip_grad_norm_(params, 1.0)        # 3) 裁剪
> opt.step()                                         # 4) 更新
> ```
> `clip_grad_norm_` 必须在 **backward 之后、step 之前**。顺序是死的，`zero_grad` 忘了会导致梯度累加、越训越离谱。

