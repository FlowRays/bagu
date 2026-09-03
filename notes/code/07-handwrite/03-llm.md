# 手撕：大模型（17 道）

> 每题一段可直接默写的 NumPy 实现，全部本地跑通并交叉验证过。
> 核心代码模式：面试里只需要写出 `class Solution` 里的这段核心计算。

> 返回 [手撕总表](00-map.md)

## 公共前置

> 下面每题的代码都默认有这几行。面试时该内联的内联，别现场造轮子。

```python
import numpy as np

def softmax(x, axis=-1):
    x = np.asarray(x, float); x = x - x.max(axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis, keepdims=True)

def _causal_mask(n, m=None):
    """(n,m) 下三角 True 表示可见；m 默认 = n。增量解码时 m>n。"""
    m = m or n
    q = np.arange(m - n, m)[:, None]          # query 的绝对位置
    return q >= np.arange(m)[None, :]
```

PyTorch 版的前置：

```python
import torch, torch.nn.functional as F
```

---

## Self-Attention

`中等` ｜ **思路**：`softmax(QKᵀ/√d)·V`。

```python
def self_attention(x, Wq, Wk, Wv, causal=False):
    """x (L,d) -> (L,dv)"""
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    s = Q @ K.T / np.sqrt(K.shape[-1])
    if causal:
        s = np.where(_causal_mask(len(Q)), s, -np.inf)
    return softmax(s) @ V
```

**复杂度**：O(L²·d)

**易错**：**必须除 √d**，否则维度一大 softmax 饱和、梯度消失；causal 时用 `-inf` 填充再 softmax。

**PyTorch 版**（与上面 NumPy 版逐个对拍，数值一致）

```python
def self_attention(x, Wq, Wk, Wv, causal=False):
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    scores = Q @ K.transpose(-2, -1) / K.size(-1) ** 0.5
    if causal:
        L = x.size(0)
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ V
```

**torch 侧注意**：causal mask 用 `scores.masked_fill(~mask, float('-inf'))`；`torch.tril` 生成下三角。

---

## Multi-Head Attention

`中等` ｜ **思路**：拆头 → 每头独立算 attention → 拼回来 → 过 `Wo`。

```python
def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads, causal=False):
    """x (L,d)；W* (d,d)；输出 (L,d)。"""
    L, d = x.shape
    dh = d // n_heads
    # (L,d) -> (L,h,dh) -> (h,L,dh)
    Q = (x @ Wq).reshape(L, n_heads, dh).transpose(1, 0, 2)
    K = (x @ Wk).reshape(L, n_heads, dh).transpose(1, 0, 2)
    V = (x @ Wv).reshape(L, n_heads, dh).transpose(1, 0, 2)
    s = Q @ K.transpose(0, 2, 1) / np.sqrt(dh)          # (h,L,L)
    if causal:
        s = np.where(_causal_mask(L), s, -np.inf)
    o = softmax(s) @ V                                   # (h,L,dh)
    return o.transpose(1, 0, 2).reshape(L, d) @ Wo       # 拼回来再过 Wo
```

**复杂度**：O(L²·d)

**易错**：reshape 成 `(L,h,dh)` 后要 `transpose` 成 `(h,L,dh)` 才能按头做批量矩阵乘；缩放用的是 **dh 不是 d**；最后必须过 `Wo`。

**PyTorch 版**（与上面 NumPy 版逐个对拍，数值一致）

```python
def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads, causal=False):
    L, d = x.shape
    dh = d // n_heads
    # (L,d) -> (L,h,dh) -> (h,L,dh)
    Q = (x @ Wq).view(L, n_heads, dh).transpose(0, 1)
    K = (x @ Wk).view(L, n_heads, dh).transpose(0, 1)
    V = (x @ Wv).view(L, n_heads, dh).transpose(0, 1)
    scores = Q @ K.transpose(-2, -1) / dh ** 0.5
    if causal:
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))
    out = F.softmax(scores, dim=-1) @ V              # (h,L,dh)
    return out.transpose(0, 1).reshape(L, d) @ Wo    # transpose 后必须 reshape/contiguous
```

**torch 侧注意**：`transpose` 之后内存不连续，必须 `reshape`（或 `contiguous().view()`），直接 `view` 会报错。

---

## Cross Attention

`中等` ｜ **思路**：和 MHA 唯一的区别：Q 来自一个序列，K/V 来自另一个。

```python
def cross_attention(x_q, x_kv, Wq, Wk, Wv, Wo, n_heads, mask=None):
    """Q 来自 x_q (Lq,d)，K/V 来自 x_kv (Lk,d)。"""
    Lq, d = x_q.shape; Lk = len(x_kv); dh = d // n_heads
    Q = (x_q @ Wq).reshape(Lq, n_heads, dh).transpose(1, 0, 2)
    K = (x_kv @ Wk).reshape(Lk, n_heads, dh).transpose(1, 0, 2)
    V = (x_kv @ Wv).reshape(Lk, n_heads, dh).transpose(1, 0, 2)
    s = Q @ K.transpose(0, 2, 1) / np.sqrt(dh)
    if mask is not None:
        s = np.where(mask, s, -np.inf)                   # mask 一般是 padding mask
    return (softmax(s) @ V).transpose(1, 0, 2).reshape(Lq, d) @ Wo
```

**复杂度**：O(Lq·Lk·d)

**易错**：mask 这里通常是 encoder 的 padding mask，不是 causal mask。

**PyTorch 版**（与上面 NumPy 版逐个对拍，数值一致）

```python
def cross_attention(x_q, x_kv, Wq, Wk, Wv, Wo, n_heads, mask=None):
    Lq, d = x_q.shape; Lk = x_kv.size(0); dh = d // n_heads
    Q = (x_q  @ Wq).view(Lq, n_heads, dh).transpose(0, 1)
    K = (x_kv @ Wk).view(Lk, n_heads, dh).transpose(0, 1)
    V = (x_kv @ Wv).view(Lk, n_heads, dh).transpose(0, 1)
    scores = Q @ K.transpose(-2, -1) / dh ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))   # padding mask
    return (F.softmax(scores, dim=-1) @ V).transpose(0, 1).reshape(Lq, d) @ Wo
```

**torch 侧注意**：padding mask 常见写法是 `scores.masked_fill(mask == 0, float('-inf'))`。

---

## Grouped Query Attention

`中等` ｜ **思路**：K/V 只投影出 `n_kv_heads` 组，再 `repeat` 广播给多个 query head 共享。

```python
def grouped_query_attention(x, Wq, Wk, Wv, Wo, n_heads, n_kv_heads, causal=False):
    """K/V 只有 n_kv_heads 组，每组被 n_heads//n_kv_heads 个 query head 共享。"""
    L, d = x.shape
    dh = d // n_heads
    g = n_heads // n_kv_heads          # 每组 KV 被几个 query head 共享
    # 注意 Wk/Wv 的输出维是 n_kv_heads*dh，比 Wq 窄，这正是 GQA 省 KV cache 的地方
    Q = (x @ Wq).reshape(L, n_heads, dh).transpose(1, 0, 2)          # (h,L,dh)
    K = (x @ Wk).reshape(L, n_kv_heads, dh).transpose(1, 0, 2)       # (hkv,L,dh)
    V = (x @ Wv).reshape(L, n_kv_heads, dh).transpose(1, 0, 2)
    K = np.repeat(K, g, axis=0)                                       # 广播到 h 组
    V = np.repeat(V, g, axis=0)
    s = Q @ K.transpose(0, 2, 1) / np.sqrt(dh)
    if causal:
        s = np.where(_causal_mask(L), s, -np.inf)
    return (softmax(s) @ V).transpose(1, 0, 2).reshape(L, d) @ Wo
```

**复杂度**：O(L²·d)

**易错**：**Wk/Wv 的输出维比 Wq 窄**（`n_kv*dh`），这正是省 KV cache 的地方；用 `repeat` 不是 `tile`，前者才让相邻 query head 共享同一组。

**PyTorch 版**（与上面 NumPy 版逐个对拍，数值一致）

```python
def grouped_query_attention(x, Wq, Wk, Wv, Wo, n_heads, n_kv_heads, causal=False):
    L, d = x.shape
    dh = d // n_heads
    g = n_heads // n_kv_heads
    Q = (x @ Wq).view(L, n_heads,    dh).transpose(0, 1)
    K = (x @ Wk).view(L, n_kv_heads, dh).transpose(0, 1)
    V = (x @ Wv).view(L, n_kv_heads, dh).transpose(0, 1)
    K = K.repeat_interleave(g, dim=0)      # 注意是 repeat_interleave 不是 repeat
    V = V.repeat_interleave(g, dim=0)
    scores = Q @ K.transpose(-2, -1) / dh ** 0.5
    if causal:
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))
    return (F.softmax(scores, dim=-1) @ V).transpose(0, 1).reshape(L, d) @ Wo
```

**torch 侧注意**：广播 KV 用 **`repeat_interleave`** 不是 `repeat`：前者是 AABBCC，后者是 ABCABC，用错了 head 就对不上。

---

## RMS Normalization

`中等` ｜ **思路**：只除均方根，**不减均值**、不用方差。

```python
def rms_norm(x, gamma, eps=1e-6):
    x = np.asarray(x, float)
    # 只除均方根，不减均值、不用方差
    return gamma * x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)
```

**复杂度**：O(N·D)

**易错**：少一次求均值和一个 β，比 LayerNorm 便宜；`eps` 在开方里面。

**PyTorch 版**（与上面 NumPy 版逐个对拍，数值一致）

```python
def rms_norm(x, gamma, eps=1e-6):
    return gamma * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
```

**torch 侧注意**：`torch.rsqrt` 比 `1/torch.sqrt` 快一点；`x.pow(2).mean(-1, keepdim=True)` 的 `keepdim` 不能漏。

---

## KV Cache 增量注意力

`中等` ｜ **思路**：把新 token 的 k/v 拼进 cache，用当前 q 对全部历史算 attention。

```python
class KVCache:
    def __init__(self): self.K = self.V = None
    def step(self, q, k, v):
        """q,k,v 是当前这一个 token 的 (h,dh)；返回该 token 的输出 (h,dh)。"""
        k = k[:, None, :]; v = v[:, None, :]                     # (h,1,dh)
        self.K = k if self.K is None else np.concatenate([self.K, k], 1)
        self.V = v if self.V is None else np.concatenate([self.V, v], 1)
        dh = q.shape[-1]
        s = (q[:, None, :] @ self.K.transpose(0, 2, 1)) / np.sqrt(dh)   # (h,1,T)
        return (softmax(s) @ self.V)[:, 0, :]                    # 新 token 能看到全部历史，无需 mask
```

**复杂度**：O(T·d) 每步

**易错**：增量解码时**不需要 causal mask**，因为新 token 本来就只能看到已有的历史；cache 沿序列维拼接。

**PyTorch 版**（与上面 NumPy 版逐个对拍，数值一致）

```python
class KVCache:
    def __init__(self): self.K = self.V = None
    def step(self, q, k, v):
        k, v = k.unsqueeze(1), v.unsqueeze(1)          # (h,dh) -> (h,1,dh)
        self.K = k if self.K is None else torch.cat([self.K, k], dim=1)
        self.V = v if self.V is None else torch.cat([self.V, v], dim=1)
        scores = (q.unsqueeze(1) @ self.K.transpose(-2, -1)) / q.size(-1) ** 0.5
        return (F.softmax(scores, dim=-1) @ self.V).squeeze(1)
```

**torch 侧注意**：`torch.cat` 沿序列维拼接；`unsqueeze(1)` 把 `(h,dh)` 变 `(h,1,dh)`。

---

## 模型分片加载

`中等` ｜ **思路**：按参数量贪心均衡：大的先放，每次放进当前最小的分片。

```python
def shard_state_dict(state_dict, n_shards):
    """按参数量贪心均衡切分（大的先放，放进当前最小的桶）。"""
    sizes = {k: int(np.asarray(v).size) for k, v in state_dict.items()}
    shards = [{} for _ in range(n_shards)]
    load = [0] * n_shards
    for k in sorted(sizes, key=lambda k: -sizes[k]):
        i = int(np.argmin(load))
        shards[i][k] = state_dict[k]
        load[i] += sizes[k]
    return shards, load

def load_from_shards(shards):
    out = {}
    for s in shards: out.update(s)
    return out
```

**复杂度**：O(n log n)

**易错**：要保证并集等于原始 state_dict，且分片间大小尽量接近。

**PyTorch 版**（与上面 NumPy 版逐个对拍，数值一致）

```python
def shard_state_dict(state_dict, n_shards):
    sizes = {k: v.numel() for k, v in state_dict.items()}
    shards = [{} for _ in range(n_shards)]
    load = [0] * n_shards
    for k in sorted(sizes, key=lambda k: -sizes[k]):
        i = load.index(min(load))
        shards[i][k] = state_dict[k]; load[i] += sizes[k]
    return shards, load
```

---

## Transformer Encoder Block

`中等` ｜ **思路**：pre-norm 残差块：`x + MHA(LN(x))`，再 `x + FFN(LN(x))`。

```python
def encoder_block(x, p, n_heads, eps=1e-5):
    """pre-norm：x + MHA(LN(x))，再 x + FFN(LN(x))。p 是参数 dict。"""
    def ln(z, g, b):
        mu = z.mean(-1, keepdims=True); var = z.var(-1, keepdims=True)
        return g * (z - mu) / np.sqrt(var + eps) + b
    h = x + multi_head_attention(ln(x, p["g1"], p["b1"]),
                                 p["Wq"], p["Wk"], p["Wv"], p["Wo"], n_heads)
    z = ln(h, p["g2"], p["b2"])
    return h + np.maximum(z @ p["W1"] + p["b1f"], 0) @ p["W2"] + p["b2f"]
```

**复杂度**：O(L²·d)

**易错**：pre-norm 和 post-norm 位置不同，现代 LLM 基本都是 pre-norm（更好训）；两个残差都不能漏。

**PyTorch 版**（与上面 NumPy 版逐个对拍，数值一致）

```python
def encoder_block(x, p, n_heads, eps=1e-5):
    def ln(z, g, b):
        return F.layer_norm(z, (z.size(-1),), weight=g, bias=b, eps=eps)
    h = x + multi_head_attention(ln(x, p["g1"], p["b1"]),
                                 p["Wq"], p["Wk"], p["Wv"], p["Wo"], n_heads)
    z = ln(h, p["g2"], p["b2"])
    return h + F.relu(z @ p["W1"] + p["b1f"]) @ p["W2"] + p["b2f"]
```

**torch 侧注意**：`F.layer_norm` 可以直接传 weight/bias，省得自己写归一化。

---

## InfoNCE 对比学习损失

`中等` ｜ **思路**：先 L2 归一化，算相似度矩阵除以温度，对角线是正样本，**双向**取平均。

```python
def info_nce(z_img, z_txt, temperature=0.07):
    """对角线是正样本，双向取平均（CLIP 的 loss）。"""
    a = z_img / np.linalg.norm(z_img, axis=-1, keepdims=True)
    b = z_txt / np.linalg.norm(z_txt, axis=-1, keepdims=True)
    logits = a @ b.T / temperature
    idx = np.arange(len(a))
    def log_softmax(z, axis):            # 别写 log(softmax(...))，会丢精度
        z = z - z.max(axis, keepdims=True)
        return z - np.log(np.exp(z).sum(axis, keepdims=True))
    li = -log_softmax(logits, 1)[idx, idx].mean()   # image -> text
    lt = -log_softmax(logits, 0)[idx, idx].mean()   # text -> image
    return float((li + lt) / 2)
```

**复杂度**：O(N²·d)

**易错**：必须先归一化；两个方向（图→文、文→图）分别沿不同轴 softmax。

**PyTorch 版**（与上面 NumPy 版逐个对拍，数值一致）

```python
def info_nce(z_img, z_txt, temperature=0.07):
    a = F.normalize(z_img, dim=-1)
    b = F.normalize(z_txt, dim=-1)
    logits = a @ b.T / temperature
    idx = torch.arange(a.size(0), device=a.device)
    return (F.cross_entropy(logits, idx) + F.cross_entropy(logits.T, idx)) / 2
```

**torch 侧注意**：两个方向直接用 `F.cross_entropy(logits, idx)` 和 `F.cross_entropy(logits.T, idx)`，比手写 log-softmax 干净。

**原理**：[对比学习：CLIP 与 SigLIP 的 loss](../../07-vlm/02-vision-encoder.md)

---

## DPO 损失函数

`困难` ｜ **思路**：`−log σ(β·[(logπ_w − logπ_l) − (logπref_w − logπref_l)])`。

```python
def dpo_loss(logp_pol_w, logp_pol_l, logp_ref_w, logp_ref_l, beta=0.1):
    """四个都是整条序列的 log prob（标量或 (N,)）。"""
    pol = np.asarray(logp_pol_w, float) - np.asarray(logp_pol_l, float)
    ref = np.asarray(logp_ref_w, float) - np.asarray(logp_ref_l, float)
    z = beta * (pol - ref)
    # -log sigmoid(z) 的稳定写法 = softplus(-z)
    return float(np.mean(np.logaddexp(0.0, -z)))
```

**复杂度**：O(1)

**易错**：四个 log prob 都是**整条序列**的和；ref 那两项要 stop-gradient；用 `logaddexp(0,−z)` 算 `−logσ(z)` 更稳。

**PyTorch 版**（与上面 NumPy 版逐个对拍，数值一致）

```python
def dpo_loss(logp_pol_w, logp_pol_l, logp_ref_w, logp_ref_l, beta=0.1):
    z = beta * ((logp_pol_w - logp_pol_l) - (logp_ref_w - logp_ref_l))
    return -F.logsigmoid(z).mean()
```

**torch 侧注意**：`-F.logsigmoid(z)` 比 `-torch.log(torch.sigmoid(z))` 稳。

---

## LoRA 低秩适配

`中等` ｜ **思路**：`y = xW + (xAB)·(α/r)`，W 冻结，**B 初始化为 0** 所以起点严格等于原模型。

```python
class LoRALinear:
    def __init__(self, W, r, alpha, seed=0):
        """W (din,dout) 冻结；只训 A (din,r) 和 B (r,dout)，B 初始化为 0。"""
        self.W = np.asarray(W, float)
        rng = np.random.default_rng(seed)
        din, dout = self.W.shape
        self.A = rng.normal(0, 0.01, (din, r))
        self.B = np.zeros((r, dout))
        self.scale = alpha / r
    def forward(self, x):
        return x @ self.W + (x @ self.A @ self.B) * self.scale
    def merge(self):
        return self.W + self.A @ self.B * self.scale      # 推理期可合并，零额外开销
```

**复杂度**：O(N·d·r)

**易错**：B 必须是 0（A 是 0 也行，但不能都非零）；缩放是 α/r；推理期可以 merge 进 W，零额外延迟。

**PyTorch 版**（与上面 NumPy 版逐个对拍，数值一致）

```python
class LoRALinear(torch.nn.Module):
    def __init__(self, W, r, alpha):
        super().__init__()
        din, dout = W.shape
        self.register_buffer("W", W)                       # 冻结，不进 parameters()
        # 新建张量要跟随 W 的 dtype/device，否则和输入相乘会报 dtype 不匹配
        self.A = torch.nn.Parameter(torch.randn(din, r, dtype=W.dtype, device=W.device) * 0.01)
        self.B = torch.nn.Parameter(torch.zeros(r, dout, dtype=W.dtype, device=W.device))
        self.scale = alpha / r
    def forward(self, x):
        return x @ self.W + (x @ self.A @ self.B) * self.scale
    def merge(self):
        return self.W + self.A @ self.B * self.scale
```

**torch 侧注意**：冻结权重用 `register_buffer` 而不是 `Parameter`，这样不进 `parameters()`、不会被优化器更新；**新建张量要跟随 `W.dtype/device`**，否则 dtype 不匹配直接报错。

---

## SFT 损失函数（Shift Right）

`中等` ｜ **思路**：logits 去掉最后一位、labels 左移一位对齐，再对 mask 后的位置算交叉熵。

```python
def sft_loss(logits, labels, ignore_index=-100):
    """logits (L,V) 是位置 0..L-1 的预测；label 要左移一位对齐。"""
    logits = np.asarray(logits, float); labels = np.asarray(labels, int)
    lg = logits[:-1]                    # 预测下一个 token，最后一位没有 target
    lb = labels[1:]                     # 真值往左挪一格
    keep = lb != ignore_index           # prompt 部分 mask 掉
    if not keep.any(): return 0.0
    lg, lb = lg[keep], lb[keep]
    z = lg - lg.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-logp[np.arange(len(lb)), lb].mean())
```

**复杂度**：O(L·V)

**易错**：**shift 是这题的全部考点**：位置 t 的 logits 预测的是 t+1 的 token；prompt 部分用 `-100` mask 掉不算 loss。

**PyTorch 版**（与上面 NumPy 版逐个对拍，数值一致）

```python
def sft_loss(logits, labels, ignore_index=-100):
    shift_logits = logits[:-1]      # 位置 t 预测 t+1
    shift_labels = labels[1:]
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index)
```

**torch 侧注意**：`F.cross_entropy` 的 `ignore_index` 直接处理 mask，不用自己筛；shift 仍然要手动做。

**原理**：[SFT loss 与 teacher forcing](../../06-post-training/distill/02-sft-and-kd.md)

---

## MLA（Multi-head Latent Attention）

`中等` ｜ **思路**：KV 先压到低维 latent `c`，需要时再升维。**KV cache 只存 c**。

```python
def mla(x, Wdkv, Wuk, Wuv, Wq, Wo, n_heads, causal=False):
    """KV 先压到低维 latent c (L,dc) 再升维，KV cache 只需缓存 c。
    Wdkv (d,dc)；Wuk/Wuv (dc, h*dh)；Wq (d, h*dh)。"""
    L, d = x.shape
    c = x @ Wdkv                                   # (L,dc)  <- 真正被 cache 的东西
    dh = Wq.shape[1] // n_heads
    Q = (x @ Wq).reshape(L, n_heads, dh).transpose(1, 0, 2)
    K = (c @ Wuk).reshape(L, n_heads, dh).transpose(1, 0, 2)
    V = (c @ Wuv).reshape(L, n_heads, dh).transpose(1, 0, 2)
    s = Q @ K.transpose(0, 2, 1) / np.sqrt(dh)
    if causal:
        s = np.where(_causal_mask(L), s, -np.inf)
    return (softmax(s) @ V).transpose(1, 0, 2).reshape(L, n_heads * dh) @ Wo, c
```

**复杂度**：O(L²·d)

**易错**：省的是 cache 不是计算；`c` 的维度 dc 远小于 `2·h·dh`，这是 MLA 相比 MHA 的核心收益。

**PyTorch 版**（与上面 NumPy 版逐个对拍，数值一致）

```python
def mla(x, Wdkv, Wuk, Wuv, Wq, Wo, n_heads, causal=False):
    L, d = x.shape
    c = x @ Wdkv                                  # 只 cache 这个低维 latent
    dh = Wq.size(1) // n_heads
    Q = (x @ Wq).view(L, n_heads, dh).transpose(0, 1)
    K = (c @ Wuk).view(L, n_heads, dh).transpose(0, 1)
    V = (c @ Wuv).view(L, n_heads, dh).transpose(0, 1)
    scores = Q @ K.transpose(-2, -1) / dh ** 0.5
    if causal:
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))
    return (F.softmax(scores, dim=-1) @ V).transpose(0, 1).reshape(L, n_heads * dh) @ Wo, c
```

---

## RQ-VAE Loss

`中等` ｜ **思路**：残差逐级量化：每级找最近码字，累加到重建、把残差传给下一级。loss = 重建 + 码本 + β·commitment。

```python
def rq_vae_loss(z, codebooks, x_rec=None, x=None, beta=0.25):
    """z (N,d) 编码器输出；codebooks: list，每级 (K,d)。残差逐级量化。"""
    z = np.asarray(z, float)
    res = z.copy(); q = np.zeros_like(z)
    cb_loss = commit = 0.0
    for C in codebooks:
        C = np.asarray(C, float)
        d = ((res[:, None, :] - C[None]) ** 2).sum(-1)     # (N,K)
        e = C[d.argmin(1)]                                 # 最近码字
        cb_loss += ((e - res_sg(res)) ** 2).sum(-1).mean()      # 拉码本靠近残差
        commit  += ((res - res_sg(e)) ** 2).sum(-1).mean()      # 拉编码器靠近码本
        q += e; res = res - e                              # 残差进入下一级
    rec = 0.0 if x is None else float(((np.asarray(x_rec, float) - x) ** 2).sum(-1).mean())
    return float(rec + cb_loss + beta * commit), q

def res_sg(a):    # numpy 没有自动微分，stop-gradient 在这里只是恒等；标注语义
    return a
```

**复杂度**：O(N·K·d·L)

**易错**：codebook loss 和 commitment loss 的 stop-gradient 方向**相反**（一个拉码本、一个拉编码器）；numpy 里没有自动微分，写的时候要标清楚哪边被 sg。

**PyTorch 版**（与上面 NumPy 版逐个对拍，数值一致）

```python
def rq_vae_loss(z, codebooks, x_rec=None, x=None, beta=0.25):
    res = z; q = torch.zeros_like(z)
    cb_loss = commit = 0.0
    for C in codebooks:
        d = torch.cdist(res, C) ** 2                       # (N,K) 平方距离
        e = C[d.argmin(1)]
        cb_loss = cb_loss + F.mse_loss(e, res.detach(), reduction="none").sum(-1).mean()
        commit  = commit  + F.mse_loss(res, e.detach(), reduction="none").sum(-1).mean()
        q = q + e; res = res - e                           # detach 决定梯度流向谁
    rec = 0.0 if x is None else F.mse_loss(x_rec, x, reduction="none").sum(-1).mean()
    return rec + cb_loss + beta * commit, q
```

**torch 侧注意**：stop-gradient 在 torch 里就是 `.detach()`，codebook loss 和 commitment loss 的 `detach` 加在**相反**的一侧。

---

## SwiGLU（LLaMA 系激活前向传播）

`中等` ｜ **思路**：`SiLU(xWg) * (xWu)` 再过 `Wd`。三个矩阵不是两个。

```python
def swiglu(x, Wg, Wu, Wd):
    """LLaMA 系 FFN：SiLU(xWg) * (xWu) 再过 Wd。"""
    def silu(z): return z / (1 + np.exp(-z))
    return (silu(x @ Wg) * (x @ Wu)) @ Wd
```

**复杂度**：O(N·d·d_ff)

**易错**：gate 分支才过 SiLU，up 分支不过；因为多了一个矩阵，`d_ff` 通常取 ⅔·4d 来对齐参数量。

**PyTorch 版**（与上面 NumPy 版逐个对拍，数值一致）

```python
def swiglu(x, Wg, Wu, Wd):
    return (F.silu(x @ Wg) * (x @ Wu)) @ Wd
```

**torch 侧注意**：`F.silu` 就是 SiLU/Swish，不用自己写 `x*sigmoid(x)`。

---

## GRPO 损失函数

`困难` ｜ **思路**：组内奖励标准化得到 advantage，广播到该序列每个 token，再走和 PPO 一样的 clip+min。

```python
def grpo_loss(logp, logp_old, rewards, clip_eps=0.2, mask=None,
              logp_ref=None, beta=0.0, eps=1e-6):
    """rewards (G,) 同一个 prompt 的 G 条 rollout；logp (G,L)。"""
    r = np.asarray(rewards, float)
    adv = (r - r.mean()) / (r.std() + eps)                # 组内归一化，除标准差
    adv = adv[:, None]                                    # 广播到该序列每个 token
    logp = np.asarray(logp, float); logp_old = np.asarray(logp_old, float)
    ratio = np.exp(logp - logp_old)
    per_tok = -np.minimum(ratio * adv,
                          np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    if beta and logp_ref is not None:                     # k3 无偏 KL 估计
        d = np.asarray(logp_ref, float) - logp
        per_tok = per_tok + beta * (np.exp(d) - d - 1)
    m = np.ones_like(per_tok) if mask is None else np.asarray(mask, float)
    return float((per_tok * m).sum() / max(m.sum(), 1e-8))
```

**复杂度**：O(G·L)

**易错**：**除标准差不是方差**（保持尺度不变）；组内 reward 全相同时 advantage 全 0，loss 为 0（这是 DAPO 要修的 dynamic sampling 问题）；KL 用 k3 无偏估计 `exp(d)−d−1`。

**PyTorch 版**（与上面 NumPy 版逐个对拍，数值一致）

```python
def grpo_loss(logp, logp_old, rewards, clip_eps=0.2, mask=None,
              logp_ref=None, beta=0.0, eps=1e-6):
    adv = ((rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)).unsqueeze(1)
    ratio = torch.exp(logp - logp_old)
    per_tok = -torch.min(ratio * adv,
                         torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    if beta and logp_ref is not None:
        d = logp_ref - logp
        per_tok = per_tok + beta * (torch.exp(d) - d - 1)     # k3 无偏 KL 估计
    m = torch.ones_like(per_tok) if mask is None else mask
    return (per_tok * m).sum() / m.sum().clamp(min=1e-8)
```

**torch 侧注意**：`rewards.std(unbiased=False)` 要显式关掉无偏修正，否则和 numpy 的 `.std()` 差一个 $\sqrt{n/(n-1)}$。

**原理**：[GRPO 为什么除标准差](../../06-post-training/rl/06-grpo.md)

---

## PPO 损失函数

`中等` ｜ **思路**：`−mean(min(r·A, clip(r,1−ε,1+ε)·A))`。

```python
def ppo_loss(logp, logp_old, adv, clip_eps=0.2, mask=None):
    logp = np.asarray(logp, float); logp_old = np.asarray(logp_old, float)
    adv = np.asarray(adv, float)
    ratio = np.exp(logp - logp_old)                       # 用 log 差再 exp，数值稳
    un = ratio * adv
    cl = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    per_tok = -np.minimum(un, cl)                         # min 取悲观侧
    if mask is None:
        return float(per_tok.mean())
    mask = np.asarray(mask, float)
    return float((per_tok * mask).sum() / max(mask.sum(), 1e-8))
```

**复杂度**：O(L)

**易错**：**min 不能省**：只写 `clip(r)A` 在 A<0 且 r 很小时会给出错误的鼓励方向；ratio 用 `exp(logp − logp_old)` 算，别直接除概率。

**PyTorch 版**（与上面 NumPy 版逐个对拍，数值一致）

```python
def ppo_loss(logp, logp_old, adv, clip_eps=0.2, mask=None):
    ratio = torch.exp(logp - logp_old)
    per_tok = -torch.min(ratio * adv,
                         torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    if mask is None:
        return per_tok.mean()
    return (per_tok * mask).sum() / mask.sum().clamp(min=1e-8)
```

**torch 侧注意**：`torch.clamp(ratio, 1-eps, 1+eps)`，`torch.min` 是逐元素取小。

**原理**：[PPO 为什么必须有 min](../../06-post-training/rl/03-clip-and-min.md)

---
