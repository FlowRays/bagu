# 手撕：Diffusion 与 Flow Matching

> 每段都是可直接默写的最小实现，**全部本地跑通并做了性质验证**（$q$ 的边缘分布对上闭式解、DDPM/DDIM 采样还原目标分布、`eta=0` 逐元素确定性、CFG 的 $w=1$ 退化、flow 完美速度场 loss 为 0、ensembling 权重归一且递增）。

> 返回 [手撕总表](00-map.md)｜原理见 [视觉生成](../02-visual-generation/00-map.md)、[Action](../05-action/00-map.md)

## 公共前置

```python
"""Diffusion / Flow matching / Action chunking 的最小实现（NumPy）。"""
import numpy as np
```

---

## Diffusion

### 噪声 schedule

**思路**：线性 $\beta$ schedule，$\bar\alpha_t=\prod_{s\le t}(1-\beta_s)$ 从 1 单调降到约 0。

```python
def make_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)                      # ᾱ_t = Π(1-β_s)
    return betas, alphas, abar
```

**易错**：$\beta$ 的范围决定加噪速度；$\bar\alpha_T$ 要足够接近 0，否则最后一步还不是纯噪声。

### 前向加噪

**思路**：前向的闭式解，**一步跳到任意 $t$**，训练才能并行。

```python
def q_sample(x0, t, abar, noise=None):
    """一步跳到任意 t: x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε"""
    if noise is None:
        noise = np.random.randn(*x0.shape)
    a = abar[t][:, None]                           # (B,1) 广播到特征维
    return np.sqrt(a) * x0 + np.sqrt(1 - a) * noise, noise
```

**易错**：`abar[t][:, None]` 那个 `None` 是为了广播到特征维；漏了就 shape 不匹配。

### 训练目标

**思路**：**随机采 $t$** 而不是遍历，一个 batch 里每条样本的 $t$ 都不同。

```python
def ddpm_loss(model, x0, abar, T):
    """L = ‖ε − ε_θ(x_t, t)‖²，随机采 t 而不是遍历"""
    t = np.random.randint(0, T, size=len(x0))
    xt, eps = q_sample(x0, t, abar)
    return float(((eps - model(xt, t)) ** 2).mean())
```

**易错**：loss 就是一个 MSE，这是 diffusion 比 GAN 稳的根本原因。

### DDPM 采样

**思路**：按 $\mu_\theta=\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta)$ 逐步去噪。

```python
def ddpm_sample(model, shape, betas, alphas, abar, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    for t in reversed(range(len(betas))):
        tt = np.full(len(x), t)
        eps = model(x, tt)
        # μ_θ = (x_t − β_t/√(1-ᾱ_t)·ε_θ) / √α_t
        mean = (x - betas[t] / np.sqrt(1 - abar[t]) * eps) / np.sqrt(alphas[t])
        x = mean + (np.sqrt(betas[t]) * rng.standard_normal(x.shape) if t > 0 else 0)
    return x
```

**易错**：**最后一步（$t=0$）不要再加噪声**，否则输出永远带一层噪。

### DDIM 采样（跳步 + 确定性）

**思路**：只在 $T$ 里挑 `steps` 个时间点；`eta=0` 时完全确定性。

```python
def ddim_sample(model, shape, abar, steps=50, eta=0.0, seed=0):
    """eta=0 完全确定性；只在 T 里挑 steps 个时间点"""
    rng = np.random.default_rng(seed)
    T = len(abar)
    ts = np.linspace(T - 1, 0, steps).round().astype(int)
    x = rng.standard_normal(shape)
    for i, t in enumerate(ts):
        eps = model(x, np.full(len(x), t))
        x0_pred = (x - np.sqrt(1 - abar[t]) * eps) / np.sqrt(abar[t])
        a_prev = abar[ts[i + 1]] if i + 1 < len(ts) else 1.0
        sigma = eta * np.sqrt((1 - a_prev) / (1 - abar[t]) * (1 - abar[t] / a_prev))
        dir_xt = np.sqrt(max(1 - a_prev - sigma ** 2, 0.0)) * eps
        x = np.sqrt(a_prev) * x0_pred + dir_xt
        if sigma > 0:
            x = x + sigma * rng.standard_normal(x.shape)
    return x
```

**易错**：先解出 $\hat x_0$ 再重新加噪到 $t-1$，这是 DDIM 的核心；`sqrt(max(..., 0))` 是防浮点误差导致负数开方。

### Classifier-free guidance

**思路**：$\tilde\epsilon=\epsilon(\varnothing)+w[\epsilon(c)-\epsilon(\varnothing)]$。

```python
def cfg_eps(model, x, t, cond, w=3.0):
    """ε̃ = ε(∅) + w·(ε(c) − ε(∅))；w=1 退化成普通条件生成"""
    e_uncond = model(x, t, None)
    e_cond = model(x, t, cond)
    return e_uncond + w * (e_cond - e_uncond)
```

**易错**：**每步要跑两次网络**，采样成本翻倍；$w=1$ 必须退化成纯条件生成，这是最好的自检。

## Flow Matching

### Flow matching 训练目标

**思路**：直线路径 $x_t=(1-t)x_0+tx_1$，目标速度是**常量** $x_1-x_0$（和 $t$ 无关）。

```python
def fm_loss(model, x1, cond=None):
    """直线路径 x_t=(1-t)x0+t·x1，目标速度是常量 x1−x0"""
    x0 = np.random.randn(*x1.shape)
    t = np.random.rand(len(x1), 1)
    xt = (1 - t) * x0 + t * x1
    u = x1 - x0
    v = model(xt, t[:, 0]) if cond is None else model(xt, t[:, 0], cond)
    return float(((v - u) ** 2).mean())
```

**易错**：$t$ 是 $[0,1]$ 的连续值不是整数步；不需要设计 noise schedule。

### Flow matching 采样（欧拉法）

**思路**：欧拉法解 ODE，`steps` 通常 4~10。

```python
def fm_sample(model, shape, steps=8, seed=0, cond=None):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    dt = 1.0 / steps
    for i in range(steps):
        t = np.full(len(x), i * dt)
        x = x + dt * (model(x, t) if cond is None else model(x, t, cond))
    return x
```

**易错**：路径接近直线所以大步走误差也小；这就是它能替代 diffusion 上机器人的唯一理由。

---

## 一个能跑的对照实验：为什么不能用 MSE 回归动作

笔记里反复说「动作多模态所以不能回归」。这段代码把它跑出来。

任务模拟绕障碍：观测固定，专家动作要么 $+1$（从左绕）要么 $-1$（从右绕），各 50%。
**正确行为是二选一，输出 0 就是直接撞上去。**

```python
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
N = 4096
obs = torch.zeros(N, 1)                                   # 观测固定，只看动作分布
act = torch.where(torch.rand(N, 1) < 0.5, -1.0, 1.0)      # 双峰: ±1

def mlp(din, dout, h=128):
    return nn.Sequential(nn.Linear(din, h), nn.SiLU(),
                         nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dout))

# ---------- 方法 A: MSE 回归 ----------
reg = mlp(1, 1)
opt = torch.optim.Adam(reg.parameters(), lr=1e-3)
for _ in range(2000):
    opt.zero_grad(); F.mse_loss(reg(obs), act).backward(); opt.step()
pred_reg = reg(torch.zeros(2000, 1)).detach()

# ---------- 方法 B: flow matching ----------
# v_θ(x_t, t, o): 输入 [动作, 时间, 观测]
fm = mlp(3, 1)
opt = torch.optim.Adam(fm.parameters(), lr=1e-3)
for _ in range(4000):
    x0 = torch.randn_like(act)
    t = torch.rand(N, 1)
    xt = (1 - t) * x0 + t * act
    u = act - x0                                          # 目标速度: 常量
    opt.zero_grad()
    F.mse_loss(fm(torch.cat([xt, t, obs], 1)), u).backward()
    opt.step()

@torch.no_grad()
def fm_sample(n, steps=8):
    x = torch.randn(n, 1); o = torch.zeros(n, 1)
    for i in range(steps):
        t = torch.full((n, 1), i / steps)
        x = x + (1 / steps) * fm(torch.cat([x, t, o], 1))
    return x
pred_fm = fm_sample(2000)

def report(name, p):
    near1 = ((p - 1).abs() < 0.3).float().mean().item()
    near_1 = ((p + 1).abs() < 0.3).float().mean().item()
    near0 = (p.abs() < 0.3).float().mean().item()
    print(f"{name:<22} 均值 {p.mean():+.3f}  标准差 {p.std():.3f}  "
          f"| 落在 -1 附近 {near_1:5.1%}  +1 附近 {near1:5.1%}  "
          f"**0 附近（撞上去）{near0:5.1%}**")
    return near0

print("专家动作分布：50% 走 -1，50% 走 +1，绝不能输出 0\n")
z_reg = report("MSE 回归", pred_reg)
z_fm  = report("Flow matching", pred_fm)

assert abs(pred_reg.mean()) < 0.15 and pred_reg.std() < 0.05, "回归应该塌成常数"
assert z_reg > 0.95, "回归几乎全部落在 0 附近"
assert z_fm < 0.15, "flow 不该塌到 0"
assert ((pred_fm + 1).abs() < 0.3).float().mean() > 0.3 and \
       ((pred_fm - 1).abs() < 0.3).float().mean() > 0.3, "flow 要覆盖两个峰"
print("\n结论：回归塌成条件均值 0（每次都撞上去）；flow matching 两个模式各占约一半。")
```

实际跑出来：

```text
专家动作分布：50% 走 -1，50% 走 +1，绝不能输出 0

MSE 回归          均值 -0.015  标准差 0.000  | -1 附近  0.0%  +1 附近  0.0%  0 附近（撞上去）100.0%
Flow matching    均值 -0.055  标准差 0.968  | -1 附近 51.1%  +1 附近 45.4%  0 附近（撞上去）  1.4%
```

$$\boxed{\text{MSE 回归塌成条件均值 0，每一次都撞上去；flow matching 两个模式各占一半}}$$

注意回归的**标准差是 0.000** —— 它不是"有时对有时错"，而是**确定性地输出了那个最差的答案**。这就是 [action head](../05-action/02-action-head.md#2-连续回归为什么不够) 那一节的全部含义。

## 自测

**1.** 写出前向加噪的闭式解，说明为什么它让训练能并行。

> **答：** $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$。
> 不需要真的一步步加噪，随机采一个 $t$ 就能直接合成 $x_t$，一个 batch 里每条样本的 $t$ 还可以不同。

**2.** DDPM 采样时最后一步要注意什么？

> **答：** **$t=0$ 时不要再加噪声**（`x = mean + (sqrt(beta)*noise if t > 0 else 0)`），否则输出永远带一层噪。

**3.** DDIM 的核心两步是什么？`eta=0` 意味着什么？

> **答：** ① 先从 $x_t$ 和 $\epsilon_\theta$ **解出 $\hat x_0$**；② 再重新加噪到 $t-1$。
> `eta=0` 时 $\sigma_t=0$，采样**完全确定性** —— 同一个初始噪声两次跑出逐元素相同的结果。

**4.** CFG 最好的自检是什么？

> **答：** **$w=1$ 必须严格退化成纯条件生成**（$\epsilon(\varnothing)+1\cdot[\epsilon(c)-\epsilon(\varnothing)]=\epsilon(c)$），$w=0$ 退化成无条件。这两条都成立才说明公式没写反。

**5.** flow matching 的目标速度为什么是常量？

> **答：** 直线路径 $x_t=(1-t)x_0+tx_1$ 对 $t$ 求导就是 $x_1-x_0$，**和 $t$ 无关**。所以训练时不需要 noise schedule，$t$ 只是插值系数。

**6.** temporal ensembling 的权重怎么算？为什么越新权重越大？

> **答：** $w_i\propto e^{-m\cdot\text{age}_i}$ 再归一化，age 是这个 chunk 已经过了几步。
> 越新的预测用的观测越接近当前实际状态，所以更可信；旧 chunk 是在更早的观测下预测的，可能已经偏了。

**7.** 用那个双峰实验说明为什么动作不能用 MSE 回归。

> **答：** 专家动作是 $\pm1$ 各 50%，MSE 的最优解是条件均值 0。实测回归 **100% 输出 0 且标准差为 0.000** —— 确定性地给出那个"撞上去"的答案；flow matching 则 51% / 45% 分到两个模式，只有 1.4% 落在 0 附近。
> 关键在于它**不是随机出错，而是稳定地输出一个哪个模式都不是的值**。
