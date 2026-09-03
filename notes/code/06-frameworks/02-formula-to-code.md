# 公式 → 代码：slime 与 verl 逐条对照

> 把 [RL 目录](../../06-post-training/rl/00-map.md) 和 [蒸馏目录](../../06-post-training/distill/00-map.md) 里的每个公式，
> 对到两个框架的**确切函数和行号**。所有引用都在本机核对过：
> `slime` commit `3778dbf`、`verl` commit `24f25b0`（均 2026-08-28），仓库在 `learn/rl/framework/`。
>
> 行号会随上游漂移，**先按函数名 grep 再看行号**。整体架构和数据流见 [01 源码导读](01-rl-framework-source-reading.md)。

## 0. 先记住两个入口文件

| | 文件 | 行数 |
|---|---|---|
| **verl** | `verl/trainer/ppo/core_algos.py` | 2549 行，**所有算法都在这一个文件里** |
| **slime** | `slime/utils/ppo_utils.py` + `slime/backends/megatron_utils/loss.py` | 767 + 1382 行 |

verl 用**注册表**组织：`@register_adv_est("grpo")`、`@register_policy_loss("gspo")`，配置里写个名字就能换算法。
slime 用**分支 + dotted-path 回调**：`if args.advantage_estimator == "grpo"`，加上一堆 `--custom-*-path` 让你注入自定义函数。

$$\boxed{\text{verl 广度取胜（算法变体最全），slime 深度取胜（主线最短最好读）}}$$

## 1. 公式在一次迭代里出现的位置

```text
rollout（SGLang / vLLM）
   └─ 记录 old_log_prob ────────────────────────┐
                                                │
reward / verifier                               │
   └─ GRPO 组内归一化  ← slime 在这里做           │
                                                │
训练侧 forward                                   │
   ├─ 算 log_prob（当前 policy）                  │
   ├─ 算 ref_log_prob（可选，给 KL 用）           │
   ├─ advantage：GAE / GRPO 组内归一化 ← verl 在这里做
   ├─ ratio = exp(log_prob − old_log_prob) ◄─────┘
   ├─ policy loss：clip + min（+ dual clip）
   ├─ loss 聚合：token-mean / seq-mean
   └─ backward
```

**最值得注意的一处架构差异**：GRPO 的组内归一化，**slime 在 rollout 侧做完再送进训练**，verl 在 **trainer 侧**做。下面第 6 条展开。

---

## 2. Importance ratio

$$r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}=\exp\big(\log\pi_\theta-\log\pi_{old}\big)$$

**verl** `core_algos.py:1336-1339`

```python
negative_approx_kl = log_prob - old_log_prob
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)   # 防 exp 溢出
ratio = torch.exp(negative_approx_kl)
```

**slime** `ppo_utils.py:132`（`ppo_kl` 由 `loss.py:1031` 传入）

```python
# loss.py:1031
ppo_kl = old_log_probs - log_probs
# ppo_utils.py:132
ratio = (-ppo_kl).exp()
```

### 卡点：两个框架的中间量符号相反

$$\boxed{\text{verl 的 negative\_approx\_kl}=\log\pi_\theta-\log\pi_{old}\qquad \text{slime 的 ppo\_kl}=\log\pi_{old}-\log\pi_\theta}$$

结果相同（slime 多取一次负号），但**读代码时极易看反**。verl 还额外 clamp 到 $[-20,20]$ 防 `exp` 溢出，slime 没有。

对应笔记：[importance sampling 与 ratio](../../06-post-training/rl/02-importance-sampling-and-ratio.md)、[为什么用 exp(对数差)](../../06-post-training/rl/05-ppo-engineering.md)。

---

## 3. Clip 与 min

$$L^{PPO}=-\mathbb E\Big[\min\big(r_tA_t,\ \mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t\big)\Big]$$

**verl** `core_algos.py:1342-1349`

```python
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
```

**slime** `ppo_utils.py:132-136`

```python
ratio = (-ppo_kl).exp()
pg_losses1 = -ratio * advantages
pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
clipfrac = torch.gt(pg_losses2, pg_losses1).float()
```

### 卡点：为什么代码里是 maximum 不是 min

公式里是 $\min$，代码里是 `torch.maximum`。因为**代码算的是 loss（已取负）**：

$$\max(-x,-y)=-\min(x,y)$$

两个框架都是这么写的，一模一样。看到 `maximum` 别以为写错了。

`clipfrac` 的定义也一致：`pg_losses2 > pg_losses1` 的比例，即「clip 生效」的 token 占比。对应 [PPO 工程](../../06-post-training/rl/05-ppo-engineering.md) 里说的监控指标。

对应笔记：[clip 与 min](../../06-post-training/rl/03-clip-and-min.md)。

---

## 4. Clip-Higher（DAPO）

$$\mathrm{clip}(r_t,\ 1-\epsilon_{\text{low}},\ 1+\epsilon_{\text{high}})$$

两个框架都是**上下界解耦**，直接支持 DAPO。

**verl** `core_algos.py:1321-1322`

```python
clip_ratio_low  = config.clip_ratio_low  if config.clip_ratio_low  is not None else clip_ratio
clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
```

**slime**：直接就是两个参数 `--eps-clip` / `--eps-clip-high`（`ppo_utils.py:128-129`）。

所以「开 DAPO 的 Clip-Higher」在这两个框架里都只是**改一个配置**，不需要换 loss 函数。

对应笔记：[DAPO](../../06-post-training/rl/08-dapo.md)。

---

## 5. Dual-clip

当 $A<0$ 且 $r$ 极大时，$-rA$ 会变成一个很大的正数（巨大的惩罚），梯度爆炸。dual-clip 给它再加一个下界：

$$\max\Big(\min\big(rA,\mathrm{clip}(r)A\big),\ c\,A\Big),\qquad c>1$$

**verl** `core_algos.py:1354-1360`

```python
pg_losses3 = -advantages * clip_ratio_c              # 默认 clip_ratio_c = 3.0
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

**slime** `ppo_utils.py:139-147`：逻辑完全一样，参数叫 `eps_clip_c`，且**默认关闭**（`None` 时不启用）。

注意 `torch.where(advantages < 0, ...)`：**dual-clip 只在 $A<0$ 时生效**，$A>0$ 那一侧不需要。

---

## 6. GRPO 的组内归一化

$$A_i=\frac{R_i-\mathrm{mean}(R)}{\mathrm{std}(R)+\epsilon}$$

**这是两个框架架构差异最大的一处。**

**verl**：在 **trainer 侧**，`core_algos.py:268-333`

```python
scores = token_level_rewards.sum(dim=-1)
id2score = defaultdict(list)
for i in range(bsz):
    id2score[index[i]].append(scores[i])          # index 是 prompt id，同 prompt 归一组
for idx in id2score:
    if len(id2score[idx]) == 1:                   # 组内只有一条 -> mean=0, std=1
        id2mean[idx], id2std[idx] = torch.tensor(0.0), torch.tensor(1.0)
    else:
        scores_tensor = torch.stack(id2score[idx])
        id2mean[idx] = torch.mean(scores_tensor)
        id2std[idx]  = torch.std(scores_tensor)
for i in range(bsz):
    if norm_adv_by_std_in_grpo:
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    else:
        scores[i] = scores[i] - id2mean[index[i]]   # Dr.GRPO：只减均值不除 std
scores = scores.unsqueeze(-1) * response_mask       # ← 序列级 A 广播到每个 token
```

**slime**：在 **rollout 侧**，`slime/ray/rollout.py:287-301`

```python
rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)   # (n_prompt, G)
mean = rewards.mean(dim=-1, keepdim=True)
rewards = rewards - mean
if self.args.advantage_estimator in ["grpo", "gspo", "cispo"] and self.args.grpo_std_normalization:
    std = rewards.std(dim=-1, keepdim=True)
    rewards = rewards / (std + 1e-6)
```

送进训练侧时 reward **已经是归一化后的**，所以 `ppo_utils.py:361-368` 的 `get_grpo_returns` 只剩广播这一件事：

```python
def get_grpo_returns(rewards, kl):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])   # 序列级标量 -> 每个 token
    return returns
```

### 三个可以直接验证笔记的点

1. **除的是 std 不是 var**（两个框架都是 `.std()`），对应 [为什么除标准差](../../06-post-training/rl/06-grpo.md#卡点-12为什么除标准差不是方差)。
2. **组内只有一条时 mean=0、std=1**（verl `:277-279`），也就是 advantage 直接为 0 —— 正是 [DAPO Dynamic Sampling](../../06-post-training/rl/08-dapo.md) 要处理的退化情形。
3. **`norm_adv_by_std_in_grpo=False` 就是 Dr.GRPO**（只减均值不除 std），verl 直接把这个开关暴露出来了。

对应笔记：[GRPO](../../06-post-training/rl/06-grpo.md)。

---

## 7. 序列级 advantage 广播到 token

$$A\cdot\nabla\log\pi(y|x)=\sum_t A\cdot\nabla\log\pi(y_t|\cdot)$$

**verl** `core_algos.py:331`：`scores.unsqueeze(-1) * response_mask`
**slime** `ppo_utils.py:367`：`torch.ones_like(kl[i]) * rewards[i]`

两行代码就是笔记里那个「**分解不是替换**」的推导落地：把同一个标量乘到每个 token 上，加起来恰好等于序列级梯度。

对应笔记：[seq-level A 怎么作用到每个 token](../../06-post-training/rl/06-grpo.md#卡点-13为什么-sequence-level-的-a-能乘到每个-token-上)。

---

## 8. GAE

$$\hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l},\qquad \delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$$

**verl** `core_algos.py:245-262`（倒序递推，标准写法）

```python
for t in reversed(range(gen_len)):
    delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
    lastgaelam_ = delta + gamma * lam * lastgaelam
    # 跳过 observation token（multi-turn 时环境返回的部分不参与 TD）
    nextvalues  = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
    lastgaelam  = lastgaelam_  * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam
    advantages_reversed.append(lastgaelam)
advantages = torch.stack(advantages_reversed[::-1], dim=1)
returns = advantages + values                       # ← R̂ = Â + V_old
advantages = verl_F.masked_whiten(advantages, response_mask)
```

**slime** `ppo_utils.py:584` `vanilla_gae` / `:693` `chunked_gae`（分块版是为了配合 context parallel）。

### 两个可以直接对上笔记的细节

- **`returns = advantages + values`**（`:264`）就是 $\hat R=\hat A+V_{old}$，critic 的回归 target。
- **`masked_whiten`**（`:265`）是 advantage 归一化，注意它在 GAE **之后**做。
- multi-turn 场景下 observation token 被 mask 掉不参与 TD 递推，这是纯数学公式里看不到的工程细节。

对应笔记：[advantage / critic / GAE](../../06-post-training/rl/04-advantage-critic-gae.md)。

---

## 9. Loss 聚合（DAPO 的 token-level）

$$\text{token-mean}:\ \frac{\sum_i\sum_t \ell_{i,t}}{\sum_i|y_i|}\qquad\text{vs}\qquad\text{seq-mean}:\ \frac1N\sum_i\frac{1}{|y_i|}\sum_t\ell_{i,t}$$

**verl** `core_algos.py:1140-1206` 的 `agg_loss`，四种模式：

| `loss_agg_mode` | 含义 | 对应 |
|---|---|---|
| **`token-mean`**（默认） | 全局所有 token 平均 | **DAPO** |
| `seq-mean-token-mean` | 先句内平均、再句间平均 | **原始 GRPO** |
| `seq-mean-token-sum` | 句内求和、句间平均 | |
| `token-sum` | 直接求和（配 dp_size 缩放） | |

```python
if loss_agg_mode == "token-mean":
    loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
elif loss_agg_mode == "seq-mean-token-mean":
    seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)   # 句内平均
    loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size
```

$$\boxed{\text{GRPO}\to\texttt{seq-mean-token-mean}\qquad\text{DAPO}\to\texttt{token-mean}}$$

这正是笔记里那个「A 有 2 个 token、B 有 10 个 token，权重比 1:1 还是 1:5」的手算题，在代码里就是换一个字符串。

**注意 `batch_num_tokens` 是 global 的**（`:1172-1176` 强制 `dp_size>1` 时必须传），否则每个 DP rank 各自按本地 token 数归一化，梯度就不对了 —— 这和笔记里 [梯度累积的 token normalization](../../03-training-fundamentals/03-gradient-accumulation.md#4-卡点真正的坑是-token-normalization) 是同一个坑，只是从 microbatch 之间变成了 DP rank 之间。

对应笔记：[DAPO 的 loss 聚合](../../06-post-training/rl/08-dapo.md#卡点-18一个-batch-内的-loss-怎么算)。

---

## 10. GSPO 的 sequence-level ratio

$$s_i(\theta)=\Big(\frac{\pi_\theta(y_i|x)}{\pi_{old}(y_i|x)}\Big)^{1/|y_i|}=\exp\Big(\frac1{|y_i|}\sum_t\big[\log\pi_\theta-\log\pi_{old}\big]\Big)$$

**verl** `core_algos.py:1583-1594`

```python
seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths   # 几何平均的 log
# straight-through：值用序列比，梯度走 token 级 log_prob
log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)
seq_importance_ratio = torch.exp(log_seq_importance_ratio)
```

**slime** `ppo_utils.py:114-120`

```python
ppo_kl = [((old_logprob - log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
          for log_prob, old_logprob, loss_mask in zip(...)]
ppo_kl = [kl.expand_as(log_prob) for kl, log_prob in zip(ppo_kl, local_log_probs)]
```

### 卡点：verl 用了 straight-through，slime 没用

verl 的 `log_prob - log_prob.detach() + seq.detach()` 是一个**恒等于 `seq` 的值、但梯度走 token 级 `log_prob`** 的技巧（前两项数值抵消、梯度不抵消）。slime 直接用序列平均，梯度自然通过那个 `sum/len` 回传。

两者数学上都实现了 GSPO，但**梯度路径不同**，这是读源码才能发现的差异。

另外 verl 的注释明确说：原论文用 `seq-mean-token-mean` 聚合，而它默认仍是 `token-mean`（`:1605-1606`）。

对应笔记：[GSPO](../../06-post-training/rl/09-gspo.md)。

---

## 11. KL 估计器 k1 / k2 / k3

$$k_1=\log r,\qquad k_2=\tfrac12(\log r)^2,\qquad k_3=e^{-\log r}-1+\log r$$

**slime** `ppo_utils.py:29-47`

```python
log_ratio = log_probs.float() - log_probs_base.float()
if kl_loss_type == "k1":
    kl = log_ratio
elif kl_loss_type == "k2":
    kl = log_ratio**2 / 2.0
elif kl_loss_type in ["k3", "low_var_kl"]:
    log_ratio = -log_ratio
    kl = log_ratio.exp() - 1 - log_ratio      # 非负、无偏、低方差
if importance_ratio is not None:
    kl = importance_ratio * kl                # DeepSeek-V3.2 的无偏 KL
if kl_loss_type == "low_var_kl":
    kl = torch.clamp(kl, min=-10, max=10)
```

**verl** `core_algos.py:2188-2215`，多了一个 `k3+` 的**直通梯度**技巧：

```python
# k1/k3 的期望是 KL，但它们的期望梯度不是 KL 的梯度；k2 的梯度才对。
# 所以 "k3+" 用值走 k3、梯度走 k2 的直通写法：
backward_score = 0.5 * (logprob - ref_logprob).square()
return backward_score - backward_score.detach() + forward_score.detach()
```

$$\boxed{k_3\ \text{的值无偏，但梯度有偏；}k_3^+\ \text{让值走 }k_3\text{、梯度走 }k_2}$$

这是一个笔记里没有、只有读源码才会遇到的精细点，面试提到会很加分。

对应笔记：[KL 散度](../../06-post-training/rl/07-kl.md)。

---

## 12. OPD：reverse KL 直接减进 advantage

$$A_t\leftarrow A_t-\beta\big(\log\pi_S(a_t|s_t)-\log\pi_T(a_t|s_t)\big)$$

**只有 slime 有**（verl 目前没有对应实现）。`loss.py:663-701`

```python
def apply_opd_kl_to_advantages(args, rollout_data, advantages, student_log_probs):
    teacher_log_probs = rollout_data.get("teacher_log_probs")
    for i, adv in enumerate(advantages):
        reverse_kl = student_log_probs[i] - teacher_log_probs[i]
        advantages[i] = adv - args.opd_kl_coef * reverse_kl     # ← 就这一行
    rollout_data["opd_reverse_kl"] = reverse_kls
```

相关参数：`--use-opd`、`--opd-type {sglang,megatron}`、`--opd-kl-coef`（默认 1.0）、`--opd-teacher-load`。监控指标是 `opd_reverse_kl`。

### 这一行完全印证了蒸馏笔记的推导

笔记里推出 $A_t=\log\pi_T-\log\pi_S$ 就是 reverse KL 的负数，可以直接当 per-token advantage。代码里写的是 `adv - coef * (student - teacher)`，展开就是

$$A_t+\beta\big(\log\pi_T-\log\pi_S\big)$$

也就是**在原有 advantage 上叠加一个 OPD 项**，而且注释明确写着 “This is orthogonal to the base advantage estimator” —— 和笔记里强调的「OPD 和 advantage estimator 是两个正交维度」完全一致。代码里的 docstring 还直接引用了 Thinking Machines 的 tinker-cookbook。

对应笔记：[reverse KL 就是 policy gradient](../../06-post-training/distill/04-reverse-kl-as-pg.md)、[OPD](../../06-post-training/distill/03-opd.md)。

---

## 13. CISPO

$$L=-\operatorname{sg}\big[\mathrm{clip}(r,1-\epsilon,1+\epsilon_{\text{high}})\big]\cdot A\cdot\log\pi_\theta$$

**slime** `ppo_utils.py:168-172`

```python
ratio = (-ppo_kl).exp()
ratio_truncated = torch.clamp(ratio, min=1.0 - eps_clip, max=1.0 + eps_clip_high)
pg_losses = -ratio_truncated.detach() * advantages * log_probs     # ← ratio 上 stop-gradient
clipfrac = (ratio_truncated != ratio).float()
```

**verl** `core_algos.py:2048` `compute_policy_loss_cispo`。

### 和 PPO 的关键区别

PPO 里被 clip 的 token **梯度为 0**（clip 后是常数）；CISPO 把 ratio 放进 `detach()`，**梯度走 `log_probs`**，所以**被截断的 token 仍然贡献梯度**。

slime 的 docstring 直接点明了这一点：“Unlike PPO, the IS ratio is clipped under stop-gradient and the gradient flows through `log_probs`, so clipped tokens still contribute gradient.”

这正好可以拿来反向理解 [clip 是怎么让参数停止更新的](../../06-post-training/rl/03-clip-and-min.md#卡点-5clip-到底怎么让参数停止更新)：CISPO 就是**故意不让它停**。

---

## 14. 两个框架的符号与命名对照

读代码时最容易看混的地方：

| 概念 | verl | slime |
|---|---|---|
| $\log\pi_\theta-\log\pi_{old}$ | `negative_approx_kl`（**正号**） | `-ppo_kl`（`ppo_kl` 是**反号**） |
| clip 上下界 | `clip_ratio_low` / `clip_ratio_high` | `eps_clip` / `eps_clip_high` |
| dual clip 系数 | `clip_ratio_c`（默认 3.0，**开启**） | `eps_clip_c`（默认 `None`，**关闭**） |
| loss mask | `response_mask` | `loss_masks` |
| 算法选择 | `@register_adv_est` / `@register_policy_loss` 注册表 | `args.advantage_estimator` 分支 |
| GRPO 归一化位置 | trainer 侧 `core_algos.py` | **rollout 侧** `ray/rollout.py` |
| loss 聚合 | `agg_loss(loss_agg_mode=...)` | `sum_of_sample_mean` 系列 |
| 自定义算法 | hydra 配置 + 注册装饰器 | `--custom-*-path` dotted path |

## 15. 动手验证理解的三个小实验

不用真跑训练，读代码 + 手算就能验证：

1. **把 `loss_agg_mode` 从 `token-mean` 换成 `seq-mean-token-mean`**，用笔记里「A 2 个 token、B 10 个 token」那个例子手算两种结果，确认权重比从 1:5 变成 1:1。
2. **在 verl 的 `compute_grpo_outcome_advantage` 里把 `norm_adv_by_std_in_grpo` 设成 `False`**，确认它就是 Dr.GRPO，并解释为什么去掉 std 会破坏尺度不变性。
3. **对比 slime 的 `compute_policy_loss` 和 `compute_cispo_loss`**，指出哪一行决定了「被 clip 的 token 还有没有梯度」。

## 自测

**1.** verl 和 slime 的 ratio 中间量符号有什么区别？各自怎么防溢出？

> **答：** verl 的 `negative_approx_kl = log_prob - old_log_prob`（正号），`ratio = exp(negative_approx_kl)`；slime 的 `ppo_kl = old_log_probs - log_probs`（**反号**），`ratio = (-ppo_kl).exp()`。结果相同但中间量差一个负号，读代码极易看反。
> verl 额外 `torch.clamp(negative_approx_kl, -20, 20)` 防 `exp` 溢出，slime 没有这一步。

**2.** 为什么两个框架的 PPO loss 代码里都是 `torch.maximum` 而不是 `min`？

> **答：** 因为代码算的是 **loss（已经取过负号）**，而 $\max(-x,-y)=-\min(x,y)$。公式里对 objective 取 $\min$，等价于对 loss 取 $\max$。看到 `maximum` 不要以为写错了。

**3.** GRPO 的组内归一化在两个框架里分别在哪做？这个差异有什么影响？

> **答：** **verl 在 trainer 侧**（`core_algos.py:268-333` 的 `compute_grpo_outcome_advantage`）；**slime 在 rollout 侧**（`ray/rollout.py:287-301`），送进训练时 reward 已经归一化好了，所以 slime 的 `get_grpo_returns` 只剩广播一件事。
> 影响：slime 的训练侧代码更薄更好读，但要改归一化逻辑得去 rollout 那边找；verl 把它和其他 advantage estimator 并列在一个注册表里，换算法更统一。

**4.** verl 里 `norm_adv_by_std_in_grpo=False` 是什么算法？为什么？

> **答：** 是 **Dr.GRPO** —— 只减均值不除 std（`scores[i] = scores[i] - id2mean[index[i]]`）。
> 去掉 std 后 advantage 不再是尺度不变的，reward 整体放大 10 倍 advantage 也放大 10 倍；Dr.GRPO 认为除 std 会引入长度/难度偏置，宁可不做。

**5.** 组内只有一条 rollout 时 verl 怎么处理？这对应笔记里的什么问题？

> **答：** `core_algos.py:277-279` 直接设 `mean=0.0, std=1.0`，于是 advantage 等于 reward 本身而不是 0。但当组内多条 reward **全相同**时，减完均值就全是 0 —— 这正是 **DAPO Dynamic Sampling** 要处理的退化情形：梯度为 0，白白浪费了这批 rollout 的采样算力。

**6.** `loss_agg_mode` 的四种模式里，GRPO 和 DAPO 各对应哪一个？

> **答：** **GRPO → `seq-mean-token-mean`**（先句内平均再句间平均，每条 response 权重相同）；**DAPO → `token-mean`**（全局所有 token 一起平均，每个 token 权重相同，verl 的默认值）。
> 换算法在代码里就是换一个字符串。注意 `token-mean` 的 `batch_num_tokens` 必须是 **global** 的，否则各 DP rank 按本地 token 数归一化，梯度就错了。

**7.** verl 的 GSPO 用了什么技巧？和 slime 的实现有什么区别？

> **答：** verl 用 **straight-through**：`log_prob - log_prob.detach() + negative_approx_kl_seq.detach()` —— 前两项数值上抵消，所以**值等于序列级 ratio**，但**梯度走 token 级 `log_prob`**。
> slime 直接用序列平均的 `ppo_kl` 再 `expand_as`，梯度自然通过那个 `sum/len` 回传。
> 两者数学上都实现了 GSPO，但**梯度路径不同**，这是只有读源码才能发现的差异。

**8.** `k3` 和 `k3+` 有什么区别？

> **答：** $k_3=e^{-\log r}-1+\log r$ 的**值**是 KL 的无偏估计，但它的**期望梯度不是 KL 的梯度**；$k_2=\frac12(\log r)^2$ 的梯度才是对的。
> verl 的 `k3+`（`core_algos.py:2210-2215`）用直通技巧：`backward_score - backward_score.detach() + forward_score.detach()`，让**值走 $k_3$、梯度走 $k_2$**，两全其美。

**9.** slime 的 OPD 是怎么实现的？为什么说它印证了笔记里的推导？

> **答：** `loss.py:663-701` 的 `apply_opd_kl_to_advantages`，核心就一行：
> ```python
> advantages[i] = adv - args.opd_kl_coef * (student_log_probs[i] - teacher_log_probs[i])
> ```
> 展开就是 $A_t+\beta(\log\pi_T-\log\pi_S)$ —— 正是笔记推出的「$A_t=\log\pi_T-\log\pi_S$ 可以直接当 per-token advantage」。
> 而且它是**在原有 advantage 上叠加**，docstring 明确写 “orthogonal to the base advantage estimator”，和笔记强调的「OPD 与 advantage estimator 是两个正交维度」完全一致。verl 目前没有对应实现。

**10.** CISPO 和 PPO 在「被 clip 的 token 还有没有梯度」上有什么区别？哪一行代码决定的？

> **答：** PPO 里被 clip 的 token **梯度为 0**（clip 之后对 $\theta$ 是常数）；CISPO 把 ratio 放进 `.detach()`、让梯度走 `log_probs`，所以**被截断的 token 仍然贡献梯度**。
> 决定这件事的是 slime `ppo_utils.py:170`：
> ```python
> pg_losses = -ratio_truncated.detach() * advantages * log_probs
> ```
> `.detach()` 和末尾那个 `* log_probs` 就是全部区别。
