# RL 后训练框架源码导读：slime 与 verl

> 目标：以 **slime**（THUDM）和 **verl**（volcengine）两个仓库为教材，搞清楚 LLM RL 算法（PPO / GRPO / GSPO / CISPO / REINFORCE++ / RLOO 等）在真实框架里是如何落地的：数据怎么流、log-prob 在哪算、advantage 在哪算、loss 怎么拼、权重怎么同步回推理引擎。
>
> 所有引用都是 `路径:行号`，相对 `learn/rl/` 目录（两个仓库克隆在 `learn/rl/framework/` 下）。代码版本：
> - `framework/slime` v0.3.2，commit `3778dbf`（2026-08-28）
> - `framework/verl` commit `24f25b0`（2026-08-28）
>
> 行号会随上游变动而漂移，但函数名不太会变，先按函数名 grep 再看行号。

---

## 0. 怎么读这份文档

两个框架做的事情是同一件：**用推理引擎采样，用训练引擎更新，两边来回同步权重**。区别在于：

| | slime | verl |
|---|---|---|
| 训练后端 | 只有 Megatron-LM | 抽象成 `engine`：FSDP / FSDP2 / Megatron / veOmni / TorchTitan |
| 推理后端 | 只有 SGLang | vLLM / SGLang / TRT-LLM |
| 编排 | Ray，驱动脚本只有 99 行 | Ray + `single_controller`（HybridFlow）+ TransferQueue |
| 数据单元 | `Sample` dataclass → `dict[str, list[Tensor]]`（变长，不 padding） | `DataProto`（TensorDict + numpy）/ TensorDict nested tensor |
| 算法可插拔点 | `--custom-*-path` 一堆 dotted-path 参数 | `@register_adv_est` / `@register_policy_loss` 注册表 + hydra 配置 |
| 代码量（核心） | `ppo_utils.py` 767 行 + `loss.py` 1382 行 | `core_algos.py` 2549 行 |

建议顺序：**先 §1 建立一次迭代的整体图景，再 §2 过一遍数学，然后 slime（§3，代码更薄，更容易看清主线）→ verl（§4，算法变体最全）→ §5 对照**。

---

## 1. 一次 RL 迭代的六个阶段

任何 on-policy LLM RL 一轮都长这样：

```
① rollout      推理引擎对 batch 里每个 prompt 采 n 条回复（GRPO 的"组"）
② reward       规则 / RM 打分，得到每条回复的标量 reward
③ log-probs    训练引擎前向：old_log_probs（行为策略 π_old）、ref_log_probs（参考策略 π_ref）、（PPO 时）values
④ advantage    reward (+KL 惩罚) → 每个 token 的 advantage / return
⑤ update       policy loss（clip 等）+ entropy + KL loss → 反向 → optimizer.step（可能多个 mini-batch）
⑥ weight sync  把更新后的参数推回推理引擎，进入下一轮
```

两个框架在每个阶段的落点：

| 阶段 | slime | verl（v1，默认） |
|---|---|---|
| 驱动循环 | `framework/slime/train.py:49-91` | `framework/verl/verl/trainer/ppo/v1/trainer_base.py:540-590` (`_step_once`) |
| ① rollout | `framework/slime/slime/rollout/sglang_rollout.py:374-470` | `framework/verl/verl/trainer/ppo/v1/agent_loop_tq.py:107-148` |
| ② reward | `framework/slime/slime/rollout/rm_hub/__init__.py:55-96` | `framework/verl/verl/experimental/reward_loop/reward_loop.py:145-155` |
| ③ log-probs | `framework/slime/slime/backends/megatron_utils/actor.py:401-454` | `trainer_base.py:1541-1626` |
| ④ advantage | `framework/slime/slime/backends/megatron_utils/loss.py:704-880` | `framework/verl/verl/trainer/ppo/ray_trainer.py:187-282` + `core_algos.py` |
| ⑤ loss / update | `loss.py:933-1172` + `model.py:512-702` | `framework/verl/verl/workers/utils/losses.py:57-144` + `engine_workers.py:242-338` |
| ⑥ weight sync | `actor.py:563-624` + `update_weight/` | `framework/verl/verl/checkpoint_engine/base.py:506-558` + `engine_workers.py:727-820` |

---

## 2. 先把数学对上（代码里会反复出现的几个量）

### 2.1 PPO clipped objective 与 dual clip

记 $r_t = \pi_\theta(a_t)/\pi_{old}(a_t) = \exp(\log\pi_\theta - \log\pi_{old})$，advantage $A_t$：

$$L^{clip} = \max\big(-r_t A_t,\; -\mathrm{clip}(r_t, 1-\epsilon_{low}, 1+\epsilon_{high}) A_t\big)$$

- 两个框架都把它写成 `max(pg_losses1, pg_losses2)`（注意是 loss 形式，所以是 max）。
- $\epsilon_{low} \ne \epsilon_{high}$ 就是 DAPO 的 clip-higher。
- **dual clip**（arXiv 1912.09729）：当 $A_t<0$ 时再套一层 $\min(-c A_t, \cdot)$，防止负 advantage 下 ratio 爆炸。slime 叫 `eps_clip_c`，verl 叫 `clip_ratio_c`，默认 3.0。

### 2.2 GRPO 组内归一化

同一个 prompt 采 $n$ 条，$A_i = (R_i - \mathrm{mean}(R_{1..n})) / (\mathrm{std}(R_{1..n}) + \epsilon)$，然后**广播到该回复每个 token**。去掉除以 std 就是 Dr.GRPO。

> 关键差异：slime 在 **rollout 侧**做组归一化（`slime/ray/rollout.py:279-303`），训练侧拿到的 reward 已经是归一化好的标量；verl 在 **driver 侧** `compute_advantage` 里做（`core_algos.py:267-331`）。

### 2.3 KL 的三种估计（Schulman 博客）

设 $\rho = \log\pi_\theta - \log\pi_{ref}$：
- k1: $\rho$
- k2: $\rho^2/2$
- k3（low_var_kl）: $e^{-\rho} - 1 + \rho$，非负、无偏、低方差，GRPO 论文用的就是它

KL 有**两个可能的位置**：
- **KL in reward**：$r_t \leftarrow r_t - \beta\,\mathrm{KL}_t$，进入 advantage（经典 RLHF-PPO）
- **KL in loss**：$L \leftarrow L + \beta\,\mathrm{KL}$，直接加到 loss（GRPO 论文做法）

slime 用 `--kl-coef` 选前者、`--use-kl-loss --kl-loss-coef` 选后者，且**二选一**（`arguments.py:1850` 有 assert）；verl 用 `algorithm.use_kl_in_reward` 和 `actor.use_kl_loss`，可同时开。

### 2.4 Loss 聚合方式（这个坑最大）

每个 token 有一个 loss，怎么合成标量？

- **token-mean**：所有有效 token 求和 / 总 token 数。长回复权重大。DAPO 推荐。
- **seq-mean-token-mean**：每条序列内先平均，再对序列平均。每条序列权重一样。GRPO 原始做法。
- **seq-mean-token-sum**：序列内求和，再对序列平均。

再叠加梯度累积（micro-batch）和数据并行（DP）之后，"分母"到底是本 micro-batch 的 token 数、还是整个 mini-batch 全局 token 数，会直接改变梯度尺度。verl 的 `agg_loss` 显式接收 `batch_num_tokens`/`global_batch_size`/`dp_size`（§4.6.3），slime 的 `loss_function` 在末尾乘 `num_microbatches / step_global_batch_size * dp_size` 来抵消 Megatron 的平均（§3.6.5）。

### 2.5 Rollout / train mismatch 与 TIS

推理引擎（bf16、不同 kernel）给出的 $\log\pi_{rollout}$ 和训练引擎重算的 $\log\pi_{old}$ 不相等。修正办法是再乘一个重要性权重 $w_t = \pi_{old}/\pi_{rollout}$ 并截断（TIS, truncated IS）或超阈值直接置零（IcePop / MIS）。两个框架都有：slime `--use-tis`（§3.6.4），verl `algorithm.rollout_correction`（§4.8）。

---

## 3. slime 导读

### 3.1 入口：99 行的 `train.py`

`framework/slime/train.py:49-91` 就是整个同步训练循环，先把它读完：

```python
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))   # ① ② 采样 + 打分
    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())                            # colocate: 腾出显存
    ...
    ray.get(actor_model.async_train(rollout_id, rollout_data_ref))           # ③ ④ ⑤ 全在 actor 里
    ...
    actor_model.update_weights()                                             # ⑥ 推权重回 SGLang
    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())
```

要点：
- 三个 Ray 对象：`rollout_manager`（CPU actor，内部管 SGLang 引擎，`slime/ray/rollout.py:37`）、`actor_model`、`critic_model`（`RayTrainGroup`，`slime/ray/actor_group.py:32`）。创建在 `slime/ray/placement_group.py:120-137`。
- `use_critic` 不是显式参数，而是 `advantage_estimator == "ppo"` 推导出来的（`slime/utils/arguments.py:1913`）。critic 走 `train.py:62-67`，先算 value 再给 actor。
- `onload_weights` 在 `update_weights` **之前**、`onload_kv` 在**之后**（`train.py:83-88`）：权重内存得先存在才能接收更新，KV cache 等生成恢复时再回来。

异步版 `train_async.py:32-40`：提前发出下一轮 `generate.remote`，训练和采样重叠，`--update-weights-interval` 控制多少轮同步一次权重；不支持 colocate（`train_async.py:11`）。

### 3.2 数据单元：`Sample` → `RolloutBatch`

`framework/slime/slime/utils/types.py:93` 的 `Sample` 贯穿 rollout 全程，关键字段：

| 字段 | 行 | 含义 |
|---|---|---|
| `group_index` / `index` | :97-98 | 组 id（同 prompt 的 n 条）/ 全局样本 id |
| `tokens` | :109 | prompt + response 的 token id |
| `response_length` | :116 | 用来切出 response 部分 |
| `reward` | :118 | float 或 dict（`--reward-key` 选字段） |
| `loss_mask` | :119 | 长度 == response_length，多轮时屏蔽 tool 输出 |
| `rollout_log_probs` | :121 | SGLang 返回的 log-prob（TIS 用） |
| `status` | :140 | `PENDING/COMPLETED/TRUNCATED/ABORTED/FAILED`（:130-138） |

进入训练前被 `_convert_samples_to_train_data`（`slime/ray/rollout.py:306-423`）转成 `RolloutBatch = dict[str, list]`（`types.py:459`）：`tokens`、`response_lengths`、`rewards`、`loss_masks`、`rollout_log_probs`、`rollout_mask_sums`……全是 **变长 list，不 padding**。这决定了后面所有 loss 代码都是 `torch.cat` + `split(response_lengths)` 的风格。

### 3.3 Rollout：over-sampling、动态过滤、abort、partial rollout

调用链：`RolloutManager.generate`（`slime/ray/rollout.py:163-182`）→ `generate_rollout`（`slime/rollout/sglang_rollout.py:627-649`）→ `generate_rollout_async`（`:374-470`）。

核心循环 `sglang_rollout.py:394-442`：

```python
while len(data) < target_data_size:                       # target = rollout_batch_size（prompt 数）
    while state.remaining_batch_size < target_data_size:
        samples = data_source(args.over_sampling_batch_size)   # 多采一些
        state.submit_generate_tasks(samples)
    done, state.pendings = await asyncio.wait(state.pendings, return_when=FIRST_COMPLETED)
    for task in done:
        group = task.result()                             # 一组 = n_samples_per_prompt 条
        dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
        if should_drop_dynamic_filter_output(...):        # DAPO 式：全对/全错的组丢掉
            state.remaining_batch_size -= 1
            continue
        if len(data) < target_data_size:
            data.append(group)
```

- **n 条/prompt 在哪展开**：`slime/rollout/data_source.py:108-117`，`get_samples` 对每个 prompt `deepcopy` n 份；并发执行在 `generate_and_rm_group`（`sglang_rollout.py:297-336`）。
- **reward 在哪算**：`generate_and_rm`（`:225-289`）末尾 `sample.reward = await async_rm(args, sample)`；`async_rm` 在 `slime/rollout/rm_hub/__init__.py:55-96`，按 `--rm-type` 分发（`math`/`dapo`/`gpqa`/`remote_rm`……）或走 `--custom-rm-path`。
- **动态过滤器**：`slime/rollout/filter_hub/dynamic_sampling_filters.py:9-23`，`check_reward_nonzero_std` 就是"组内 reward 方差为 0 则丢"。
- **abort**：凑够 batch 后 `abort`（`:339-371`）向所有 SGLang worker 发 `/abort_request`，未完成的组若开了 `--partial-rollout` 会带着已生成的 token 塞回 buffer（`data_source.py:198-211`），下一轮 `generate`（`:166-173`）用 `max_new_tokens -= response_length` 续写。
- **组归一化在这里**：`RolloutManager._post_process_rewards`（`slime/ray/rollout.py:279-303`）对 `grpo/gspo/cispo/reinforce_plus_plus_baseline` 做 `reshape(-1, n)` 减均值、（grpo 系）除 std。所以训练侧的 `rewards` 已经是 advantage 标量了。

### 3.4 数据切分与"一轮几个 optimizer step"

`_split_train_data_by_dp`（`slime/ray/rollout.py:428-495`）调 `build_dp_schedule`（`slime/utils/dp_schedule.py:82`）：

- 同一 prompt 的 n 条样本必须在同一个 step（`dp_schedule.py:129-134`）。
- `num_steps = len(rollout_ids) // global_batch_size`（`:135`）。`--num-steps-per-rollout` 只是语法糖：`global_batch_size = rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout`（`arguments.py:1974-1984`）。
- 每个 step 再切 micro-batch（`--micro-batch-size` 或 `--use-dynamic-batch-size --max-tokens-per-gpu`）。
- 返回值按 DP rank 各自 `Box(ray.put(...))`（`rollout.py:486-490`），`Box` 只是防止 Ray 自动解引用的壳（`slime/utils/misc.py:131-137`）；每个训练 rank 只 `ray.get` 自己那份（`slime/utils/data.py:304-329`）。

### 3.5 训练侧主线：`train_actor`

`framework/slime/slime/backends/megatron_utils/actor.py:391-535`，按顺序：

1. `_switch_model("ref")` → `compute_log_prob(store_prefix="ref_")`（`:401-412`）。slime 没有独立的 ref 进程，而是同一个 Megatron 模型**换权重**（`TensorBackuper`，CPU 快照）。
2. 老 actor 前向 `compute_log_prob(store_prefix="")` 得 `log_probs`（`:448-454`），除非命中 `can_reuse_log_probs_in_loss`（`:428-439`）：

```python
can_reuse_log_probs_in_loss = (
    len(num_microbatches) == 1          # 一轮只有一个 optimizer step
    and self.args.loss_type == "policy_loss"
    and self.args.kl_coef == 0
    and not self.args.use_rollout_logprobs
    and not self.args.get_mismatch_metrics
    and not self.args.use_critic
    and not self.args.keep_old_actor
    and not self.args.use_opd
    and ... and self.args.advantage_estimator != "gspo"
)
```

   命中时省掉一次前向，loss 里 `old_log_probs = log_probs.detach()`（`loss.py:984-985`），ratio 恒为 1，clip 失效，退化为 REINFORCE 梯度。**这是 slime 一个很重要的默认行为**：单 step、无 KL 的 GRPO 实际上跑的是 on-policy policy gradient。

3. `compute_advantages_and_returns(self.args, rollout_data)`（`:470`），整轮一次性算完（因为 `--normalize-advantages` 要全局统计）。
4. `train(...)`（`:492-500`）→ `model.py:827-830` 循环 `num_steps_per_rollout` 次 `train_one_step`。
5. 训完 `weights_backuper.backup("actor")`（`:518`）刷新 CPU 快照；`--ref-update-interval` 到点则把 ref 也刷新（`:520-529`）。

`compute_log_prob` 本身是 `forward_only(get_log_probs_and_entropy, ...)`（`:324-339`），走 Megatron 的 `forward_backward_func(forward_only=True)`。

### 3.6 算法核心

#### 3.6.1 advantage：`compute_advantages_and_returns`

`framework/slime/slime/backends/megatron_utils/loss.py:704-880`。先算每 token KL（`:743-756`，`kl_coef==0` 时全零），然后按 `--advantage-estimator` 分发：

| 值 | 代码 | 公式 |
|---|---|---|
| `grpo` / `gspo` / `cispo` | `loss.py:763-767` → `ppo_utils.py:361-368` | `A_t = R`（已归一化的标量广播到每 token） |
| `ppo` | `loss.py:769-781` → `ppo_utils.py:476-581` | token reward = `-kl_coef * KL_t`，末 token 加 `R`；GAE（`chunked_gae` `:693-713`，分块并行扫描，默认；`vanilla_gae` `:584-605` 是教科书递推） |
| `reinforce_plus_plus` | `loss.py:783-794` → `ppo_utils.py:371-445` | 折扣累计回报，`rewards_for_seq = -kl_coef*KL; rewards_for_seq[last] += R`（`:413-420`） |
| `reinforce_plus_plus_baseline` | `loss.py:796-803` → `ppo_utils.py:448-473` | `A_t = R - kl_coef * KL_t` |

之后可选 OPD（`:809-815`）和全局白化 `--normalize-advantages`（`:818-877`，在 DP+CP 组上 all-reduce 统计量，`distributed_masked_whiten` 在 `slime/utils/distributed_utils.py:111-171`）。

`chunked_gae` 值得看一眼（`ppo_utils.py:705-711`）：

```python
next_values = torch.cat([values[:, 1:], zeros(B,1)], dim=1)
deltas = rewards + gamma * next_values - values
advantages = chunked_discounted_returns(deltas, gamma * lambd, chunk_size)   # 把 O(T) 递推变成分块矩阵乘
returns = advantages + values
```

#### 3.6.2 policy loss：`policy_loss_function`

`loss.py:933-1172`，一共 8 步，建议对着源码逐段读：

```
(1) get_log_probs_and_entropy(logits)         :969-977   当前策略 log π_θ + entropy（一次算完再按样本切）
(2) old_log_probs 选择                         :963-988   rollout_log_probs / log_probs / log_probs.detach()
(3) ppo_kl = old - new（GSPO 用序列平均）       :1019-1032
(4) compute_policy_loss / compute_cispo_loss  :1034-1043 → ppo_utils.py:124-148 / :151-171
(5) OPSM mask                                  :1045-1046
(6) TIS 权重                                   :1049-1091 → vanilla_tis_function :883-904
(7) sum_of_sample_mean 归约                     :1103-1105
(8) - entropy_coef * H + kl_loss_coef * KL     :1108-1129
```

核心公式 `ppo_utils.py:124-148`：

```python
ratio = (-ppo_kl).exp()                                # ppo_kl := old - new，所以 ratio = π_θ/π_old
pg_losses1 = -ratio * advantages
pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
if eps_clip_c is not None:                             # dual clip
    pg_losses3 = -eps_clip_c * advantages
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

GSPO 只改第 (3) 步（`ppo_utils.py:95-121`）：序列级 `mean_t(old - new)` 再 `expand_as` 到每个 token，于是 ratio 变成 $(\prod_t r_t)^{1/|y|}$。CISPO 只改第 (4) 步（`ppo_utils.py:167-170`）：`-sg(clip(ratio)) * A * log_prob`，梯度走 `log_prob`，被 clip 的 token 仍有梯度。

KL 估计 `compute_approx_kl`（`ppo_utils.py:11-51`），k1/k2/k3 三种，`--use-unbiased-kl` 时再乘 IS ratio（DeepSeek-V3.2 做法）。

#### 3.6.3 entropy 的计算方式

`_VocabParallelLogProbEntropy`（`ppo_utils.py:187-336`）是一个自定义 autograd Function，在 TP 切分的 vocab 上一次前向同时算 `log_prob` 和 `entropy = logsumexp(z) - Σ softmax(z)·z`（`:245-250`），手写 backward（`:298-336`），避免保存 `[T, V]` 的计算图。`--entropy-coef == 0` 时 entropy 只做指标不留梯度（`loss.py:551`）。

#### 3.6.4 TIS / OPSM

`vanilla_tis_function`（`loss.py:883-904`）：

```python
tis = torch.exp(old_log_probs - rollout_log_probs)              # π_train / π_sglang
tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
pg_loss = pg_loss * tis_weights
```

`icepop_function`（`:907-930`）改成超阈值置零。`--custom-tis-function-path` 可换成 `examples/train_infer_mismatch_helper/mis.py`（token / sequence / geometric 三种粒度 × truncate / clip / mask 三种模式）。

OPSM（`ppo_utils.py:54-92`）：序列级 KL 超过 `--opsm-delta` 且 advantage < 0 的整条序列屏蔽。

#### 3.6.5 loss 归约与 Megatron 缩放

`get_sum_of_sample_mean`（`slime/backends/megatron_utils/cp_utils.py:47-124`）：默认对每条样本 `(x * mask).sum() / denom` 再相加，`denom` 是 `rollout_mask_sums`（同一 prompt 组内所有样本的 mask 总和，`slime/ray/rollout.py:362-371`），即**按 rollout 组做 token 加权平均**。`--calculate-per-token-loss` 改成纯 token 求和。

`loss_function`（`loss.py:1282-1382`）分发 `policy_loss / value_loss / sft_loss / custom_loss`（`:1326-1336`），末尾做尺度修正（`:1351-1360`）：

```python
if not args.calculate_per_token_loss:
    loss = loss * num_microbatches / step_global_batch_size * dp_world_size(with_cp)
else:
    loss = loss * cp_world_size
```

这一步是为了抵消 Megatron `forward_backward_func` 内部按 micro-batch 数平均的行为。

#### 3.6.6 critic / value loss

`value_loss_function`（`loss.py:1175-1229`），PPO 标准 clipped value loss（`:1211-1218`）：

```python
values_clipped = old_values + (values - old_values).clamp(-value_clip, value_clip)
loss = torch.max((values_clipped - returns)**2, (values - returns)**2)
```

critic 是单独的 Ray actor 角色，`train_critic`（`actor.py:363-389`）先 `forward_only(get_values)`，算 advantage，把 `loss_type` 改成 `value_loss` 再 `train`，返回的 CPU values 通过 `external_data` 交给 actor（`train.py:63-65`，`actor.py:458-464`）。

### 3.7 Megatron 训练循环

`framework/slime/slime/backends/megatron_utils/model.py`：

- `train`（`:710`）→ `for step_id in range(num_steps_per_rollout): train_one_step(...)`（`:827-830`）
- `train_one_step`（`:512-702`）：`zero_grad` → 定义 `forward_step`（`get_batch` 取 micro-batch，`:568-604`；模型前向；返回 `partial(loss_function, ...)`）→ `forward_backward_func(...)`（`:647-657`）→ `optimizer.step()` + `opt_param_scheduler.step(increment=step_global_batch_size)`（`:679-686`）
- `get_batch`（`data.py:14-150`）：把变长样本 `cat` 成 `[1, T]` 的 packed 序列，构造 `PackedSeqParams(qkv_format="thd")`，CP 时做 zigzag 切片。这就是 slime 不 padding 的实现基础。

### 3.8 权重同步与显存腾挪

`MegatronTrainRayActor.update_weights`（`actor.py:563-624`）→ `weight_updater.update_weights()`。updater 由 `create_weight_updater`（`update_weight/__init__.py:10-56`）选：

| 条件 | 类 | 机制 |
|---|---|---|
| `--colocate` | `UpdateWeightFromTensor`（`update_weight_from_tensor.py:51`） | CUDA IPC，同卡 SGLang 直接拿张量 |
| 分离部署 | `UpdateWeightFromDistributed`（`update_weight_from_distributed.py:24`） | 建 NCCL 组，按 bucket broadcast（`:101-133`） |
| `--update-weight-transport disk` | `UpdateWeightFromDisk` | 写 HF checkpoint，SGLang `update_weights_from_disk` |
| `--update-weight-mode delta` | `UpdateWeightFromDiskDelta` | 只写 delta |

Megatron → HF 命名转换在 `update_weight/../megatron_to_hf/*.py`，按模型族一个文件。

colocate 时的显存腾挪：训练侧 `sleep/wake_up`（`actor.py:174-212`，用 `torch_memory_saver`），推理侧 `release_memory_occupation / resume_memory_occupation`（`slime/backends/sglang_utils/sglang_engine.py:328-339`）。placement group 布局在 `slime/ray/placement_group.py:100-117`：colocate 时 PG 大小 = `max(actor, rollout)`，rollout offset 0；否则 actor + rollout 首尾相接。

### 3.9 异步

- `train_async.py`：一步流水线（§3.1）。
- `slime/rollout/fully_async_rollout.py`：常驻后台线程 + asyncio loop（`AsyncRolloutWorker`，`:76-208`），rollout 永不停，训练只从输出队列取够 `rollout_batch_size` 组就走；无 abort、无动态过滤，ABORTED 组直接回 buffer（`:199-206`）。通过 `--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async` 启用。

### 3.10 扩展点速查

| 参数 | 加载处 | 签名 |
|---|---|---|
| `--custom-loss-function-path` | `loss.py:1334` | `f(args, batch, logits, sum_of_sample_mean) -> (loss, metrics)` |
| `--custom-advantage-function-path` | `loss.py:759` | `f(args, rollout_data) -> None`，原地写 `advantages/returns` |
| `--custom-tis-function-path` | `loss.py:1074` | `f(*, args, pg_loss, train_log_probs, rollout_log_probs, loss_masks, ...) -> (pg_loss, masks, metrics)` |
| `--custom-rm-path` | `rm_hub/__init__.py:58` | `async def f(args, sample, **kw) -> float \| dict` |
| `--custom-reward-post-process-path` | `slime/ray/rollout.py:61` | `f(args, samples) -> (raw_rewards, rewards)` |
| `--rollout-function-path` | `slime/ray/rollout.py:57` | `f(args, rollout_id, data_source, evaluation=False)` |
| `--custom-generate-function-path` | `sglang_rollout.py:252` | `f(args, sample, sampling_params)`（多轮 / agent 在这里接） |
| `--dynamic-sampling-filter-path` | `sglang_rollout.py:395` | `f(args, group) -> bool \| DynamicFilterOutput` |

---

## 4. verl 导读

### 4.1 入口与"v0 / v1"

`framework/verl/verl/trainer/main_ppo.py:166-193`：hydra 读 `config/ppo_trainer.yaml`，`trainer.use_v1`（默认 `true`，`ppo_trainer.yaml:228`）决定走 `TaskRunnerV1` 还是 legacy `main_ppo_v0.TaskRunner`。

- **v1（当前默认）**：`TaskRunnerV1.run`（`main_ppo.py:133-162`）→ `get_trainer_cls(config.trainer.v1.trainer_mode)` → `PPOTrainerSync`（`v1/trainer_sync.py:25`，默认）/ `PPOTrainerColocateAsync` / `PPOTrainerSeparateAsync`，基类 `PPOTrainer`（`v1/trainer_base.py:119`）。v1 **强制开启 TransferQueue**（`main_ppo.py:143`）：batch 不再作为 `DataProto` 在 driver 上流转，而是 `KVBatchMeta`（一堆 key），数据在分布式 KV 里，worker 通过 `tqbridge` 按需拉取。
- **v0（legacy）**：`RayPPOTrainer`（`ppo/ray_trainer.py:286`），已标 `@deprecated`（`:285`，"will be removed in v0.9.0"）。但它的纯函数 `apply_kl_penalty`（`:78`）、`compute_advantage`（`:187`）仍被 v1 复用。

如果你看过老版 verl 文档，注意：`verl/workers/fsdp_workers.py`、`megatron_workers.py`、`workers/actor/dp_actor.py`、`workers/critic/`、`sharding_manager/` **都已不存在**。对应关系：

| 老位置 | 现在 |
|---|---|
| `dp_actor.py::update_policy` | `workers/engine_workers.py:242` `train_mini_batch` + `workers/engine/base.py:113` `train_batch` |
| `dp_actor.py` 里的 loss 拼装 | `workers/utils/losses.py:57` `ppo_loss` |
| `dp_actor.py::_forward_micro_batch` | `workers/engine/fsdp/transformer_impl.py:1127-1425` |
| `fsdp_workers.py::ActorRolloutRefWorker` | `workers/engine_workers.py:451` |
| `sharding_manager/` | `checkpoint_engine/` + `rollout/*/update_weights` |

### 4.2 数据结构

- `DataProto`（`framework/verl/verl/protocol.py:317-328`）：`batch: TensorDict` + `non_tensor_batch: dict[str, np.ndarray(object)]` + `meta_info: dict`。关键方法 `union`（`:778`）、`chunk`（`:861`，按 DP 切）、`repeat`（`:968`，v0 的 rollout.n 在这展开）、`concat`（`:914`）、`select`（`:597`）。
- v1 用 TensorDict 的 **nested tensor**（变长）在 worker 内部流转，loss 前用 `no_padding_2_padding` / `to_padded_tensor()` 转成 `(bs, response_len)` 的 padded 张量（`losses.py:59, 91`）。
- 关键字段名（`ppo_loss` 里 select 的）：`response_mask`、`old_log_probs`、`advantages`、`ref_log_prob`、`rollout_is_weights`；driver 侧还有 `token_level_scores`（原始 reward）、`token_level_rewards`（加了 KL 惩罚）、`values`、`returns`、`uid`（组 id）。

### 4.3 `single_controller`：一个 driver 调用如何扇出

这是 HybridFlow 论文的核心工程点。`framework/verl/verl/single_controller/base/decorator.py`：

- `Dispatch` 模式注册在 `:38-47`（`ONE_TO_ALL`、`DP_COMPUTE_PROTO`……）。现代路径用 `make_nd_compute_dataproto_dispatch_fn(mesh_name)`（`:300`）：按命名 mesh（`"actor"`/`"ref"`/`"train"`）的 DP rank 映射切数据，TP/PP 重复 rank 拿同一份。
- `@register(dispatch_mode=..., blocking=...)`（`:398-444`）把 dispatch/collect 函数挂到方法的 `MAGIC_ATTR` 上，并用 `tqbridge` 包一层（`KVBatchMeta` ↔ TensorDict 的自动转换）。
- driver 侧 `RayWorkerGroup._bind_worker_method`（`base/worker_group.py:185-250`）扫描这些属性，生成 `func_generator`（`ray/base.py:49-67`）：`dispatch_fn → execute_fn（N 个 .remote）→ ray.get → collect_fn`。

所以 `self.actor_rollout_wg.compute_log_prob(batch)` 一行 = 切分 → N 个远程调用 → 拼回。一个用法例子：`engine_workers.py:699-705`。

### 4.4 训练循环：`_step_once` 九步

`framework/verl/verl/trainer/ppo/v1/trainer_base.py:540-590`，这就是 verl 版的"一次迭代"：

```python
batch, off_policy_metrics = self.replay_buffer.sample(...)          # 1 从 TransferQueue 取够 batch（等 rollout）
self.on_sample_end()                                                # sync 模式: sleep_replicas 释放推理显存
if self.reward_loop_manager.reward_loop_worker_handles is None:
    batch = self._compute_reward_colocate(batch, metrics=metrics)   # 2 仅 colocated RM 时；规则 reward 已在 rollout 里算完
batch = self._balance_batch(batch, metrics=metrics)                 # 3 按 token 数在 DP 间均衡
batch = self._compute_old_log_prob(batch, metrics=metrics)          # 4 actor 前向 → old_log_probs（或 bypass 用 rollout_log_probs）
if self.use_reference_policy:
    batch = self._compute_ref_log_prob(batch, metrics=metrics)      # 5
if self.use_critic:
    batch = self._compute_values(batch, metrics=metrics)            # 6
batch = self._compute_advantage(batch, metrics=metrics)             # 7 KL-in-reward → rollout correction → adv estimator
if self.use_critic:
    batch = self._update_critic(batch, metrics=metrics)             # 8
if self.config.trainer.critic_warmup <= self.global_steps:
    batch = self._update_actor(batch, metrics=metrics)              # 9
```

外层 `fit`（`:389-509`）：`step()`（`:511-538`，把 `train_batch_size` 按 `parameter_sync_step` 拆成多次 `_step_once`）→ 存 ckpt → `on_step_end()`（**权重同步在这**：`trainer_sync.py:35-38` 调 `checkpoint_manager.update_weights`）→ 验证 → `tq.kv_clear`。

`prepare_step`（`:1432`）→ `_submit_batch_to_rollout`（`:1403`）把 prompt 交给 `agent_loop_manager.generate_sequences`，**非阻塞**，`replay_buffer.sample` 再去等。

### 4.5 Rollout：AgentLoop 与 rollout.n

层次：`AgentLoopManagerTQ`（`v1/agent_loop_tq.py:230`）→ `AgentLoopWorkerTQ`（`:53`，Ray actor）→ `AgentLoopBase`（`experimental/agent_loop/agent_loop.py:207`，单轮/多轮/tool 逻辑）→ `LLMServerClient`（`workers/rollout/llm_server.py:197`，HTTP `/v1/chat/completions`）→ `RolloutReplica`（`workers/rollout/replica.py:70`）→ vLLM/SGLang `ServerAdapter`。

- **n 条/prompt 在 worker 内展开**（v1）：`agent_loop_tq.py:115-129`，`for i in range(n): _run_agent_loop(session_id=i)`；写回 TQ 的 key 是 `{uid}_{session_id}_{index}`（`:193`）。v0 是 driver 上 `gen_batch.repeat(rollout.n)`（`ray_trainer.py:1491`）。
- **reward 在 rollout 内流式算**（默认）：`AgentLoopWorker._compute_score`（`agent_loop.py:937-1000`）→ `RewardLoopWorker.compute_score`（`experimental/reward_loop/reward_loop.py:145-155`）→ `reward_manager.run_single` → `default_compute_score`（`utils/reward_score/__init__.py:19`，按 `data_source` 分发）。
- `RolloutMode`（`replica.py:54-67`）：`HYBRID`（与训练同进程同卡，需要 weight sync）/ `COLOCATED` / `STANDALONE`。

### 4.6 算法核心：`core_algos.py`

`framework/verl/verl/trainer/ppo/core_algos.py`，两个注册表：`register_adv_est`（`:116`）/ `get_adv_estimator_fn`（`:137`），`register_policy_loss`（`:53`）/ `get_policy_loss_fn`（`:70`）。

#### 4.6.1 advantage estimators（`algorithm.adv_estimator`）

所有函数签名统一：`(token_level_rewards, response_mask, index=uid, config, ...) -> (advantages, returns)`，形状 `(bs, response_len)`。

| 名字 | 行 | 一句话 |
|---|---|---|
| `gae` | `:215` | 唯一用 `values` 的；标准 GAE 递推 + `masked_whiten` |
| `grpo` | `:267` | 按 `uid` 分组，`(R - mean) / (std + eps)`；`norm_adv_by_std_in_grpo=False` 即 Dr.GRPO |
| `grpo_vectorized` | `:334` | 同上，scatter 向量化 |
| `grpo_passk` | `:471` | 组内只给 top-1 非零 advantage（`r_max - r_second`） |
| `reinforce_plus_plus` | `:693` | 折扣回报 + 全 batch 白化 |
| `reinforce_plus_plus_baseline` | `:533` | 减组均值 + 全 batch 白化 |
| `rloo` | `:587` | leave-one-out：`R*n/(n-1) - mean*n/(n-1)` |
| `remax` | `:734` | 减 greedy 解码的 baseline（trainer 额外发一次 greedy rollout） |
| `opo` | `:639` | 长度加权的最优 baseline |
| `gpg` | `:770` | `alpha * (R - mean)`，alpha = bsz / 非零 reward 数 |
| `gdpo` | `:361` | 多维 reward 各自 GRPO 归一化后加权 |
| `optimal_token_baseline` | `:871` | 逐 token 的方差最优 baseline，需要 `sum_pi_squared` |

GRPO 的核心几行（`:324-329`）：

```python
for i in range(bsz):
    if norm_adv_by_std_in_grpo:
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    else:
        scores[i] = scores[i] - id2mean[index[i]]
scores = scores.unsqueeze(-1) * response_mask       # 标量广播到每个 token
```

GAE（`:250-262`）注意 `response_mask` 为 0 的 observation token（多轮的 tool 输出）会被跳过，`nextvalues`/`lastgaelam` 直接透传。

driver 侧分发：`ray_trainer.compute_advantage`（`:187-282`）；v1 再包一层 `compute_advantage_for_multi_trajectories`（`v1/utils.py:148`），GRPO 时只对每个 session 的最终输出算分再广播。

#### 4.6.2 policy losses（`actor.policy_loss.loss_mode`）

统一签名 `(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, config, rollout_is_weights) -> (loss, metrics)`：

| 名字 | 行 | 要点 |
|---|---|---|
| `vanilla` | `:1285` | 标准 PPO + 非对称 clip + dual clip（`clip_ratio_c`） |
| `gspo` | `:1545` | 序列级 ratio `exp(mean_t log r_t)`，用 `log_prob - log_prob.detach() + seq.detach()` 技巧让梯度流经每个 token（`:1583-1597`） |
| `cispo` | `:2047` | `-sg(clip(ratio)) * A * log_prob` |
| `sapo` | `:1622` | 用 sigmoid 门控替代硬 clip，正负 advantage 用不同 tau |
| `gpg` | `:1707` | 纯 REINFORCE `-log_prob * A` |
| `clip_cov` | `:1743` | 随机屏蔽一小部分高协方差 token 的梯度（熵机制论文） |
| `kl_cov` | `:1848` | 只对高协方差 token 加 KL 惩罚，无 clip |
| `geo_mean` | `:1928` | GMPO，log 空间 clip 后取几何平均；**绕过 `agg_loss`** |
| `dro` | `:2014` | `-(log_prob * A - 0.5 * beta * log_ratio^2)` |
| `dppo_tv` / `dppo_kl` | `:1379` / `:1460` | 用有效性 mask × 截断 IS 替代 clip |
| `bypass_mode` | `:2413` | rollout correction bypass 模式的分发器（§4.8） |

`vanilla` 核心（`:1336-1358`）和 slime 的 `compute_policy_loss` 几乎逐行对应，多了 `negative_approx_kl.clamp(-20, 20)` 和最后 `pg_losses * rollout_is_weights`。

clip 参数在 `ActorConfig`（`workers/config/actor.py:158-163`），默认 `0.2 / 0.2 / 0.2 / 3.0`；yaml 在 `trainer/config/actor/actor.yaml:36-42, 83`。

#### 4.6.3 `agg_loss` 与全局归一化

`core_algos.py:1140-1206`。五种 `loss_agg_mode`：`token-mean` / `token-sum` / `seq-mean-token-sum` / `seq-mean-token-mean` / `seq-mean-token-sum-norm`。关键是它接收全局量：

```python
if loss_agg_mode == "token-mean":
    loss = masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size     # 分母是整个 mini-batch 跨 DP 的 token 数
elif loss_agg_mode == "seq-mean-token-mean":
    seq_losses = sum(loss_mat * mask, -1) / (seq_len + 1e-8)
    loss = masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size   # 分母是整个 mini-batch 的序列数
```

链路：`batch_num_tokens` 在每个 mini-batch 开头跨 DP all-reduce（`engine/fsdp/transformer_impl.py:711-716`）；`global_batch_size = ppo_mini_batch_size * rollout.n` 由 driver 塞进 `extra_info`（`trainer_base.py:1758`）；`ppo_loss` 把它们打包进 `config.global_batch_info`（`losses.py:65-68`），每个 policy loss 都 `agg_loss(..., **config.global_batch_info)`。这样**每个 micro-batch 都除以全局分母，梯度累加后天然等于全局均值**，不需要再除 micro-batch 数（Megatron 例外，`engine/megatron/transformer_impl.py:1428-1430` 乘回 `num_micro_batch`）。

#### 4.6.4 loss 拼装：`ppo_loss`

`framework/verl/verl/workers/utils/losses.py:57-144`，verl 版的 `policy_loss_function`：

```python
policy_loss_fn = get_policy_loss_fn(config.policy_loss.loss_mode)
pg_loss, pg_metrics = policy_loss_fn(old_log_prob=..., log_prob=..., advantages=..., response_mask=..., ...)
policy_loss = pg_loss
if entropy is not None:
    policy_loss -= config.entropy_coeff * agg_loss(entropy, ...)          # :123-129
if config.use_kl_loss:
    kld = kl_penalty(log_prob, ref_log_prob, kl_penalty=config.kl_loss_type)
    policy_loss += agg_loss(kld, ...) * config.kl_loss_coef               # :132-141
```

它在 `ActorRolloutRefWorker.init_model` 里被 `partial(ppo_loss, config=actor_config)` 绑定为 engine 的 `loss_fn`（`engine_workers.py:638-645`）。

#### 4.6.5 KL

`kl_penalty`（`core_algos.py:2188`）→ `kl_penalty_forward`（`:2216-2251`）：`kl/k1`、`abs`、`mse/k2`、`low_var_kl/k3`。带 `+` 后缀（如 `k3+`）是 straight-through：前向值用 k3，梯度用 k2（`:2200-2213`）。

- KL in reward：`apply_kl_penalty`（`ray_trainer.py:78-117`），`token_level_rewards = token_level_scores - beta * kld`，`beta` 来自 `FixedKLController` / `AdaptiveKLController`（`core_algos.py:153-210`）。v1 调用点 `trainer_base.py:1661-1667`。
- KL in loss：上面的 `losses.py:132-141`，默认 `kl_loss_type=low_var_kl`，`kl_loss_coef=0.001`。

#### 4.6.6 critic

`compute_value_loss`（`core_algos.py:2125-2185`），`vpredclipped = clip(vpreds, values ± cliprange_value)`，`0.5 * max((vpreds-returns)^2, (vpredclipped-returns)^2)`，默认 `cliprange_value=0.5`。包装在 `losses.py:147` `value_loss`。critic 是另一个 `TrainingWorker`（value head 模型，`engine/fsdp/transformer_impl.py:1573`）。

### 4.7 actor 更新：三层循环

```
driver  _update_actor                 trainer_base.py:1734-1773   extra_info = {mini_batch_size = ppo_mini_batch_size * n, epochs = ppo_epochs, shuffle, ...}
  └─ worker  train_mini_batch         engine_workers.py:242-338   for epoch: for mini_batch: train_batch(mini_batch)   ← 每个 mini-batch 一次 optimizer.step
       └─ engine  train_batch         engine/base.py:113-132      zero_grad → forward_backward_batch → optimizer_step
            └─ forward_backward_batch transformer_impl.py:705-758  for micro_batch: loss.backward()（最后一个 micro-batch 才 sync 梯度）
```

- micro-batch 切分：`engine/utils.py:92` `prepare_micro_batches`，`use_dynamic_bsz` 时按 `ppo_max_token_len_per_gpu` 打包。
- 梯度裁剪与 NaN 跳过：`transformer_impl.py:769-815`。
- **old_log_prob 的来源**：默认（decoupled）每个 data batch 单独前向一次算 `old_log_probs`（`trainer_base.py:1541-1600`），`ppo_epochs` × 多个 mini-batch 共享它；`rollout_correction.bypass_mode=True` 时直接 `old_log_probs = rollout_log_probs` 省掉这次前向（`:1548-1554`）。verl **没有** slime 那种"单 step 就复用"的捷径。

### 4.8 Rollout correction（TIS / IcePop / 拒绝采样 / bypass）

`framework/verl/verl/trainer/ppo/rollout_corr_helper.py`，配置 `algorithm.rollout_correction.*`（`trainer/config/algorithm.py:63`，yaml `config/algorithm/rollout_correction.yaml`，还有 `decoupled_token_is` / `bypass_ppo_clip` 等约 20 个预设 `:186-616`）。

- IS 权重 `compute_rollout_correction_weights`（`:522`）：`log_ratio = old_log_prob - rollout_log_prob`，`rollout_is=token|sequence`，阈值字符串不含 `_` 是 TIS（clamp 上限），含 `_` 是 IcePop（区间外置零）（`:592-604`）。权重 `.detach()`，最后乘进 policy loss（`rollout_is_weights` 参数）。
- 拒绝采样 `compute_rollout_rejection_mask`（`:197`）：`{token,seq_sum,seq_mean,seq_max}_k{1,2,3}` 组合，直接改 `response_mask`（`:1060-1061`）。
- `bypass_mode`（`:1109`）：不重算 old_log_prob，强制 `loss_mode=bypass_mode`，在 loss 里对着不断变化的 π_θ 现算 IS（`core_algos.py:2413-2541`）；`loss_type=ppo_clip` 时故意不再乘 IS 权重，因为 PPO ratio 本身就是 π_θ/π_rollout。

### 4.9 权重同步

driver：`CheckpointEngineManager.update_weights`（`checkpoint_engine/base.py:506-558`）。`backend=naive`（hybrid 默认，`trainer_base.py:359` 强制）时直接 `actor_wg.update_weights(...)`；其他 backend（`nccl`/`nixl`/`mooncake`……）先 abort 未完成请求、释放 KV cache、建进程组再发。

worker：`ActorRolloutRefWorker.update_weights`（`engine_workers.py:727-820`）：`rollout.resume(tags=["weights"])` → `actor.engine.get_per_tensor_param()`（FSDP 在 `transformer_impl.py:959-1010`）→ `rollout.update_weights(per_tensor_param)`（vLLM 侧 `workers/rollout/vllm_rollout/vllm_rollout.py:209-247`，CUDA IPC）→ 参数 offload 回 CPU → `rollout.resume(tags=["kv_cache"])`。

反方向（训练前释放推理显存）是 `on_sample_end` → `sleep_replicas`（`trainer_sync.py:40-42`）。engine 层还有按阶段自动 offload 的 `_context_switch`（`engine/base.py:314-328`）。

### 4.10 异步模式与 replay buffer

- `ReplayBuffer`（`v1/replay_buffer.py:63`）：sync 版 `sample`（`:405`）实际是无缓冲，等所有在飞请求完成；DAPO 的 filter_groups 在这里做（组内 reward std 为 0 则驱逐并补发，`:293-298, 452-469`）。
- `ReplayBufferAsync`（`:497`）：按 `max_off_policy_threshold` 丢弃或等待过期的组（`:503-539`）。
- `PPOTrainerColocateAsync`（`v1/trainer_colocate_async.py:26`）：同卡，靠 `num_warmup_batches` 预热 + abort/resume 保留 partial rollout。
- `PPOTrainerSeparateAsync`（`v1/trainer_separate_async.py:43`）：训练和推理分卡，hybrid 卡在 rollout/trainer 间自适应切换（`:261-360`）；`parameter_sync_step` 内多个 mini-batch 共享一份 CPU 快照的 π_old（`:154-178`，Decoupled PPO）。

---

## 5. 两个框架对照：值得琢磨的设计差异

| 问题 | slime | verl |
|---|---|---|
| GRPO 组归一化在哪 | rollout 侧 `_post_process_rewards`，训练侧只广播 | driver 侧 `compute_grpo_outcome_advantage`，按 `uid` 分组 |
| old_log_prob 什么时候算 | 整轮一次；单 step + 无 KL 时**直接复用训练前向**（ratio ≡ 1） | 每个 data batch 一次独立前向；bypass 模式用 rollout log-prob |
| 一轮几个 optimizer step | `--num-steps-per-rollout`，每个 step 一次 `forward_backward_func` | `ppo_epochs` × (`train_batch_size * n / ppo_mini_batch_size`) 个 mini-batch |
| loss 分母 | 按 rollout 组的 token 总数（`rollout_mask_sums`），末尾乘 `num_microbatches/step_global_batch_size*dp` | `agg_loss` 直接用全局 `batch_num_tokens`/`global_batch_size` |
| 变长序列 | 不 padding，`cat` 成 packed `[1,T]` + `cu_seqlens`，list[Tensor] 到处传 | nested tensor 内部流转，loss 前 pad 成 `(bs, L)`；FSDP 用 `use_remove_padding` + flash-attn varlen |
| KL 位置 | `--kl-coef`（进 reward）与 `--kl-loss-coef`（进 loss）二选一 | `use_kl_in_reward` 与 `use_kl_loss` 可同时 |
| 算法变体 | 6 种 advantage、3 种 loss 形态（PPO/GSPO/CISPO），扩展靠 `--custom-*-path` | 14 种 advantage、12 种 policy loss，注册表 + hydra |
| ref 模型 | 同一 Megatron 实例换权重（CPU 快照） | 独立 `TrainingWorker`（或 LoRA 时同权重关 adapter） |
| 权重同步 | Megatron→HF 转换 + IPC/NCCL/disk | `get_per_tensor_param` + `checkpoint_engine` 多后端 |
| 并行策略 | Megatron TP/PP/CP/EP 全套，CP 用 zigzag 切 log-prob | 由 engine 决定：FSDP + Ulysses SP，或 Megatron |
| 异步 | `train_async.py` 一步流水线 / fully async rollout 线程 | replay buffer + 三种 trainer_mode |

---

## 6. 按算法主题的阅读路径

**A. 标准 PPO（带 critic、GAE、KL in reward）**
1. verl `core_algos.py:215-263`（GAE）→ `:2125-2185`（value loss）→ `ray_trainer.py:78-117`（KL in reward）→ `:1285-1376`（clip loss）
2. slime `ppo_utils.py:693-713`（chunked GAE）→ `loss.py:769-781`（token reward 构造）→ `loss.py:1175-1229`（value loss）→ `train.py:59-67`（critic/actor 编排）

**B. GRPO（无 critic、组归一化、KL in loss）**
1. slime `slime/ray/rollout.py:279-303`（归一化）→ `loss.py:763-767`（广播）→ `loss.py:933-1172`（loss 全流程）→ `actor.py:428-442`（何时复用 log-prob）
2. verl `core_algos.py:267-331` → `losses.py:57-144` → `core_algos.py:1140-1206`（agg_loss）

**C. GSPO / CISPO（改 ratio 或改梯度路径）**
- slime `ppo_utils.py:95-121`、`:151-171`；verl `core_algos.py:1545-1620`、`:2047-2105`。对比两边 GSPO 让梯度流过每个 token 的写法。

**D. 训推不一致（TIS / MIS / 拒绝采样）**
- slime `loss.py:883-930` + `examples/train_infer_mismatch_helper/mis.py`；verl `rollout_corr_helper.py:262-265, 522-604, 786-878`。

**E. 从 log-prob 到 entropy 的数值实现（大 vocab 省显存）**
- slime `ppo_utils.py:187-336`（自定义 autograd）；verl `utils/torch_functional.py:73`（`logprobs_from_logits`）+ `:231-260`（entropy）+ `utils/kernel/linear_cross_entropy.py`（fused kernel）。

**F. 权重同步与显存腾挪**
- slime `update_weight/update_weight_from_tensor.py:278`、`actor.py:174-212`；verl `engine_workers.py:727-820`、`checkpoint_engine/base.py:506-558`。

**练习建议**：分别在两个框架里加一个新的 advantage estimator（比如 RLOO 加到 slime，或把 slime 的 OPSM 加到 verl），走一遍各自的扩展机制，比读十遍代码更能理解数据流。slime 改 `loss.py:758-803` 的分发 + `arguments.py:952` 的 choices；verl 只要在 `core_algos.py` 里 `@register_adv_est("xxx")` 并在 `AdvantageEstimator` 枚举加一项。
