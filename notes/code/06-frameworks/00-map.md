# RL 框架源码导读（slime / verl）

> 对应 [bagu.md](../../../bagu.md) 的 code (6) framework familiarity。
> 代码版本：`slime` commit `3778dbf`、`verl` commit `24f25b0`（均 2026-08-28），仓库在 `learn/rl/framework/`（不入 git）。

| 篇 | 内容 |
|---|---|
| [01 整体导读](01-rl-framework-source-reading.md) | 两个框架的架构、一次迭代的六个阶段、数据流、权重同步、各自的可插拔点 |
| [02 公式 → 代码逐条对照](02-formula-to-code.md) | 把笔记里的每个公式对到确切的函数和行号：ratio / clip+min / Clip-Higher / dual-clip / GAE / GRPO 组内归一化 / loss 聚合 / GSPO / KL 估计器 / OPD / CISPO（10 题自测） |

## 一句话对比

$$\boxed{\text{verl 广度取胜（算法变体最全，注册表 + hydra），slime 深度取胜（主线最短，最好读）}}$$

| | verl | slime |
|---|---|---|
| 算法都在哪 | `verl/trainer/ppo/core_algos.py`（2549 行，**一个文件装下全部**） | `slime/utils/ppo_utils.py` (767) + `backends/megatron_utils/loss.py` (1382) |
| 训练后端 | FSDP / FSDP2 / Megatron / veOmni / TorchTitan | 只有 Megatron-LM |
| 推理后端 | vLLM / SGLang / TRT-LLM | 只有 SGLang |
| 扩展方式 | `@register_adv_est` / `@register_policy_loss` 注册表 | `--custom-*-path` dotted path 回调 |
| GRPO 归一化 | trainer 侧 | **rollout 侧** |
| 有 OPD 吗 | 没有 | **有**（`--use-opd`） |

## 读之前先建立的三个预期

1. **公式里的 $\min$，代码里是 `torch.maximum`** —— 因为代码算的是已取负的 loss。
2. **两个框架的 ratio 中间量符号相反** —— verl 的 `negative_approx_kl` 是正号，slime 的 `ppo_kl` 是反号。
3. **同一个算法可能落在完全不同的阶段** —— GRPO 的组内归一化 slime 在 rollout 做、verl 在 trainer 做。

## 相关

- [RL 目录](../../06-post-training/rl/00-map.md) — 这些公式的推导
- [蒸馏目录](../../06-post-training/distill/00-map.md) — OPD 的推导，slime 里就一行代码
- [SFT/OPD/RL 显存对比](../../06-post-training/memory-sft-opd-rl.md) — 为什么 verl 里会同时出现 FSDP2、vLLM、engine sleep、offload
