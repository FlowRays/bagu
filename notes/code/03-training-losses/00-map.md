# 训练 loss（代码）

> 对应 [bagu.md](../../../bagu.md) 的 code (3) training losses。
> **实现都在 [手撕代码合集](../07-handwrite/00-map.md) 里**，这一页只做索引。

| Loss | 可默写实现 | 理论 |
|---|---|---|
| Softmax / 交叉熵 / BCE | [手撕：深度学习](../07-handwrite/02-dl.md) | [熵、交叉熵、KL、JSD](../../06-post-training/distill/01-entropy-ce-kl-jsd.md) |
| KL 散度 | [手撕：深度学习](../07-handwrite/02-dl.md#kl-散度) | [forward vs reverse KL](../../06-post-training/rl/07-kl.md) |
| SFT loss（shift right） | [手撕：大模型](../07-handwrite/03-llm.md#sft-损失函数shift-right) | [SFT 工程](../../06-post-training/sft/01-sft-basics.md) |
| InfoNCE | [手撕：大模型](../07-handwrite/03-llm.md#infonce-对比学习损失) | [CLIP 与 SigLIP](../../07-vlm/02-vision-encoder.md) |
| DPO | [手撕：大模型](../07-handwrite/03-llm.md#dpo-损失函数) | [DPO 完整推导](../../06-post-training/preference/02-dpo.md) |
| PPO | [手撕：大模型](../07-handwrite/03-llm.md#ppo-损失函数) | [clip 与 min](../../06-post-training/rl/03-clip-and-min.md) |
| GRPO | [手撕：大模型](../07-handwrite/03-llm.md#grpo-损失函数) | [GRPO](../../06-post-training/rl/06-grpo.md) |
| IoU / MSE / RQ-VAE | [手撕](../07-handwrite/02-dl.md) | — |

## 写 loss 时的通用铁律

1. **减最大值再 exp**：softmax、log_softmax、交叉熵、InfoNCE 全靠它防溢出。
2. **别写 `log(softmax(x))`**，用 `log_softmax`；别写 `log(sigmoid(x))`，用 `logsigmoid`。
3. **交叉熵对 logits 的梯度就是 $p-q$**，不用再乘激活函数的导数。
4. **ratio 用 `exp(logp - logp_old)`**，别直接除概率。
5. **归一化按总有效 token 数**，不是按样本平均（见 [梯度累积](../../03-training-fundamentals/03-gradient-accumulation.md#4-卡点真正的坑是-token-normalization)）。
