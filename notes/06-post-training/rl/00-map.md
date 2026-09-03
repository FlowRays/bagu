# RL 算法主线：PPO → GRPO → DAPO → GSPO（总图 + 卡点索引）

> 本目录把 PPO 从最原始的 policy gradient 一路推到 GSPO，重点标注**推导链条上最容易卡住的地方**。

## 一句话演化线

> **PPO**：critic 估 advantage；**GRPO**：用同题多条 rollout 的相对 reward 代替 critic；**DAPO**：把 GRPO 修到能大规模 reasoning RL；**GSPO**：把 token-level ratio 改成 sequence-level ratio。

```text
Policy Gradient
      │
      ▼
     PPO
      │  critic 很贵；LLM 可以同 prompt 多采样
      ▼
    GRPO
      │  scaling 暴露：clip 太保守 / 无效 prompt / 长度偏置 / 超长截断
      ▼
    DAPO
      │  token-level ratio 本身合理吗？
      ▼
    GSPO
```

四者只改三样东西：**A 怎么来、ratio 怎么算、loss 怎么聚合。**

| 方法 | Advantage 从哪来 | Critic | Ratio/Clip 粒度 | 核心 |
|---|---|---|---|---|
| PPO | $R-V$ / GAE | 有 | token/timestep | 稳定的 policy gradient |
| GRPO | group-relative reward | 无 | token | 同 prompt 多 rollout 当 baseline |
| DAPO | group-relative reward | 无 | token | 把 GRPO 的 scaling 问题逐个修好 |
| GSPO | group-relative reward | 无 | **sequence** | sequence-level policy optimization |

## 必背的五个公式

$$A = Q - V$$

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$\hat A_t^{GAE} = \sum_{l\ge 0} (\gamma\lambda)^l \delta_{t+l}$$

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$$

$$L^{PPO} = \mathbb E\Big[\min\big(r_tA_t,\ \mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t\big)\Big]$$

一句话直觉：**critic 告诉 actor 这次 action 比预期好还是差；actor 按 advantage 调概率；clip 保证每批旧数据只让 policy 走一小步。**

## 卡点索引（推导链条上最容易卡住的 19 处）

| # | 卡点 | 在哪 |
|---|---|---|
| 1 | J、E、Loss、gradient 混在一起分不清 | [01](01-from-J-to-loss.md#卡点-1核心j--e--l--l-到底是什么关系) |
| 2 | 为什么 $J \ne \mathbb E[A\log\pi]$ | [01](01-from-J-to-loss.md#3-关键澄清pg-定理给的是-j不是另一个-j) |
| 3 | 为什么把 $A\log\pi$ 换成 $Ar$ | [02](02-importance-sampling-and-ratio.md#卡点-3a-log-pi-换成-a-r-不是代数替换) |
| 4 | 为什么是 $\mathbb E_{old}[rA]$ 而不是 $\mathbb E_{old}[rA\log\pi]$ | [02](02-importance-sampling-and-ratio.md#卡点-4最关键为什么-surrogate-里没有-log-pi) |
| 5 | 为什么要 clip | [03](03-clip-and-min.md#1-为什么要-clip) |
| 6 | 加了 clip 之后 L → gradient → θ 的关系 | [03](03-clip-and-min.md#卡点-5clip-到底怎么让参数停止更新) |
| 7 | 为什么必须有 min，不能只写 clip(r)A | [03](03-clip-and-min.md#3-为什么必须有-min) |
| 8 | GAE 是什么，λ=1 为什么退化成 MC | [04](04-advantage-critic-gae.md#5-gae1-的完整推导telescoping) |
| 9 | 偏差和方差到底指什么 | [04](04-advantage-critic-gae.md#卡点-8bias-和-variance-到底在说什么) |
| 10 | critic 讲得太含糊：学什么、target 哪来 | [04](04-advantage-critic-gae.md#3-critic-到底学什么) |
| 11 | 为什么用 $\hat R$，$V_{old}$ 又是什么 | [04](04-advantage-critic-gae.md#卡点-10v_old-是什么为什么-target-叫-r-hat) |
| 12 | 为什么 $r=\exp(\log\pi_\theta-\log\pi_{old})$ | [05](05-ppo-engineering.md#卡点-11为什么用-exp对数差-算-ratio) |
| 13 | 为什么除标准差不是方差，一般 norm 怎么做 | [06](06-grpo.md#卡点-12为什么除标准差不是方差) |
| 14 | 为什么 seq-level loss 可以"替代" token-level | [06](06-grpo.md#卡点-13为什么-sequence-level-的-a-能乘到每个-token-上) |
| 15 | 为什么必须同 prompt 分组，不能整个 batch norm | [06](06-grpo.md#5-为什么必须同-prompt-分组) |
| 16 | GRPO 是 forward 还是 backward KL；KL 为什么 ≥0；怎么记住 | [07](07-kl.md) |
| 17 | A=0 是不是训练和不训练没区别 | [08](08-dapo.md#卡点-17a0-等于没训练吗) |
| 18 | 一个 batch 内的 loss 到底怎么聚合 | [08](08-dapo.md#卡点-18一个-batch-内的-loss-怎么算) |
| 19 | 原始 PPO 的 KL penalty 罚的是 $\pi_{old}$，而且是 forward | [07](07-kl.md#卡点原始-ppo-的-kl-penalty-罚的是-pi_old方向是-forward) |

## 阅读顺序

1. [01 从 J 到 loss](01-from-J-to-loss.md) — 概念地基，卡点最集中，**必须先通这一篇**
2. [02 importance sampling 与 ratio](02-importance-sampling-and-ratio.md) — PPO 的第一层
3. [03 clip 与 min](03-clip-and-min.md) — PPO 的第二层，四象限
4. [04 advantage / critic / GAE](04-advantage-critic-gae.md) — A 从哪来
5. [05 PPO 工程细节](05-ppo-engineering.md) — 完整 loss、训练循环、超参
6. [06 GRPO](06-grpo.md) — 高频考点
7. [07 KL 散度](07-kl.md) — 高频考点
8. [08 DAPO](08-dapo.md)
9. [09 GSPO](09-gspo.md)
10. [自测题](self-test.md) — 关掉笔记口述 / 纸上推导

相关：

- [verl / slime 源码导读](../../code/06-frameworks/00-map.md) — 这些公式在真实框架里落在哪些文件
- [蒸馏 / OPD / OPSD](../distill/00-map.md) — reverse KL 为什么**就是** policy gradient，把 RL 和蒸馏接起来
