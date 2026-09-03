# SFT / OPD / GRPO / PPO 的显存构成对比

> 前置：[显存账本](../03-training-fundamentals/01-memory-accounting.md)、[DDP/ZeRO](../04-distributed-infra/01-ddp-and-zero.md)。
> 这一篇只回答一个问题：**每个算法相对 SFT 多了什么。**

## 1. 统一拆成 6 类

$$\boxed{M=M_{\text{param}}+M_{\text{grad}}+M_{\text{optim}}+M_{\text{activation}}+M_{\text{KV/cache}}+M_{\text{extra models}}}$$

真正拉开差距的是**最后两项**。

| 显存来源 | SFT | OPD | GRPO | PPO |
|---|---|---|---|---|
| Student/Actor 参数 | ✅ | ✅ | ✅ | ✅ |
| Gradient | ✅ | ✅ | ✅ | ✅ |
| Adam optimizer state | ✅ | ✅ | ✅ | ✅ |
| Training activation | ✅ | ✅ | ✅ | ✅ |
| Teacher / Ref 参数 | ❌ | ✅ Teacher | ✅ Ref | ✅ Ref |
| Teacher activation | ❌ | 少量 | ❌ | ❌ |
| **Rollout KV cache** | ❌ | on-policy 时 ✅ | ✅ 很大 | ✅ 很大 |
| Critic（训练态） | ❌ | ❌ | ❌ | ✅ |
| Reward Model | ❌ | ❌ | 可选 | 通常 ✅ |
| old logprob | ❌ | ❌ | $[B,L]$ | $[B,L]$ |

$$\boxed{\text{SFT}<\text{OPD}<\text{GRPO}<\text{PPO}}$$

## 2. SFT

$$M_{\text{SFT}}=P_S+G_S+O_S+A_S$$

7B BF16 + AdamW：$14+14+56=84$ GB（+ activation）。8 卡 ZeRO-3 后 model states 约 10.5 GB/卡。

## 3. OPD = SFT + frozen teacher（+ rollout）

### teacher 便宜在哪

teacher 不训练，所以只有 $P_T$，**没有** $G_T$（14 GB）和 $O_T$（56 GB）。7B teacher 裸权重约 14 GB。

即使 teacher 是 32B：BF16 权重 64 GB，而如果它也训练则要约 384 GB。

teacher 的 forward 在 `torch.no_grad()` 下，不建 backward graph，用完即释放：

$$\boxed{A_T^{\text{inference}}\ll A_S^{\text{training}}}$$

### teacher 要 KV cache 吗

**不要**（就 Thinking Machines 那种 recipe 而言）。teacher 是对 student 已经生成好的 trajectory 做一次 **teacher-forcing forward** 拿 logprob，不是 autoregressive 生成。这和 RL 的 rollout engine 完全不同。

### 卡点：supervision tensor 的 $[B,L]$ vs $[B,L,V]$

reverse-KL sampled-token：teacher 只返回 student 采样 token 的一个标量 logprob。$B=8,L=8192$、FP32：

$$8\times8192\times4\approx0.26\text{ MB}$$

精确 full forward KL 需要整个词表分布，$V=150\text{K}$、BF16：

$$8\times8192\times150000\times2\approx19.7\text{ GB}$$

$$\boxed{[B,L]\quad vs\quad[B,L,V]}$$

这是蒸馏显存里最关键的一个区别，也是 reverse-KL OPD 工程友好的重要原因（见 [distill/05](distill/05-kl-estimation.md)）。

实际做 forward KL 时也不会傻存完整 logits：top-K 蒸馏（$[B,L,K]$）、chunked loss（算完即释放）、从 teacher 采样近似。

## 4. 卡点：逻辑组件 ≠ 物理峰值

**不能**直接写 $M_{\text{OPD}}=M_{\text{train}}+M_{\text{rollout}}+M_{\text{teacher}}$ 然后说它们一定同时占显存。

on-policy OPD 一个 step 是分阶段的：

```text
Phase 1  Rollout         : P_S^infer + KV_S
Phase 2  Teacher scoring : P_T + A_T^infer
Phase 3  Student training: P_S + G_S + O_S + A_S
```

如果 infra 做得好（rollout engine sleep、free KV cache、offload/gather weights、分时复用 GPU）：

$$\boxed{M^{\text{peak}}=\max(M_{\text{rollout}},\ M_{\text{teacher}},\ M_{\text{training}})}$$

而不是三者相加。这就是 verl 里 `engine sleep` 的思路。

反过来，如果部署成三个独立 engine 占不同 GPU group，那三者相加说的是**集群总资源**，不是**单卡峰值**。这个区别在分析 RL 时同样重要。

$$\boxed{\begin{aligned}M_{\text{train}}&=P_S+G_S+O_S+A_S\\ M_{\text{rollout}}&=P_S^{\text{infer}}+KV_S\\ M_{\text{teacher}}&=P_T+A_T^{\text{infer}}\\ M_{\text{supervision}}&\approx[B,L]\end{aligned}}$$

## 5. RL 为什么突然特别吃显存：rollout

RL 相比 SFT/OPD 最大的区别是**需要 rollout**，也就是要一个 inference engine（vLLM/SGLang）和它的 **KV cache**：

$$M_{KV}\propto B_{\text{rollout}}\times L\times N_{\text{layer}}\times d_{KV}$$

long CoT、大 rollout batch、GRPO 每题采 $G=8/16/32$、32K/64K context —— KV cache 会非常恐怖。这就是"模型只有几 B，rollout engine 却吃掉几十 GB"的原因。

### GRPO

$$M_{\text{GRPO}}\approx\underbrace{M_{\text{actor}}^{\text{train}}}_{\text{类似 SFT}}+\underbrace{M_{\text{ref}}}_{\text{frozen}}+\underbrace{M_{\text{rollout}}}_{\text{vLLM/KV}}$$

规则 reward（math verifier / code execution / 环境 reward）时不需要 reward model。相比 PPO 最大的优势是没有 critic。

### PPO

四个模型：**Actor（训练）+ Critic（训练）+ Reference（frozen）+ Reward Model（frozen）**，再加 rollout。critic 是完整的训练态模型（$P+G+O+A$），所以特别重。

## 6. 卡点：old policy 占不占一个模型

$$r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$$

看起来需要一个 $\pi_{\text{old}}$ 模型，但工程上**不一定常驻第二份权重**：rollout 时直接把 $\log\pi_{\text{old}}(a_t|s_t)$ 存下来，只有 $[B,L]$；训练时

$$r_t=\exp(\log\pi_\theta-\log\pi_{\text{old}})$$

$$\boxed{\text{逻辑上有 old policy，物理上不一定有第二份 actor 权重}}$$

别把 GRPO 说成 "actor + old actor + ref 三个完整模型"。

## 7. 一句话总结

> **SFT 的显存核心是单模型训练状态（参数、梯度、optimizer state、activation）；OPD 在此基础上额外常驻一个 frozen teacher，但 teacher 没有梯度和 optimizer；LLM RL 还必须承担 rollout 的 inference/KV cache，其中 GRPO 通常是 actor + ref + rollout，PPO 还要一个训练的 critic 和 frozen reward model，因此显存通常 PPO > GRPO > OPD > SFT。**

$$\boxed{\text{SFT/OPD 的瓶颈更偏 training memory}}\qquad\boxed{\text{RL 同时有 training memory + rollout memory}}$$

这就是为什么 verl 里 FSDP2、vLLM、engine sleep、offload、resharding 会同时出现 —— 它们分别在解决这两套生命周期完全不同的显存。

从 infra 结构看 OPD 已经很像 RL（都是 `rollout → score → train`），区别主要在 scoring：**OPD 给的是每 token 的 dense logprob，RL 给的往往是 sequence-level reward。**

## 自测（口述版）

1. 把训练显存拆成 6 类，指出哪两类拉开了 SFT/OPD/RL 的差距。
2. teacher 相比一个"训练中的第二个模型"便宜在哪？7B 和 32B 各差多少？
3. teacher 需要 KV cache 吗？为什么？
4. sampled-token 和 full-vocab 的 supervision tensor 各多大？用 $B=8,L=8192,V=150$K 算。
5. 为什么不能把 rollout / teacher / training 的显存直接相加？峰值应该怎么写？
6. GRPO 和 PPO 各要几个模型？critic 为什么特别重？
7. $\pi_{\text{old}}$ 要不要常驻一份权重？工程上怎么做？

> 带答案的题库在 [显存 / 训练工程 / 分布式 自测](../03-training-fundamentals/self-test.md)。
