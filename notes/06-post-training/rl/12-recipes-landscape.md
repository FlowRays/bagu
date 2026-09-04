# 各家 RL recipe 横向：Qwen / GLM / Kimi 分别用什么

> ⚠️ **这一篇的性质和别的笔记不一样**：它有时效性，而且大量内容是**推测**。
> 记录时间：2026-09。面试里问"XX 模型用什么 RL 算法"时，**把"官方明确披露"和"根据技术演进推测"分开说**，比背一个名字得分高得多。

## 0. 最重要的一条：不能再用一个名字概括一家

$$\boxed{\text{"Qwen = GSPO，GLM = GRPO，Kimi = PPO"}\quad\text{这种记法已经过时了}}$$

各家现在都进入了「**自定义 policy-gradient recipe + 大规模异步 agent RL**」的阶段，而且**越新的模型越不公开 objective**。

| 模型 | 公开的 RL 算法 / recipe | 确定程度 |
|---|---|---|
| Qwen2.5 | SFT + **DPO**（offline）+ **GRPO**（online） | ★★★★★ 官方 |
| Qwen3 首发版 | **GRPO** | ★★★★★ 官方 |
| Qwen3-2507 / Coder / Thinking | **GSPO** | ★★★★★ 官方 |
| Qwen3.5 | 只说大规模异步 RL + agent environment scaling，**没点名 objective** | ⚠️ 未公开 |
| Qwen3.6 / 3.7 / 3.8 | 同样**没公开 policy optimizer** | ⚠️ 未公开 |
| GLM-5.x | **GRPO + IcePop** + async agent RL | ✅ 系列 recipe 明确 |
| Kimi K2.5 / K3 | **K2.5-style group-relative PG + token-level mask + $(\log\rho)^2$** + partial rollout + MOPD | ✅ 明确 |

## 1. Qwen 这条线：GRPO → GSPO → SAPO

### Qwen2.5

$$\text{SFT}\rightarrow\text{Offline RL}\rightarrow\text{Online RL}$$

Offline RL 实际上是 **DPO**（$(y^+,y^-)$ 对），Online RL 明确写的是 **GRPO**，每个 query 采 8 条 response 用 reward model 打分。

$$\boxed{\text{Qwen2.5}=\text{SFT}+\text{DPO}+\text{GRPO}}$$

### Qwen3 首发版：⚠️ 还是 GRPO，不是 GSPO

**这是最容易记错的一处。** Qwen3 Technical Report（2025-05）的 Reasoning RL 明确写的是 **GRPO**，3995 个 query-verifier pair、大 batch、单 query 大量 rollout、有 off-policy training、控制 entropy、rule-based verifier reward。

四阶段 pipeline：

```text
1. Long CoT Cold Start
2. Reasoning RL          ← GRPO
3. Thinking / Non-Thinking Fusion
4. General RL
```

### 2025-07-27 GSPO 发布，才进入后续 Qwen3

官方明确说 GSPO 已应用于**当时最新的** Qwen3 models（Instruct / Coder / Thinking）。所以必须区分：

$$\boxed{\text{Qwen3 2025-04/05 首发}=\text{GRPO}}\qquad\boxed{\text{Qwen3-2507 / Coder / Thinking}=\text{GSPO}}$$

$$\boxed{\text{Qwen2.5: GRPO}\rightarrow\text{Qwen3: GRPO}\rightarrow\text{Qwen3 后期: GSPO}}$$

### 2025-12-04 SAPO

见 [10-sapo](10-sapo.md)。它把 GRPO 和 GSPO 都当 baseline，用 smooth gating 代替 hard clipping。官方报告 SAPO > GSPO > GRPO（在 Qwen3-30B-A3B、Qwen3-VL 系列上）。

### Qwen3.5 之后：重心从 algorithm scaling 转向 environment scaling

Qwen3.5 官方说 post-training 的提升主要来自：

> scaling virtually all RL tasks and environments

也就是疯狂扩大 reasoning / coding / agent / multimodal / tool-use / multi-turn RL，并专门建了 $\boxed{\text{scalable asynchronous RL framework}}$，train 和 rollout 完全解耦。

$$\boxed{\text{RL algorithm scaling}\ \longrightarrow\ \text{RL environment scaling}}$$

**但官方没有写"we use GSPO"。** 所以 Qwen3.5 更该理解成"大规模异步 environment RL"，而不是"用了某个 optimizer"。

### 一个反证：不能说"Qwen3.5 = SAPO"

Qwen 在 2026-06 发布 Qwen-AgentWorld 时仍然明确写 **RL uses GSPO**（reward $=R_{\text{LLM judge}}+R_{\text{rule verifier}}$）。说明 SAPO 出来之后 Qwen 内部并没有把 GSPO 全废掉。更可能是：

$$\boxed{\text{不同任务用不同 RL optimizer}}$$

甚至 Qwen-VLA 在 2026 年做 simulation closed-loop RL 时仍然用 **PPO**。所以**不要认为一家公司有一个统一的 RL 算法**。

### ⚠️ Muon 不是 RL 算法

Qwen3.8-Flash-Next 技术报告详细披露了 pretraining 用 **Muon + AdamW**，但**没有公开 post-training 的 RL objective**。看到 Muon 不要答成 RL 算法：

| | 是什么 | 形式 |
|---|---|---|
| Muon | **parameter optimizer** | $\theta\leftarrow\theta-\eta\,\text{Muon}(\nabla L)$ |
| GSPO / SAPO | **policy optimization objective** | $L_{RL}(\theta)$ |

**完全不同的层面。**

### ⚠️ 第三方框架的 recipe ≠ 官方训练用的算法

有资料写"Qwen3.8 用 GRPO"，基本上是因为 NeMo-RL / ms-swift 这类框架**支持拿 Qwen3.8 做 GRPO**。那是**后训练 Qwen3.8 的开源 recipe，不是 Qwen 官方训练时用了 GRPO 的证据**。这个区别面试里很值钱。

### Qwen 时间线总图

$$\boxed{\text{DPO+GRPO}\overset{Qwen3}{\longrightarrow}\text{GRPO}\overset{2025.7}{\longrightarrow}\text{GSPO}\overset{2025.12}{\longrightarrow}\text{SAPO}\overset{2026}{\longrightarrow}\text{大规模 Async Agent RL}+\text{多种 optimizer}}$$

| Qwen2.5 | Qwen3 | Qwen3 late | Qwen3.5 | 3.6 | 3.7 | 3.8 |
|---|---|---|---|---|---|---|
| GRPO | GRPO | **GSPO** | ? | ? | ? | ? |
| | | | ← GSPO / SAPO / internal descendants → | | | |

## 2. GLM-5.x：最接近"GRPO 的现代增强版"

GLM-5 技术报告明确：

$$\boxed{\text{Reasoning RL}=\text{GRPO}+\text{IcePop}}$$

它保留了 GRPO 的核心结构（$A_i=\frac{r_i-\mu_r}{\sigma_r}$、token-level ratio $\rho_{i,t}=\pi_\theta/\pi_{old}$），但加了 **IcePop** 来解决

$$\pi_{train}\ne\pi_{infer}$$

带来的 **training-inference mismatch**。也就是说除了正常的 $\pi_\theta/\pi_{old}$，它还额外关注 **rollout inference engine 和 training engine 之间的概率差异**，把偏差异常大的 token/sample 抑制掉。GLM-5 还去掉了原 IcePop 的 KL 正则以加快 RL improvement。

> 这和 [GSPO 解决 MoE routing 抖动](09-gspo.md#6-为什么对-moe-尤其重要)、[Kimi 的 off-policy token mask](11-kimi-k25-k3.md#3-改动三ppo-clipping--与-advantage-符号无关的-off-policy-masking) 是**同一类问题的三种不同解法**：训推不一致 / 数值 mismatch 会污染 token-level ratio。
> - GSPO：把 token ratio 平均成 sequence ratio，让噪声互相抵消
> - IcePop：直接盯 train/infer 的概率差，抑制异常 token
> - Kimi：盯 $\pi_\theta/\pi_{old}$ 的偏离程度，超界就 mask

Agent 部分不是简单"跑 GRPO"，而是 $\boxed{\text{asynchronous group-based agent RL}}$，靠 **slime** 让 rollout workers 和 training workers 异步执行，解决 SWE / terminal / search 这类超长 agent trajectory 的 rollout 长尾。

GLM-5.3 官方说得很明确：**5.3 和 5.2 用同一个 base model，能力提升全部来自 post-training**，但没有新的完整技术报告重新定义一个"GLM-5.3 algorithm"。

$$\boxed{\text{GLM-5.x}\approx\text{GRPO}+\text{IcePop}+\text{Async Agent RL}}$$

再加 General RL 和最后的 **On-Policy Cross-Stage Distillation**。

## 3. Kimi K2.5 / K3

详见 [11-kimi-k25-k3](11-kimi-k25-k3.md)。一句话：

$$\boxed{\text{Kimi-style Group-Relative Policy Gradient}+\text{Token-level Off-policy Regularization}}$$

**不要硬说它是 GRPO** —— 它没有 std normalization、clip 与 advantage 符号无关、多了 $(\log\rho)^2$、还有 partial rollout 和 MOPD。

## 4. 一张血统图

```text
DeepSeek R1
    │
    └── GRPO
         │
         ├── Qwen
         │     GRPO → GSPO → SAPO / stable PG + IS
         │     Qwen3.5 / 3.8：production recipe 未公开
         │
         ├── GLM-5.x
         │     GRPO + IcePop + Async Agent RL
         │
         └── Kimi K2.5 / K3
               group-relative PG
                 + token-level clipping
                 + log-ratio² regularization
                 + partial rollout
                 + MOPD (K3)
```

## 5. 面试怎么答才严谨

问 **"Qwen3.5 用什么 RL？"**：

> Qwen3.5 官方公开了大规模异步 RL 和 agent-environment scaling，但**没有公布最终 production recipe 对应哪一个单一命名算法**。Qwen 同期的 RL 技术栈已经从 GRPO 演进到 GSPO、SAPO，以及带 IS correction / clipping / Routing Replay 的稳定 policy-gradient recipe，所以不能直接断言 Qwen3.5 = GSPO。

问 **"Qwen3.7 用什么 RL？"**：

> 具体 optimizer 官方未披露。比起单纯说 GSPO，更准确的是它属于 Qwen3.5 之后的大规模异步 agent RL 范式（long-horizon agent training、几十小时级 autonomous execution、verifiable environment reward），底层大概率继承 GSPO/SAPO 系的 sequence-aware policy optimization。

三条通用原则：

1. **把"官方明确"和"我的推测"分开说**，并给出推测的依据（发布时间、技术演进、模型规模是不是 MoE）
2. **不要把一家公司说成只有一个算法** —— 同一家的 reasoning / agent / world model / VLA 可能用不同 optimizer
3. **区分 parameter optimizer（Muon/AdamW）和 policy objective（GRPO/GSPO/SAPO）**，也区分**官方 recipe 和第三方框架的 recipe**

## 自测

**1.** ⭐ Qwen3 首发版用的是 GRPO 还是 GSPO？

> **答：** **GRPO**。Qwen3 Technical Report（2025-05）的 Reasoning RL 明确写的是 GRPO。GSPO 是 2025-07-27 才发布的，官方说应用于**当时最新的** Qwen3 models（Instruct/Coder/Thinking，即 2507 之后那批）。**不要把 Qwen3 全部记成 GSPO。**

**2.** 写出 Qwen 的 RL 算法演进线。

> **答：** Qwen2.5（SFT+DPO+GRPO）→ Qwen3 首发（GRPO）→ 2025.7 GSPO（Qwen3 后期版本）→ 2025.12 SAPO → 2026 起转向大规模异步 agent RL，且**production objective 不再公开**。

**3.** ⭐ 为什么不能说"Qwen3.5 = SAPO"？给一个反证。

> **答：** 因为官方 Qwen3.5 release **没有点名 objective**，只说了 RL task/environment scaling 和异步 rollout-training 解耦架构。
> 反证：Qwen 在 2026-06 发布 Qwen-AgentWorld 时仍明确写 **RL uses GSPO**，说明 SAPO 出来后 GSPO 并没被废掉；Qwen-VLA 甚至还在用 PPO。**同一家不同任务用不同 optimizer** 是更合理的判断。

**4.** ⭐ 看到 Qwen3.8-Flash 技术报告里的 Muon，能不能答"它的 RL 算法是 Muon"？

> **答：** **不能。** Muon 是 **parameter optimizer**（$\theta\leftarrow\theta-\eta\,\text{Muon}(\nabla L)$，pretraining 用的），GSPO/SAPO 是 **policy optimization objective**（$L_{RL}(\theta)$），完全不同层面。那份报告讲的是 architecture + pretraining + optimizer，**没有披露 post-training RL objective**。

**5.** 有资料说"Qwen3.8 用 GRPO"，怎么判断？

> **答：** 大概率是因为 NeMo-RL / ms-swift 这类框架**支持拿 Qwen3.8 做 GRPO 后训练**。那是**第三方的开源 recipe**，不是 Qwen 官方训练 Qwen3.8 时用了 GRPO 的证据。两者要严格区分。

**6.** ⭐ GLM-5.x 的 RL 是什么？IcePop 解决什么问题？

> **答：** $\text{GRPO}+\text{IcePop}+\text{Async Agent RL}$（靠 slime 做 rollout/training worker 异步）。
> IcePop 解决 $\pi_{train}\ne\pi_{infer}$ 的 **training-inference mismatch**：除了正常的 $\pi_\theta/\pi_{old}$，还关注 **rollout inference engine 和 training engine 之间的概率差异**，把偏差异常大的 token/sample 抑制掉。GLM-5 去掉了原 IcePop 的 KL 正则以加快 RL improvement。

**7.** ⭐⭐ 训推不一致这个问题，GSPO / IcePop / Kimi 三家分别怎么解？

> **答：** 同一类问题的三种解法：
> **GSPO**：把 token ratio 平均成 sequence ratio（几何平均），让 routing 抖动互相抵消，因此不再需要 Routing Replay；
> **IcePop**：直接盯 train engine 和 infer engine 的概率差，抑制偏差异常大的 token/sample；
> **Kimi**：盯 $\pi_\theta/\pi_{old}$ 的偏离，超出区间就 mask 掉这个 token 的梯度，另加 $(\log\rho)^2$ 软约束。

**8.** 被问"XX 最新模型用什么 RL 算法"，怎么答不容易翻车？

> **答：** ① 把**官方明确披露**的和**自己的推测**分开说，推测要给依据（发布时间、算法演进、是不是大 MoE）；② 不要把一家公司说成只有一个算法（reasoning / agent / world model / VLA 可能各用各的）；③ 区分 **parameter optimizer**（Muon/AdamW）和 **policy objective**（GRPO/GSPO/SAPO），也区分**官方 recipe 和第三方框架 recipe**。
