# Reward Model 与 RLHF

> RLHF 的三阶段：SFT → 训 RM → 用 RM 做 RL。
> 这一篇讲前两阶段和整体框架，PPO/GRPO 的算法细节在 [RL 目录](../rl/00-map.md)。

## 1. 为什么需要 RM

SFT 只能教「模仿一条正确答案」，但很多目标**没有唯一正确答案**：有用性、无害性、风格、简洁度。这类目标：

- 难以写成规则
- 但人类**很容易在两个回答之间做比较**

$$\boxed{\text{绝对打分难，相对比较容易}}$$

所以数据形态是 **pairwise 偏好**：$(x,\ y_w,\ y_l)$，$y_w$ 比 $y_l$ 好。

## 2. Bradley-Terry 模型

假设存在一个潜在的 reward $r(x,y)$，人类选择 $y_w$ 的概率是：

$$\boxed{P(y_w\succ y_l\mid x)=\sigma\big(r(x,y_w)-r(x,y_l)\big)=\frac{e^{r_w}}{e^{r_w}+e^{r_l}}}$$

对这个概率做最大似然，就得到 RM 的训练目标：

$$\mathcal L_{\text{RM}}=-\mathbb E_{(x,y_w,y_l)}\Big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\Big]$$

**只依赖分数差**，所以 $r$ 有一个**平移不变性**：整体加常数不影响 loss。这意味着 RM 的绝对分数没有意义，只有相对大小有意义。

### 结构

拿 SFT 模型，把 LM head 换成一个输出标量的 head，**取最后一个 token 的隐状态**打分：

```text
[prompt + response] → LM backbone → 最后一个位置的 hidden → Linear(d,1) → r
```

通常用 SFT 模型初始化，训练 1 个 epoch（多了会过拟合）。

## 3. RM 的问题

| 问题 | 说明 |
|---|---|
| **奖励黑客（reward hacking）** | policy 找到 RM 的漏洞，产出 RM 高分但人类不喜欢的东西（比如超长、堆砌套话、迎合） |
| **分布漂移** | RM 在 SFT 分布上训练，但 RL 中 policy 越走越远，RM 在新分布上不准 |
| **长度偏置** | 人类标注天然偏好长回答，RM 学到"长 = 好" |
| **过优化** | RL 优化 RM 分数超过某个点后，真实质量反而下降（Goodhart's law） |

应对：

- **KL 惩罚**限制 policy 不要跑太远（见下）
- 长度惩罚 / 长度去偏
- 定期用新数据重训 RM（iterative RLHF）
- **rule-based reward** 代替 RM（数学答案对错、代码测试通过率），这是 reasoning RL 的主流

## 4. RLHF 的目标函数

$$\boxed{\max_\pi\ \mathbb E_{y\sim\pi}\big[r(x,y)\big]-\beta\,D_{KL}\big(\pi\,\|\,\pi_{\text{ref}}\big)}$$

工程上 KL 项通常被折进逐 token 的 reward：

$$r_t=\underbrace{r^{\text{RM}}\cdot\mathbb 1[t=T]}_{\text{只在最后一个 token}}-\beta\big(\log\pi_\theta(a_t|s_t)-\log\pi_{\text{ref}}(a_t|s_t)\big)$$

**为什么要 KL 项**：

1. 防止 reward hacking —— 跑太远就要付出代价
2. 保住 pretrain/SFT 学到的语言能力和格式
3. 保证 importance sampling 的 surrogate 还可信

$\pi_{\text{ref}}$ 通常是**冻结的 SFT 模型**，和 $\pi_{\text{old}}$ 不是一回事（见 [KL 散度](../rl/07-kl.md#再次强调pi_old-和-pi_ref-不是一回事)）。

### 现代趋势：$\beta=0$

reasoning RL（R1、DAPO 风格）常常直接去掉 reference KL：rule-based reward 很干净不需要兜底，而且强 KL 会限制探索、阻止模型偏离 SFT policy 去发展长 CoT。

## 5. 四个模型

经典 RLHF-PPO 要同时存在：

| 模型 | 是否训练 | 作用 |
|---|---|---|
| **Actor** | 训练 | 被优化的 policy |
| **Critic** | 训练 | 估 $V(s)$，算 advantage |
| **Reference** | 冻结 | 算 KL 惩罚 |
| **Reward Model** | 冻结 | 打分 |

显存非常重，详见 [SFT/OPD/RL 显存对比](../memory-sft-opd-rl.md)。GRPO 去掉 critic，rule-based reward 再去掉 RM，就只剩 actor + ref + rollout。

## 6. 一个关键联系：RLHF 本身就是 reverse KL

定义隐式目标分布

$$\pi_R^*(y)=\frac1Z\pi_{\text{ref}}(y)\,e^{r(y)/\beta}$$

可以证明

$$\max_\pi\big\{\mathbb E_\pi[r]-\beta D_{KL}(\pi\|\pi_{\text{ref}})\big\}\iff\min_\pi D_{KL}(\pi\|\pi_R^*)$$

**RLHF 就是在向一个由 reward + reference 隐式定义的分布做 sequence-level reverse KL。** 完整推导见 [reverse KL = policy gradient](../distill/04-reverse-kl-as-pg.md#6-更深的一层llm-rl-本来就可以写成-reverse-kl)。

这个等价关系正是下一篇 [DPO](02-dpo.md) 的出发点。

## 自测（口述版）

**1.** 为什么需要 RM 而不是直接 SFT？为什么数据是 pairwise 而不是打分？

> **答：** 很多目标（有用性、无害性、风格、简洁度）**没有唯一正确答案**，也难以写成规则，SFT 那种「模仿一条标准答案」教不了。
> 数据是 pairwise 是因为 **绝对打分难、相对比较容易** —— 让人给一个回答打 7.5 分很难且不一致，让人从两个回答里选一个好的则简单可靠。所以数据形态是 $(x,y_w,y_l)$。

**2.** 写出 Bradley-Terry 和 RM 的 loss。$r$ 的平移不变性意味着什么？

> **答：** $$P(y_w\succ y_l\mid x)=\sigma\big(r(x,y_w)-r(x,y_l)\big)$$
> 最大似然得 $\mathcal L_{\text{RM}}=-\mathbb E\big[\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))\big]$。
> 它**只依赖分数差**，整体加一个常数 loss 不变，所以 **RM 的绝对分数没有意义，只有相对大小有意义**（跨 prompt 比较 RM 分数是不合理的）。

**3.** RM 的结构是什么？取哪个位置的隐状态？

> **答：** 拿 SFT 模型，把 LM head 换成一个输出标量的 `Linear(d,1)`，**取最后一个 token 的隐状态**打分：`[prompt + response] → LM backbone → 最后位置 hidden → Linear(d,1) → r`。
> 通常用 SFT 模型初始化，训练 1 个 epoch（多了会过拟合）。

**4.** RM 的四个问题分别是什么？各自怎么缓解？

> **答：** ① **reward hacking**：policy 找到 RM 的漏洞，产出高分但人类不喜欢的东西（超长、堆砌套话、迎合）；
> ② **分布漂移**：RM 在 SFT 分布上训练，RL 中 policy 越走越远后 RM 不准；
> ③ **长度偏置**：人类标注天然偏好长回答，RM 学到「长 = 好」；
> ④ **过优化**：优化 RM 分数超过某点后真实质量反而下降（Goodhart's law）。
> 缓解：KL 惩罚限制 policy 不跑太远、长度惩罚/去偏、定期用新数据重训 RM（iterative RLHF）、用 **rule-based reward** 代替 RM（reasoning RL 的主流）。

**5.** 写出 RLHF 的目标函数。KL 项在工程上怎么实现？为什么需要它（三条）？

> **答：** $$\max_\pi\ \mathbb E_{y\sim\pi}[r(x,y)]-\beta\,D_{KL}(\pi\|\pi_{\text{ref}})$$
> 工程上折进逐 token reward：$r_t=r^{\text{RM}}\cdot\mathbb 1[t=T]-\beta\big(\log\pi_\theta(a_t|s_t)-\log\pi_{\text{ref}}(a_t|s_t)\big)$。
> 三条理由：① 防 reward hacking（跑太远要付代价）；② 保住 pretrain/SFT 学到的语言能力和格式；③ 保证 importance sampling 的 surrogate 还可信。

**6.** $\pi_{\text{ref}}$ 和 $\pi_{\text{old}}$ 的区别？

> **答：** $\pi_{\text{ref}}$ 是训练开始时**冻结的 SFT 模型**，全程不动，用于 KL penalty，是**长期锚点**；
> $\pi_{\text{old}}$ 是**产生当前这批 rollout 的 policy 快照**，每轮都更新，用于 importance ratio，是**短期基准**。

**7.** 为什么现代 reasoning RL 常把 $\beta$ 设成 0？

> **答：** ① rule-based reward 很干净，不需要 KL 兜底防 hacking；② 强 KL 会**限制探索**；③ reasoning model 本来就希望**允许明显偏离 SFT policy** 去发展长 CoT。

**8.** RLHF-PPO 需要几个模型？哪些训练哪些冻结？GRPO 少了哪个？

> **答：** **四个**：**Actor**（训练）、**Critic**（训练）、**Reference**（冻结，算 KL）、**Reward Model**（冻结，打分），再加一个 rollout engine。
> **GRPO 去掉 Critic**（用组内相对 reward 当 baseline）；如果再用 rule-based reward 就连 RM 也去掉，只剩 actor + ref + rollout。

**9.** 为什么说 RLHF 本身就是一个 reverse KL？写出 $\pi_R^*$。

> **答：** 定义隐式目标分布 $\boxed{\pi_R^*(y)=\frac1Z\pi_{\text{ref}}(y)\,e^{r(y)/\beta}}$，则
> $$\max_\pi\big\{\mathbb E_\pi[r]-\beta D_{KL}(\pi\|\pi_{\text{ref}})\big\}\iff\min_\pi D_{KL}(\pi\|\pi_R^*)$$
> 也就是在向一个由 **reward + reference 隐式定义**的分布做 **sequence-level reverse KL**。这正是 DPO 的出发点。

