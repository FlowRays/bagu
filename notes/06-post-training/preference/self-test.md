# 偏好学习自测题库（关掉笔记用）

> 侧边栏顶部有 **「答案」开关**。标 ⭐ 的是高频考点。

## A. RM 与 RLHF（[01](01-rm-and-rlhf.md)）

**1.** ⭐ 为什么需要 RM？为什么数据是 pairwise？

> **答：** 有用性、无害性、风格这类目标没有唯一正确答案，难写成规则，SFT 教不了。但人类**很容易在两个回答之间比较**：绝对打分难，相对比较容易。所以数据是 $(x,y_w,y_l)$。

**2.** ⭐⭐ 写出 Bradley-Terry 和 RM 的 loss。$r$ 的平移不变性意味着什么？

> **答：** $P(y_w\succ y_l|x)=\sigma(r_w-r_l)$，最大似然得
> $\mathcal L_{\text{RM}}=-\mathbb E[\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))]$。
> 只依赖**分数差**，整体加常数不变，所以 RM 的**绝对分数没有意义，只有相对大小有意义**。

**3.** RM 的结构？

> **答：** 拿 SFT 模型，LM head 换成输出标量的 Linear(d,1)，取**最后一个 token** 的隐状态打分。通常用 SFT 模型初始化，训 1 个 epoch（多了过拟合）。

**4.** ⭐ RM 的四个问题？怎么缓解？

> **答：** ① **reward hacking**（找 RM 漏洞产出高分但人类不喜欢的东西）；② **分布漂移**（RM 在 SFT 分布上训，policy 越走越远后不准）；③ **长度偏置**（人类偏好长回答，RM 学到"长=好"）；④ **过优化**（超过某点真实质量反而下降，Goodhart's law）。
> 缓解：KL 惩罚、长度去偏、定期重训 RM（iterative RLHF）、用 **rule-based reward** 代替 RM。

**5.** ⭐⭐ 写出 RLHF 目标函数。KL 项工程上怎么实现？为什么需要它？

> **答：** $\max_\pi \mathbb E_{y\sim\pi}[r(x,y)]-\beta D_{KL}(\pi\|\pi_{\text{ref}})$。
> 工程上折进逐 token reward：$r_t=r^{\text{RM}}\mathbb 1[t=T]-\beta(\log\pi_\theta(a_t|s_t)-\log\pi_{\text{ref}}(a_t|s_t))$。
> 三个理由：① 防 reward hacking；② 保住 pretrain/SFT 的语言能力和格式；③ 保证 surrogate 还可信。

**6.** $\pi_{\text{ref}}$ 和 $\pi_{\text{old}}$ 的区别？

> **答：** $\pi_{\text{ref}}$ 是训练开始时冻结的 SFT 模型，长期锚点，用于 KL penalty；$\pi_{\text{old}}$ 是产生当前这批 rollout 的 policy 快照，每轮都变，用于 importance ratio。

**7.** 为什么现代 reasoning RL 常设 $\beta=0$？

> **答：** rule-based reward 很干净不需要兜底；强 KL 限制探索；reasoning model 本来就希望允许明显偏离 SFT policy 去发展长 CoT。

**8.** RLHF-PPO 需要几个模型？

> **答：** 四个：**Actor**（训练）、**Critic**（训练）、**Reference**（冻结，算 KL）、**Reward Model**（冻结，打分），再加 rollout engine。GRPO 去掉 critic，rule-based reward 再去掉 RM，只剩 actor + ref + rollout。

**9.** ⭐⭐ 为什么说 RLHF 本身就是 reverse KL？

> **答：** 定义 $\pi_R^*(y)=\frac1Z\pi_{\text{ref}}(y)e^{r(y)/\beta}$，则
> $\max_\pi\{\mathbb E_\pi[r]-\beta D_{KL}(\pi\|\pi_{\text{ref}})\}\iff\min_\pi D_{KL}(\pi\|\pi_R^*)$。
> 也就是在向一个由 reward + reference 隐式定义的分布做 **sequence-level reverse KL**。这正是 DPO 的出发点。

## B. DPO（[02](02-dpo.md)）

**10.** DPO 想解决什么？

> **答：** RLHF 要训 RM、四个模型、rollout、调 PPO 一堆超参，太重。DPO 问：能不能跳过 RM 和 RL，**直接用偏好数据训 policy**。

**11.** ⭐⭐⭐ **完整推导 DPO**。

> **答：** 三步。
> ① KL-regularized RL 的闭式最优解：$\pi^*(y|x)=\frac1{Z(x)}\pi_{\text{ref}}(y|x)\exp(r(x,y)/\beta)$，其中 $Z(x)=\sum_y\pi_{\text{ref}}e^{r/\beta}$。（因为目标等价于 $\min_\pi D_{KL}(\pi\|\pi^*)$，在 $\pi=\pi^*$ 取 0。）
> ② 反解 reward：$r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\log Z(x)$。**policy 自己隐式就是 reward model**。
> ③ 代入 Bradley-Terry 做差，$\beta\log Z(x)$ 只依赖 $x$，**在做差时完全消掉**：
> $$\mathcal L_{\text{DPO}}=-\mathbb E\Big[\log\sigma\big(\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\big)\Big]$$
> 最漂亮的地方就是那个算不出来的配分函数自己没了。

**12.** $h=0$ 时 loss 多少？梯度里 $\sigma(-h)$ 起什么作用？

> **答：** $-\log\sigma(0)=\log2\approx0.693$。
> $\nabla\mathcal L=-\beta\sigma(-h)[\nabla\log\pi_\theta(y_w)-\nabla\log\pi_\theta(y_l)]$，$\sigma(-h)$ 是**自适应权重**：已经分对的样本权重趋近 0，分错的趋近 1，自动聚焦在没学会的偏好对上。

**13.** ⭐⭐ DPO 和 PPO 的本质区别？导致什么局限？

> **答：** **DPO 是 off-policy（离线固定数据），PPO/GRPO 是 on-policy（持续采样）**。
> 局限：DPO 只在给定的 $(y_w,y_l)$ 上学，**无法探索数据里没有的更好回答**，上限受限于离线数据分布。和 SFT vs OPD 的区别是同一个道理。

**14.** ⭐ DPO 「同时压低两者概率」怎么解释？

> **答：** loss 只约束 $\log\pi(y_w)$ 和 $\log\pi(y_l)$ 的**差值**，不约束绝对值。所以完全可能两者一起下降、只是 $y_l$ 降得更快，loss 照样下降。实践中经常观察到。

**15.** DPO 的其他问题？

> **答：** 分布外失效（偏好数据不是当前 policy 采的，$\pi_\theta$ 在那些 $y$ 上概率本来就低，梯度不可靠）；长度偏置；对 $\beta$ 敏感（小了跑飞、大了学不动）。

**16.** ⭐ 五个变体各改了什么？

> **答：** **IPO** 把 sigmoid 换成平方损失缓解过拟合；**KTO** 只需单边「好/坏」标注不需要成对；**SimPO** 去掉 reference model，用长度归一化的平均 logp 当隐式 reward；**ORPO** 把 SFT loss 和偏好项合成一个，不需要单独 SFT；**Online / 迭代 DPO** 每轮用当前 policy 采样标注再 DPO，把 off-policy 变近似 on-policy —— 这是弥补 DPO 最大短板的方向。

**17.** ⭐ 实现要点？

> **答：** 四个 log prob 都是整条 response 的 token logp **之和**（除非用 SimPO 那种归一化变体）；ref 两项要 detach，最好**离线预先算好缓存**，训练时显存里只需一个模型；用 `F.logsigmoid` 而不是 `log(sigmoid(x))`；$\beta$ 典型 0.1，lr 比 SFT 还小（5e-7~5e-6）。

**18.** ⭐ 一分钟讲清 DPO。

> **答：** RLHF 的 KL-regularized 目标有闭式最优解 $\pi^*\propto\pi_{\text{ref}}e^{r/\beta}$，反解出 reward 就是 $\beta\log\frac{\pi^*}{\pi_{\text{ref}}}$ 加一个只依赖 prompt 的配分项。把它代进 Bradley-Terry 的偏好概率里做差，配分项消掉，于是整个 RLHF 塌缩成一个关于 policy 的二分类最大似然。好处是不用训 RM、不用 rollout、只要两个模型；代价是它是 off-policy 的，无法探索离线数据之外的回答，所以后来又有 online DPO 把 on-policy 找回来。
