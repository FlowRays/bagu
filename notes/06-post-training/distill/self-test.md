# 蒸馏 / OPD / OPSD 自测题库（关掉笔记用）

> 对应学习流程第三步：**跟 GPT 口述 / 纸上手写验证**。
> 用法：先关掉所有笔记，口述或纸上写；卡住的题记下来，回填对应笔记。
> 标 ⭐ 的是**高频考点**或最容易卡的题。
>
> 侧边栏顶部有 **「答案」开关**，可以全局显示 / 隐藏所有答案。自测时先隐藏。

## A. 信息论地基（[01](01-entropy-ce-kl-jsd.md)）

**1.** 写出熵、交叉熵、KL 三个定义。

> **答：** $H(q)=-\sum_v q(v)\log q(v)$；$H(q,p)=-\sum_v q(v)\log p(v)$；$D_{KL}(q\|p)=\sum_v q(v)\log\frac{q(v)}{p(v)}$。
> 语义：熵 = 用 $q$ 自己的最优编码编 $q$ 的样本；交叉熵 = 数据来自 $q$ 却用 $p$ 的编码；KL = 多付出的那部分。

**2.** ⭐ **纸上推导** $H(q,p)=H(q)+D_{KL}(q\|p)$，并说明它为什么重要。

> **答：** $H(q,p)=-\sum q\log p=-\sum q\log q+\sum q\log\frac{q}{p}=H(q)+D_{KL}(q\|p)$。
> 重要性：训练时 $q$（label / teacher）固定，$H(q)$ 是常数，所以 $\min_p H(q,p)\iff\min_p D_{KL}(q\|p)$ —— **最小化交叉熵就是在做 forward KL**（数据/teacher 在前，模型在后）。

**3.** one-hot label 时，CE 和 KL 是"等价"还是"相等"？

> **答：** **相等**。因为 one-hot 分布的熵 $H(\delta_{y^*})=0$，恒等式退化成 $\text{CE}=D_{KL}(\delta_{y^*}\|\pi_\theta)$，不只是差一个常数。

**4.** ⭐ 写出 softmax+CE 对 logits 的梯度，并用 $p=[0.4,0.3,0.3]$ 分别对 $q=[1,0,0]$ 和 $q=[0.7,0.2,0.1]$ 算一次。

> **答：** $\dfrac{\partial L}{\partial z}=p-q$，hard 和 soft 通用。
> hard：$p-q=[-0.6,0.3,0.3]$，A 的 logit 升、B/C 降 → 把概率全压到唯一 GT token。
> soft：$p-q=[-0.3,0.1,0.2]$，A 升、B 略降、C 降更多 → 目标是 $p\to q$（拟合 teacher 的形状），而不是 $p\to[1,0,0]$。

**5.** ⭐ 从 teacher 采样做 hard CE，和直接用 teacher 分布做 soft CE，是什么关系？

> **答：** 期望相等。$\mathbb E_{a\sim q}[-\log p(a)]=-\sum_v q(v)\log p(v)=H(q,p)$，所以采样版是 soft CE 的**无偏 Monte-Carlo 估计**。
> soft CE 是"把所有可能的 teacher sample 一次性平均掉"，方差更小，但需要 teacher 的完整分布；hard 采样只需要 teacher 能生成。

**6.** 写出 JSD 的定义，解释为什么要引入 mixture $m$。

> **答：** $m=\beta p_T+(1-\beta)p_S$，$\mathrm{JSD}_\beta=\beta D_{KL}(p_T\|m)+(1-\beta)D_{KL}(p_S\|m)$。
> 引入 $m$ 是因为两个 KL 都有"分母趋 0 就爆炸"的问题；混合后只要有一边有概率，$m$ 就不为 0（例如 $p_S(v)=0$ 时 $p_T(v)/m(v)=2$）。所以 JSD 对称且有界（$\beta=0.5$、自然对数时 $0\le\mathrm{JSD}\le\ln2$），**即使两个分布 support 完全不重叠也有限**。

**7.** $\beta\to1$ 时 JSD 的行为偏向哪一侧？

> **答：** $m\to p_T$，行为偏向 forward KL 那一侧。$\beta$ 就是 teacher / student 两边权重的旋钮。

## B. SFT 与 KD（[02](02-sft-and-kd.md)）

**8.** ⭐ 写出 SFT 的 loss，指出哪一项是 exposure bias 的根源。

> **答：** $L_{\text{SFT}}=-\sum_t\log\pi_\theta(y_t^*|x,y^*_{<t})$。根源是条件 prefix $y^*_{<t}$ —— 训练时永远喂 GT 的正确历史（teacher forcing），模型从没见过"自己前面走错之后"的状态。

**9.** ⭐ 用一个例子解释 exposure bias。

> **答：** $17\times6=?$，GT 是 `17 × 6 = 102`。SFT 只训练 prefix `17 × 6 =`。但推理时 student 可能生成 `17 × 6 = 96, therefore ...`，这个状态在 SFT 数据里从没出现过，模型没学过"我已经犯错了现在怎么办" → train-test distribution mismatch。

**10.** 证明 $\min L_{\text{SFT}}\equiv\min D_{KL}(p_{\text{data}}\|\pi_S)$。

> **答：** $L_{\text{SFT}}=\mathbb E_{y\sim p_{\text{data}}}[-\log\pi_S(y)]=H(p_{\text{data}},\pi_S)=H(p_{\text{data}})+D_{KL}(p_{\text{data}}\|\pi_S)$，第一项与 $\theta$ 无关。所以 SFT = 在 expert/data states 上做 empirical **forward** KL。

**11.** ⭐ "teacher 生成数据再 SFT" 和 "用 teacher 分布做 logit 蒸馏" 是什么关系？

> **答：** 期望下是**同一个目标**。$\mathbb E_{a\sim\pi_T}[-\log p_\theta(a)]=H(\pi_T,p_\theta)=H(\pi_T)+D_{KL}(\pi_T\|p_\theta)$，采样只是 teacher distribution 的无偏 MC 估计。区别在方差和成本，不在目标。

**12.** 传统 logit KD 和 SFT 差在哪个维度？和 OPD 又差在哪个维度？

> **答：** 和 SFT 差在 **target**（teacher 完整 soft 分布 vs one-hot hard token），prefix 都是 teacher/data 的。和 OPD 差在 **state distribution**（teacher/data prefix vs student rollout prefix）。

## C. OPD 的定义（[03](03-opd.md)）

**13.** ⭐⭐ 给出 OPD 的定义，指出公式里哪一部分对应 "on-policy"。

> **答：** $\mathcal L=\mathbb E_{\hat y\sim\pi_S}\big[\sum_t D(\pi_T(\cdot|h_t),\pi_S(\cdot|h_t))\big]$，$h_t=(x,\hat y_{<t})$。
> **外层的 $\hat y\sim\pi_S$ 才是 on-policy**，严格讲是 $s_t\sim d^{\pi_S}$（prefix 由 student 自己一步步生成）。里面的 $D$ 是什么是另一个维度。
> 一句话：teacher 从"示范者"变成"判卷者"，不重新写答案，只在 student 走到的每个位置回答"下一步该怎么走"。

**14.** ⭐⭐ "OPD 就是 reverse KL 蒸馏" 错在哪？画出四象限表。

> **答：** 混淆了两个**正交**维度。on-policy 描述的是 prefix/state 从谁来；forward/reverse KL 描述的是在这个 state 上两个分布怎么比。
> 四象限：off-policy+forward = 传统 KD（hard 特例是 SFT）；off-policy+reverse = 少见；**on-policy+forward = GKD / OPSD 主 recipe**；**on-policy+reverse = Thinking Machines OPD recipe**。
> OPSD 论文主实验用的正是 forward KL，这就是最直接的反例。

**15.** ⭐ forward-KL OPD 会不会退化成 SFT？

> **答：** 不会。两者内部都带 forward-KL/CE 的味道，但 (a) **state 不同**：$L_{\text{SFT}}\approx\mathbb E_{h\sim d_{\text{expert}}}[\cdot]$ vs $L_{\text{fwd-OPD}}=\mathbb E_{h\sim d_{\pi_S}}[\cdot]$；(b) **target 不同**：SFT 是 hard token，forward-KL OPD 是 teacher 的完整 soft distribution。
> 记：SFT = teacher state + hard token；Forward OPD = student state + teacher soft distribution；Reverse OPD = student state + teacher 对 student action 打分。

**16.** ⭐ 用一个具体表格解释 reverse KL 为什么 mode-seeking。

> **答：** student $\{A:0.8,B:0.1,C:0.1\}$，teacher $\{A:0.001,B:0.6,C:0.399\}$。reverse KL 的期望在 $\pi_S$ 下，A 的贡献是 $0.8\log\frac{0.8}{0.001}$，极大。于是 student 被强烈要求"**不要把概率放在 teacher 认为很差的位置**"，概率质量撤回 teacher 的高概率 mode。
> 严谨表述是 "student-weighted"，不要说成"student 有值 teacher 没值"。

**17.** 一句话对照 forward / reverse KL 在蒸馏里查的是什么。

> **答：** Forward KL：**teacher 看 student 有没有漏**（mode covering）。Reverse KL：**student 看自己有没有乱跑**（mode seeking）。

**18.** ⭐ "reverse-KL OPD 只能强化已有能力"这句话怎么改才严谨？

> **答：** 三点：① LM 的 softmax 让几乎每个 token 都有 $\pi(v)>0$，严格数学意义上的 support 几乎是全词表，真正的问题是 **low-probability / 几乎不会被访问的 trajectory**，不是 $p=0$；② 准确说法是 "SFT 适合 capability injection / cold start，on-policy OPD 主要在 student 当前 visitation distribution 上做 policy refinement"；③ Rethink OPD 发现 OPD 真正有效时 teacher **还必须提供新能力**，所以不能说 OPD 传不了新能力。

**19.** ⭐ 现代 OPD 偏爱 reverse KL 的三个理由？forward KL 唯一的杀手锏是什么？

> **答：** ① 便宜：只要 teacher 返回 student 采样 token 的一个标量 logprob，不用 100K+ 维完整 logits；② 小 student 容量有限，被 forward KL 逼着 mode-cover 会把概率摊开、生成质量下降，reverse 的 mode-seeking 更聚焦；③ 和 RL pipeline 同构，$A_t=\log\pi_T-\log\pi_S$ 直接当 advantage。
> forward KL 的杀手锏：**能主动增加 support**。$\pi_T(B)=0.3$ 而 $\pi_S(B)\approx0$ 时，reverse KL 下 student 几乎永远采不到 B，teacher 没机会说"B 很好"；forward KL 直接产生 $-\pi_T(B)\log\pi_S(B)$ 的梯度硬推。所以典型 recipe 是先 SFT/forward 扩 support，再 reverse-KL OPD mode-seek。

## D. Reverse KL = policy gradient（[04](04-reverse-kl-as-pg.md)）

**20.** ⭐⭐ **纸上推导** $\nabla_\theta[-D_{KL}(p_\theta\|q)]$。

> **答：** $J=\sum_a p_\theta(a)[\log q(a)-\log p_\theta(a)]$。product rule 得两项：
> (I) $\sum_a\nabla p_\theta(a)[\log q-\log p_\theta]$；(II) $-\sum_a p_\theta(a)\nabla\log p_\theta(a)$。
> (II) 中 $p_\theta\nabla\log p_\theta=\nabla p_\theta$，故 $\sum_a\nabla p_\theta(a)=\nabla(\sum_a p_\theta(a))=\nabla1=0$，**直接消失**。
> (I) 再用 log trick：$\nabla J=\mathbb E_{a\sim p_\theta}[(\log q(a)-\log p_\theta(a))\nabla\log p_\theta(a)]$ —— 就是 policy gradient。

**21.** 为什么 "reward 里含 $\theta$" 不破坏这个推导？

> **答：** 因为多出来的那一项 $-\mathbb E[\nabla\log p_\theta]$ 恰好是 score function 的期望，等于 0（归一化常数求导）。所以可以直接把 $A(a)=\log q(a)-\log p_\theta(a)$ 当作（stop-gradient 的）advantage 用。

**22.** ⭐ 写出 OPD 的 advantage，用两组数值解释含义。

> **答：** $A_t=\log\pi_T(a_t|s_t)-\log\pi_S(a_t|s_t)=\log\frac{\pi_T}{\pi_S}$。
> $(\pi_S,\pi_T)=(0.1,0.4)$：$A=\log4\approx1.386>0$ → 提高该 token 概率（"你这个 token 其实挺好，你自己还不够相信它"）。
> $(0.4,0.1)$：$A\approx-1.386<0$ → 降低（"你特别喜欢它，但我认为它不该出现"）。相等则 $A=0$ 不更新。
> 关键：看的是**比值**，不是 teacher 概率的绝对值。

**23.** ⭐ 描述 Thinking Machines 的实现流程，为什么 RL trainer 几乎不用改？

> **答：** rollout 时已记录 $\log\pi_{\text{old}}(a_t|s_t)$；teacher 对同一个 token 算 $\log\pi_T(a_t|s_t)$；令 $A_t=\log\pi_T-\log\pi_{\text{old}}$（即负的 sampled reverse KL），送进现成的 importance-sampling loss $L=-\mathbb E[\rho_t A_t]$，$\rho_t=\pi_\theta/\pi_{\text{old}}$。
> rollout 刚结束时 $\rho=1$，梯度正是推导出的 $-\mathbb E[A_t\nabla\log\pi_\theta]$。所以只是把 `reward → advantage` 换成 `teacher logprob → KL advantage`；对已有的 KL-regularized RL trainer 来说约等于"把 regularizer model 换成 teacher"。

**24.** OPD 为什么比 outcome RL 更 dense？为什么可以取 $\gamma=0$？

> **答：** outcome RL 一条 1000 token 的 CoT 只有最后一个 0/1 reward，credit assignment 很难；OPD 每个 token 都有 $A_t$，是 token-level dense advantage。
> $\gamma=0$ 表示第 $t$ 个 token 只优化当前位置的 teacher mismatch，不把未来 reward 折回来，即 $A_t=r_t$，不需要 $G_t=r_t+\gamma r_{t+1}+\cdots$。实验上更大的 discount 没带来改善。

**25.** ⭐⭐ **推导** $\max_\pi\{\mathbb E_\pi[R]-\beta D_{KL}(\pi\|\pi_{\text{ref}})\}\iff\min_\pi D_{KL}(\pi\|\pi_R^*)$。

> **答：** 定义 $\pi_R^*(y)=\frac1Z\pi_{\text{ref}}(y)e^{R(y)/\beta}$。则
> $D_{KL}(\pi\|\pi_R^*)=\mathbb E_\pi[\log\pi-\log\pi_{\text{ref}}-R/\beta+\log Z]$，两边乘 $-\beta$ 得
> $-\beta D_{KL}(\pi\|\pi_R^*)=\mathbb E_\pi[R]-\beta D_{KL}(\pi\|\pi_{\text{ref}})-\beta\log Z$，最后一项与 $\pi$ 无关。
> 含义：**KL-regularized RL 本身就在做 sequence-level reverse KL**，只是它的 teacher 分布没有显式模型，由 reward model + reference model 隐式定义。OPD 相当于把这个隐式 teacher 换成真实存在、且能逐 token 给 logprob 的 $\pi_T$。

**26.** ⭐⭐⭐ 把 SFT 和 OPD（reverse KL）的梯度都推到 logits 层面，说明为什么一个不用 PG、一个必须用。

> **答：** 记 $\dfrac{\partial\log\pi_S(v)}{\partial z_k}=\delta_{vk}-\pi_S(k)$，$\dfrac{\partial\pi_S(v)}{\partial z_k}=\pi_S(v)(\delta_{vk}-\pi_S(k))$，核心恒等式 $\sum_v\pi_S(v)(\delta_{vk}-\pi_S(k))=0$。
>
> **SFT**：$L=-\log\pi_S(y^*)\Rightarrow\nabla_z L=\pi_S-e_{y^*}$。**Forward KL**：$L=-\sum_v\pi_T(v)\log\pi_S(v)\Rightarrow\nabla_z L=\pi_S-\pi_T$（$\pi_T$ 全程当常数拎出，从未被求导）。
>
> **Reverse KL**：$L=\sum_v\pi_S(v)r(v)$，$r=\log\frac{\pi_S}{\pi_T}$。乘法法则两项：(II) $=\sum_v\pi_S(v)(\delta_{vk}-\pi_S(k))=0$ 精确消掉；(I) $=\pi_S(k)r(k)-\pi_S(k)\sum_v\pi_S(v)r(v)$，得 $\boxed{\nabla_{z_k}L=\pi_S(k)\big(r(k)-D_{KL}(\pi_S\|\pi_T)\big)}$。注意 baseline $-D_{KL}$ 是 softmax Jacobian 自己长出来的，不是手工加的。
>
> **分水岭**：写成 $L=\sum_v w(v)f_\theta(v)$，SFT/forward KL 的权重 $w$ 与 $\theta$ 无关 $\Rightarrow$ (I) 不存在，梯度全在 (II)；reverse KL 的 $w=\pi_S$ 含 $\theta$ $\Rightarrow$ 梯度全在 (I)。**两者恰好互换。**
>
> **为什么采样后必须 PG**：$\hat L=\frac1N\sum_i r(v_i)$ 的**值无偏**，但 $v_i$ 是常数下标，权重 $\pi_S$ 被"哪些下标被抽中"吸收，autograd 只算得到 (II)，而 $\mathbb E[\partial\hat L/\partial z_k]=\pi_S(k)-\pi_S(k)=0$ —— 梯度是零的无偏估计，纯噪声（症状：loss 数值正常，模型不动）。PG 用 $\nabla\pi_S=\pi_S\nabla\log\pi_S$ 把 (I) 改写成 $\mathbb E_{v\sim\pi_S}[r(v)\nabla\log\pi_S(v)]$，可以采样估；代码里用 surrogate $\tilde L=\frac1N\sum_i\text{sg}[r(v_i)-b]\log\pi_S(v_i)$。
>
> forward KL 采样时从 $\pi_T$ 采（与 $\theta$ 无关），本来就没有 (I)，所以什么都不丢。详见 [04 第 7 节](04-reverse-kl-as-pg.md#7-完整梯度推导sft-与-opd-从-logits-一路推到底)。

**27.** ⭐ OPD 的三档 KL 估计是什么？各自的取舍？

> **答：** sampled-token（1 个采样 token，KL 的**单样本无偏估计**，最便宜、监督最稀疏、方差最大）；top-k（student top-k 子集，重新归一化后算 KL，折中，$O(TK)$）；full-vocabulary（整词表精确 KL，监督最密、方差最低，$O(BTV)$ 显存最贵）。
> 注意这是**第三个维度**，和 on-policy、forward/reverse 都正交。

**28.** 证明 sampled-token estimator 无偏。

> **答：** $\mathbb E_{\hat y_t\sim p_t}[\log p_t(\hat y_t)-\log q_t(\hat y_t)]=\sum_v p_t(v)\log\frac{p_t(v)}{q_t(v)}=D_{KL}(p_t\|q_t)$。

**29.** ⭐ 用一个例子说明 sampled-token 的稀疏性问题。

> **答：** student $\{A:0.4,B:0.3,C:0.2,D:0.1\}$，teacher $\{A:0.1,B:0.6,C:0.2,D:0.1\}$。teacher 想说的是"把概率从 A 搬到 B"。如果这次采到 A，估计量只看到 $\log0.4-\log0.1$，只知道"A 该压"，**完全没看到 B 该升**，要等下次采到 B 才知道。期望正确但方差大 —— 和 REINFORCE 一个味道。

**30.** 估算 full-vocab 的显存，说明痛点是 FLOPs 还是显存/带宽。

> **答：** $T=2048$、$V=150\text{K}$、bf16：$2048\times150000\times2\approx614$ MB 一份，teacher+student 约 1.2 GB/sample；$T=16$K 时约 4.8 GB。
> 痛点主要是**显存和带宽**：正常 LM 训练算 CE 本来就要过 $h_tW_{\text{vocab}}$ 产生完整 logits，多出来的是 teacher 的额外一次 forward、两边完整 logits、整个 $V$ 上的 softmax+KL、以及为 backward 保留的中间量。

**31.** ⭐ Top-k KL 为什么必须重新归一化？它的含义是什么？

> **答：** 因为 $\sum_{v\in S_t}p(v)<1$，截断后不再是分布。两边都要 $\bar p(v)=p(v)/\sum_{u\in S_t}p(u)$、$\bar q$ 同理，再算 $D_{KL}(\bar p\|\bar q)$。
> 含义：**不要求 student 学 teacher 的整个词表分布，只要求它在"自己真正在考虑的几个候选"之间学会像 teacher 一样排序和分配概率** —— 只训练 decision boundary。

**32.** per-entry KL clipping 和 top-k 的本质区别？OPSD 为什么需要它？

> **答：** clipping 是 $\sum_{v}\min(\ell_{t,v},\tau)$，**仍然遍历整个 vocab**，只限制单项贡献；top-k 是根本不看其余 token。
> OPSD 需要它是因为 `wait / therefore / however / think` 这类 **style token 的 KL 特别大**，会淹掉真正数学 reasoning token 的梯度。论文附录里的 `Top-k=-1` 是生成 sampling 参数，不是 KL 的 top-k。

**33.** ⭐ "loss 的 MC estimator" 和 "policy gradient 的估计量" 差在哪？

> **答：** 前者是 $\ell_t=\log p_t(\hat y_t)-\log q_t(\hat y_t)$，是 KL **数值**的无偏估计；后者是 $A_t=\operatorname{sg}[\log q_t-\log p_t]$，$L_{\text{PG}}=-A_t\log p_\theta(\hat y_t)$。
> 不能对采样 token 直接做普通 CE backprop，因为 $\hat y_t\sim p_\theta$ 这个 sampling 不可导，$\theta$ 同时出现在期望的分布和被积函数里，必须走 score-function 路线，且 $A_t$ 要 stop-gradient。

## F. OPSD（[06](06-opsd.md)）

**34.** ⭐⭐ 写出 OPSD 的 student / teacher 分布，指出 teacher 的优势来源。

> **答：** $p_S(\cdot|x)$ 与 $p_T(\cdot|x,y^*)$，同一个模型、不同 context。
> **OPD 是能力优势**（7B ← 72B）；**OPSD 是信息优势**（privileged information：teacher 多看了 GT）。teacher 本身未必更强。

**35.** ⭐ 为什么"自己教自己"会有训练信号？不给 GT 会怎样？

> **答：** 因为 $p_\theta(\cdot|x)\ne p_\theta(\cdot|x,y^*)$ —— 同一个模型信息不同。利用的现象是"知道答案后解释怎么推，比不知道答案时自己找出来容易得多"。
> 不给 GT 的话 teacher 就是 $p_\theta(\cdot|x,\hat y_{<t})$，和 student 完全一样，没有任何额外信息，信号为 0。

**36.** privileged information 具体怎么给 teacher？

> **答：** 直接**拼进 teacher 的 context**（Problem + Reference solution + Student attempt），student 的 context 里没有 reference solution 那一段。teacher **不继续生成**，只做一次 forward 取该位置的 next-token 分布，rationalization 是隐式完成的。

**37.** ⭐⭐ teacher 拿到 GT 后会不会只是照抄 GT？

> **答：** 不会，因为它算的是 $p_T(\cdot|x,y^*,\hat y_{<t})$ 而不是 $p_T(\cdot|x,y^*)$ —— **三样都给**：题目、GT、student 当前 prefix。
> 例：GT 是 `2x=4 → x=2`，但 student 写成 `2x = 10. Therefore ...`。照抄 GT 就该输出 `2x = 4`，可是接在这个 prefix 后面语言上都不自然。teacher 实际会倾向 `however` / `this is incorrect` / `we made an error` 再修正。
> 这叫 **GT-aware trajectory correction**，不是 GT imitation。

**38.** 用开车类比说明 SFT 和 OPSD 的差别。

> **答：** SFT 是驾校老师给一条标准路线，你只学"正确状态 → 正确动作"，走错了就没辙。OPSD 是你自己开、拐错了，老师坐副驾且知道目的地，他不说"你本来应该左转"，而说"既然你已经到了这个路口，现在该怎么走才能回到正确方向"。即在自己的 state distribution 上学 recovery。

**39.** ⭐ OPSD 训练时 teacher 是实时跟着 student 更新的吗？

> **答：** 不是。teacher branch **stop-gradient**，而且固定在 **initial policy snapshot**，student 持续更新。这是为了训练稳定。所以准确说法是"同源模型、不同 context"，不是"实时的自己"。

**40.** ⭐⭐ OPSD 主实验用的是 forward 还是 reverse KL？这和"OPD 是 reverse KL"矛盾吗？

> **答：** **forward KL**。论文 §4.3.1 的消融（Qwen3-1.7B / AIME25）：forward KL 36.7 → 43.9（step 50）；reverse KL 只有 37.5；JSD 36.9。作者写 "We therefore adopt forward KL in all remaining experiments."
> 不矛盾，恰恰证明 on-policy 和 forward/reverse 是**正交**的：OPSD = on-policy prefix + full-vocab **forward** KL。把 Thinking Machines 的 sampled-token reverse-KL recipe 当成 "OPD 的定义"才是错的；在 OPSD 论文里它是被当作 *alternative objective* 讨论的。

**41.** OPSD 的两个 objective 怎么取舍？

> **答：** full-vocabulary（主 recipe，逐词表 forward KL + per-entry clipping）效果更好；sampled-token policy-gradient 版更便宜。论文结论是 full-vocab > sampled-token，代价是要保存 vocab-sized logits，peak memory 更高 —— 明确的 performance–memory tradeoff。

**42.** OPSD 是第一个提出 self-distillation 的工作吗？

> **答：** 不是。learning with privileged information、teacher-privileged distillation、各种 LLM self-training 都更早。它真正新的是**组合**：same model + privileged teacher context + student on-policy rollout + token-level distribution distillation，并正式命名为 OPSD（Zhao et al., 2026-01）。
> 时间线：GKD → Thinking Machines OPD → **OPSD (2026-01)** → Rethinking OPSD (2026-07) → U-OPSD (2026-08，进一步去掉 GT supervision)。

## G. Rethink OPD 与 MOPD（[07](07-rethink-and-mopd.md)）

**43.** ⭐⭐ "teacher 越强蒸馏效果越好"错在哪？给一个实验证据。

> **答：** $\text{Teacher capability}\ne\text{Distillability}$。证据：MOPD 把同源 RL teacher 换成更强的 Qwen3-235B 后，teacher benchmark 更强但效果反而更差 —— 初始 KL 从约 0.04 升到约 0.19，policy-gradient 版出现 entropy 收缩，top-k 版甚至训练发散。

**44.** ⭐ 成功 OPD 的两个条件是什么？各举一个反例。

> **答：** ① **thinking pattern compatible**（在 student-visited states 上两者高概率 continuation 有重叠）；② **teacher 真的提供新能力**。
> 反例一（只有新能力、不兼容）：换一个完全异构的超强 reasoning teacher，support 几乎不重合，大量 token 拿负 advantage，可能 collapse。
> 反例二（只兼容、无新能力）：Qwen 1.5B ← Qwen 7B 但 teacher 没做额外 post-training，论文的 weak-to-strong reverse distillation 显示两者在相关状态上的分布几乎 indistinguishable，收益很小。

**45.** ⭐ Top-K overlap 的实验发现是什么？

> **答：** 成功 OPD 表现为"初期已有较大 overlap → 训练中 overlap 持续升高"，而且这个**很小的共享 token 集合覆盖 97%–99% 的 probability mass**。
> 结论：OPD 不是靠 teacher 硬拽 student 完全不知道的 token，而是在两者共同认为 plausible 的候选里**重新分配概率**（progressive alignment on high-probability tokens at student-visited states）。这也正是 top-k OPD 合理的原因。

**46.** ⭐ 不 compatible 时的两个 recipe？为什么 cold start 用 off-policy SFT？

> **答：** ① **off-policy cold start**：teacher 先生成 trajectory，做一小段 SFT 把 student 拉进 teacher 的区域、提高 Top-K overlap，再切 OPD；② **teacher-aligned prompt**：让 OPD 的 prompt 尽量贴近 teacher RL 时的 prompt 格式（但过度 in-distribution 会降 entropy，仍需多样性）。
> 用 SFT 是因为它的 **support-covering** 能力正好补上 reverse-KL OPD 缺的那一环：SFT = support expansion，OPD = on-policy refinement。

**47.** MOPD 和普通 OPD 算法上差多少？写出它的 advantage。

> **答：** 几乎没差，只是 teacher 按 domain 切换：$\hat A_{\text{MOPD},t}=\operatorname{sg}[\log\pi_{\phi_d}(y_t)-\log\pi_\theta(y_t)]$，$d$ 是 domain。哪个 domain 的题就用哪个 specialist teacher 监督。

**48.** 为什么不直接 Mix-RL？MOPD 解耦了什么？

> **答：** 各 domain 的 reward 形式、rollout 长度、RL 难度、超参、收敛速度都不同，混在一起互相干扰。
> MOPD 把**能力生产**（各 domain 独立 RL 出 specialist）和**能力集成**（multi-teacher OPD 蒸回统一 student）解耦，而且 merge 的是 **policy behavior 而不是 parameter**。各领域团队可以独立迭代。

**49.** ⭐ 为什么 MOPD 的 teacher 必须从同一个 SFT checkpoint 出发？

> **答：** 因为 $T_d=\pi_0\xrightarrow{\text{domain RL}}\pi_{T_d}$，与 student（也来自 $\pi_0$）天生 **thinking pattern 高度兼容**；同时 domain RL 又赋予了 teacher **真正的新能力**。恰好同时满足 Rethink OPD 的两个成功条件。这也解释了为什么"找一个 benchmark 更强的巨大外部 teacher"未必更好。

## H. 串联题

**50.** ⭐⭐ 把 SFT / 传统 KD / Forward-KL OPD / Reverse-KL OPD 放进一张表。

> **答：**
> | state / prefix | target | 方法 |
> |---|---|---|
> | teacher/data | hard GT token | SFT |
> | teacher/data | teacher soft distribution | 普通 logit KD |
> | **student** | teacher soft distribution | Forward-KL OPD（GKD / OPSD） |
> | **student** | student action + teacher 打分 | Reverse-KL OPD（Thinking Machines） |
>
> 还有一种 **sampled forward-KL OPD**：student state + $a\sim\pi_T$ + $-\log\pi_S(a)$，期望下也是 soft CE。

**51.** ⭐⭐ 写出"两层采样"的一般形式，并说明 forward/reverse OPD 各是什么。

> **答：** $\mathcal L=\mathbb E_{s\sim d^?}\big[\mathbb E_{a\sim ?}[\cdots]\big]$：外层决定 **prefix 谁生成**，内层决定 **KL 期望对谁取**。
> Reverse-KL OPD：$s_t\sim d^{\pi_S},\ a_t\sim\pi_S$（两层都是 student，所以和 RL 天然契合）。
> Forward-KL OPD：$s_t\sim d^{\pi_S},\ a_t\sim\pi_T$（**状态是 student 的，监督方向是 teacher → student**）。

**52.** ⭐ 看到一篇新的蒸馏论文，你先问哪两个问题？

> **答：** ① **prefix 谁生成？**（on-policy / off-policy）② **KL 是 $T\|S$ 还是 $S\|T$（还是 JSD）？** 再补第三个工程问题：**用几个 token 估这个 divergence？**（sampled / top-k / full-vocab）。三个答案定下来，方法基本就定位清楚了。

**53.** ⭐⭐ 一分钟讲清 OPD 和 OPSD。

> **答：** OPD 的核心不是某一种 KL，而是 on-policy trajectory：student 先自己 rollout，teacher 不重新生成答案，而是在 student 实际访问到的每个 prefix 上给 next-token 分布，提供 dense token-level supervision，从而消除 SFT 的 exposure bias。reverse KL 是 student-weighted、mode-seeking，且 $A_t=\log\pi_T-\log\pi_S$ 可直接当 per-token advantage 塞进 RL trainer；forward KL 是 teacher-weighted、mode-covering，更擅长扩 support。
> OPSD 再进一步，把 external teacher 换成 privileged self-teacher：同一个模型，student 只看 question，teacher 额外看到 GT solution，沿 student rollout 给逐 token supervision；teacher stop-gradient 并固定在 initial snapshot，主实验用 full-vocab forward KL + per-entry KL clipping。
