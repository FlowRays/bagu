# SFT 与传统 KD：off-policy 那一侧的基准

> 要讲清楚 OPD "新"在哪，必须先把 SFT 和普通 KD 用同一套语言写出来。
> 结论先给：**SFT 和 OPD 最本质的区别不是 KL 方向，而是 state distribution。**

## 1. SFT 的形式：teacher forcing

给定一条 GT / teacher trajectory $y^*=(y_1^*,\dots,y_T^*)$，自回归分解：

$$\pi_\theta(y^*|x)=\prod_t \pi_\theta(y_t^*\mid x,y^*_{<t})$$

取负对数似然：

$$\boxed{L_{\text{SFT}}=-\sum_t\log\pi_\theta\big(y_t^*\mid x,\ \underbrace{y^*_{<t}}_{\text{注意这里}}\big)}$$

最关键的是条件 prefix 是 $y^*_{<t}$，也就是**GT / teacher 自己的正确历史**。训练时永远有人把正确前缀喂给模型，这就是 **teacher forcing**。

## 2. SFT 其实是 one-hot cross entropy

在每个位置上，label 是 one-hot 分布 $q=\delta_{y_t^*}$，于是

$$L_t=-\sum_v q(v)\log\pi_\theta(v|s_t)=-\log\pi_\theta(y_t^*|s_t)$$

再用 [01](01-entropy-ce-kl-jsd.md#3-交叉熵--熵--kl最该背的一条恒等式) 的恒等式：

$$H(q,\pi_\theta)=H(q)+D_{KL}(q\|\pi_\theta),\qquad H(\delta)=0$$

$$\boxed{L_{\text{SFT}}=D_{KL}(\delta_{y^*}\|\pi_\theta)}$$

更一般地，如果数据来自某个分布 $p_{\text{data}}$：

$$L_{\text{SFT}}=\mathbb E_{y\sim p_{\text{data}}}[-\log\pi_S(y)]=H(p_{\text{data}},\pi_S)$$

$$\boxed{\min L_{\text{SFT}}\ \equiv\ \min D_{KL}(p_{\text{data}}\|\pi_S)}$$

所以：**SFT = 在 expert/data states 上做 empirical forward KL。**

## 3. Exposure bias：SFT 真正的病

题目：$17\times6=?$，GT trajectory 是 `17 × 6 = 102`。

SFT 训练时看到的 prefix 永远是：

```text
17 × 6 =        →  要求输出 102
```

但推理时 student 可能自己生成：

```text
17 × 6 = 96, therefore ...
```

它现在处在状态 $s=$ `17 × 6 = 96, therefore`。

这个状态**在 SFT 数据里从来没出现过**。所以模型根本没学过：

> "我前面已经犯错了，现在怎么办？"

$$\boxed{\text{train-test distribution mismatch / exposure bias}}$$

这就是 OPD 要解决的核心问题：**训练分布 = 推理时 student 自己的状态分布。**

## 4. 从 teacher 采样做 SFT = forward KL 的蒙特卡洛估计

这个结论在面试里很加分。

teacher 在某个 state $s$ 有分布 $q(a)=\pi_T(a|s)$。从 teacher 采样 $a\sim q$，然后做普通 SFT $L=-\log p_\theta(a)$，取期望：

$$\mathbb E_{a\sim q}[-\log p_\theta(a)]=-\sum_a q(a)\log p_\theta(a)=H(q,p)=H(q)+D_{KL}(q\|p)$$

$$\boxed{\text{teacher 生成数据 + SFT}\ \approx\ \text{Monte-Carlo forward-KL distillation}}$$

也就是说：

- **用 teacher 的完整 next-token 分布做 logit 蒸馏**
- **从 teacher 分布采样 sequence 再做 SFT**

这两件事**在期望下是同一个目标**，采样只是 teacher distribution 的无偏 MC 估计。区别只在方差和成本。

## 5. 传统 logit KD

把 hard label 换成 teacher 的完整分布，prefix 仍然来自 teacher/dataset：

$$s_t=(x,y^T_{<t}),\qquad L_t=-\sum_v \pi_T(v|s_t)\log\pi_S(v|s_t)$$

即 **off-policy prefix + forward KL soft CE**。SFT 就是它把 $\pi_T(\cdot|s_t)\to\delta_{y^*_t}$ 的 hard-label 特例。

## 6. 三者的坐标

| | prefix / state | target | 等价形式 |
|---|---|---|---|
| SFT | teacher / data | hard GT token | $D_{KL}(\delta_{y^*}\|\pi_S)$ |
| logit KD | teacher / data | teacher 完整分布 | $D_{KL}(\pi_T\|\pi_S)$，soft CE |
| **OPD** | **student rollout** | 取决于 divergence | 见 [03](03-opd.md) |

$$\boxed{\text{SFT 学"老师遇到的状态"；OPD 学"自己真正会遇到的状态"。}}$$

## 7. 面试一句话

> **SFT 是 imitation on expert trajectories，OPD 是 distillation on student trajectories。**
> SFT 教的是"标准答案怎么做"，OPD 教的是"你实际走到这里之后应该怎么做"。

## 自测（口述版）

1. 写出 SFT 的 loss，指出哪一项是 exposure bias 的根源。
2. 证明 one-hot 情况下 $L_{\text{SFT}}=D_{KL}(\delta_{y^*}\|\pi_\theta)$ 是**相等**而不只是等价。
3. 用 $17\times6$ 的例子解释 exposure bias。
4. "从 teacher 采样做 SFT" 和 "用 teacher 分布做 soft CE" 是什么关系？写出期望推导。
5. 传统 logit KD 和 SFT 的差别在哪一个维度上？和 OPD 的差别又在哪一个维度上？
