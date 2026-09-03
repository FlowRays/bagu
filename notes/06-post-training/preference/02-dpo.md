# DPO：把 RLHF 变成一个分类问题

> **高频考点**，而且推导很漂亮：从 RLHF 的目标出发，一步步消掉 reward model 和 RL，
> 最后只剩一个二分类 loss。
> 手写实现见 [手撕：DPO 损失](../../code/07-handwrite/03-llm.md#dpo-损失函数)。

## 1. 动机

RLHF 要训 RM、要四个模型、要 rollout、要调 PPO 的一堆超参，非常重。

DPO 的问题是：**能不能跳过 RM 和 RL，直接用偏好数据训 policy？**

## 2. 推导（三步）

### 第一步：写出 KL-regularized RL 的最优解

$$\max_\pi\ \mathbb E_{y\sim\pi}[r(x,y)]-\beta D_{KL}(\pi\|\pi_{\text{ref}})$$

这个问题有**闭式解**：

$$\boxed{\pi^*(y|x)=\frac1{Z(x)}\pi_{\text{ref}}(y|x)\exp\Big(\frac{r(x,y)}{\beta}\Big)}$$

其中 $Z(x)=\sum_y\pi_{\text{ref}}(y|x)e^{r(x,y)/\beta}$ 是配分函数。

（推导：目标等价于 $\min_\pi D_{KL}(\pi\|\pi^*)$，见 [reverse KL 那一节](../distill/04-reverse-kl-as-pg.md#6-更深的一层llm-rl-本来就可以写成-reverse-kl)，KL 在 $\pi=\pi^*$ 时取 0。）

### 第二步：反解出 reward

把上式两边取对数、移项：

$$\boxed{r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\log Z(x)}$$

**这是关键一步**：reward 可以用 policy 和 reference 表示出来。既然如此，就不需要单独训一个 RM 了 —— **policy 自己隐式地就是一个 reward model**。

### 第三步：代入 Bradley-Terry，$Z$ 消掉

$$P(y_w\succ y_l\mid x)=\sigma\big(r(x,y_w)-r(x,y_l)\big)$$

代入第二步的表达式：

$$r_w-r_l=\beta\log\frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\beta\log\frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}+\underbrace{\beta\log Z(x)-\beta\log Z(x)}_{=0}$$

$$\boxed{Z(x)\ \text{只依赖}\ x\text{，在做差时完全消掉}}$$

这就是整个推导最漂亮的地方 —— 那个算不出来的配分函数自己没了。

于是把 $\pi^*$ 换成待优化的 $\pi_\theta$，做最大似然：

$$\boxed{\mathcal L_{\text{DPO}}=-\mathbb E_{(x,y_w,y_l)}\left[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Big)\right]}$$

## 3. 怎么理解这个 loss

记 $h=\beta\big[(\log\pi_\theta(y_w)-\log\pi_{\text{ref}}(y_w))-(\log\pi_\theta(y_l)-\log\pi_{\text{ref}}(y_l))\big]$，则 $\mathcal L=-\log\sigma(h)$。

- $h$ 就是**隐式 reward 的差**
- $h\to+\infty$（模型比 ref 更偏好 $y_w$）→ loss → 0
- $h=0$（没有偏好差）→ loss $=\log 2\approx0.693$
- $h\to-\infty$ → loss 爆炸

梯度：

$$\nabla\mathcal L=-\beta\,\sigma(-h)\Big[\nabla\log\pi_\theta(y_w)-\nabla\log\pi_\theta(y_l)\Big]$$

$\sigma(-h)$ 是一个**自适应权重**：模型已经分对的样本权重接近 0，分错的样本权重接近 1。所以它自动聚焦在还没学会的偏好对上。

$$\boxed{\text{DPO = 提高 }y_w\text{ 的概率、压低 }y_l\text{ 的概率，权重由「当前错得多离谱」决定}}$$

## 4. DPO vs RLHF-PPO

| | DPO | RLHF-PPO |
|---|---|---|
| 需要 RM | **不需要** | 需要 |
| 需要 rollout | **不需要**（离线数据） | 需要 |
| 模型数 | policy + frozen ref（2 个） | actor+critic+ref+RM（4 个）+ rollout |
| 实现难度 | 一个 loss 函数 | 一整套 RL 基建 |
| 数据 | 固定的偏好对 | 可以持续用新采样的数据 |
| 上限 | 受限于离线数据分布 | 更高（能探索） |

$$\boxed{\text{DPO 是 off-policy 的，PPO/GRPO 是 on-policy 的}}$$

这也是 DPO 最核心的局限：它只在给定的 $(y_w,y_l)$ 上学，无法探索数据里没有的更好回答。这和 [SFT vs OPD 的区别](../distill/03-opd.md#5-support-这个词要小心用) 是同一个道理。

## 5. DPO 的已知问题

| 问题 | 说明 |
|---|---|
| **同时压低两者的概率** | 实践中经常观察到 $\log\pi(y_w)$ 和 $\log\pi(y_l)$ **一起下降**，只是 $y_l$ 降得更快。loss 只约束**差值**，不约束绝对值 |
| **分布外失效** | 偏好数据如果不是当前 policy 采出来的，$\pi_\theta$ 在这些 $y$ 上本来概率就低，梯度信号不可靠 |
| **长度偏置** | 和 RM 一样会学到"长 = 好" |
| **对 $\beta$ 敏感** | $\beta$ 小容易跑飞，大了学不动 |

### 常见变体

| 变体 | 改了什么 |
|---|---|
| **IPO** | 把 BT 的 sigmoid 换成平方损失，缓解过拟合到确定性偏好 |
| **KTO** | 只需要「好/坏」的**单边标注**，不需要成对，数据更好收集 |
| **SimPO** | 去掉 reference model，用**长度归一化的平均 logp** 当隐式 reward |
| **ORPO** | 把 SFT loss 和偏好项合成一个，一步到位不需要单独 SFT |
| **Online DPO / 迭代 DPO** | 每轮用当前 policy 采样、标注、再 DPO，把 off-policy 变成近似 on-policy |

**Online DPO 是弥补 DPO 最大短板的方向**：既保留实现简单，又拿回了 on-policy 的好处。

## 6. 实现要点

```python
def dpo_loss(logp_pol_w, logp_pol_l, logp_ref_w, logp_ref_l, beta=0.1):
    z = beta * ((logp_pol_w - logp_pol_l) - (logp_ref_w - logp_ref_l))
    return -F.logsigmoid(z).mean()
```

- 四个 log prob 都是**整条 response 的 token log prob 之和**（不是平均，除非用 SimPO 那种归一化变体）
- ref 那两项要 `detach`（或者干脆在 `no_grad` 下预先算好缓存起来，省一个模型的前向）
- 用 `F.logsigmoid` 而不是 `log(sigmoid(x))`，数值更稳
- $\beta$ 典型值 0.1，lr 比 SFT 还要小（5e-7 ~ 5e-6）

**工程技巧**：ref 的 log prob 可以在训练前**离线算好存下来**，这样训练时只需要一个模型在显存里。

## 自测

1. ⭐ DPO 想解决 RLHF 的什么问题？
2. ⭐⭐ **完整推导 DPO**：写出 KL-regularized RL 的闭式最优解，反解 reward，代入 Bradley-Terry，说明 $Z(x)$ 为什么消掉。
3. ⭐ 为什么说「policy 自己隐式就是一个 reward model」？
4. $h=0$ 时 loss 是多少？梯度里的 $\sigma(-h)$ 起什么作用？
5. ⭐⭐ DPO 和 PPO 在 on/off-policy 上的本质区别？这导致 DPO 的什么局限？
6. ⭐ DPO 「同时压低两者概率」的现象怎么解释？
7. IPO / KTO / SimPO / ORPO / Online DPO 各改了什么？
8. 实现时四个 log prob 是求和还是平均？ref 项怎么处理最省显存？
