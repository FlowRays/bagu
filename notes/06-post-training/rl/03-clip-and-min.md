# PPO 的 clip 与 min：四象限

> 承接 [02](02-importance-sampling-and-ratio.md)，此时已有 $L=\mathbb E_{old}[rA]$。这一篇解决三个卡点：为什么 clip、clip 怎么让参数停下来、为什么必须有 min。

$$\boxed{L^{CLIP}(\theta)=\mathbb E_t\Big[\min\big(r_tA_t,\ \mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t\big)\Big]}$$

通常 $\epsilon=0.2$，即 $r\in[0.8,1.2]$。

## 1. 为什么要 clip

只优化 $rA$ 有个新问题：**它会不停把概率往极端推。**

设 $A=+1$，那么 $L=rA=r$，为了最大化它，optimizer 当然希望 $r$ 越大越好：

$$r:\ 1\to1.2\to2\to5\to\dots$$

目标一直变大，**完全没有告诉 optimizer "差不多了，别再推了"**。

但这批数据始终来自 $\pi_{old}$。当 $\pi_\theta$ 已经离 $\pi_{old}$ 很远时，这批旧数据越来越不能代表当前 policy，继续榨它就很危险。

所以要分清两层职责：

| 机制 | 解决什么 |
|---|---|
| $r$（importance ratio） | old data **如何**用于 current policy（分布修正） |
| clip | 就算能用，也**不能无限榨**同一批 old data（更新幅度） |

> **clip 不是在修正 importance sampling**，IS 已经由 $r$ 完成了。clip 是第二层：限制 policy 在一批 old data 上更新得太远。这就是 PPO 里 "Proximal（近端）"的含义：**新 policy 要待在 old policy 附近。**

### 一个重要误区

PPO **并没有**硬约束 $r\le1.2$。更新后 $r=2$ 完全可能。它只是让"超出有害方向后，objective 不再提供继续推远的梯度收益"：

$$\boxed{\text{objective clipping}} \ne \boxed{\text{parameter hard constraint}}$$

（TRPO 才是真的加约束 $D_{KL}(\pi_{old}\|\pi_\theta)\le\delta$；PPO 是用 clipping 简单近似 trust-region 的效果。）

## 卡点 5：clip 到底怎么让参数停止更新

> **常见卡点**："加了 clip 之后，我又不理解 L 到 gradient 到 update theta 的关系了。"
>
> **正确的理解**："clip 的目的是让 π 相比 old π 变化超过一定范围之后就不更新参数了，而不更新参数的方式就是让 loss 在那个范围外保持不变，而保持不变就是让 r 变成常量了，也就是 loss 里面没有 theta 了。" ✅

精确版本，以 $A>0$ 为例：

**$r=1.1$（范围内）**：$\mathrm{clip}(r)=r$，所以 $L=rA$，仍然依赖 $\theta$：

$$\nabla_\theta L=rA\nabla_\theta\log\pi \ne 0 \quad\Rightarrow\quad \text{继续更新}$$

**$r=1.5$（超出）**：$\mathrm{clip}(r)=1.2$，所以 $L=1.2A$，这是一个**与 $\theta$ 无关的常数**（$A$ 在 rollout 后就固定了）：

$$\boxed{\nabla_\theta L=0} \quad\Rightarrow\quad \text{这个 sample 不再推动参数}$$

链条是：

$$\boxed{\text{clip} \rightarrow \text{改变 } L \text{ 的形状} \rightarrow \text{某些区域梯度为 0} \rightarrow \text{停止更新 } \theta}$$

一维图像最直观。没有 clip 时 $L(r)=Ar$ 是一条一直往上的直线，$dL/dr=A>0$ 永远继续推：

```text
L
|        /
|      /
|    /
|  /
+---------- r
```

加 clip 后 $L(r)=A\min(r,1.2)$，右边变平：

```text
L
|       ________
|      /
|    /
|  /
+---------- r
       1.2
```

通过链式法则 $\dfrac{\partial L}{\partial\theta}=\dfrac{\partial L}{\partial r}\cdot\dfrac{\partial r}{\partial\theta}$，当 $\partial L/\partial r=0$ 时自然 $\partial L/\partial\theta=0$。

> 一句话：**PPO 不是 clip 参数，也不是 clip gradient，而是 clip surrogate objective 对 ratio 的依赖，让某些方向的 loss 变平，从而梯度为 0。**

## 3. 为什么必须有 min

> **常见卡点**："既然要 clip，直接写 $\mathrm{clip}(r)A$ 不就完了吗？"

不行。因为我们**只想阻止"在有利方向上走太远"，不想阻止模型纠正错误**。

关键反例：$A>0$（好动作），但 $r=0.5$，说明模型居然把好动作的概率降了一半，这是**错误方向**，应该继续纠正。

如果直接用 $\mathrm{clip}(r)A$：$\mathrm{clip}(0.5)=0.8$，loss 会把 0.5 假装成 0.8，**掩盖了错误有多严重**，削弱纠错梯度。

所以 PPO 同时算两个值再取 min，选那个**更保守、对 policy 改进更悲观**的估计：

- $A>0,\ r=0.5$：$\min(0.5A,\ 0.8A)=0.5A$ → **不 clip**，保留梯度继续纠正 ✅
- $A>0,\ r=1.5$：$\min(1.5A,\ 1.2A)=1.2A$ → **clip**，变平停止 ✅

## 4. 四象限（PPO 最核心的一张表）

$\epsilon=0.2$，范围 $[0.8,1.2]$。

| # | 情况 | 含义 | 数值验算（$A=\pm1$） | 结果 |
|---|---|---|---|---|
| ① | $A>0,\ r>1+\epsilon$ | 好动作，已提太多 | $\min(1.5,\ 1.2)=1.2$ | **clip**，梯度 0 |
| ② | $A>0,\ r<1-\epsilon$ | 好动作，反而降了概率（错方向） | $\min(0.5,\ 0.8)=0.5$ | 不 clip，继续纠正 |
| ③ | $A<0,\ r<1-\epsilon$ | 坏动作，已降太多 | $\min(-0.5,\ -0.8)=-0.8$ | **clip**，梯度 0 |
| ④ | $A<0,\ r>1+\epsilon$ | 坏动作，反而提了概率（错方向） | $\min(-1.5,\ -1.2)=-1.5$ | 不 clip，继续纠正 |

> ③④ 容易绕晕，注意 $A<0$ 时大小关系反过来：$-0.8 < -0.5$，所以 $\min$ 选到的是 clipped 分支。

**所以 PPO 只 clip 两种情况**：

$$\boxed{A>0,\ r>1+\epsilon} \qquad \boxed{A<0,\ r<1-\epsilon}$$

也就是 **"朝正确方向走太远"**。其余情况都保留梯度。

两句话记忆：

$$A>0:\ r \text{ 太大才 clip} \qquad\qquad A<0:\ r \text{ 太小才 clip}$$

"根据 $A$ 的正负自动决定该截哪一边"，正是 `min` 巧妙实现的。

## 5. min 的另一种理解：pessimistic surrogate

PPO 同时看 $rA$ 和 $\mathrm{clip}(r)A$，永远选**更差、更保守**的那个 improvement estimate。目的是：

> 不让 optimizer 因为 policy 已经大幅偏离 old policy，而获得虚假的额外收益。

## 6. 完整直觉

$$\boxed{\text{好动作：提高概率，但别提高太多}}$$
$$\boxed{\text{坏动作：降低概率，但别降低太多}}$$

更精确的一句：

> **如果你正在朝正确方向改变，但已经相对 old policy 改得太多，就停止从这个 sample 获得更多梯度；如果你朝错误方向走，则继续纠正。**

注意这**不是** "ratio 超范围就不更新"，而是"只有正确方向走太远才停止更新"。

## 7. 别混淆：三种 clip

| 名字 | 对象 | 作用 |
|---|---|---|
| PPO ratio clip | $r_t$ | 限制 surrogate objective，间接限制 policy 更新幅度 |
| gradient clipping | $\|g\|\le c$（如 `max_grad_norm=0.5`） | 防止 optimizer 的 gradient norm 爆炸 |
| value clipping | $V_\theta - V_{old}$ | critic 不要一次更新太猛（各实现差异大，非核心） |

## 一次完整 PPO update 的链条

$$\boxed{\theta \rightarrow \pi_\theta \rightarrow r \rightarrow L \rightarrow \nabla_\theta L \rightarrow \theta'}$$

- $\theta$：网络参数
- $\pi_\theta$：action 概率
- $r$：这个概率相对 old policy 变了多少
- $A$：这个概率应该升还是降
- $L$：把"升/降 + 不要走太远"编码成一个可微目标
- $\nabla L$：告诉 optimizer 往哪走
- $\theta'$：policy 真正改变

代码里一般写成最小化 $-L$ 再做 gradient descent，本质一样。

## 自测

**1.** clip 和 importance ratio 各自解决什么问题？

> **答：** **importance ratio** 解决「数据来自旧分布」的**分布修正**问题，是第一性的。
> **clip** 是顺手利用这个 ratio 来限制**单步更新幅度** —— ratio 偏离 1 太多说明 policy 已经走远，surrogate 不再可信，就把梯度收益截断掉。
> 两者不是一回事：ratio 是必需的数学修正，clip 是可选的 proximal 机制（PPO-Penalty 用 KL 也能达到同样目的）。

**2.** PPO 有没有硬约束 $r\le 1.2$？更新后 $r=2$ 可能吗？

> **答：** **没有硬约束**，$r=2$ 完全可能。
> PPO 是 **objective clipping** 而不是 parameter hard constraint：超出区间后只是不再提供继续推远的**梯度收益**，但并不阻止参数走到那里（比如一次大的 minibatch 更新，或者别的 token 的梯度把它带过去）。
> TRPO 才是真的加约束 $D_{KL}(\pi_{\text{old}}\|\pi_\theta)\le\delta$。

**3.** clip 是怎么让参数停止更新的？说出完整链条。

> **答：** 链条：**$r$ 超出区间 → `clip(r)` 对 $r$ 的导数为 0 → 该项 loss 变成常数（对 $\theta$ 的梯度为 0）→ 这个 token 不再贡献梯度 → 参数不再因它而更新**。
> 注意是「这个样本不再贡献梯度」，不是「参数被锁住」，别的样本照样推动参数。

**4.** 为什么不能只写 $\mathrm{clip}(r)A$？举一个具体反例。

> **答：** 只写 clip 会在「坏方向」上失去约束。反例：$A=-1$（坏 action），$r=0.5$（新 policy 已经把它压得很低）。
> 此时 $\text{clip}(0.5,0.8,1.2)=0.8$，$\text{clip}(r)A=-0.8$；而 $rA=-0.5$。取 $\min$ 得 $-0.8$。
> 如果只用 $\text{clip}(r)A=-0.8$，等于在已经压够了的方向上**继续给激励**。$\min$ 保证取「更悲观」的一侧，让已经走对方向的样本不再被过度推。

**5.** 默写四象限表，并对 $A=-1$、$r=0.5$ 和 $r=1.5$ 手算 min。

> **答：** 四象限：$A>0$ 时上界被 clip 住（不许过度提高），$A<0$ 时下界被 clip 住（不许过度压低）；在「回到区间内」的方向上不设限。
> $A=-1,r=0.5$：$rA=-0.5$，$\text{clip}(r)A=0.8\times(-1)=-0.8$，$\min=-0.8$（**clip 生效**）。
> $A=-1,r=1.5$：$rA=-1.5$，$\text{clip}(r)A=1.2\times(-1)=-1.2$，$\min=-1.5$（**取未裁剪的**，因为它更悲观 —— 这个方向要继续惩罚）。

**6.** PPO ratio clip 和 gradient clipping 有什么区别？

> **答：** **PPO ratio clip** 作用在 **objective** 上，裁的是 importance ratio，目的是限制 policy 相对 $\pi_{\text{old}}$ 的变化幅度，是算法的一部分。
> **gradient clipping** 作用在 **梯度** 上，按 global L2 norm 等比例缩放，目的是防止偶发的梯度爆炸，是通用的训练保险丝，和 RL 无关。
> 两者完全正交，通常同时使用。

