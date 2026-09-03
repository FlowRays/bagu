# DAPO：把 GRPO 修到能大规模 reasoning RL

> DAPO = **D**ecoupled Clip and Dynamic s**A**mpling **P**olicy **O**ptimization（ByteDance）。
> 注意区分：不是另一篇同名的 Direct Advantage-Based Policy Optimization。
> 官方报告用 Qwen2.5-32B Base 做 reasoning RL，报告比 DeepSeek-R1-Zero-Qwen-32B 更高的 AIME 2024 表现且训练步数更少。

$$\boxed{\text{DAPO}=\text{GRPO}+\text{4 个关键修正}}$$

$$\boxed{\text{Clip-Higher}+\text{Dynamic Sampling}+\text{Token-level Loss}+\text{Overlong Reward Shaping}}$$

DAPO 不是另起炉灶，而是：**真的把 GRPO/R1-style reasoning RL 跑大之后，发现几个具体问题，然后逐个修。**

## 1. Clip-Higher

PPO/GRPO 原本是**对称** clip：$r\in[1-\epsilon,1+\epsilon]$，比如 $[0.8,1.2]$。DAPO 认为 reasoning RL 里这个**上界太保守**。

典型情况：某个正确 reasoning token 在 old policy 下概率极低（$\pi_{old}=0.001$），这次偶然 sample 出来而且答对了（$A>0$）。训练后 $\pi_\theta=0.0012$：

$$r=\frac{0.0012}{0.001}=1.2 \quad\text{已经撞到 clip 上界}$$

也就是说：**这个极其稀有但被验证正确的行为，概率只从 0.1% 提到 0.12%，这批数据就不再继续推它了。**

而 reasoning RL 里最有价值的恰恰是：

$$\boxed{\text{低概率探索}\rightarrow\text{偶然成功}\rightarrow\text{应该强力强化}}$$

所以改成**不对称** clip：

$$\boxed{r\in[1-\epsilon_{low},\ 1+\epsilon_{high}],\qquad \epsilon_{high}>\epsilon_{low}}$$

概念上比如 $[0.8,1.28]$ 而不是 $[0.8,1.2]$。重点不是具体数字，而是：

> **允许正 advantage 的 action 概率往上涨更多，但负 advantage 的下界仍然保守。**

为什么只放宽上界？

- 低概率但正确的新 reasoning pattern 可能是宝贵探索
- 一个坏 trajectory 被过度压低，一般没那么值得冒险

$$\text{PPO：正负两个方向同样保守} \quad\Rightarrow\quad \text{DAPO：正方向稍微大胆，负方向继续保守}$$

⚠️ 注意 DAPO **仍然保留 `min` 逻辑**，只是 clip 区间从对称变成不对称：$\mathrm{clip}(r,1-\epsilon_{low},1+\epsilon_{high})$。这不是新算法结构，就是改 PPO 的 proximal constraint。

一句话：**PPO/GRPO 说"别走太远"；Clip-Higher 说"如果你发现了一个稀有但正确的 reasoning，可以允许往这个方向多走一点"。**

## 2. Dynamic Sampling

解决 GRPO 的浪费问题。同一 prompt 一组 rollout 如果 reward 是 $[1,1,1,1]$ 或 $[0,0,0,0]$，组内标准化后：

$$A_i=0 \quad\text{基本没有学习信号}$$

DAPO 动态筛掉这类"全对/全错"的 prompt，优先保留：

$$\boxed{0<\text{success rate}<1}$$

比如 $[1,1,0,0]$ 最有价值，因为能直接比较"哪些 rollout 比同题其他 rollout 更好"。

$$\boxed{\text{Dynamic Sampling}=\text{少浪费 rollout 在全会/全不会的题上}}$$

本质是提高 $\boxed{\text{有效 rollout}/\text{总 rollout}}$ 的比例。直觉：**不要浪费 rollout compute 在已经全会或者完全不会的问题上。**

### 卡点 17：A=0 等于没训练吗

> **常见卡点**："A=0 是不是训练和不训练没有任何区别？这样 rollout 的数据虽然浪费了 rollout 的步骤，但是节省了参数更新的时间？"

基本正确，但要加两个修正。

若组内 reward 全一样，$A_i=0$，则 policy loss 项 $r_iA_i=0$，clip 后也是 0：

$$\boxed{\nabla_\theta L_{policy}=0}$$

所以从 **policy update 角度**，训练它和跳过它几乎没区别。

**修正一**：真正浪费的**不是 backward**，而是前面的：

- 生成这些 response 的推理算力
- reward / verifier 计算
- logprob forward、数据搬运

**修正二**：如果训练目标里还有 **KL、entropy 或其他辅助 loss**，即使 $A=0$，这些样本仍可能产生梯度。所以"完全等价于不训练"只对**纯 policy-gradient 那部分**严格成立。

因此 Dynamic Sampling 的主要价值不是省 backward，而是：

$$\boxed{\text{避免先花昂贵 rollout 算力采到一堆最终零梯度的数据}}$$

## 3. Token-level Policy Gradient Loss

### 卡点 18：一个 batch 内的 loss 怎么算

> **常见卡点**："怎么理解一个 batch 内的 loss 计算这件事，就是 GRPO 的时候？"

把一个 batch 想成**两层**：$\boxed{\text{prompt 层}}$ 和 $\boxed{\text{response/token 层}}$。

比如 2 个 prompt、每个采 3 条 response：

$$x_1\to y_{11},y_{12},y_{13} \qquad x_2\to y_{21},y_{22},y_{23}$$

每条 response 又有很多 token：$y_{ij}=(y_{ij,1},\dots,y_{ij,T_{ij}})$。

**第一步**：在**每个 prompt 的 group 内**算 baseline 和 advantage。比如 $x_1: R=[1,1,0]$ 得到 $A_{11},A_{12},A_{13}$。

**第二步**：一条 response 的所有 token 共用同一个 $A_{ij}$，但每个 token 有自己的 ratio，于是得到很多 **token loss**：

$$\ell_{ij,t}=\min\big(r_{ij,t}A_{ij},\ \mathrm{clip}(r_{ij,t})A_{ij}\big)$$

**第三步（关键分歧点）**：怎么把这些 token loss 聚成一个标量 batch loss？

**GRPO 常见做法**：先在每条 response 内部平均，再对所有 response 平均：

$$L_{ij}=\frac{1}{T_{ij}}\sum_{t=1}^{T_{ij}}\ell_{ij,t} \qquad\Rightarrow\qquad \boxed{L_{batch}=\frac{1}{N_{resp}}\sum_{i,j}L_{ij}}$$

> **每条 response 先自己算一个平均表现，再让所有 response 等权投票**，这就是 sequence-level averaging。

**DAPO 改成**：直接对整个 batch 的所有有效 token 一起平均：

$$\boxed{L_{batch}=\frac{\sum_{i,j,t}\ell_{ij,t}}{\sum_{i,j}T_{ij}}}$$

### 具体例子看清区别

response A 有 2 个 token，$\ell=[1,1]$ → $L_A=1$
response B 有 10 个 token，$\ell=[1,\dots,1]$ → $L_B=1$

| | batch loss | A:B 权重 |
|---|---|---|
| GRPO | $\dfrac{L_A+L_B}{2}=1$ | $1:1$（虽然 B 有 5 倍 token） |
| DAPO | $\dfrac{2+10}{12}$ | $2:10$，即 B 的影响约是 A 的 **5 倍** |

$$\boxed{\text{GRPO：每条 response 等权}} \qquad \boxed{\text{DAPO：每个 token 等权}}$$

> 把"batch loss"理解成 **"反向传播前，要把这一批所有样本产生的很多小 loss 聚合成一个标量"**，这件事就很自然了。

### GRPO 和 DAPO 哪个倾向生成更长的回答

**通常 DAPO 更容易鼓励长回答。**

GRPO 先做 sequence 内平均，无论 50 还是 500 token，整条 response 总权重大致一样，长回答不会因 token 更多获得更多总梯度。DAPO 全 batch token-level average 后，长 response 包含更多 token 就贡献更多 gradient；一条长 reasoning 若是正 advantage，会有更多 token 被强化。

$$\boxed{\text{DAPO 比 GRPO 更偏向保留/强化长 reasoning}}$$

但**不是"直接奖励长度"**：如果长回答 reward 不高，或触发 overlong penalty，一样会被抑制。准确说法是：

> GRPO 对每条 sequence 更等权；DAPO 对每个 token 更等权，因此长的**高质量** response 相对更容易获得更大的总训练影响。

### DAPO 这种方式好吗

在 reasoning RL 里 token-level aggregation **往往更合理，但不是无条件更好**。

| 优点 | 副作用 |
|---|---|
| 每个 token 权重一致，长短回答不会在总梯度里被强行拉平 | 长回答天然占更多权重 |
| 一条高质量长 reasoning 贡献更多训练信号，适合需要长 CoT 的任务 | reward 设计不好时容易被长序列主导，出现"越写越长"倾向 |

所以 DAPO 通常需要配合 overlong penalty、长度控制、dynamic sampling 一起用。

> 哪种更好，取决于你想让**什么成为基本训练单位**。长链 reasoning → DAPO 更自然；不希望长度影响训练权重 → GRPO 式 sequence averaging 更稳。

## 4. Overlong Reward Shaping

reasoning RL 会设最大生成长度 $T_{\max}$（比如 8k）。超过就截断，最粗暴的做法是 $R=0$。问题是 reward 在长度边界处**突然跳变**：7999 token 时 $R=1$，8001 token 时 $R=0$，制造非常不连续的训练信号。

### Soft Overlong Punishment（具体公式）

设 $L_{\max}$ 为期望最大长度、$L_{cache}$ 为软惩罚缓冲区、$|y|$ 为实际生成长度：

$$R_{length}(y)=\begin{cases}
0, & |y|\le L_{\max}-L_{cache}\\[8pt]
\dfrac{(L_{\max}-L_{cache})-|y|}{L_{cache}}, & L_{\max}-L_{cache}<|y|\le L_{\max}\\[10pt]
-1, & |y|>L_{\max}
\end{cases}$$

$$\boxed{R_{total}=R_{correct}+R_{length}}$$

三段直觉：安全区完全不罚 → 软惩罚区随长度线性从 0 降到 $-1$ → 超过 $L_{\max}$ 直接 $-1$。

### 论文实际设置

$$L_{\max}=16384,\qquad L_{cache}=4096,\qquad \text{实际 generation 上限}=20480$$

对应区间：

- $0\sim12288$：不罚
- $12288\sim16384$：线性罚 $0\to-1$
- $>16384$：$-1$

**算例**：生成 $|y|=14336$（刚好软区一半）：

$$R_{length}=\frac{12288-14336}{4096}=-0.5$$

若 $R_{correct}=1$，则 $R_{total}=0.5$。也就是"答对了，但太啰嗦，所以不给满分"。若生成到 16384，$R_{length}=-1$，即使答对最终也是 0。

### 演化过程

$$\text{粗暴惩罚 truncated sample} \rightarrow \text{reward noise 很大}$$
$$\downarrow$$
$$\text{Overlong Filtering：直接把截断样本的 loss mask 掉，不训练} \rightarrow \text{训练明显更稳}$$
$$\downarrow$$
$$\boxed{\text{Soft Overlong Punishment：用长度相关的连续 penalty}}$$

这也解释了为什么叫 **reward shaping**：不是简单判断"超长/不超长"，而是把长度接近上限时的 reward 做成一个**连续坡度**。目的有二：避免 reward discontinuity；抑制模型无限拉长 reasoning。

## 总结

> **DAPO 不是推翻 GRPO，而是把 GRPO 在大规模 reasoning RL 里几个很实际的训练问题逐个修掉。**

## 自测

**1.** DAPO 全称是什么？四个改动分别是什么？

> **答：** **D**ecoupled clip **a**nd dynamic s**a**mpling **P**olicy **O**ptimization。
> 四个改动：① **Clip-Higher**（上下界解耦，放宽上界）；② **Dynamic Sampling**（过滤掉组内 reward 全同的 prompt）；③ **Token-level loss 聚合**（按 token 而不是按 response 平均）；④ **Soft Overlong Punishment**（对超长回答用连续的软惩罚）。

**2.** Clip-Higher 改了什么？为什么只放宽上界？举那个 $0.001\to0.0012$ 的例子。

> **答：** 把 $\text{clip}(r,1-\epsilon,1+\epsilon)$ 改成上下界解耦的 $\text{clip}(r,1-\epsilon_{\text{low}},1+\epsilon_{\text{high}})$，并把 $\epsilon_{\text{high}}$ 放大（如 0.28）。
> 只放宽上界是为了**保护低概率 token 的探索**：一个概率 0.001 的 token 即使 $r$ 顶到上界 1.2，也只能涨到 0.0012，涨幅微乎其微；而高概率 token（如 0.9）同样的 $r$ 就能涨很多。统一的上界实际上**不成比例地压制了低概率 token**，导致熵快速塌缩、模型不再探索。放宽上界给低概率 token 更大的成长空间。

**3.** Dynamic Sampling 解决什么问题？$A=0$ 时到底浪费的是什么？

> **答：** 解决**组内 reward 全相同导致 advantage 全为 0** 的问题（简单题全对、难题全错）。
> $A=0$ 时梯度确实为 0，「训练和不训练没区别」这句话就**梯度而言是对的**；但浪费的是**这批 rollout 的采样算力** —— 你花了 $G$ 次生成的成本，却没换来任何学习信号。而 rollout 恰恰是 RL 里最贵的一环。
> DAPO 的做法是持续采样直到凑够足够多「组内有差异」的 prompt 再进入训练。

**4.** 手算：response A 有 2 个 token、B 有 10 个 token，GRPO 和 DAPO 的 batch loss 分别怎么算？权重比各是多少？

> **答：** 设每个 token 的 loss 都是 1。
> **GRPO（sample-level）**：先对每条 response 内部求平均，$L_A=1$、$L_B=1$，再对两条平均 → $L=1$。权重比 **A : B = 1 : 1**，也就是每个 token 的实际权重 A 是 B 的 5 倍。
> **DAPO（token-level）**：所有 token 放在一起除以总 token 数，$L=\frac{2\times1+10\times1}{12}=1$。权重比 **A : B = 2 : 10 = 1 : 5**，每个 token 权重相同。

**5.** GRPO 和 DAPO 哪个倾向生成更长的回答？为什么？这是"直接奖励长度"吗？

> **答：** **DAPO** 更倾向长回答。因为 token-level 聚合下长回答贡献的 token 多、占的总权重大，梯度更容易被长序列主导。
> 但**不是「直接奖励长度」**：如果长回答本身 reward 不高，或触发了 overlong punishment，一样会被压制。准确说法是「**长的高质量 response 相对更容易获得更大的总训练影响**」。所以 DAPO 需要配合 overlong penalty、长度控制、dynamic sampling 一起用。

**6.** 默写 Soft Overlong Punishment 的三段公式，并算 $|y|=14336$ 时的 $R_{length}$。

> **答：** 设 $L_{\max}=16384$、$L_{\text{cache}}=4096$，安全区上界是 $L_{\max}-L_{\text{cache}}=12288$：
> $$R_{\text{length}}(|y|)=\begin{cases}0 & |y|\le L_{\max}-L_{\text{cache}}\\[4pt] \dfrac{(L_{\max}-L_{\text{cache}})-|y|}{L_{\text{cache}}} & L_{\max}-L_{\text{cache}}<|y|\le L_{\max}\\[6pt] -1 & |y|>L_{\max}\end{cases}$$
> $|y|=14336$ 落在中间段：$\frac{12288-14336}{4096}=\frac{-2048}{4096}=\mathbf{-0.5}$。
> 三段直觉：安全区完全不罚 → 软惩罚区随长度线性从 0 降到 $-1$ → 超过 $L_{\max}$ 直接 $-1$。演进路径是「粗暴给截断样本 $R=0$ → Overlong Filtering 直接 mask 掉 → 长度相关的连续 soft penalty」。

