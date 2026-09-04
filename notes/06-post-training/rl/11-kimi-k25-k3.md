# Kimi K2.5 / K3：一条 production RL recipe 里改了 GRPO 的哪八处

> K3 技术报告直接说：**policy optimization follows the algorithm in Kimi K2.5**。所以真正要看的是 K2.5 的 objective，K3 在上面加了 partial rollout、reasoning budget、9 experts 和 MOPD。
> 这篇的重点不是背公式，而是**从"它要解决什么问题"反推出每一处改动**。

## 0. 基线：原版 GRPO 是什么

对一个 prompt $x$，旧策略生成 $G$ 条 response $y_1,\dots,y_G\sim\pi_{old}$，得到 sequence reward $r_1,\dots,r_G$：

$$A_i=\frac{r_i-\bar r}{\sigma_r+\epsilon},\qquad \rho_{i,t}=\frac{\pi_\theta(y_{i,t}|s_{i,t})}{\pi_{old}(y_{i,t}|s_{i,t})}$$

$$J_{GRPO}=\frac1G\sum_i\frac1{|y_i|}\sum_t\min\big[\rho_{i,t}A_i,\ \mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i\big]-\beta D_{KL}(\pi_\theta\|\pi_{ref})$$

记成四件事：

$$\boxed{\text{std-normalized group advantage}+\text{sequence-level averaging}+\text{PPO clipping}+\text{reference KL}}$$

**Kimi 四处都动了。**

## 1. 改动一：去掉 reward 的 std normalization

$$A_i^{GRPO}=\frac{r_i-\bar r}{\sigma_r}\qquad\Longrightarrow\qquad\boxed{A_i^{Kimi}=r_i-\bar r}$$

### 为什么

std normalization 不只是"做个归一化"，它会**改变不同 prompt 之间的相对训练权重**。

reward 是 0/1，一组结果 $[1,1,1,0]$，$\bar r=0.75$：

- 不除 std：$A_{correct}=0.25$，$A_{wrong}=-0.75$，天然就在 $[-1,1]$ 附近
- 除以 $\sigma_r$：不同 pass rate 的题被重新缩放。**$\sigma_r\to0$ 时（几乎全对或几乎全错），再小的 reward 差异也会被放大**

### 影响

$$\text{GRPO：每个 prompt 被 std 强行重新定标}\quad\Longrightarrow\quad\text{Kimi：reward gap 本身决定梯度强弱}$$

代价：不同 reward scale 的任务混在一起时，需要 reward 本身设计得比较可控。

## 2. 改动二：sequence-level averaging → token-level loss

$$J_{GRPO}=\frac1G\sum_i\boxed{\frac1{|y_i|}}\sum_t L_{i,t}\qquad\Longrightarrow\qquad J_{Kimi}=\boxed{\frac1N}\sum_i\sum_t L_{i,t},\quad N=\sum_i|y_i|$$

也就是 $\boxed{\text{每个 token 等权}}$，形式上和 [DAPO 的 token-level loss](08-dapo.md#3-token-level-policy-gradient-loss) 一致。

### 为什么

GRPO 的 $\frac1{|y_i|}$ 是**先在每条 response 内部平均**，所以 $|y_1|=100$ 和 $|y_2|=10000$ 两条在最终 loss 里权重差不多。反过来看单个 token：

$$|y|=100\ \Rightarrow\ \text{每 token }\tfrac1{100};\qquad |y|=100000\ \Rightarrow\ \text{每 token }\tfrac1{100000}$$

差了 1000 倍。也就是：

> trajectory 越长，每一个 decision token 的梯度反而被稀释得越严重。

普通数学题还好，但 K3 的 agent trajectory 有几百上千次 tool call、上百万 context，这就非常不自然。改成 token-level 后：

$$\boxed{\text{一个 decision token 就是一个训练单位}}$$

### 影响（必须一起记的副作用）

两条 trajectory 都是 $A=+1$，一条 100 token、一条 1000 token，则第二条的总梯度贡献约为第一条的 **10 倍**：

$$\boxed{\text{长 response 获得更多总训练权重}}$$

所以 **token-level loss 必须和 length / budget control 一起看**，否则会产生

$$\text{long response}\rightarrow\text{更多 positive token}\rightarrow\text{更多梯度}$$

的长度偏置。这正是后面第 6 条 reasoning budget 的动机之一。

## 3. 改动三：PPO clipping → 与 advantage 符号无关的 off-policy masking

### 先看 PPO 在防什么

$$\min[\rho_tA,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)A]$$

PPO 的 clip **和 $A$ 的正负绑定**：$A>0$ 时主要防 $\rho_t\gg1$（涨过头），$A<0$ 时主要防 $\rho_t\ll1$（跌过头）。核心思想是：

$$\boxed{\text{别沿着当前 advantage 方向更新过头}}$$

### Kimi 要解决的是另一个问题

Kimi 遇到的不是"policy update 太大"，而是：

> **这条 token 数据本身已经太 stale / 太 off-policy，我根本不相信它了。**

所以它做 token-level masking，而且**与 advantage 正负无关**：

$$m_t=\mathbf 1\big[\alpha\le\rho_t\le\beta\big],\qquad \boxed{J_{PG}=\frac1N\sum_t m_t\,\rho_t\,A_t}$$

ratio 在允许区间内就正常算 PG；超出区间，**直接把这个 token 的 policy gradient 置零**。

### 为什么

$\pi_{rollout}$ 和 $\pi_\theta$ 差太远时，importance sampling 本身的方差就非常大。$\rho_t=20$ 意味着这个 token 在当前 policy 下的概率已经是 rollout 时的 20 倍，$\rho_tA$ 会大得离谱。Kimi 干脆：

$$\boxed{\text{太 off-policy}\rightarrow\text{这个 token 不学}}$$

### 影响

$$\text{PPO 问：这个方向还能不能继续更新？}\qquad \text{Kimi 问：这条数据还可不可信？}$$

$$\boxed{\text{更强的稳定性}\ \longleftrightarrow\ \text{丢掉一部分 stale sample 的梯度}}$$

官方说这对 long-horizon multi-step tool-use 的训练稳定性非常关键。

## 4. 改动四：reference KL → squared log-ratio 正则

原版：$-\beta D_{KL}(\pi_\theta\|\pi_{ref})$，$\pi_{ref}$ 通常是冻结的 pretrained/SFT 模型，目的是"别 RL 得离原始模型太远"。

Kimi 换成：

$$\boxed{-\tau\Big(\log\frac{\pi_\theta(y_t|s_t)}{\pi_{old}(y_t|s_t)}\Big)^2}=-\tau(\log\rho_t)^2$$

### 为什么换：两者根本在解决不同问题

| | 约束谁和谁 | 解决什么 |
|---|---|---|
| GRPO reference KL | $\pi_\theta\leftrightarrow\pi_{ref}$ | 别 RL 得太离谱、别忘掉原始能力 |
| Kimi $(\log\rho)^2$ | $\pi_\theta\leftrightarrow\pi_{old}$ | **当前 policy 别迅速脱离产生这批 trajectory 的行为策略** |

后者就是 $\boxed{\text{off-policy stability}}$，对 partial rollout 尤其重要。

### 数学效果

$\rho_t=1\Rightarrow$ penalty $=0$；$\rho_t=e^2$ 和 $\rho_t=e^{-2}$ 都得到 $(\log\rho_t)^2=4$ —— **在 log-prob space 里是对称的**：

$$\boxed{\text{涨太多和跌太多都罚}}$$

梯度 $\frac{\partial}{\partial\log\pi_\theta}\tau(\log\rho_t)^2=2\tau\log\rho_t$，离得越远拉回去的力越强。

## 5. 为什么既要 mask 又要 $(\log\rho)^2$

因为两者的作用区域不同：

```text
ρ 接近 1
    ↓
正常 PG + 很小的 regularization

ρ 开始偏离 1
    ↓
regularization 越来越强        ← (log ρ)² 软约束

ρ 偏离得非常离谱
    ↓
PG 直接被 mask 掉              ← ratio mask 硬保险
```

$$\boxed{(\log\rho)^2=\text{软约束}}\qquad+\qquad\boxed{\text{ratio mask}=\text{硬保险}}$$

一软一硬，很合理的组合。

## 6. 到这里 Kimi 的 RL objective 可以写成

$$\boxed{J_{Kimi}=\frac1N\sum_{i,t}\Big[M(\rho_{i,t})\,\rho_{i,t}\,(r_i-\bar r)-\tau\big(\log\rho_{i,t}\big)^2\Big]},\qquad N=\sum_i|y_i|$$

$M(\rho)$ 表示：ratio 合理 $\Rightarrow1$，太 off-policy $\Rightarrow0$。这是理解 K2.5/K3 RL optimizer 最干净的形式。

三个部件：

$$\boxed{A_j=r_j-\bar r}\quad+\quad\boxed{\rho_t\text{ 的 mask}}\quad+\quad\boxed{(\log\rho_t)^2}$$

## 7. 改动五：同步 rollout → Partial Rollout

这不是 loss 公式的改动，但**它正是前面两个 off-policy 技术变得必要的原因**。

原 GRPO：$N_pK$ 条 rollout 全部跑完 → 算 group reward → train。

Agent rollout 的长度长尾极其严重：

```text
1000 条 trajectory：
  绝大多数： 2 min
  少部分：  10 min
  最慢：    50 min
```

同步 GRPO $\Rightarrow$ $\boxed{\text{所有 GPU 等最慢那几条}}$，大量 idle。

**Partial rollout**：完成 $\lambda N_pK$ 条就开始训练，剩下 $(1-\lambda)N_pK$ 条暂停，下一 iteration 接着 resume。

$$\boxed{\text{不等 straggler，先更新}}$$

## 8. 但它制造了 GRPO 原本没这么严重的问题

一条 trajectory 在 $\pi_0$ 下生成了一半，模型更新到 $\pi_1$，下一 iteration 接着 rollout，可能又到 $\pi_2$。最后整条 trajectory 跨了好几个 policy version：

$$\boxed{\pi_{behavior}\ne\pi_{current}}\quad\text{而且可能差得非常远}$$

这就是 $\boxed{\text{data staleness}}$。所以逻辑链是：

$$\boxed{\text{partial rollout}\rightarrow\text{stale data}\rightarrow\text{ratio mask}+(\log\rho)^2}$$

**这三件事必须一起理解，单看任何一个都会觉得"为什么要搞这么多花样"。**

## 9. 改动六：普通 task reward → 加 reasoning budget

原始 $r(y)=\mathbf 1[\text{correct}]$ 没有告诉模型"你用了多少 token"。如果"思考更久 $\Rightarrow P(\text{correct})\uparrow$"，RL 自然会学出 $\boxed{\text{能想多久就想多久}}$。

做法：先估计任务的 baseline budget $b_0(x)$，乘以 effort multiplier $\gamma$：

$$B(x)=\gamma\, b_0(x),\qquad T(y)>B(x)\ \Rightarrow\ \boxed{r(y)=-1}$$

### 为什么不是统一给个固定 10k token

因为题目难度不同：简单题可能 $b_0(x)=500$，难题 $b_0(x)=10000$。所以做成 $\boxed{\text{problem-dependent budget}}$ 而不是一个全局 max\_tokens。

### 影响

reward 从单目标 $\max$ accuracy 变成隐式多目标：

$$\boxed{\max\{\text{quality},\ -\text{compute}\}}$$

用不同的 $\gamma_{low}<\gamma_{high}<\gamma_{max}$ 训出三个 policy $\pi_{low},\pi_{high},\pi_{max}$。注意这**不是 inference 时简单截断 token**，而是：

$$\boxed{\text{policy 本身学会在不同计算预算下怎么推理}}$$

### Agent 任务为什么连 tool-call 参数都算进 token

普通 reasoning：$T(y)=\#\text{thinking tokens}$。Agent：

$$\boxed{T(y)=\#(\text{reasoning}+\text{model-generated tool arguments})}$$

否则模型可以：

```text
thinking 很短  →  不停调工具  →  实际用了巨量计算
                              但账面 reasoning budget 很低
```

$$\boxed{\text{防止通过 tool calling 绕过 reasoning budget}}$$

## 10. 改动七：单一 RL policy → 9 个 specialized policy

$$3\ \text{domains}\times3\ \text{efforts}=\boxed{9\ \text{experts}}\quad \pi_{D,E}$$

$$D\in\{\text{General},\ \text{General Agent},\ \text{Coding Agent}\},\qquad E\in\{\text{low},\ \text{high},\ \text{max}\}$$

### 为什么

因为不同 RL objective 之间很可能有**梯度冲突**：

| | 倾向 |
|---|---|
| coding | 长工具交互、验证、改文件 |
| general chat | 简洁、少工具、快速回答 |
| low effort | 少算 |
| max effort | 尽可能把题做对 |

一个 policy 同时 RL 所有 domain + 所有 compute regime，reward 信号会互相拉扯。Kimi 的办法是 $\boxed{\text{先分别把能力峰值训出来}}$，再想办法合并 —— 这就是 MOPD 的来历。

## 11. 改动八：最后不是继续 GRPO，而是 MOPD

$$\text{GRPO：environment reward}\rightarrow A\qquad\Longrightarrow\qquad\boxed{\text{MOPD：teacher/student 概率差}\rightarrow\text{dense token reward}}$$

student 自己 rollout $y_t\sim\pi_S$，对应 domain/effort 的 teacher $\pi_T^{(d,e)}$ 只给它实际采到的那个 token 打分：

$$\boxed{r_t^{OPD}=\mathrm{clip}\Big(\mathrm{sg}\Big[\log\frac{\pi_T(y_t|s_t)}{\pi_S(y_t|s_t)}\Big],\ -R_{\max},\ R_{\max}\Big)}$$

$\mathrm{sg}$ 是 stop-gradient。细节（为什么是 sampled token 而不是全词表/top-k、为什么 clip、为什么对应 reverse KL）见 [MOPD](../distill/07-rethink-and-mopd.md#三kimi-k3-的-mopd把-9-个-expert-合回一个模型)。

### 相比 GRPO 的本质变化：credit assignment 的粒度

GRPO 的 $A_i$ 是 **sequence-level** 的，一条一万 token 的回答基本共享同一个 $A_i$，credit assignment 很粗。MOPD 的 $r_t$ 是 **token-specific** 的：

| token | teacher vs student | reward |
|---|---|---:|
| `def` | teacher 更喜欢 | +0.8 |
| `foo` | 差不多 | +0.05 |
| 错误 API | teacher 极不喜欢 | −3.2 |
| 正确参数 | teacher 更喜欢 | +1.4 |

$$\boxed{\text{dense token-level credit}}$$

信息密度比最终的 sequence correctness reward 高得多。

## 12. ⚠️ 最容易混的两个 clip

这是这篇最值得单独记的一条。

$$\boxed{\underbrace{\mathrm{clip}\big(\log\pi_T-\log\pi_S\big)}_{\text{MOPD：限制 teacher 的监督信号}}}\qquad \text{vs}\qquad \boxed{\underbrace{\mathrm{clip/mask}\Big(\frac{\pi_\theta}{\pi_{old}}\Big)}_{\text{RL optimizer：限制 off-policy drift}}}$$

| | clip 的对象 | 控制什么 | 哪一层 |
|---|---|---|---|
| 第一层 | teacher/student 的 log-prob 差 | teacher 给我的监督信号不能太离谱 | MOPD reward |
| 第二层 | $\pi_\theta/\pi_{old}$ | 当前 policy 和 rollout policy 不能差太远 | K2.5 policy optimizer |
| 第三层 | —— | policy drift 的软约束 $(\log\rho)^2$ | K2.5 policy optimizer |

**这三个东西目的完全不同**，被追问时千万别混。

一个 MOPD token 完整走一遍：

```text
student rollout 得到 y_t
        ↓
问 teacher：π_T(y_t|s_t)
        ↓
Δ_t = log π_T(y_t) − log π_S(y_t)
        ↓
第一道保险：A_t = clip(sg(Δ_t), −R_max, R_max)      监督信号别太大
        ↓
拿 A_t 做 token-level PG
        ↓
同时算 ρ_t = π_θ(y_t)/π_old(y_t)
        ↓
第二道保险：ρ_t 离 1 太远 → mask 掉这个 token 的梯度   数据别太 off-policy
        ↓
第三道：−τ(log ρ_t)²                                policy drift 软约束
```

## 13. 全部改动的因果链（真正该背的版本）

| GRPO 原设计 | Kimi 改动 | 为什么 | 最直接影响 |
|---|---|---|---|
| $(r-\bar r)/\sigma_r$ | $r-\bar r$ | 避免 group std 改变样本权重 | reward gap 更直接，少 std 放大 |
| sequence average | **token average** | 长 trajectory 里 token 不该被 $1/L$ 稀释 | 长 trajectory 总权重更高 |
| PPO sign-dependent clip | **ratio-based token mask** | stale token 本身不可信 | off-policy 更稳，但丢数据 |
| reference KL | **$(\log\pi_\theta-\log\pi_{old})^2$** | 控制 behavior/current mismatch | soft trust region |
| 全同步 rollout | **partial rollout** | 避免 long-tail straggler | 吞吐↑，staleness↑ |
| correctness reward | **budgeted reward** | 防无限 overthinking | token efficiency + effort control |
| 一个 policy | **9 个 RL expert** | 减少 domain/effort 梯度冲突 | 各专家峰值能力↑ |
| sequence reward | **MOPD token reward** | 合并专家 + dense credit | 精细 token supervision |
| 无 teacher reward clip | **$\pm R_{\max}$** | teacher/student log-ratio 无界 | 稳定↑，极端纠正被截断 |

## 14. 所以 K3 不是"GRPO + 一堆 trick"

更准确的拆法：

$$\boxed{\begin{aligned}\text{GRPO 的骨架：}&\quad\text{group sampling}+\text{group baseline}+\text{critic-free PG}\\[2mm]\text{DAPO-like 改造：}&\quad\text{no std norm}+\text{token-level aggregation}\\[2mm]\text{Kimi 的核心改造：}&\quad\text{off-policy token mask}+(\log\rho)^2+\text{partial rollout}\\[2mm]\text{K3 的能力扩展：}&\quad\text{budget RL}+9\ \text{experts}+\text{MOPD}\end{aligned}}$$

两条最关键的因果链：

$$\boxed{\text{million-token agent RL}\Rightarrow\text{partial rollout}\Rightarrow\text{严重 stale/off-policy}\Rightarrow\text{token mask}+(\log\rho)^2}$$

$$\boxed{\text{token-level loss}\Rightarrow\text{长 trajectory 权重更大}\Rightarrow\text{必须认真控制 reasoning/tool budget}}$$

理解了这两条，K3 的 recipe 就不是背公式，而是能从问题反推出设计。

## 面试版

> K3 的 policy optimization 沿用 K2.5：group-relative advantage 但**不除 std**，loss 按 batch 总生成 token 数做 token-level 聚合（形式上和 DAPO 一致），用**与 advantage 符号无关的 token-level ratio mask** 取代 PPO 的 sign-dependent clip，再加一个 $(\log\pi_\theta/\pi_{old})^2$ 的 soft trust region。后两者是被 partial rollout 逼出来的 —— 一条 agent trajectory 可能跨多个 policy version，staleness 非常严重。K3 在此之上加了 problem-dependent 的 reasoning budget reward（超预算直接 $r=-1$，agent 场景把 tool-call 参数也计入 token），按 3 domain × 3 effort 训出 9 个 RL expert，最后用 MOPD 把它们蒸馏回一个统一模型。注意 MOPD 的 clip（clip teacher/student 的 log-prob 差）和 optimizer 的 clip（mask $\pi_\theta/\pi_{old}$）是完全不同的两件事。

## 自测

**1.** ⭐ Kimi 为什么去掉 GRPO 的 std normalization？

> **答：** 因为 $\sigma_r$ 会**改变不同 prompt 的相对训练权重**。0/1 reward 下一组 $[1,1,1,0]$ 不除 std 时 $A$ 天然在 $[-1,1]$ 附近；一除 $\sigma_r$，不同 pass rate 的题被重新缩放，**$\sigma_r\to0$ 时（几乎全对/全错）再小的 reward 差异都会被放大**。去掉后由 reward gap 本身决定梯度强弱。代价是不同 reward scale 的任务混训时需要 reward 设计得可控。

**2.** ⭐ token-level loss 相比 sequence average 改了什么？副作用是什么？

> **答：** 从 $\frac1G\sum_i\frac1{|y_i|}\sum_t$ 改成 $\frac1N\sum_{i,t}$（$N=$ batch 总生成 token 数），即**每个 token 等权**。
> 动机：GRPO 里 $|y|=100000$ 的 trajectory 每个 token 只有 $\frac1{100000}$ 权重，比 $|y|=100$ 的稀释 1000 倍，agent 场景很不自然。
> 副作用：**长 response 总权重更大**（1000 token 的 trajectory 总梯度约是 100 token 的 10 倍），会产生长度偏置，所以必须配合 length/budget control 一起用。

**3.** ⭐⭐ Kimi 的 token mask 和 PPO 的 clip 有什么本质区别？

> **答：** PPO 的 clip **和 advantage 符号绑定** —— $A>0$ 防 $\rho\gg1$、$A<0$ 防 $\rho\ll1$，问的是"这个方向还能不能继续更新"。
> Kimi 的 mask **与 advantage 正负无关**：$m_t=\mathbf 1[\alpha\le\rho_t\le\beta]$，超出区间就把这个 token 的 PG 直接置零，问的是"**这条数据还可不可信**"。
> 因为 $\rho_t=20$ 这种 stale token 的 importance sampling 方差极大，Kimi 宁可丢掉它的梯度。

**4.** 为什么把 reference KL 换成 $(\log\rho)^2$？两者约束的是谁和谁？

> **答：** reference KL 约束 $\pi_\theta\leftrightarrow\pi_{ref}$（冻结的 SFT 模型），解决"别 RL 得忘掉原始能力"；$(\log\rho)^2$ 约束 $\pi_\theta\leftrightarrow\pi_{old}$（产生这批 trajectory 的行为策略），解决 **off-policy stability**。partial rollout 下后者才是真问题。
> 它在 log-prob space 对称（$\rho=e^2$ 和 $e^{-2}$ 都罚 4），梯度 $2\tau\log\rho_t$，离得越远拉得越狠。

**5.** 既然有了 mask，为什么还要 $(\log\rho)^2$？

> **答：** 作用区域不同，一软一硬。$\rho$ 接近 1 → 正常 PG + 极小正则；$\rho$ 开始偏离 → 正则越来越强（**软约束**）；$\rho$ 偏离得非常离谱 → PG 直接 mask 掉（**硬保险**）。

**6.** ⭐⭐ partial rollout 是什么？为什么它是前面那些 off-policy 技术的根源？

> **答：** 完成 $\lambda N_pK$ 条 rollout 就开始训练，剩下的暂停、下一 iteration resume，避免所有 GPU 等 long-tail straggler（agent trajectory 从 2 min 到 50 min 都有）。
> 代价：一条 trajectory 的前半段由 $\pi_0$ 生成、后半段可能已经是 $\pi_2$，$\pi_{behavior}\ne\pi_{current}$ 且差得很远。
> 因果链：$\text{partial rollout}\rightarrow\text{stale data}\rightarrow\text{ratio mask}+(\log\rho)^2$。这三件事必须一起理解。

**7.** reasoning budget 怎么做的？为什么不是统一一个 max\_tokens？agent 场景有什么特殊处理？

> **答：** 估计任务的 baseline budget $b_0(x)$，乘 effort multiplier $\gamma$ 得 $B(x)=\gamma b_0(x)$，$T(y)>B(x)$ 时直接 $r(y)=-1$。
> 不用固定值是因为题目难度不同（简单题 500、难题 10000），要 **problem-dependent**。
> agent 场景把 **model-generated tool arguments 也计入 $T(y)$**，否则模型可以"thinking 很短但疯狂调工具"来绕过预算。
> 用 $\gamma_{low}<\gamma_{high}<\gamma_{max}$ 训出三个 effort expert —— 是 **policy 学会在不同预算下推理**，不是 inference 时截断。

**8.** 为什么要 9 个 expert 而不是一个 policy 全学？

> **答：** 3 domain（General / General Agent / Coding Agent）× 3 effort（low/high/max）。因为不同 RL objective 有**梯度冲突**：coding 倾向长工具交互和验证，general chat 倾向简洁少工具；low effort 要少算，max effort 要不惜代价做对。混在一起 reward 信号互相拉扯。做法是先分别训出能力峰值，再用 MOPD 合并。

**9.** ⭐⭐ K3 里有几个 clip？分别 clip 什么？

> **答：** **三层，目的完全不同**：
> ① **MOPD reward clip**：$\mathrm{clip}(\mathrm{sg}[\log\pi_T-\log\pi_S],-R_{\max},R_{\max})$，控制 **teacher 给的监督信号不能太离谱**；
> ② **off-policy mask**：$\mathrm{mask}(\pi_\theta/\pi_{old})$，控制 **当前 policy 和 rollout policy 不能差太远**；
> ③ **policy drift 正则**：$-\tau(\log\pi_\theta/\pi_{old})^2$，软约束。
> ①clip 的是 teacher/student 的差，②③管的是 $\theta$ 和 $old$ 的差，**千万别混**。

**10.** ⭐ 写出 K2.5/K3 的 RL objective。

> **答：** $J_{Kimi}=\frac1N\sum_{i,t}\big[M(\rho_{i,t})\,\rho_{i,t}\,(r_i-\bar r)-\tau(\log\rho_{i,t})^2\big]$，其中 $N=\sum_i|y_i|$，$\rho_{i,t}=\pi_\theta/\pi_{old}$，$M(\rho)$ 在 ratio 合理时为 1、太 off-policy 时为 0。
> 三个部件：group-relative advantage（不除 std）+ ratio mask + squared log-ratio 正则。

**11.** 一句话说清 K3 的血统。

> **答：** **GRPO 的骨架**（group sampling + group baseline + critic-free PG）+ **DAPO-like 改造**（no std norm + token-level aggregation）+ **Kimi 的核心改造**（off-policy token mask + $(\log\rho)^2$ + partial rollout）+ **K3 的能力扩展**（budget RL + 9 experts + MOPD）。不是"GRPO 加一堆 trick"。
