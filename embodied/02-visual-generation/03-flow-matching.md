# Flow Matching 与 Rectified Flow

> **VLA 的 action head 正在大规模从 diffusion 转向 flow matching**（π0 系列是代表）。
> 核心动机只有一个：**少步采样**。机器人要 50 Hz 闭环，跑不起 50 步去噪。

## 1. 换个视角：不是去噪，是搬运

diffusion 的视角是"逐步去噪"。flow matching 的视角是：

$$\boxed{\text{学一个速度场 }v_\theta(x,t)\text{，把噪声分布沿着 ODE 连续地搬到数据分布}}$$

$$\frac{dx_t}{dt}=v_\theta(x_t,t),\qquad x_0\sim p_{\text{noise}},\quad x_1\sim p_{\text{data}}$$

采样就是解这个 ODE。**路径越直，需要的求解步数越少。**

## 2. Conditional Flow Matching

直接回归边缘速度场是不可解的（要对所有数据积分），但有一个漂亮的结论：

$$\boxed{\text{回归「条件速度场」的梯度，等于回归「边缘速度场」的梯度}}$$

于是只要给每一对 $(x_0,x_1)$ 指定一条路径，就能训练。最简单的选择是**直线**：

$$x_t=(1-t)\,x_0+t\,x_1,\qquad \frac{dx_t}{dt}=x_1-x_0$$

训练目标：

$$\boxed{\mathcal L_{\text{CFM}}=\mathbb E_{t\sim\mathcal U[0,1],\ x_0\sim\mathcal N(0,I),\ x_1\sim p_{\text{data}}}\Big[\big\|v_\theta\big((1-t)x_0+tx_1,\ t\big)-(x_1-x_0)\big\|^2\Big]}$$

训练循环同样只有四行：

```text
1) 取数据 x1，采噪声 x0 ~ N(0,I)，采 t ~ U[0,1]
2) 线性插值 x_t = (1-t)·x0 + t·x1
3) 目标速度 u = x1 - x0            ← 常量，和 t 无关
4) 最小化 ‖v_θ(x_t, t) − u‖²
```

和 diffusion 一样是个 MSE，一样稳。

## 3. Rectified Flow：为什么能一步

Rectified Flow 就是上面这个"直线路径"版本，名字强调的是**路径被拉直**这件事。

- diffusion 的采样轨迹是**弯的**（由 SDE/概率流决定），必须小步走才准
- flow matching 用直线插值训练，学到的轨迹接近直线，**大步走误差也小**

$$\boxed{\text{路径直} \Rightarrow \text{欧拉法几步就够} \Rightarrow \text{可以 1–10 步采样}}$$

**Reflow** 可以进一步拉直：用训练好的模型生成 $(x_0,x_1)$ 配对，再用这些配对重新训练，路径会越来越直，最终逼近一步生成。

## 4. 和 diffusion 的对照

| | Diffusion (DDPM) | Flow Matching / Rectified Flow |
|---|---|---|
| 视角 | 逐步去噪 | 学速度场解 ODE |
| 训练目标 | $\|\epsilon-\epsilon_\theta\|^2$ | $\|(x_1-x_0)-v_\theta\|^2$ |
| 训练难度 | MSE，稳 | MSE，一样稳 |
| 采样步数 | 20–1000 | **1–10** |
| 路径 | 弯 | **直** |
| 噪声 schedule | 要设计 $\beta_t$ | **不需要**，$t$ 就是线性插值系数 |
| 理论关系 | 是 flow 的一个特例（特定路径 + SDE） | 更一般的框架 |

**它们并不对立**：diffusion 可以写成一个概率流 ODE，flow matching 换一条路径也能复现 diffusion。区别在于**路径的选择**。

## 5. 为什么 VLA 特别需要它

机器人控制频率 10–50 Hz，意味着每次推理只有 20–100 ms 预算，而且还要留给 VLM backbone。

$$\boxed{\text{diffusion 50 步} \times \text{每步一次 head 前向} \gg \text{预算}}$$

flow matching 只要 **1–10 步**，才让「大 VLM + 生成式 action head」这个组合真正跑得起来。这就是 π0 用 flow matching action expert 的直接原因（见 [VLA 模型](../06-vla/01-vla-models.md)）。

而且它**保留了多模态建模能力** —— 这是相对直接回归的关键优势，不能为了快就退回 MSE。

## 6. 条件化

和 diffusion 一样，把观测 $o$、语言 $\ell$、本体状态 $s$ 作为条件喂给 $v_\theta$：

$$v_\theta(x_t,\ t,\ o,\ \ell,\ s)$$

CFG 同样适用（训练时随机丢条件，采样时外推），但在 VLA 里用得比图像生成少，因为条件（观测）几乎总是必须的，而且要省那一倍的前向。

## 自测

**1.** flow matching 的核心视角和 diffusion 有什么不同？

> **答：** diffusion 是「逐步去噪」，flow matching 是「**学一个速度场 $v_\theta(x,t)$，沿 ODE $\frac{dx_t}{dt}=v_\theta$ 把噪声分布连续搬到数据分布**」。采样就是解这个 ODE，**路径越直需要的步数越少**。

**2.** 写出 Conditional Flow Matching 的训练目标和四行训练循环。

> **答：** 用直线路径 $x_t=(1-t)x_0+tx_1$，目标速度就是常量 $x_1-x_0$：
> $$\mathcal L=\mathbb E_{t,x_0,x_1}\big[\|v_\theta((1-t)x_0+tx_1,\ t)-(x_1-x_0)\|^2\big]$$
> 循环：① 取数据 $x_1$、采噪声 $x_0$、采 $t\sim\mathcal U[0,1]$；② 线性插值出 $x_t$；③ 目标速度 $u=x_1-x_0$；④ 最小化 $\|v_\theta(x_t,t)-u\|^2$。
> 和 diffusion 一样是 MSE，一样稳。

**3.** 为什么 CFM 可以只回归「条件速度场」？

> **答：** 因为有一个结论：**回归条件速度场的梯度，等于回归边缘速度场的梯度**。直接回归边缘速度场需要对所有数据积分、不可解；换成给每一对 $(x_0,x_1)$ 指定一条路径后就能训练了。

**4.** Rectified Flow 为什么能少步甚至一步？Reflow 又做了什么？

> **答：** diffusion 的采样轨迹是**弯的**，必须小步走才准；flow matching 用**直线插值**训练，学到的轨迹接近直线，欧拉法大步走误差也小，所以 1–10 步就够。
> **Reflow**：用训练好的模型生成 $(x_0,x_1)$ 配对，再拿这些配对重新训练，路径会越来越直，逼近一步生成。

**5.** 列出 diffusion 和 flow matching 的对照表。它们对立吗？

> **答：** 视角（去噪 vs 速度场）、目标（$\|\epsilon-\epsilon_\theta\|^2$ vs $\|(x_1-x_0)-v_\theta\|^2$）、采样步数（20–1000 vs **1–10**）、路径（弯 vs 直）、是否需要噪声 schedule（要 vs **不要**）。
> **不对立**：diffusion 可以写成概率流 ODE，是 flow 的一个特例（特定路径 + SDE）；flow matching 换条路径也能复现 diffusion。区别在**路径的选择**。

**6.** 为什么 VLA 特别需要 flow matching？

> **答：** 机器人控制频率 10–50 Hz，每次推理只有 20–100 ms 预算，还要分给 VLM backbone。diffusion 50 步 × 每步一次 head 前向远超预算；flow matching 只要 1–10 步。
> 这才让「大 VLM + 生成式 action head」跑得起来（π0 就是这么做的）。而且它**保留了多模态建模能力**，不能为了快退回 MSE 回归。
