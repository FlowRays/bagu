# 视觉生成总图

> 这一册是 [世界模型](../03-world-model/00-map.md) 和 [diffusion / flow action head](../05-action/02-action-head.md) 的共同地基。
> 学它不是为了做图像生成，而是因为**具身里两个最核心的模块都建在它上面**。

| 篇 | 内容 |
|---|---|
| [01 生成模型谱系](01-generative-models.md) | VAE / GAN / AR / Diffusion / Flow 五条路线的取舍、VQ-VAE 与离散化、为什么具身选了 diffusion 和 flow |
| [02 Diffusion](02-diffusion.md) | 前向闭式解、反向去噪、**目标为什么塌成 MSE**、与 score matching 等价、DDIM、CFG、$v$-pred |
| [03 Flow Matching](03-flow-matching.md) | 速度场视角、Conditional FM、**Rectified Flow 为什么能少步**、与 diffusion 的对照、为什么 VLA 需要它 |

## 三条主线

1. **离散化是通往语言模型的桥**：VQ-VAE 把连续信号变 token，图像/视频/动作就都能当 LLM 做（RT-2、OpenVLA 的动作 tokenization 就来自这里）。
2. **训练目标塌成 MSE 是 diffusion 稳定的根本原因**，也是它能取代 GAN 的关键。
3. **路径直不直决定采样步数**，这是 flow matching 在 VLA 里迅速取代 diffusion 的唯一理由。

## 一句话回答面试

> **为什么机器人策略要用 diffusion / flow 而不是直接回归动作？**
> 因为动作分布是**多模态**的 —— 绕障碍可以从左也可以从右，MSE 回归输出的是两者的平均，会直接撞上去。生成式建模才能表达"要么左要么右"。
> 而在 diffusion 和 flow 之间选 flow，是因为**控制频率**：机器人 50 Hz 闭环跑不起 50 步去噪，flow matching 的直线路径让 1–10 步就够。

## 相关

- [手撕：RQ-VAE Loss](../../notes/code/07-handwrite/03-llm.md#rq-vae-loss) — 残差量化的可默写实现
- [熵、交叉熵、KL、JSD](../../notes/06-post-training/distill/01-entropy-ce-kl-jsd.md) — ELBO 里的 KL 项
- [VLM 架构](../../notes/07-vlm/01-architecture.md) — 视觉表示怎么进 LLM
