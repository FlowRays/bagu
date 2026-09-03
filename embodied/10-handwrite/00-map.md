# 手撕：具身相关实现

> 和 [LLM 手撕合集](../../notes/code/07-handwrite/00-map.md) 互补：那边是模型组件，这边是具身特有的生成与控制。
> 全部本地跑通并做了性质验证，不是"看起来对"。

| 篇 | 内容 |
|---|---|
| [01 Diffusion 与 Flow Matching](01-diffusion-flow.md) | schedule、前向闭式解、训练目标、DDPM/DDIM 采样、CFG、flow 训练与欧拉采样，外加**一个证明「不能用 MSE 回归动作」的可运行实验** |
| [02 Action Chunking](02-action-chunking.md) | 一次预测 H 步 + temporal ensembling 的最小实现 |

## 验证了哪些性质

不是只看 shape 对不对：

- $q(x_t|x_0)$ 的**边缘分布**（均值 $\sqrt{\bar\alpha}x_0$、标准差 $\sqrt{1-\bar\alpha}$）逐个 $t$ 对上闭式解
- DDPM 和 DDIM 采样都能**还原目标分布** $\mathcal N(3,0.5^2)$
- DDIM `eta=0` 时同 seed 两次结果**逐元素相同**，`eta=1` 则不同
- CFG 的 $w=1$ 严格退化成条件生成、$w=0$ 退化成无条件
- flow matching 在完美速度场下 loss 为 0；线性速度场 1 步和 8 步都能把均值搬到位
- temporal ensembling 权重**归一且随新鲜度递增**
- **双峰动作实验**：MSE 回归 100% 落在 0（标准差 0.000），flow matching 51%/45% 分到两个模式

## 相关

- [视觉生成](../02-visual-generation/00-map.md) — 这些公式的推导
- [Action head](../05-action/02-action-head.md) — 这些实现装在哪
- [Chunking 与实时性](../05-action/03-chunking-and-realtime.md) — ensembling 的代价
