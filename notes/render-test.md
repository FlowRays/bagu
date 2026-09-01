# 渲染测试

确认阅读器各元素渲染正常，确认后可删除本文件。

## 数学公式

行内公式：attention score $s_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}}$，softmax 后加权求和。

块级公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

RoPE 旋转矩阵作用在 $q, k$ 上：$\langle R_m q, R_n k \rangle = \langle q, R_{n-m} k \rangle$，只依赖相对位置 $n-m$。

## 代码

```python
import torch

def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6):
    # x: (B, S, H)
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return x * rms * gamma
```

行内代码：`temperature=0`、`top_p=0.95`，代码块里的 `$HOME` 不应被当成公式。

```bash
echo "$HOME"   # dollar sign in code
```

## 表格

| 归一化 | 公式核心 | 参数 |
|---|---|---|
| LayerNorm | 减均值除标准差 | $\gamma, \beta$ |
| RMSNorm | 除以均方根 | $\gamma$ |

## 其他

> 引用块：pre-norm 结构下残差流保持恒等通路，训练更稳定。

- [x] 列表 + 任务框
- [ ] 未完成项
  - 嵌套列表

**加粗**、*斜体*、~~删除线~~、[链接](https://example.com)。

---

分割线之后结束。
