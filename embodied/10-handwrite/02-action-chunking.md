# 手撕：Action Chunking

> 每段都是可直接默写的最小实现，**全部本地跑通并做了性质验证**（$q$ 的边缘分布对上闭式解、DDPM/DDIM 采样还原目标分布、`eta=0` 逐元素确定性、CFG 的 $w=1$ 退化、flow 完美速度场 loss 为 0、ensembling 权重归一且递增）。

> 返回 [手撕总表](00-map.md)｜原理见 [视觉生成](../02-visual-generation/00-map.md)、[Action](../05-action/00-map.md)

## 公共前置

```python
"""Diffusion / Flow matching / Action chunking 的最小实现（NumPy）。"""
import numpy as np
```

---

## Action chunking

### Action chunking

**思路**：一次预测 $H$ 步依次下发；重叠 chunk 按指数衰减加权，**越新的预测权重越大**。

```python
class ChunkExecutor:
    """一次预测 H 步，依次下发；重叠 chunk 做 temporal ensembling。"""
    def __init__(self, horizon, m=0.1):
        self.H, self.m = horizon, m
        self.buf = {}                              # 绝对时刻 -> [(chunk 年龄, 动作)]

    def push(self, t0, chunk):
        """t0 时刻预测出的 chunk，覆盖 t0..t0+H-1"""
        for i, a in enumerate(chunk):
            self.buf.setdefault(t0 + i, []).append(a)

    def act(self, t):
        """取 t 时刻的动作：多个 chunk 的预测按指数衰减加权，越新权重越大"""
        preds = self.buf.pop(t, None)
        if not preds:
            return None
        k = len(preds)
        # preds[0] 最旧, preds[-1] 最新 -> 年龄 age = k-1-j
        w = np.exp(-self.m * np.arange(k - 1, -1, -1))
        w = w / w.sum()
        return sum(wi * a for wi, a in zip(w, preds))
```

**易错**：权重要归一化；`buf.pop` 用完即删，否则内存无限涨。真实部署还要处理异步推理的延迟偏移。
