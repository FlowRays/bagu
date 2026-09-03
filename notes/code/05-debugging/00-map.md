# 调试

> 对应 [bagu.md](../../../bagu.md) 的 code (5) debugging。

| 篇 | 内容 |
|---|---|
| [01 OOM 与 NaN](01-oom-and-nan.md) | OOM 分类与处置顺序、显存泄漏、NaN 排查清单、性能 profiling 与 MFU、结果不对时的检查顺序、经典 bug 清单 |

## 四句话

1. **OOM**：先把 batch 减半。显存变了是 activation 问题，没变是 model states 问题，两者的解法完全不同。
2. **NaN**：先看 grad norm 曲线（比 loss 更早暴露），再用 FP32 跑几步区分「逻辑错」还是「数值问题」。
3. **慢**：先 profile，别猜。CUDA 异步，计时必须 `synchronize`。用 MFU 判断到底有没有问题。
4. **结果不对**：先 **overfit 一个 batch**。做不到就是实现有 bug，别调超参。

## 相关

- [显存账本](../../03-training-fundamentals/01-memory-accounting.md) — OOM 时该动哪一项
- [DDP / ZeRO](../../04-distributed-infra/01-ddp-and-zero.md) — 分布式下的显存与通信
- [Python / PyTorch 速成](../01-python-pytorch/00-map.md) — 三个静默的坑
