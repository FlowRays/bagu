# 分布式代码

> 对应 [bagu.md](../../../bagu.md) 的 code (4) distributed code。
> 原理在 [分布式训练](../../04-distributed-infra/01-ddp-and-zero.md)，这一页记**实际写代码时的要点**。

## 起手式

```python
import torch.distributed as dist
dist.init_process_group(backend="nccl")          # GPU 用 nccl，CPU 用 gloo
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)                 # 必须设，否则所有进程挤在 cuda:0
...
dist.destroy_process_group()
```

`torchrun --nproc_per_node=8 train.py` 会自动注入 `RANK` / `LOCAL_RANK` / `WORLD_SIZE`。

| 变量 | 含义 |
|---|---|
| `RANK` | 全局进程号 |
| `LOCAL_RANK` | 本机内的进程号，**用它 set_device** |
| `WORLD_SIZE` | 总进程数 |

## 常用 collective

| 操作 | 语义 | 用在哪 |
|---|---|---|
| `all_reduce` | 所有 rank 规约后**都拿到**结果 | DDP 同步梯度 |
| `reduce_scatter` | 规约后**每个 rank 只拿一片** | ZeRO-2/3 的梯度 |
| `all_gather` | 每个 rank 的分片**拼成完整的**发给所有人 | ZeRO-3 的参数、收集 eval 结果 |
| `broadcast` | 从一个 rank 发给所有人 | 同步初始权重 |
| `all_to_all` | 每个 rank 给每个 rank 发不同的数据 | **MoE 的 token dispatch** |
| `barrier` | 同步点 | 确保某步所有 rank 都完成 |

$$\text{all\_reduce}=\text{reduce\_scatter}+\text{all\_gather}\ \Rightarrow\ \text{通信量}\ 2M$$

这个恒等式是理解 [ZeRO 通信量](../../04-distributed-infra/01-ddp-and-zero.md#3-通信量为什么-zero-2-是-sweet-spot) 的关键。

## 实际写代码时最容易错的

| 坑 | 后果 |
|---|---|
| 忘了 `set_device(local_rank)` | 所有进程都在 cuda:0，直接 OOM |
| 只在 rank 0 保存 / 打印，但**其他 rank 也执行了 barrier 之外的分支** | 死锁 |
| **collective 调用不对齐**（某个 rank 因为 `if` 跳过了一次 all_reduce） | 死锁，而且现象是"卡住不报错" |
| 变长数据导致不同 rank 的 step 数不同 | 最后几步死锁 → 要 pad 到相同步数或用 `join()` |
| eval 时忘了 all_gather | 每个 rank 报自己那份的指标，数字偏小 |
| **梯度裁剪只在本 rank 算 norm** | ZeRO/FSDP 下梯度是分片的，必须跨卡聚合平方和，用 `clip_grad_norm_` 的分布式版本 |
| 随机种子所有 rank 一样 | 数据增强 / dropout 完全相同，等于白做数据并行的多样性 |
| 日志每个 rank 都打 | 刷屏，用 `if rank == 0` |

$$\boxed{\text{死锁的最常见原因：某个 rank 少调了一次 collective}}$$

排查方法：设 `TORCH_DISTRIBUTED_DEBUG=DETAIL`，或者在每个 collective 前后打日志看哪个 rank 卡住了。

## 保存 checkpoint

- DDP：`model.module.state_dict()`，只在 rank 0 存
- FSDP：要用 `FullStateDictConfig` 或者存 sharded state dict，直接 `state_dict()` 拿到的是分片
- 大模型建议存 **sharded checkpoint**，加载快很多

## 相关

- [DDP / ZeRO / FSDP](../../04-distributed-infra/01-ddp-and-zero.md)
- [并行全图](../../04-distributed-infra/02-parallelism-map.md)
- [调试](../05-debugging/00-map.md)
