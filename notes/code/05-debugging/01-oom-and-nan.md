# 调试：OOM 与 NaN

> 训练岗面试很爱问「训练崩了你怎么查」。这一篇是可执行的排查顺序。

## 一、OOM

### 第一步：先分清是哪一类显存

$$M=\underbrace{P+G+O}_{\text{model states，正比于参数量}}+\underbrace{A}_{\text{activation，正比于 }B\times L}+M_{\text{temp}}$$

**判断方法**：把 `batch_size` 减半。

| 现象 | 结论 | 该动谁 |
|---|---|---|
| 显存明显下降 | 瓶颈是 **activation** | gradient checkpointing、减 microbatch + 梯度累积、SP/CP |
| 几乎不变 | 瓶颈是 **model states** | ZeRO-2 → ZeRO-3/FSDP、TP、offload、LoRA |

$$\boxed{\text{batch 减半显存不变} \Rightarrow \text{别再折腾 activation 了}}$$

详见 [显存账本](../../03-training-fundamentals/01-memory-accounting.md#9-各优化技术分别在治哪一块)。

### 第二步：按顺序上手段

代价从小到大：

1. **减 microbatch + 梯度累积**（不改有效 batch，几乎无损）
2. **gradient checkpointing**（activation ↓40–70%，计算 +33%）
3. **ZeRO-2**（通信量和 DDP 一样，显存省很多，sweet spot）
4. **ZeRO-3 / FSDP**（通信 +50%）
5. **TP**（高频通信，最好在 node 内）
6. **CPU offload**（很慢，最后手段）

### 常见的"意外"OOM

| 现象 | 原因 |
|---|---|
| 第一步就 OOM | 模型 states 本身放不下，和 batch 无关 |
| 跑几百步后才 OOM | **显存碎片**，或者有东西在累积（见下） |
| eval 时 OOM | 忘了 `torch.no_grad()`，建了计算图 |
| 变长数据偶发 OOM | 撞上了最长的那个 batch，要按 **token 数**而不是样本数组 batch |
| loss 记录导致 OOM | `losses.append(loss)` 存的是**带计算图的张量**，要 `loss.item()` |

$$\boxed{\text{任何往 list 里存的张量都要先 .item() 或 .detach()}}$$

这是最经典的显存泄漏，因为它把整个计算图一直挂着。

### 排查工具

```python
torch.cuda.memory_allocated()   # 当前张量占用
torch.cuda.max_memory_allocated()
torch.cuda.memory_reserved()    # allocator 向驱动要的总量
torch.cuda.memory_summary()     # 详细分类
torch.cuda.reset_peak_memory_stats()
```

`reserved` 远大于 `allocated` 说明**碎片严重**，可以试 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

`torch.cuda.memory._record_memory_history()` + `_dump_snapshot()` 能导出可视化的显存时间线，定位到具体是哪一行分配的。

## 二、NaN / Inf

### 第一步：定位是前向还是反向

```python
# 逐层检查前向
for name, module in model.named_modules():
    module.register_forward_hook(
        lambda m, i, o, n=name: print(n) if torch.isnan(o).any() else None)

# 检查梯度
for n, p in model.named_parameters():
    if p.grad is not None and not torch.isfinite(p.grad).all():
        print("bad grad:", n)
```

或者直接 `torch.autograd.set_detect_anomaly(True)`（很慢，只在定位时开）。

### 常见原因清单

| 原因 | 症状 / 检查 |
|---|---|
| **log(0)** | 交叉熵、KL 里没做数值稳定 → 用 `log_softmax`、`logsigmoid`，别写 `log(softmax(x))` |
| **除 0** | 归一化时分母没加 eps；GRPO 里组内 reward 全相同导致 std=0 |
| **整行被 mask** | attention 某行全是 `-inf` → softmax 分母为 0 → **nan**。变长 batch 要保证每行至少一个可见位置 |
| **FP16 溢出** | 换 BF16，或用 loss scaling |
| **lr 太大 / 没 warmup** | 前几十步就崩 |
| **脏数据** | 某个特定 batch 必崩 → 固定 seed 复现，把那个 batch dump 出来看 |
| **梯度爆炸** | grad norm 在崩之前会先飙升 |

$$\boxed{\text{先看 grad norm 的曲线，它比 loss 更早暴露问题}}$$

### 稳定性排查顺序

1. 关掉混合精度，用 FP32 跑几步 —— **还崩就是逻辑错，不崩就是数值问题**
2. 检查 loss 里所有的 `log` 和除法
3. 检查 mask 有没有整行为空
4. 把 lr 降 10 倍看还崩不崩
5. 固定 seed 定位到具体 batch，dump 出来检查数据

## 三、性能问题（跑得慢）

### 先测，别猜

```python
with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log")) as prof:
    for step, batch in enumerate(loader):
        train_step(batch); prof.step()
```

计时一定要 `torch.cuda.synchronize()`，否则测的是**下发 kernel 的时间**不是执行时间。

$$\boxed{\text{CUDA 是异步的，不 synchronize 的计时全是错的}}$$

### 常见瓶颈

| 现象 | 原因 |
|---|---|
| GPU 利用率低、时间花在 CPU | **数据加载**是瓶颈 → 加 `num_workers`、`pin_memory`、预处理离线做 |
| GPU 利用率高但 MFU 低 | kernel 效率低 → 用 FlashAttention、融合算子；或者 batch 太小 |
| 多卡比单卡慢很多 | 通信没和计算 overlap；或者跨 node 走了慢网络做 TP |
| 偶发卡顿 | 有同步点（`.item()`、`print`、`torch.cuda.synchronize()` 在循环里） |

**MFU** 是判断的核心指标：

$$\text{MFU}=\frac{6ND/\text{step 时间}}{\text{GPU 峰值 FLOPS}\times\text{卡数}}$$

大规模训练做到 40%–50% 算正常，低于 30% 就该查了。见 [scaling](../../05-pretraining/02-scaling.md#5-训练-flops-的估算)。

## 四、结果不对（不报错但效果差）

最难查的一类。检查顺序：

1. **先过拟合一个 batch**。拿 8 条数据训到 loss 接近 0。做不到 → 实现有 bug，别再调超参了。
   $$\boxed{\text{overfit-one-batch 是最有效的一个 sanity check}}$$
2. **检查 loss mask**：是不是该 mask 的没 mask（tool 输出、prompt）、该算的没算（`<|im_end|>`）
3. **检查 shift**：logits 和 labels 有没有错位一格
4. **检查 chat template**：训练和推理是否完全一致（差一个换行都会掉点）
5. **检查数据**：随机抽 20 条 decode 回文本，人眼看一遍。这一步经常能直接发现问题
6. **检查 packing**：position_ids 有没有 reset，attention mask 是不是 block-diagonal
7. **对拍**：小规模下和一个已知正确的实现（HF Trainer）比 loss 曲线

## 五、几个经典 bug

| bug | 症状 | 原因 |
|---|---|---|
| 忘了 `zero_grad()` | loss 越训越离谱 | 梯度累加 |
| 模型没 `.eval()` | eval 结果抖动 | dropout / BN 还在训练模式 |
| shift 错位 | loss 下降但生成乱套 | 学的是错位映射 |
| tool 输出算了 loss | 模型幻想工具返回 | 学会了编造 observation |
| position_ids 没 reset | packing 后效果掉 | 位置编码错乱 |
| `repeat` 写成 `repeat_interleave` | GQA 效果异常但不报错 | head 和 KV 组对不上 |
| `var` 的无偏差异 | numpy 和 torch 实现对不上 | 默认 `unbiased` 相反 |
| 学习率 schedule 按 step 还是 epoch | lr 曲线不对 | 梯度累积时 step 数是 optimizer step 不是 forward 次数 |

后三个是**静默错误**，不报错但结果不对，见 [Python/PyTorch 速成](../01-python-pytorch/00-map.md#三个静默的坑)。

## 自测

**1.** 怎么快速判断 OOM 是 activation 还是 model states 的问题？

> **答：** **把 `batch_size` 减半**：显存明显下降 → 瓶颈是 **activation**（该上 gradient checkpointing、减 microbatch + 梯度累积、SP/CP）；几乎不变 → 瓶颈是 **model states**（该上 ZeRO-2/3、TP、offload、LoRA）。
> 一句话：**batch 减半显存不变，就别再折腾 activation 了。**

**2.** 按代价从小到大列出解决 OOM 的手段。

> **答：** ① 减 microbatch + 梯度累积（不改有效 batch，几乎无损）；② gradient checkpointing（activation ↓40–70%，计算 +33%）；③ **ZeRO-2**（通信量和 DDP 一样，sweet spot）；④ ZeRO-3 / FSDP（通信 +50%）；⑤ TP（高频通信，最好在 node 内）；⑥ CPU offload（很慢，最后手段）。

**3.** 「跑几百步后才 OOM」的两种可能原因？往 list 里存 loss 为什么会泄漏？

> **答：** 两种：**显存碎片**（`reserved` 远大于 `allocated`，可试 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`）；或者**有东西在累积**。
> `losses.append(loss)` 存的是**带计算图的张量**，整张图被一直挂着无法释放，几百步后就爆了。**任何往 list 里存的张量都要先 `.item()` 或 `.detach()`。**
> 其他常见「意外」OOM：eval 忘了 `torch.no_grad()`、变长数据撞上最长的 batch（应按 token 数而不是样本数组 batch）。

**4.** `reserved` 远大于 `allocated` 说明什么？

> **答：** 说明**显存碎片严重** —— allocator 向驱动要了很多显存，但因为碎片化拼不出连续的大块来满足新的分配请求。
> 可以试 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，或者用 `torch.cuda.memory._record_memory_history()` + `_dump_snapshot()` 导出显存时间线定位。

**5.** NaN 的常见原因列表？为什么整行被 mask 会出 nan？

> **答：** **log(0)**（交叉熵、KL 没做数值稳定 → 用 `log_softmax`、`logsigmoid`）、**除 0**（归一化分母没加 eps；GRPO 组内 reward 全同导致 std=0）、**整行被 mask**、**FP16 溢出**、**lr 太大/没 warmup**、**脏数据**、**梯度爆炸**。
> 整行被 mask 出 nan 是因为：attention 某一行全是 `-inf`，softmax 的分母 $\sum e^{-\infty}=0$，$0/0=$ **nan**。变长 batch 要保证每行至少一个可见位置。

**6.** 稳定性排查的五步顺序？为什么先用 FP32 跑？

> **答：** ① **关掉混合精度用 FP32 跑几步**；② 检查 loss 里所有的 `log` 和除法；③ 检查 mask 有没有整行为空；④ 把 lr 降 10 倍看还崩不崩；⑤ 固定 seed 定位到具体 batch，dump 出来检查数据。
> 先用 FP32 是为了**一刀切开两类问题**：FP32 下还崩就是**逻辑错**（公式写错、mask 错），不崩就是**数值问题**（精度/溢出）。这一步能省掉大量瞎猜。

**7.** 为什么 CUDA 计时必须 synchronize？

> **答：** **CUDA kernel 的下发是异步的** —— Python 端 `t0=time()` 到 `t1=time()` 之间只是把 kernel 塞进了 stream，GPU 可能还没开始算。不 `torch.cuda.synchronize()` 测到的是**下发时间**而不是执行时间，数字完全没意义。

**8.** MFU 怎么算？多少算正常？

> **答：** $$\text{MFU}=\frac{6ND/\text{step 时间}}{\text{GPU 峰值 FLOPS}\times\text{卡数}}$$
> 大规模训练做到 **40%–50%** 算正常，**低于 30% 就该查了**（数据加载瓶颈、通信没 overlap、kernel 效率低、batch 太小）。

**9.** 结果不对时最有效的第一个 sanity check 是什么？为什么？

> **答：** **overfit 一个 batch**：拿 8 条数据训到 loss 接近 0。
> 做不到就说明**实现有 bug**（loss mask 错、shift 错位、梯度没回传、数据处理错），此时再调超参是浪费时间。这个检查成本极低但排除了绝大多数实现错误。
> 之后再按顺序查：loss mask → shift → chat template → 随机抽 20 条 decode 回文本人眼看 → packing 的 position_ids → 和 HF Trainer 对拍 loss 曲线。

**10.** 说出三个静默错误（不报错但结果不对）的例子。

> **答：** ① **`repeat` 写成 `repeat_interleave`**（或反过来）：GQA 的 head 和 KV 组对不上，效果异常但不报错；
> ② **`var` 的无偏默认差异**：numpy 除 $n$、torch 除 $n-1$，两边实现对不上；
> ③ **`F.kl_div` 的方向**：第一个参数要 log-prob 且方向是 `KL(target‖input)`，写反了算出来是另一个量。
> 还有：**position_ids 没 reset** 导致 packing 后效果掉、**tool 输出算了 loss** 导致模型幻想工具返回、**lr schedule 按 forward 次数而不是 optimizer step 计数**。

