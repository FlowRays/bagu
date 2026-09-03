# Python / NumPy / PyTorch（给 C++ 选手）

> 面向已经很熟 C++ / STL、但 Python 和 PyTorch 不熟的人。不讲基础语法，
> 只讲**和 C++ 直觉冲突的地方**，以及 [手撕代码合集](../07-handwrite/00-map.md) 里实际用到的每一个 API。
>
> 笔记里的每条论断都跑过验证（`repeat` 语义、`var` 无偏默认、`view` 报错条件、
> `masked_fill` 方向、整行 mask 出 nan、`F.kl_div` 的方向、梯度累加、numpy 视图共享…）。

| 篇 | 内容 |
|---|---|
| [01 给 C++ 选手的 NumPy](01-numpy-for-cpp.md) | ndarray = buffer + shape + strides、赋值是引用不是拷贝、广播、axis/keepdims、view vs copy、索引三件套、np API 全量速查、C++ 循环到 NumPy 的翻译表 |
| [02 给 C++ 选手的 PyTorch](02-torch-for-cpp.md) | tensor↔ndarray 对照、dtype/device、view/reshape/contiguous、**causal mask 与 masked_fill**、autograd 与 detach、nn.Module、F.* 速查、常见报错对照 |

## 最该先记住的六条

1. **`b = a` 是绑定不是拷贝。** C++ 的 `vector<int> b = a` 深拷贝，Python 是指针别名。要副本写 `a.copy()`。
2. **`reshape` / `transpose` / 切片默认共享内存**，改一个另一个跟着变。因为它们只改 shape 和 strides。
3. **广播是唯一没有 C++ 对应物的概念。** 从右往左逐维比，相等或其中一个是 1 才行。看到循环先问能不能用广播加规约表达。
4. **规约完还要跟原数组运算，就必须 `keepdims=True`**（torch 里叫 `keepdim`，没有 s）。
5. **mask 加在 softmax 之前，填 $-\infty$ 不是 0**：`scores.masked_fill(mask == 0, float('-inf'))`。
6. **梯度是累加的**，每步必须 `opt.zero_grad()`。

## 三个静默的坑

这几个不报错，但结果是错的：

| 坑 | 后果 |
|---|---|
| `np.repeat` vs `torch.repeat` | 名字一样语义相反。torch 要 `repeat_interleave` 才等于 `np.repeat`。GQA 里用错，head 和 KV 组对不上 |
| `var` 的无偏默认 | numpy 除 n，torch 除 n-1。GRPO 归一化 advantage 时两边会差 $\sqrt{n/(n-1)}$ |
| `F.kl_div(input, target)` | 第一个参数要 log-prob，方向是 `KL(target ‖ input)`，和数学写法反着 |

## 相关

- [手撕代码合集](../07-handwrite/00-map.md) — 41 道，每道都给了 NumPy 和 PyTorch 两个版本
- [LeetCode 39 题（C++）](../08-leetcode/00-map.md) — 算法题走 C++，不受这一篇影响
- [显存账本](../../03-training-fundamentals/01-memory-accounting.md) — `no_grad` 为什么能省显存
