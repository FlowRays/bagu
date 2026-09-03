# 给 C++ 选手的 NumPy

> 面向已经很熟 C++ / STL 的人。不讲「什么是数组」，只讲**和 C++ 直觉不一样的地方**，以及
> [手撕代码合集](../07-handwrite/00-map.md) 里实际用到的每一个 API。
>
> 一句话总纲：**NumPy 的核心不是「数组库」，是「把循环写进类型系统」。**
> 你在 C++ 里写三重 for，在这里要写成一次带广播的表达式。

## 一、心智模型：ndarray 到底是什么

**不是** `vector<vector<double>>`。它是：

```text
ndarray = 一块连续的 buffer  +  shape（各维长度）  +  strides（各维步长，单位是字节）  +  dtype
```

对应到 C++，最接近的是你自己手写的：

```cpp
double* data;              // 连续内存
int shape[2] = {3, 4};
int stride[2] = {4, 1};    // 以元素为单位
double& at(int i, int j) { return data[i*stride[0] + j*stride[1]]; }
```

**为什么必须知道 strides**：因为 `reshape` / `transpose` / 切片**大多不复制数据，只改 shape 和 strides**。这就是 C++ 里的指针别名，只是它是默认行为。

```python
a = np.arange(12).reshape(3, 4)   # 不复制，只是换了个 (shape, strides) 的视图
b = a.T                            # 也不复制，strides 互换
b[0, 0] = 999                      # a[0,0] 也变成 999
```

$$\boxed{\text{和 C++ 的值语义相反：这里默认是共享，不是拷贝}}$$

要真拷贝：`a.copy()`。

### 赋值是绑定，不是拷贝

这是 C++ 选手最容易踩的第一个坑，和 ndarray 无关，是 Python 本身：

```python
a = [1, 2, 3]
b = a          # b 和 a 指向同一个 list，不是副本
b.append(4)    # a 也变成 [1,2,3,4]
```

C++ 的 `vector<int> b = a;` 是深拷贝，Python 的 `b = a` 是 `int* b = a;`。

## 二、广播：唯一没有 C++ 对应物的概念

两个形状不同的数组做逐元素运算时，NumPy 会自动把维度「撑开」。规则只有两条，**从右往左**逐维比较：

1. 长度相等 → OK
2. 其中一个是 1 → 把这一维复制到另一个的长度（逻辑上复制，不真占内存）

否则报错。

```text
(3,4)   +  (4,)      → (4,) 先补成 (1,4) → (3,4)     OK
(3,1)   +  (1,4)     → (3,4)                          OK
(3,4)   +  (3,)      → (3,) 补成 (1,3)，4≠3           报错
```

手撕题里的实际用法，K-Means 一行算出全部点到全部中心的距离：

```python
d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
#     (n,1,d)         (1,k,d)   →  (n,k,d)  →  sum 掉最后一维  →  (n,k)
```

`X[:, None, :]` 就是在第 1 维插一个长度 1 的轴（等价于 `np.expand_dims(X, 1)`）。这一行替代了 C++ 里的双重循环。

**代价**：中间那个 `(n,k,d)` 是真实存在的临时数组。n、k、d 都大的时候会炸内存，那时候要换成 $\|x\|^2 - 2x^\top c + \|c\|^2$ 的矩阵乘写法。

## 三、axis 和 keepdims

`axis=i` 的含义：**沿着第 i 维做规约，第 i 维消失**。

```python
x.shape                  # (2, 3, 4)
x.sum(axis=0).shape      # (3, 4)      第 0 维没了
x.sum(axis=-1).shape     # (2, 3)      最后一维没了
x.sum(axis=(0,1)).shape  # (4,)
```

`keepdims=True` 保留成长度 1，**为的是后面还能广播回去**：

```python
x.mean(-1).shape                  # (2,3)      —— 不能和 (2,3,4) 相减
x.mean(-1, keepdims=True).shape   # (2,3,1)    —— 可以广播成 (2,3,4)
```

所有归一化都靠这个，LayerNorm：

```python
mu  = x.mean(-1, keepdims=True)      # 漏了 keepdims 就 shape 不匹配
var = x.var(-1, keepdims=True)
out = gamma * (x - mu) / np.sqrt(var + eps) + beta
```

$$\boxed{\text{规约完还要和原数组运算 → 一定要 keepdims=True}}$$

## 四、view 还是 copy

| 操作 | 共享内存？ |
|---|---|
| 基本切片 `a[1:3, :]` | **共享** |
| `reshape` / `ravel` | 通常共享（做不到时才复制） |
| `transpose` / `.T` / `swapaxes` | **共享**（只换 strides） |
| `np.newaxis` / `None` 插维 | 共享 |
| 花式索引 `a[[0,2,1]]` | **复制** |
| 布尔索引 `a[a > 0]` | **复制** |
| `astype` / `copy` / 算术运算 | 复制（产生新数组） |

`transpose` 之后 `reshape` 有时会隐式复制一次，因为内存已经不连续了。NumPy 会自动处理，但 PyTorch 会直接报错（见 [torch 篇](02-torch-for-cpp.md#三view--reshape--contiguous)）。

## 五、索引三件套

```python
a = np.arange(12).reshape(3, 4)

# 1) 切片：和 C++ 迭代器一样是左闭右开，支持负数和步长
a[1:3]        # 第 1、2 行
a[:, -1]      # 最后一列
a[::2]        # 隔行取
a[::-1]       # 整个反转

# 2) 花式索引：用整数数组当下标，可以乱序、可以重复
idx = np.array([2, 0, 0])
a[idx]        # 取第 2、0、0 行

# 3) 布尔索引
a[a % 2 == 0]        # 所有偶数，展平成一维
```

**最重要的一个组合**：两个整数数组同时索引 = 按坐标对取，这是取「每行指定列」的标准写法：

```python
logp[np.arange(N), labels]     # 取第 i 行的第 labels[i] 列，共 N 个数
```

交叉熵就是靠这一行取出正确类别的 log 概率。C++ 里对应 `for (i) out[i] = logp[i][labels[i]];`。

同样的写法也能**写入**：

```python
dz2 = p.copy()
dz2[np.arange(N), y] -= 1      # 就地把 one-hot 减掉，得到 p - q
```

## 六、手撕题里用到的 np API 速查

按实际出现频次排。**这张表覆盖了 41 道手撕题里出现的每一个 numpy 调用。**

### 创建与转换

| API | 作用 | C++ 类比 |
|---|---|---|
| `np.asarray(x, float)` | 转成 ndarray，**已经是就不复制** | 比 `np.array` 更该用（后者总是复制） |
| `np.array(x)` | 总是复制一份 | `vector<T> v(other)` |
| `np.zeros(n)` / `np.ones(n)` | 全 0 / 全 1 | `vector<T> v(n, 0)` |
| `np.empty(n)` | **不初始化**，内容是垃圾 | `new T[n]`，要自己全部写一遍 |
| `np.zeros_like(a)` / `np.empty_like(a)` | 形状 dtype 都跟 a 一样 | — |
| `np.arange(n)` | `0..n-1` | `iota` |
| `np.atleast_2d(x)` | 一维就补成 `(1,n)` | 统一维度，省掉特判 |
| `np.hstack([A, b])` | 水平拼接 | 线性回归拼偏置列 |
| `np.concatenate([a,b], axis)` | 沿指定轴拼接 | KV cache 拼新 token |
| `np.pad(x, ((0,0),(p,p)))` | 补零，每维给 `(前,后)` | 卷积 padding |

### 逐元素与规约

| API | 说明 |
|---|---|
| `np.exp` / `np.log` / `np.sqrt` / `np.abs` | 逐元素 |
| `np.log1p(x)` | 算 `log(1+x)`，x 很小时精度远好于 `log(1+x)` |
| `np.logaddexp(a, b)` | 稳定地算 `log(exp(a)+exp(b))`，DPO 里算 `-logσ(z)` 用 |
| `np.maximum(a, b)` / `np.minimum` | **逐元素**取大/小（两个数组比） |
| `a.max()` / `a.min()` | **规约**成一个数（或沿某轴） |
| `np.clip(x, lo, hi)` | 截断；`np.clip(x, 0, None)` = ReLU |
| `a.sum/mean/var(axis, keepdims)` | 规约，`var` 默认是**有偏**（除 n） |
| `np.where(cond, a, b)` | 逐元素三元表达式 |
| `np.allclose(a, b)` | 浮点近似相等，写测试用 |

$$\boxed{\texttt{np.maximum(a,b)}\ \text{逐元素}\quad\text{vs}\quad\texttt{a.max()}\ \text{规约}}$$

这两个名字太像，是最常见的笔误来源。

### 形状操作

| API | 说明 |
|---|---|
| `a.reshape(...)` | 换形状，`-1` 表示这一维自动算 |
| `a.transpose(1,0,2)` | 按给定顺序重排轴 |
| `a.T` | 二维转置 |
| `x[:, None, :]` | 插一个长度 1 的新轴（广播必备） |
| `np.repeat(a, g, axis=0)` | 每个元素**重复 g 次**：AABBCC。GQA 广播 KV 用这个 |

`np.repeat` vs `np.tile`：前者 AABBCC，后者 ABCABC。GQA 里用错，head 就对不上。

### 排序与查找

| API | 说明 |
|---|---|
| `np.argsort(a, axis)` | 返回排序后的**下标** |
| `a.argmin(axis)` / `a.argmax(axis)` | 最小/最大值的下标 |
| `np.bincount(x)` | 统计非负整数出现次数，KNN 投票用 |

### 线性代数与随机

| API | 说明 |
|---|---|
| `a @ b` | 矩阵乘。**高维时是批量矩阵乘**：`(h,L,d) @ (h,d,L) → (h,L,L)`，前面的维当 batch |
| `np.tensordot(a, b, axes=([1,2,3],[1,2,3]))` | 指定多个轴同时收缩，卷积一次算完所有输出通道 |
| `np.linalg.norm(x, axis=-1, keepdims=True)` | 求范数，L2 归一化用 |
| `np.linalg.svd(X, full_matrices=False)` | SVD，PCA 用 |
| `np.linalg.lstsq(A, y, rcond=None)` | 最小二乘，比 `inv(AᵀA)Aᵀy` 稳 |
| `rng = np.random.default_rng(seed)` | **新式随机数接口**，别用老的 `np.random.seed` |
| `rng.normal(loc, scale, size)` / `rng.choice(n, k, replace=False)` | 正态 / 不放回抽样 |

## 七、Python 语法里 ACM 选手会踩的坑

```python
# 1) 整数除法是 //，/ 永远返回 float
7 // 2      # 3      —— 对应 C++ 的 int 除法
7 / 2       # 3.5    —— 注意 shape 计算里要用 //
-7 // 2     # -4     —— 向下取整，不是向零截断！C++ 是 -3

# 2) 没有 ++，用 += 1
# 3) 链式比较是合法的
if 0 <= i < n:  ...        # 等价于 0 <= i and i < n

# 4) 三元表达式顺序和 C++ 相反
x = a if cond else b       # C++: cond ? a : b

# 5) 可变默认参数是共享的（经典陷阱）
def f(acc=[]):             # 每次调用用的是同一个 list
    acc.append(1); return acc

# 6) 列表推导 = 一行的 for
sq = [x*x for x in range(n) if x % 2 == 0]

# 7) 解包
a, b = b, a                          # 交换，不用 temp
mu, *rest = [1, 2, 3]                # 星号收集剩余
for i, c in enumerate(s): ...        # 带下标遍历
for x, y in zip(A, B): ...           # 并行遍历

# 8) f-string
print(f"{loss:.4f}")

# 9) 类：self 必须显式写，__init__ 是构造函数
class Adam:
    def __init__(self, lr=1e-3):     # 默认参数直接写在签名里
        self.lr = lr                 # 成员变量在这里才创建
    def step(self, p, g):            # 每个方法第一个参数都是 self
        return p - self.lr * g

# 10) 类型注解只是给人和 IDE 看的，运行时不检查
def f(x: List[int]) -> int: ...
```

## 八、把 C++ 循环翻译成 NumPy 的套路

| 你想写的 C++ | NumPy |
|---|---|
| `for(i) c[i] = a[i] + b[i];` | `c = a + b` |
| `for(i) if(a[i]<0) a[i]=0;` | `a = np.maximum(a, 0)` |
| `for(i) s += a[i];` | `s = a.sum()` |
| `for(i) for(j) c[i][j] = a[i] * b[j];` | `c = a[:, None] * b[None, :]` |
| `for(i) out[i] = m[i][idx[i]];` | `out = m[np.arange(n), idx]` |
| `for(i) for(j) d[i][j] = dist(x[i], y[j]);` | `d = ((x[:,None,:] - y[None])**2).sum(-1)` |
| `for(i) if(a[i]>0) v.push_back(a[i]);` | `v = a[a > 0]` |

$$\boxed{\text{看到循环先问：能不能用广播 + 规约表达}}$$

## 自测

1. `a = np.arange(12).reshape(3,4)`，`b = a.T`，改 `b[0,0]` 会影响 `a` 吗？为什么？
2. 写出广播的两条规则。`(3,4)` 和 `(3,)` 能不能相加？怎么改才行？
3. `keepdims=True` 什么时候必须加？举 LayerNorm 的例子。
4. `np.maximum(a, b)` 和 `a.max()` 的区别？
5. 用一行取出 `logp` 每行第 `labels[i]` 列。这在交叉熵里是干什么的？
6. `np.repeat` 和 `np.tile` 的区别？GQA 里为什么必须用前者？
7. `-7 // 2` 在 Python 和 C++ 里分别是多少？
8. 把 `for(i) for(j) d[i][j] = |x[i]-y[j]|` 翻译成 NumPy。
