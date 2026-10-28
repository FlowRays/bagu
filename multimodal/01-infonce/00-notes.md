# 模块 1：表征与对比学习（InfoNCE）

## 1. 什么是"表征"
编码器 $f_\theta$ 把输入 $x$（图像、文本……）映射成向量 $z = f_\theta(x) \in \mathbb{R}^D$。好的表征满足：
**语义相近 → 向量相近**，并且下游任务用一个简单模型（比如线性分类器 $\hat y = Wz + b$）就能读出所需信息。

## 2. 对比学习的思路
手里没有标签，但知道哪些样本是"一对"（正样本对）：
- 同一张图的两种增强 $(x, x')$（SimCLR，本模块用这个）
- 一张图和它的描述文字 $(x^{\text{img}}, x^{\text{txt}})$（CLIP，模块 2/3）

目标：正样本对拉近，非配对样本（负样本）推远。

## 3. InfoNCE
一个 batch 有 $N$ 对 $(u_i, v_i)$，$i = 1,\dots,N$。先做 L2 归一化 $\hat u_i = u_i / \lVert u_i \rVert$，相似度取余弦：

$$
s_{ij} = \hat u_i^\top \hat v_j \in [-1, 1]
$$

对第 $i$ 个样本，把它看作一个 **$N$ 选 1 的分类问题**，正确答案是 $j = i$：

$$
\mathcal{L}_i = -\log \frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{N} \exp(s_{ij}/\tau)}
$$

写成矩阵形式：logits 矩阵是 $S/\tau = \hat U \hat V^\top / \tau \in \mathbb{R}^{N\times N}$，标签是对角线 $y_i = i$，loss 就是逐行的 cross-entropy。
CLIP 用对称版本，行方向（图→文）和列方向（文→图）各做一次：

$$
\mathcal{L} = \frac{1}{2}\Big[\mathrm{CE}(S/\tau,\ y) + \mathrm{CE}(S^\top/\tau,\ y)\Big]
$$

**temperature $\tau$**：对 $\mathcal{L}_i$ 求导，

$$
\frac{\partial \mathcal{L}_i}{\partial s_{ij}} = \frac{1}{\tau}\big(p_{ij} - \mathbb{1}[j=i]\big), \qquad p_{ij} = \frac{\exp(s_{ij}/\tau)}{\sum_k \exp(s_{ik}/\tau)}
$$

所以每个负样本受到的推力正比于 $p_{ij}$。$\tau$ 越小，softmax 越尖锐，梯度越集中在最难（最相似）的负样本上；$\tau$ 太小训练会不稳定，太大则区分度不够。常见取值是 $0.05$–$0.5$，CLIP 把 $\log(1/\tau)$ 设成可学习参数，初始 $\tau = 0.07$。

**为什么需要很多负样本**：InfoNCE 给出互信息的下界，

$$
I(u; v) \ \ge\ \log N - \mathcal{L}_{\text{InfoNCE}}
$$

这个界最大不超过 $\log N$。$N$ 越大，界越紧，任务也越难，学到的表征通常越好，所以 CLIP 用 $N = 32768$ 的 batch。顺带一个基准值：随机初始化时所有 $s_{ij}$ 都差不多，$\mathcal{L} \approx \log N$。

**为什么要归一化**：不归一化的话，$s_{ij}$ 没有上界，模型可以靠放大 $\lVert u \rVert$ 把 logits 拉开来降低 loss，而不是去学语义。归一化之后，相似度被限制在 $[-1, 1]$，尺度只由 $\tau$ 控制。

## 4. 练习
1. 在 [infonce.py](infonce.py) 里补全 `info_nce`（大约 4 行）
2. `python check_infonce.py`（[check_infonce.py](check_infonce.py)）：和参考实现对比，并做几个直觉测试
3. `python train_mnist.py --tsne`（[train_mnist.py](train_mnist.py)）：用你写的 loss 在 MNIST 上做 SimCLR（不用标签），
   然后用 linear probe 评估表征，并和随机初始化的编码器对比。
   `--tsne` 会在当前目录保存 `tsne_random.png` 和 `tsne_simclr.png`，跑完后贴到下面的「结果」一节
4. 实验：分别用 `--temperature 0.05 / 0.2 / 1.0` 和 `--batch-size 64 / 1024` 跑，
   把 probe 准确率记在下面的表里

| $\tau$ | batch | probe acc |
|---|---|---|
|   |   |   |

## 5. 自测问题
- 如果一个 batch 里有两张图其实是同一个数字（"假负样本"），会发生什么？
- 为什么 linear probe 用的是 encoder 的输出，而不是 projection head 之后的 z？

## 6. 参考数字（用参考答案（[_solution.py](_solution.py)）跑 3 个 epoch，每个 epoch 约 3 秒）
随机初始化 CNN 的 probe acc 约 0.957，SimCLR 训练后约 0.988。
注意：随机初始化的 CNN 在 MNIST 上已经很强了，这是卷积的归纳偏置（inductive bias）带来的。
所以评估表征一定要和随机初始化做对比，模块 4 换到更难的数据上，差距会更明显。

## 7. 结果

（跑完后记在这里：probe 准确率表、t-SNE 图、观察到的现象）
