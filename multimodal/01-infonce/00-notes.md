# 模块 1：表征与对比学习（InfoNCE）

## 1. 什么是"表征"
编码器 f 把输入 x（图像、文本……）映射成向量 z = f(x)。好的表征满足：
**语义相近 → 向量相近**，并且下游任务用一个简单模型（比如线性分类器）就能读出所需信息。

## 2. 对比学习的思路
手里没有标签，但知道哪些样本是"一对"（正样本对）：
- 同一张图的两种增强（SimCLR，本模块用这个）
- 一张图和它的描述文字（CLIP，模块 2/3）

目标：正样本对拉近，非配对样本（负样本）推远。

## 3. InfoNCE
一个 batch 有 N 对 (z1_i, z2_i)，先做 L2 归一化，相似度取余弦 s_ij = z1_i · z2_j。
对第 i 个样本，把它看作一个 **N 选 1 的分类问题**：正确答案是 j = i。

    L_i = -log( exp(s_ii / τ) / Σ_j exp(s_ij / τ) )

也就是：logits = (Z1 @ Z2ᵀ) / τ，labels = arange(N)，然后做 cross_entropy。
CLIP 用对称版本：(CE(logits, labels) + CE(logitsᵀ, labels)) / 2。

**temperature τ**：τ 越小，softmax 越尖锐，越关注最难的负样本；τ 太小训练会不稳定，太大则区分度不够。常见取值是 0.05–0.5，CLIP 把它设成可学习参数（初始 0.07）。

**为什么需要很多负样本**：InfoNCE 是互信息的下界，这个界最大不超过 log N。batch 越大，任务越难，学到的表征通常越好。所以 CLIP 用 32k 的 batch。

**为什么要归一化**：不归一化的话，模型可以靠放大向量模长来降低 loss，而不是去学语义。

## 4. 练习
1. 在 [infonce.py](infonce.py) 里补全 `info_nce`（大约 4 行）
2. `python check_infonce.py`（[check_infonce.py](check_infonce.py)）：和参考实现对比，并做几个直觉测试
3. `python train_mnist.py --tsne`（[train_mnist.py](train_mnist.py)）：用你写的 loss 在 MNIST 上做 SimCLR（不用标签），
   然后用 linear probe 评估表征，并和随机初始化的编码器对比。
   `--tsne` 会在当前目录保存 `tsne_random.png` 和 `tsne_simclr.png`，跑完后贴到下面的「结果」一节
4. 实验：分别用 `--temperature 0.05 / 0.2 / 1.0` 和 `--batch-size 64 / 1024` 跑，
   把 probe 准确率记在下面的表里

| τ | batch | probe acc |
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
