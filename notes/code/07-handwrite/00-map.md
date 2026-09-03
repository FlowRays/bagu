# 手撕代码合集（41 道）

> 算法岗手撕的核心代码模式：面试里通常是白板或共享文档，只写 `class Solution` 里那段核心计算，统一 Python + NumPy。
> 这里每题一段可直接默写的实现，**全部本地跑通**，并做了交叉验证（MHA 单头 ≡ self-attention、GQA 全头 ≡ MHA、
> KV cache 增量 ≡ 整段 causal attention、反向传播 ≡ 数值梯度、AUC ≡ 暴力枚举）。

> 题目清单参考 sspoffer 的公开题单（<https://sspoffer.com/oj/48>），**题解是自己写的**，和原站无关。

## 分册

| 册 | 内容 |
|---|---|
| [01 机器学习](01-ml.md) | K-Means、逻辑回归、PCA、线性回归、KNN、AUC |
| [02 深度学习](02-dl.md) | 卷积 / 池化 / im2col、Sigmoid / Softmax / CE / BCE / KL、LayerNorm / BatchNorm、SGD / Adam、反向传播、MLP |
| [03 大模型](03-llm.md) | Self-Attention / MHA / Cross / GQA / MLA、KV Cache、RMSNorm、SwiGLU、Encoder Block、InfoNCE、SFT / DPO / PPO / GRPO、LoRA、RQ-VAE、模型分片 |

## 按优先级刷

> 按面试出现频次分档。同档内先刷通过率低的。

### 第一梯队：必须闭着眼默写

| 题目 | 板块 | 难度 | 题解 |
|---|---|---|---|
| Multi-Head Attention | 大模型 | 中等 | [写法](03-llm.md#multi-head-attention) |
| Self-Attention | 大模型 | 中等 | [写法](03-llm.md#self-attention) |
| 交叉熵损失 | 深度学习 | 中等 | [写法](02-dl.md#交叉熵损失) |
| Transformer Encoder Block | 大模型 | 中等 | [写法](03-llm.md#transformer-encoder-block) |
| AUC 计算 | 机器学习 | 中等 | [写法](01-ml.md#auc-计算) |
| InfoNCE 对比学习损失 | 大模型 | 中等 | [写法](03-llm.md#infonce-对比学习损失) |
| MLP训练 | 深度学习 | 中等 | [写法](02-dl.md#mlp训练) |
| Grouped Query Attention | 大模型 | 中等 | [写法](03-llm.md#grouped-query-attention) |

### 第二梯队：要能现场推出来

| 题目 | 板块 | 难度 | 题解 |
|---|---|---|---|
| IoU Loss | 深度学习 | 简单 | [写法](02-dl.md#iou-loss) |
| Cross Attention | 大模型 | 中等 | [写法](03-llm.md#cross-attention) |
| Softmax | 深度学习 | 中等 | [写法](02-dl.md#softmax) |
| FFN（前馈网络前向传播） | 深度学习 | 中等 | [写法](02-dl.md#ffn前馈网络前向传播) |
| RMS Normalization | 大模型 | 中等 | [写法](03-llm.md#rms-normalization) |
| 梯度下降求平方根 | 深度学习 | 中等 | [写法](02-dl.md#梯度下降求平方根) |
| KL 散度 | 深度学习 | 中等 | [写法](02-dl.md#kl-散度) |
| GRPO 损失函数 | 大模型 | 困难 | [写法](03-llm.md#grpo-损失函数) |
| K-Means 聚类 | 机器学习 | 中等 | [写法](01-ml.md#k-means-聚类) |
| 反向传播（两层全连接网络） | 深度学习 | 中等 | [写法](02-dl.md#反向传播两层全连接网络) |
| SGD 随机梯度下降 | 深度学习 | 中等 | [写法](02-dl.md#sgd-随机梯度下降) |
| DPO 损失函数 | 大模型 | 困难 | [写法](03-llm.md#dpo-损失函数) |
| SFT 损失函数（Shift Right） | 大模型 | 中等 | [写法](03-llm.md#sft-损失函数shift-right) |
| 多通道卷积计算 | 深度学习 | 中等 | [写法](02-dl.md#多通道卷积计算) |

### 第三梯队：见过、能补出来

| 题目 | 板块 | 难度 | 题解 |
|---|---|---|---|
| Sigmoid | 深度学习 | 中等 | [写法](02-dl.md#sigmoid) |
| BCE（二元交叉熵损失） | 深度学习 | 中等 | [写法](02-dl.md#bce二元交叉熵损失) |
| PPO 损失函数 | 大模型 | 中等 | [写法](03-llm.md#ppo-损失函数) |
| 线性回归 | 机器学习 | 中等 | [写法](01-ml.md#线性回归) |
| Layer Normalization | 深度学习 | 中等 | [写法](02-dl.md#layer-normalization) |
| 逻辑回归 | 机器学习 | 中等 | [写法](01-ml.md#逻辑回归) |
| 一维卷积 im2col 矩阵展开 | 深度学习 | 中等 | [写法](02-dl.md#一维卷积-im2col-矩阵展开) |
| Max Pooling（二维最大池化前向传播） | 深度学习 | 中等 | [写法](02-dl.md#max-pooling二维最大池化前向传播) |
| LoRA 低秩适配 | 大模型 | 中等 | [写法](03-llm.md#lora-低秩适配) |
| MLA（Multi-head Latent Attention） | 大模型 | 中等 | [写法](03-llm.md#mlamulti-head-latent-attention) |
| SwiGLU（LLaMA 系激活前向传播） | 大模型 | 中等 | [写法](03-llm.md#swiglullama-系激活前向传播) |
| Adam 优化器 | 深度学习 | 中等 | [写法](02-dl.md#adam-优化器) |
| RQ-VAE Loss | 大模型 | 中等 | [写法](03-llm.md#rq-vae-loss) |
| KNN | 机器学习 | 中等 | [写法](01-ml.md#knn) |
| Batch Normalization | 深度学习 | 中等 | [写法](02-dl.md#batch-normalization) |
| Linear Layer | 深度学习 | 简单 | [写法](02-dl.md#linear-layer) |
| PCA | 机器学习 | 中等 | [写法](01-ml.md#pca) |
| KV Cache 增量注意力 | 大模型 | 中等 | [写法](03-llm.md#kv-cache-增量注意力) |
| 模型分片加载 | 大模型 | 中等 | [写法](03-llm.md#模型分片加载) |

## 几个反复出现的套路

把这几条记住，41 道里大部分能现推：

1. **减最大值再 exp** —— softmax、log_softmax、交叉熵、InfoNCE 全靠它防溢出。
2. **交叉熵对 logits 的梯度就是 `p − q`** —— 逻辑回归、MLP 反向、SFT loss 都直接用，不用再乘 sigmoid/softmax 的导数。见 [推导](../../06-post-training/distill/01-entropy-ce-kl-jsd.md#5-一个统一的梯度logits-梯度等于-p-减-q)。
3. **attention 三件套**：除 √dh、拆头要 transpose 成 `(h,L,dh)`、最后过 `Wo`。MHA/Cross/GQA/MLA 只是在这个骨架上换 K/V 从哪来。
4. **ratio 用 `exp(logp − logp_old)`** —— PPO、GRPO 都是，别直接除概率。
5. **归一化三兄弟**：LayerNorm 沿特征维、BatchNorm 沿 batch 维、RMSNorm 不减均值。`eps` 一律在开方**里面**。
6. **滑窗类**（conv / pooling / im2col）输出尺寸都是 `(H + 2p − k) // s + 1`。

## 和理论笔记的对照

手撕考的是「能不能写出来」，理论笔记考的是「为什么这么写」，两边配着看：

| 手撕题 | 对应理论 |
|---|---|
| 交叉熵 / BCE / Softmax | [熵、交叉熵、KL、JSD](../../06-post-training/distill/01-entropy-ce-kl-jsd.md) |
| KL 散度 | [forward vs reverse KL](../../06-post-training/rl/07-kl.md) |
| SFT 损失 | [SFT 与传统 KD](../../06-post-training/distill/02-sft-and-kd.md) |
| PPO 损失 | [clip 与 min](../../06-post-training/rl/03-clip-and-min.md) |
| GRPO 损失 | [group-relative advantage](../../06-post-training/rl/06-grpo.md) |
| Adam / SGD | [显存账本](../../03-training-fundamentals/01-memory-accounting.md)、[梯度累积](../../03-training-fundamentals/03-gradient-accumulation.md) |
| 反向传播 | [为什么要存 activation](../../03-training-fundamentals/01-memory-accounting.md#6-activation为什么必须存) |
| InfoNCE | [CLIP 与 SigLIP 的 loss](../../07-vlm/02-vision-encoder.md) |

还没有理论笔记、但 bagu.md 已规划的：attention 家族（Self/MHA/GQA/MLA/KV Cache）、norm 家族、FFN/SwiGLU、DPO、LoRA。
这几块正好可以拿手撕当入口反过来补。
