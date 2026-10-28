# 多模态表征：一个下午的实践课

**目标：用"基础知识 + 代码实践"的方式，一个下午（约 4 小时）弄懂多模态表征的基础。**
每个模块都有一页笔记（概念、公式、练习、实验记录）和可以直接运行的代码；代码文件在左侧目录里点开就能看。

学完要能回答四个问题：

1. 为什么对比学习能让图像和文本落到同一个空间？
2. temperature 和 batch size 分别影响什么？
3. CLIP 的表征和纯视觉自监督表征（DINOv2）各擅长什么？
4. 多模态大模型是怎么"接上"视觉表征的？

## 计划与进度

| # | 模块 | 时长 | 基础知识 | 代码实践 | 状态 |
|---|---|---|---|---|---|
| 0 | [环境](multimodal/00-env/00-setup.md) | 15 min | – | uv 环境，缓存放在 /storage 下 | ✅ |
| 1 | [表征与 InfoNCE](multimodal/01-infonce/00-notes.md) | 40 min | embedding 空间、InfoNCE、temperature、负样本数量 | 手写 InfoNCE，在 MNIST 上做 SimCLR，用 linear probe 评估 | 进行中 |
| 2 | CLIP：图文对齐 | 60 min | 双塔结构、对称对比损失、zero-shot 分类、prompt 模板 | open_clip ViT-B/32：CIFAR-10 zero-shot、图文检索 Recall@K、prompt 对比 | 待开始 |
| 3 | 从零训练 mini-CLIP | 60 min | 冻结 encoder + 投影头；batch size / τ 的影响 | 冻结的视觉和文本编码器 + 两个投影层，在 Flickr 上训练 | 待开始 |
| 4 | 表征质量怎么评估 | 40 min | linear probe 和 zero-shot 的区别、CLIP vs DINOv2、modality gap | 两种特征的 probe 对比，UMAP 可视化 modality gap | 待开始 |
| 5 | 从表征到 MLLM（选学） | 30 min | LLaVA 式 encoder → projector → LLM，visual token | 拆开一个小 VLM，看视觉 token 和 projector 的结构 | 待开始 |

## 怎么学

先读笔记 → 补全代码里标 `TODO` 的部分（参考答案在 `_solution.py`，先自己写再看）→ 运行 → 把实验结果和图记回笔记里。

和 [LLM/VLM 八股](bagu.md) 的关系：八股里的 VLM 章节讲架构和训练细节，这里是动手版，重点是"表征"本身。
