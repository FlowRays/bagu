# VLM 总图

> 目标不是背模型名，而是能回答"**为什么这么设计**"。

## 一条主线

$$I\xrightarrow{\ \text{ViT}\ }Z_v\xrightarrow{\ \text{Projector}\ }H_v\ \longrightarrow\ [\,H_v;H_t\,]\xrightarrow{\ \text{LLM}\ }\text{next token}$$

```text
raw image → patchify → ViT → semantic visual tokens → projector
         → LLM-compatible visual tokens → [visual tokens + text tokens]
         → LLM self-attention → next-token prediction
```

训练侧：

```text
CLIP/SigLIP/DINO pretrain → projector alignment → multimodal pretrain → SFT → distill → RL
```

## 三个核心矛盾

| 矛盾 | 展开在 |
|---|---|
| 一个简单 MLP 凭什么就能把视觉接进 LLM | [01 架构](01-architecture.md) |
| 一张图具体怎么变成 N 个 token | [01b Patch Embedding](01b-patch-embedding.md) |
| LLM 已经几百 B，vision encoder 为什么还只有 0.4B | [02 vision encoder](02-vision-encoder.md) |
| 高分辨率 → visual token 爆炸，怎么压 | [03 visual token 压缩](03-visual-token-compression.md) |

## 卡点索引

| # | 卡点 | 在哪 |
|---|---|---|
| 1 | projector 不只是维度变换 | [01](01-architecture.md#1-projector-不是把-1024-维硬变成-4096-维) |
| 2 | LLM 冻结了，梯度怎么训 projector | [01](01-architecture.md#5-卡点llm-冻结了梯度还能穿过它吗) |
| 3 | visual token 凭什么能和 text token 拼起来 | [01](01-architecture.md#7-统一-self-attention四个象限) |
| 4 | CLIP 和 SigLIP 的 loss 差在哪 | [02](02-vision-encoder.md#1-clip-到-siglip为什么换掉-softmax) |
| 5 | DINO 学的东西和 CLIP 有什么不同 | [02](02-vision-encoder.md#3-dino无标签的自蒸馏) |
| 6 | 为什么没人把 vision encoder scale 上去 | [02](02-vision-encoder.md#6-卡点为什么大家不-scale-vision-encoder) |
| 7 | Kimi K3 凭什么敢不用 SigLIP 初始化 | [02](02-vision-encoder.md#5-kimi-k3从零训练的-moonvit-v2) |
| 8 | 9 patch 合 1 token 和平均池化的区别 | [03](03-visual-token-compression.md#2-spatial-merge--pixel-shuffle) |
| 9 | pretrain 的 CE 和 SFT 的 CE 有什么不同 | [04](04-training-stages.md#5-pretrain-的-ce-和-sft-的-ce-差在哪) |
| 10 | 三个模块是不是各有一个 loss | [04](04-training-stages.md#6-卡点三个模块并不各有一个-loss) |
| 11 | 语言 CE 凭什么能训练 ViT | [04](04-training-stages.md#7-为什么语言-ce-能更新-vit) |
| 12 | 图片不是输出，那它有梯度吗 | [04](04-training-stages.md#8-卡点image-token-是-condition不是-prediction-target) |
| 13 | RL 为什么倾向冻结 ViT | [04](04-training-stages.md#10-rl-为什么更倾向-freeze-vit) |
| 14 | conv 凭什么等价于 patch embedding | [01b](01b-patch-embedding.md#7-为什么-stride-也要等于-patch-size) |
| 15 | out_channels 和 token 维度的关系 | [01b](01b-patch-embedding.md#9--channel-和-token-维度别搞混) |
| 16 | ViT 里的 2D RoPE 和 LLM 里的 MRoPE 是不是一回事 | [05](05-qwen25-vl.md#4-vit-内部是-2d-rope不是-mrope) |
| 17 | visual embedding 到底怎么进 LLM 序列 | [05](05-qwen25-vl.md#6--visual-token-怎么真的进到-llm-的序列里) |
| 18 | window attention 是不是把 token 数减少了 | [05](05-qwen25-vl.md#3-window-attention28-层局部--4-层全局) |
| 19 | DeepStack 是不是 concat，会不会撑长序列 | [06](06-deepstack-and-qwen-evolution.md#4--最关键的一点不是-concat是逐元素相加) |
| 20 | Qwen3.5 为什么反而删掉 DeepStack | [06](06-deepstack-and-qwen-evolution.md#8-那-qwen35-为什么把-deepstack-删了) |
| 21 | native multimodal 是不是就没有 ViT 了 | [06](06-deepstack-and-qwen-evolution.md#9--native-multimodal-不等于没有-vit) |
| 22 | 混纯文本数据时 ViT 有没有梯度 | [04](04-training-stages.md#-混纯文本数据时vit-拿不到梯度) |

## 阅读顺序

1. [01 架构：图片怎么进 LLM](01-architecture.md)
2. [01b Patch Embedding](01b-patch-embedding.md) — **数学前置**，conv 忘干净了就先看这篇
3. [02 Vision Encoder：CLIP / SigLIP / DINO 与主流模型配置](02-vision-encoder.md)
4. [03 visual token 压缩](03-visual-token-compression.md)
5. [04 多阶段训练：每阶段的 loss 和更新谁](04-training-stages.md)
6. [05 Qwen2.5-VL 完整走查](05-qwen25-vl.md) — 从 pixel 到 LLM 序列，每步都有 shape
7. [06 DeepStack 与 Qwen 演进](06-deepstack-and-qwen-evolution.md) — 2.5-VL → 3-VL → 3.5
8. [自测题](self-test.md)

相关：[大模型参数构成](../04-distributed-infra/03-model-param-breakdown.md)（vision 塔在超大 MoE 里只占万分之几）。
