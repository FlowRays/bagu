# 学习优先级

> 整体知识体系见 [bagu.md](../bagu.md)。这一页只记录**当前优先攻的主题**和进度。
> 学习流程：跟 GPT 交互理解 → 整理结构化笔记 → 关掉笔记口述 / 纸上推导 → 暴露问题 → 回填笔记。

## 优先级 1：RL / Post-training

对应目录 [`06-post-training/rl/`](06-post-training/rl/00-map.md)

- [x] **PPO 全链条**：$J \to \nabla J \to$ surrogate $\to$ importance sampling $\to$ clip/min $\to$ GAE/critic $\to$ 工程细节
- [x] **GRPO**：group-relative advantage、为什么不用 critic、同 prompt 分组、seq→token 分解
- [x] **KL 散度**：定义与三性质、$\ge 0$ 的证明、forward vs reverse、mode-covering/seeking、免背记忆法
- [x] **DAPO**：Clip-Higher、Dynamic Sampling、token-level loss 聚合、overlong reward shaping
- [x] **GSPO**：sequence-level importance ratio（内容较薄，待深挖）

自测题库：[rl/self-test.md](06-post-training/rl/self-test.md)（82 题）

## 优先级 1b：蒸馏 / OPD / OPSD

对应目录 [`06-post-training/distill/`](06-post-training/distill/00-map.md)

- [x] **信息论地基**：熵 / 交叉熵 / KL / JSD，$H(q,p)=H(q)+D_{KL}(q\|p)$，$\partial L/\partial z=p-q$
- [x] **SFT 与传统 KD**：teacher forcing、exposure bias、SFT 为什么是 forward KL、teacher 采样 ≈ MC forward KL
- [x] **OPD**：on-policy 指 state distribution、四象限（prefix 来源 × KL 方向）、forward-KL OPD ≠ SFT
- [x] **reverse KL = policy gradient**：梯度推导、$A_t=\log\pi_T-\log\pi_S$、KL-regularized RL ≡ sequence-level reverse KL
- [x] **KL 估计粒度**：sampled-token / top-k / full-vocab，显存账、per-entry KL clipping
- [x] **OPSD**：privileged information、GT-aware correction、teacher 固定 snapshot、主实验用 forward KL
- [x] **Rethink OPD / MOPD**：distillability ≠ capability、Top-K overlap、cold start、多教师能力集成

自测题库：[distill/self-test.md](06-post-training/distill/self-test.md)（53 题）

待办：
- [ ] slime 里 `--use-opd` / `--opd-kl-coef` 的实现和笔记里的公式逐条对照

## 优先级 2：VLM

对应目录 [`07-vlm/`](07-vlm/00-map.md)

- [x] **Vision architecture**：image → ViT → projector → LLM，visual token，统一 self-attention
- [x] **Vision Encoder**：CLIP / SigLIP / SigLIP2 / DINO，Qwen3-VL / Qwen3.5 / Kimi K3 / DeepSeek-V4-Flash-Vision 配置，为什么不 scale vision encoder
- [x] **visual token 压缩**：2×2 merge、3×3 aligner、resampler、pruning、token budget
- [x] **VLM 训练阶段划分**：Stage −1/0/1 → SFT → distill → RL，各阶段冻结什么、更新什么
- [x] **VLM SFT loss 设计**：三个模块共享一个 loss，语言 CE 怎么反传到 ViT，RL 为什么倾向 freeze ViT

自测题库：[07-vlm/self-test.md](07-vlm/self-test.md)（36 题）

## 优先级 2b：显存 / 训练工程 / 分布式

对应目录 [`03-training-fundamentals/`](03-training-fundamentals/01-memory-accounting.md)、[`04-distributed-infra/`](04-distributed-infra/01-ddp-and-zero.md)

- [x] **显存账本**：2/4 byte 从哪来、12 vs 16 B/param、activation、logits
- [x] **Gradient Checkpointing**：实现粒度、省多少、增加 33%
- [x] **Gradient Accumulation**：等价性、token normalization 坑、和 checkpointing 的区别
- [x] **Sequence packing / gradient clipping**
- [x] **DDP → ZeRO-1/2/3 → FSDP1/FSDP2**：通信量 2M/2M/3M
- [x] **并行全图**：DP/TP/PP/SP/CP/EP 各切什么维度，8 卡怎么排
- [x] **大模型参数构成**：MoE 占 98%，total vs activated
- [x] **SFT / OPD / GRPO / PPO 显存对比**

自测题库：[03-training-fundamentals/self-test.md](03-training-fundamentals/self-test.md)（52 题）

## 优先级 2c：手撕代码

对应目录 [`code/07-handwrite/`](code/07-handwrite/00-map.md)

- [x] **机器学习 6 道**：K-Means、逻辑回归、PCA、线性回归、KNN、AUC
- [x] **深度学习 18 道**：卷积/池化/im2col、Sigmoid/Softmax/CE/BCE/KL、LayerNorm/BatchNorm、SGD/Adam、反向传播、MLP
- [x] **大模型 17 道**：Self-Attention/MHA/Cross/GQA/MLA、KV Cache、RMSNorm、SwiGLU、Encoder Block、InfoNCE、SFT/DPO/PPO/GRPO、LoRA、RQ-VAE、模型分片

全部本地跑通并交叉验证。优先级排序见 [手撕总表](code/07-handwrite/00-map.md#按优先级刷)。

## 优先级 3：框架代码

对应目录 [`code/06-frameworks/`](code/06-frameworks/rl-framework-source-reading.md)

- [ ] slime / verl 里 PPO / GRPO / DAPO / GSPO 核心算法的实现位置与细节
- [ ] 公式 → 代码的逐条对照
