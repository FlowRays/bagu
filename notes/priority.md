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
- [ ] **OPD**（On-Policy Distillation）
  - 已知线索：slime 有实现，`--use-opd` / `--opd-kl-coef`，做的是 **reverse KL 蒸馏**（把 student 对 teacher 的 reverse KL 当惩罚减进 advantage），与 advantage estimator 正交
- [ ] **OPSD**

自测题库：[rl/self-test.md](06-post-training/rl/self-test.md)（82 题）

## 优先级 2：VLM

对应目录 `07-vlm/`

- [ ] **Vision architecture**：Qwen3 / Qwen3.5 的视觉部分，视觉信息如何进入 LLM
- [ ] **VLM 训练阶段划分**：各阶段冻结什么、优化什么
- [ ] **VLM SFT loss 设计**：Vision Encoder 不冻结时，分阶段各用什么 loss

## 优先级 3：框架代码

对应目录 [`code/06-frameworks/`](code/06-frameworks/rl-framework-source-reading.md)

- [ ] slime / verl 里 PPO / GRPO / DAPO / GSPO 核心算法的实现位置与细节
- [ ] 公式 → 代码的逐条对照
