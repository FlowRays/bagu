# 评测：benchmark 与 setting

> 面试里问评测，考的往往不是"你知道哪些榜"，而是**你知不知道分数怎么被做出来的**。

## 1. 常见 benchmark

### LLM

| 类别 | 代表 | 考什么 |
|---|---|---|
| 知识 | MMLU、MMLU-Pro、CMMLU | 多选题，广度知识 |
| 数学 | GSM8K、MATH、AIME | 多步推理 |
| 代码 | HumanEval、MBPP、LiveCodeBench | 生成可执行代码 |
| 推理 | BBH、GPQA、ARC | 复杂推理 |
| 长文本 | LongBench、RULER、大海捞针 | 长上下文利用 |
| 对话 | MT-Bench、AlpacaEval、Arena | 主观质量，LLM-as-judge 或人评 |
| 指令 | IFEval | 能否严格遵守格式约束 |

### VLM

| 类别 | 代表 |
|---|---|
| 综合 | MMMU、MMBench、MMStar |
| OCR / 文档 | DocVQA、ChartQA、OCRBench |
| 幻觉 | POPE、HallusionBench |
| 定位 | RefCOCO |
| 视频 | Video-MME |

### Agent

| 类别 | 代表 |
|---|---|
| 软件工程 | SWE-bench（真实 GitHub issue） |
| 工具调用 | BFCL、ToolBench |
| 网页 | WebArena、Mind2Web |
| GUI | OSWorld、AndroidWorld |

## 2. Setting 比 benchmark 更影响分数

同一个模型同一个 benchmark，setting 不同分数可以差十几个点。

| 维度 | 选项 | 影响 |
|---|---|---|
| **shot 数** | 0-shot / 5-shot | few-shot 教格式，对弱模型帮助巨大 |
| **CoT** | 直接答 / 让它 step by step | 推理类任务差距极大 |
| **答案提取** | 正则 / 让模型输出固定格式 / LLM 判 | 提取失败会被算成答错 |
| **打分方式** | exact match / **log-likelihood 选项比较** | 多选题用 loglikelihood 比生成再匹配稳得多 |
| **采样** | greedy / $T>0$ 多次 | 影响可复现性 |
| **prompt 模板** | 是否套 chat template | base 模型和 instruct 模型要求不同 |

$$\boxed{\text{报分数必须报 setting，否则不可比}}$$

### pass@k

代码和数学常用。采样 $n$ 条，其中 $c$ 条正确，无偏估计：

$$\text{pass@}k=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}$$

含义是「采 $k$ 条至少一条对」的概率。$k=1$ 时就是准确率。

**注意**：pass@k（$k>1$）衡量的是**模型有没有能力**，pass@1 才衡量**能不能稳定做对**。RL 的一个典型效果是把 pass@k 的能力压缩进 pass@1，但 pass@k 本身可能不涨甚至略降。

### maj@k / self-consistency

采 $k$ 条，取出现最多的答案。比 pass@k 更接近实际可用性（因为不需要知道正确答案）。

## 3. 污染

**训练数据里混进了评测集** → 分数虚高但没有真实能力。

检测手段：

- n-gram 重叠比对
- 让模型续写 benchmark 题目（能背出来就是背过）
- 看在**同分布但全新**的题上是否掉点（比如 GSM8K vs GSM1K）

$$\boxed{\text{去污染是数据流水线的必做项，见}}$$ [预训练数据](../05-pretraining/01-data.md#2-清洗流水线) 与 [SFT 数据](../06-post-training/sft/02-data-and-cot.md#3-数据去重与过滤)。

## 4. LLM-as-judge

用强模型给回答打分或做 pairwise 比较。

**已知偏置**：

| 偏置 | 说明 | 缓解 |
|---|---|---|
| **位置偏置** | 偏好排在前面的那个 | 交换顺序各评一次取平均 |
| **长度偏置** | 偏好更长的回答 | 长度控制 / 长度去偏 |
| **自我偏好** | 偏好自己家族模型的输出 | 用多个不同家族的 judge |
| 格式偏置 | 偏好带 markdown、分点的 | 统一格式要求 |

和 [RM 的问题](../06-post-training/preference/01-rm-and-rlhf.md#3-rm-的问题) 高度重合，本质是同一类问题。

## 5. 怎么判断一个评测结果可不可信

面试可以按这个清单答：

1. setting 是否完整披露（shot / CoT / 提取方式 / 采样）
2. 是否做了去污染
3. 是否在**新题**或**私有题**上验证过
4. 主观评测是否处理了位置和长度偏置
5. 是否只报了自己擅长的榜（cherry-picking）
6. 多次采样的方差有多大

## 自测

1. LLM / VLM / Agent 各举三个代表 benchmark。
2. ⭐ 哪些 setting 会显著影响分数？为什么多选题推荐用 log-likelihood 而不是生成后匹配？
3. ⭐⭐ 写出 pass@k 的无偏估计公式。pass@1 和 pass@k 分别衡量什么？RL 对两者的典型影响？
4. maj@k 和 pass@k 的区别？哪个更接近实际可用性？
5. ⭐ 污染怎么检测？三种手段。
6. ⭐ LLM-as-judge 的四种偏置和缓解办法？它和 RM 的问题有什么关系？
7. 判断一个评测结果可信度的清单？
