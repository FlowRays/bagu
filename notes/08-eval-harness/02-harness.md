# Agent harness：从 ReAct 到编码智能体

> harness 指的是**把模型包成一个能干活的智能体**的那层框架：怎么给工具、怎么循环、怎么管上下文。

## 1. ReAct：最基础的循环

**Reason + Act** 交替：

```text
Thought: 我需要先查一下天气
Action: weather(city="北京")
Observation: {"temp":18,"rain":true}
Thought: 有雨，应该建议带伞
Answer: 明天北京有雨，建议带伞。
```

循环直到模型输出最终答案或达到步数上限。

**关键点**：
- `Observation` 是**环境返回的**，不是模型生成的。训练时绝对不能算 loss（见 [agentic SFT](../06-post-training/sft/02-data-and-cot.md#5-agentic-sft)）
- 现在的实现基本都用**原生 function calling**（结构化 JSON）而不是让模型输出 `Action: xxx` 再正则解析，因为格式错误率低得多

## 2. 一个 harness 要解决什么

| 问题 | 常见做法 |
|---|---|
| **工具定义** | JSON schema 放 system prompt；工具太多时先做检索只放相关的 |
| **上下文膨胀** | 历史越来越长 → 截断、摘要、把长输出存成文件只保留引用 |
| **错误恢复** | 工具报错要把错误信息喂回去让模型改，而不是直接失败 |
| **循环终止** | 最大步数、重复动作检测、无进展检测 |
| **并行** | 无依赖的工具调用可以并发 |
| **权限与安全** | 危险操作（删文件、发请求）需要确认或沙箱 |

**上下文膨胀是最核心的工程问题**。agent 跑几十步后 context 就满了，而且前面的内容大多已经无关。

$$\boxed{\text{agent 场景每轮都带长历史}\Rightarrow\text{prefix caching 收益极大}}$$

见 [prefix caching](../02-inference-serving/02-serving-optimization.md#3-prefix-caching复用相同前缀的-kv)。

## 3. 编码智能体（Claude Code / Codex / OpenHands 这类）

在通用 harness 之上还要解决：

| 问题 | 做法 |
|---|---|
| **代码库太大放不进 context** | 不是全读进来，而是给**搜索工具**（grep / glob / 语义检索），让模型按需读 |
| **编辑要精确** | 用「查找-替换」式的结构化编辑而不是让模型重写整个文件 |
| **验证** | 跑测试 / 编译 / lint，把结果喂回去形成闭环 |
| **多文件改动的一致性** | 让模型先做计划，再逐个执行 |

**核心设计哲学**：不要试图把所有信息塞进 context，而是给模型**探索环境的工具**，让它自己按需获取。这和人类程序员的工作方式一致。

## 4. 评测 agent 的难点

| 难点 | 说明 |
|---|---|
| **环境不可复现** | 网页会变、依赖会更新，同一个 benchmark 不同时间跑分数不同 |
| **多路径正确** | 同一个任务有多种正确解法，难以用固定答案判 |
| **部分成功** | 完成了 80% 算不算成功？需要设计分级指标 |
| **成本差异大** | 一个 agent 跑 5 步、另一个跑 50 步，都成功了但成本差 10 倍 |
| **判定本身要模型** | 用 LLM 判成功与否，又引入 judge 的偏置 |

所以 agent benchmark 通常要求**报告 pass rate + 平均步数 + 平均 token 消耗**三个数，只报成功率是不完整的。

SWE-bench 之所以受重视，正是因为它用**真实的单元测试**做判定，绕开了 LLM-as-judge。

## 5. Agent 的 RL

把 harness 当环境，用 [RL](../06-post-training/rl/00-map.md) 训 policy。特有的难点：

- **reward 稀疏**：只有最后成功/失败一个信号，中间几十步没有反馈
- **credit assignment 极难**：失败了到底是哪一步的错
- **rollout 极慢**：每步都要真的调工具、等环境返回
- **reward hacking**：模型可能学会绕过任务本身（比如直接改测试文件让测试通过）

对应的手段：过程奖励（PRM）、把长任务拆成子任务、用 [OPD](../06-post-training/distill/00-map.md) 这类 dense 监督替代稀疏 reward。

## 自测

1. ReAct 的循环是什么？Observation 为什么不能算 loss？
2. 为什么现在用原生 function calling 而不是正则解析？
3. ⭐ 一个 harness 要解决哪些问题？哪个是最核心的工程问题？
4. 为什么 agent 场景 prefix caching 收益特别大？
5. ⭐ 编码智能体怎么处理"代码库放不进 context"？核心设计哲学是什么？
6. ⭐ 评测 agent 有哪五个难点？应该报哪三个数？SWE-bench 为什么受重视？
7. Agent RL 的四个特有难点？各有什么应对手段？
