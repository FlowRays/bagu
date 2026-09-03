# SFT 自测题库（关掉笔记用）

> 侧边栏顶部有 **「答案」开关**。标 ⭐ 的是高频考点。

## A. 工程（[01](01-sft-basics.md)）

**1.** ⭐ 写出 SFT 的 loss，loss mask 覆盖哪些位置？

> **答：** $\mathcal L=-\sum_{t\in\text{assistant}}\log\pi_\theta(y_t|x,y_{<t})$。system 和 user 段全部 mask（填 `ignore_index=-100`），只在 assistant 段算 loss。

**2.** ⭐⭐ shift 一位怎么做？错了会怎样？

> **答：**
> ```python
> shift_logits = logits[:-1]   # 位置 t 的 logits 预测 t+1，最后一位没 target
> shift_labels = labels[1:]
> loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
> ```
> 错了 loss 看起来正常，但模型学的是错位的映射。这是 SFT 实现最容易错的一步。

**3.** `<|im_end|>` 要不要算 loss？

> **答：** **要**。否则模型不知道什么时候停，会一直生成。这是「模型不会停」这类 bug 的常见原因。

**4.** ⭐ 多轮对话两种做法？为什么推荐一次算完？等价吗？

> **答：** ① 拆成多条样本，每条训一个 assistant 段 —— 前缀被重复前向 $n$ 次；② **一条样本多段 loss**，一次前向所有 assistant 段一起算 —— 省 $n$ 倍计算，推荐。
> **数学上等价**，因为 causal mask 保证第 $i$ 轮的 assistant 看不到第 $i+1$ 轮。

**5.** ⭐ packing 必须配套哪三件事？

> **答：** ① block-diagonal causal mask（或 `cu_seqlens` 走 FlashAttention 变长接口）；② **position_ids 每条重新从 0 开始**；③ loss 不跨样本边界算 next-token。少了任何一条就不等价。

**6.** ⭐⭐ packing 后 loss 怎么归一化？

> **答：** 按**总有效 token 数**：$\mathcal L=\frac{\sum_i\sum_t \ell_{i,t}}{\sum_i|\text{assistant}_i|}$。
> 按「每条样本的平均 loss 再平均」会把长短样本强行拉平，microbatch 之间权重不对。

**7.** SFT 的 lr 和 epoch 为什么都小？

> **答：** lr 比 pretrain 小 1–2 个量级（1e-5~2e-5），因为是在已经很好的分布上微调，大 lr 会破坏 pretrain 能力（灾难性遗忘）。epoch 只 1–3，SFT 极易过拟合。

**8.** ⭐ LoRA vs 全参？$B$ 为什么初始化为 0？

> **答：** LoRA 只有 adapter 有 $G,O$，base 只占 $P$，显存省很多；效果接近但改变深层能力时弱；多任务可以一个 base + 多个 adapter 热插拔；推理可 merge 回 base 零延迟。
> $B=0$ 使起点 $x@W+(x@A@B)\cdot s=x@W$ **严格等于原模型**，训练从原模型平滑出发。

**9.** 灾难性遗忘的缓解手段？

> **答：** ① 数据混合（掺通用/pretrain 数据 replay）；② 小 lr + 少 epoch；③ LoRA（参数改动受限）；④ 加对 base 的 KL 惩罚（这已经接近 OPD 的思路）。

## B. 数据（[02](02-data-and-cot.md)）

**10.** ⭐⭐ LIMA 的结论？怎么解释？

> **答：** 1000 条精心挑选的数据好过几十万条噪声数据，**质 ≫ 量**。
> 解释（Superficial Alignment Hypothesis）：知识和能力几乎全在 pretrain 获得，SFT 主要教**用什么格式、风格、行为模式**表达已有能力。学"表达方式"不需要太多样本，但必须一致且高质量。这也解释了为什么 1–3 个 epoch 就够。

**11.** 五种数据合成方法？

> **答：** 蒸馏强模型、Self-Instruct（种子指令自扩写）、Evol-Instruct（迭代进化变难）、**拒绝采样 RFT**、Persona/场景扩展。

**12.** ⭐⭐ 拒绝采样为什么好用？为什么是 SFT 和 RL 之间的桥？

> **答：** 流程是「prompt → 自己采 n 条 → verifier 判对错 → 只留对的 → SFT」。
> 好用是因为：数据来自**模型自己的分布**更好学；只要有 verifier 就能自动化；可以反复迭代。
> 它其实是最简单的一种 RL：**只保留 reward=1 的样本做 SFT，等价于 advantage 只有 0/1 的 policy gradient**。

**13.** 数据过滤手段？哪个是诚信问题？

> **答：** 去重（MinHash/SimHash）、长度过滤、质量打分（RM 或 LLM-as-judge）、多样性采样、**去污染**。最后一项必须做，否则 benchmark 分数没有意义。

**14.** ⭐ CoT 为什么有效？

> **答：** ① **测试时计算量变长**，每个中间 token 是一次前向，等于更多思考步数；② 把一个难映射拆成若干容易的映射；③ 中间结果写进 context 可被后续引用，相当于外部工作记忆。

**15.** CoT 数据的四种来源？哪种有隐患？

> **答：** 人工写；**强模型生成 + 答案验证**（隐患：只保证结果对，**过程可能错但蒙对**）；**反向合成 rationalization**（给题目和答案让模型补推理过程，就是 OPSD 的 privileged information 思路）；过程监督 PRM（每步打分，最贵最好）。

**16.** 长 CoT 的反思行为靠什么获得？

> **答：** 主要靠 **RL 自己涌现**（`wait`、`let me recheck`、回溯、多路验证），SFT 只做 cold start 把格式教会。

**17.** ⭐⭐ agentic SFT 的 loss mask 规则？算错会怎样？

> **答：** **所有 assistant 段（含 tool_call）算 loss，user 和 tool 段都 mask**。
> **tool 返回的内容绝对不能算 loss** —— 它是环境给的观测。如果算了，模型会开始**幻想工具输出**，推理时自己编造 observation 而不真的调用。

**18.** 为什么 agent 场景 prefix caching 收益特别大？

> **答：** 工具定义放在 system prompt 里，每轮都带着，而且多轮历史越来越长，前缀重复度极高。

**19.** 只留成功轨迹有什么问题？

> **答：** 模型没见过错误恢复。掺一些「调用失败 → 纠正 → 成功」的轨迹更鲁棒。这和 OPD 强调「学自己会犯错的状态」是同一个道理。

**20.** 数据配比怎么定？

> **答：** 是要调的超参。经验：想要的能力数据占比要够；但某类占比过高会挤掉别的能力；通用数据是地基不能太少，否则专项上去了通用能力崩。
