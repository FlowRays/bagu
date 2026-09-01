# RL 自测题库（关掉笔记用）

> 对应学习流程第三步：**跟 GPT 口述 / 纸上手写验证**。
> 用法：先关掉所有笔记，口述或纸上写；卡住的题记下来，回填对应笔记。
> 标 ⭐ 的是**高频考点**或最容易卡的题。

## A. 概念地基（[01](01-from-J-to-loss.md)）

1. ⭐ $J$、$\mathbb E$、$L$、$\nabla L$ 四个东西分别是什么？关系是什么？
2. ⭐ 为什么说 "$J=\mathbb E[A\log\pi]$" 是错的？正确的说法是什么？
3. 写出 policy gradient 定理。它给出的是 $J$ 还是 $\nabla J$？
4. 为什么代码里要写 $A\log\pi$ 这个 loss？它和真正目标是什么关系？
5. 用监督学习类比解释 surrogate objective。
6. policy gradient 成立的隐藏前提是什么？

## B. Importance Sampling（[02](02-importance-sampling-and-ratio.md)）

7. PPO 为什么要重复利用 rollout？重用之后出了什么问题？（注意：不是"$J$ 变了"）
8. 写出 importance sampling 恒等式。
9. ⭐ 用 500L/500R → 900L/100R 的例子解释 $r$ 在做什么。
10. **纸上推导**：$\nabla r=r\nabla\log\pi$。
11. ⭐⭐ 为什么 surrogate 是 $\mathbb E_{old}[rA]$ 而不是 $\mathbb E_{old}[rA\log\pi]$？把后者的梯度算出来，指出多了什么。
12. $A\log\pi$ 和 $rA$ 是代数等价的吗？它们真正的联系是什么？
13. 为什么 rollout 刚结束时所有 $r=1$？

## C. Clip 与 min（[03](03-clip-and-min.md)）

14. 只用 $rA$ 会出什么问题？
15. $r$ 和 clip 各自解决什么问题？（分清两层职责）
16. ⭐ clip 是怎么让参数停止更新的？说出 clip → loss → gradient → θ 的完整链条。
17. PPO 有硬约束 $r\le1.2$ 吗？更新后 $r=2$ 可能吗？
18. ⭐ 为什么不能只写 $\mathrm{clip}(r)A$？举一个具体反例（$A>0$、$r=0.5$）。
19. ⭐⭐ **默写四象限表**，并对 $A=-1,r=0.5$ 和 $A=-1,r=1.5$ 手算 $\min$。
20. PPO 只 clip 哪两种情况？用两句话概括。
21. PPO ratio clip、gradient clipping、value clipping 三者的区别。
22. PPO 和 TRPO 是什么关系？

## D. Advantage / Critic / GAE（[04](04-advantage-critic-gae.md)）

23. 为什么不能直接拿 $G_t$ 当 advantage？baseline 解决了什么？举 $G=100,V=95$ 和 $G=20,V=2$ 的例子。
24. ⭐ critic 学的目标是什么？真实值不知道，用什么当 target？为什么 noisy target 能训出来？
25. 写出 TD error，解释它为什么近似 advantage。
26. 什么是 bootstrapping？
27. ⭐ 用自己的话解释 bias 和 variance。它们描述的是单个样本还是估计器？
28. MC target 和 TD(0) target 分别的 bias/variance 特点？
29. ⭐⭐ GAE 全称是什么？写出定义。**纸上推导** $\lambda=1$ 时如何 telescope 成 $G_t-V_t$。
30. $\lambda$ 控制什么？$\lambda\uparrow$ 时 bias 和 variance 分别怎么变？常用值？
31. ⭐ $V_{old}$ 是什么？和 $V_\phi$ 有什么区别？为什么算 GAE 必须用 $V_{old}$？
32. ⭐ $r_t$、$G_t$、$\hat R_t$ 三者的区别？为什么 $\hat R=\hat A+V_{old}$？
33. $\lambda=0$ 和 $\lambda=1$ 时 critic target 分别退化成什么？
34. actor 用哪个量？critic 用哪个量？

## E. PPO 工程（[05](05-ppo-engineering.md)）

35. 默写完整 PPO loss 三项，说明各自作用和符号方向。
36. ⭐ 按时间顺序说出一次 PPO iteration 的完整步骤，指出哪些量在什么时候被冻结。
37. 用一句话概括 PPO 的时间结构。
38. ⭐ 为什么用 `exp(log_prob - old_log_prob)` 而不是直接相除？
39. rollout batch 和 SGD minibatch 的区别？`ppo_epochs=4` 是什么意思？epoch 越多有什么代价？
40. 写出经典 PPO 默认超参（$\gamma,\lambda,\epsilon,c_v,c_e$, max_grad_norm）。LLM RL 的 lr 和经典 RL 差多少量级？
41. 为什么要做 advantage normalization？
42. 三个必看指标是什么？clipfrac 很高说明什么？列举可能原因。
43. ⭐ LLM 里 $s_t$、$a_t$ 分别是什么？reward 什么时候给？
44. ⭐ $\pi_{old}$ 和 $\pi_{ref}$ 的区别？
45. PPO 和 DQN 的本质区别？

## F. GRPO（[06](06-grpo.md)）⭐ 高频考点

46. ⭐⭐ **一分钟讲清 GRPO 的原理和训练方式。**
47. ⭐ GRPO 的 advantage 怎么算？写出公式。
48. ⭐ 为什么可以不用 critic？为什么这在 LLM reasoning 里可行、在 Atari 里不可行？
49. ⭐ 为什么除标准差而不是方差？如果 reward 整体放大 10 倍分别会怎样？
50. 常见的 norm 有哪几种？为什么 advantage 偏好 z-score？
51. ⭐⭐ 一条 response 的 sequence-level advantage 是怎么作用到每个 token 上的？**推导**（提示：$\log\prod=\sum\log$）。这是"替代"还是"分解"？
52. ⭐ GRPO 解决 credit assignment 了吗？为什么？
53. ⭐ 为什么必须同 prompt 分组，不能整个 batch 一起 norm？举难易两道题的例子。
54. GRPO 的 group baseline 对应 PPO 里的什么？
55. GRPO 什么时候完全没有学习信号？
56. ⭐ GRPO 相对 PPO 的收益和代价各是什么？
57. GRPO loss 里有哪两种"不要走太远"？时间尺度有何不同？

## G. KL（[07](07-kl.md)）⭐ 高频考点

58. ⭐ 写出 KL 定义，说出三个性质。
59. ⭐⭐ **纸上证明** KL ≥ 0（提示 $\log x\le x-1$）。这个不等式叫什么？
60. ⭐⭐ Forward KL 和 Reverse KL 的定义、区别、各自导致什么行为？
61. ⭐⭐ **不许背名字**，只用"谁在 expectation 下面"现场推出 mode-covering / mode-seeking。
62. ⭐ GRPO 的 reference KL 是 forward 还是 reverse？当场展开推一遍。
63. reference KL 在目标函数里怎么写？直觉是什么？
64. 为什么现在很多 reasoning RL 把 $\beta$ 设成 0？
65. PPO 里即使不加 KL penalty 也要监控 KL，为什么？怎么近似算？

## H. DAPO（[08](08-dapo.md)）

66. DAPO 全称？四个改动分别是什么？
67. ⭐ Clip-Higher 改了什么？为什么只放宽上界？举 $0.001\to0.0012$ 的例子。
68. Dynamic Sampling 解决什么问题？保留什么样的 prompt？
69. ⭐ $A=0$ 时训练和不训练有区别吗？真正浪费的是什么？有什么例外？
70. ⭐⭐ **手算**：response A 有 2 个 token、B 有 10 个 token（每个 token loss 都是 1），GRPO 和 DAPO 的 batch loss 分别怎么算？两者权重比各是多少？
71. ⭐ GRPO 和 DAPO 哪个倾向生成更长回答？为什么？这算"直接奖励长度"吗？
72. DAPO 的 token-level aggregation 是无条件更好吗？副作用是什么？
73. ⭐ 默写 Soft Overlong Punishment 的三段公式，算 $|y|=14336$ 时的 $R_{length}$（$L_{\max}=16384,L_{cache}=4096$）。
74. overlong 处理的演化过程是怎样的？为什么叫 reward shaping？

## I. GSPO（[09](09-gspo.md)）

75. GSPO 质疑 GRPO 的什么？
76. sequence-level ratio 怎么写？为什么要长度归一化？可以理解成什么平均？
77. GRPO 和 GSPO 的 clip 分别在什么粒度？
78. 为什么说"把整条 sequence 当 action"更自然？

## J. 整体串讲（压轴）

79. ⭐⭐ **把 PPO → GRPO → DAPO → GSPO 的演化线讲一遍**，说明每一步改的是什么（提示：A 怎么来、ratio 怎么算、loss 怎么聚合）。
80. ⭐ 默写五个必背公式。
81. ⭐ 从 $J(\theta)$ 一路推到 $L^{CLIP}$，中间不跳步。
82. 画出经典 RLHF PPO 的架构图（actor / critic / reward / reference 四个模型的关系）。

## 记录卡住的题

> 每次自测把卡住的题号记在这里，回填笔记后划掉。

- [ ] （待填）
