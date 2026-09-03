# LLM 架构自测题库（关掉笔记用）

> 侧边栏顶部有 **「答案」开关**，可全局显示 / 隐藏。标 ⭐ 的是高频考点。

## A. Tokenizer（[01](01-tokenizer.md)）

**1.** BPE / BBPE / WordPiece / Unigram 的依据和方向各是什么？

> **答：** BPE 自底向上合并，依据是相邻 pair 的**频次**；BBPE 同 BPE 但基本单元是 **byte**；WordPiece 自底向上合并，依据是互信息式打分 $\frac{f(ab)}{f(a)f(b)}$；Unigram **自顶向下**，从大词表开始删掉似然损失最小的 token。

**2.** ⭐ BBPE 为什么永远不会 OOV？代价是什么？

> **答：** 任何字符串都是 UTF-8 byte 序列，而基本单元是全部 256 个 byte，所以一定能表示。代价是起点更碎（一个中文字符 3 个 byte），需要更多 merge 才能合成有意义的单元。

**3.** 为什么 decode 简单而 encode 有优先级问题？

> **答：** decode 是纯查表：id → token → 拼接 → UTF-8 解码。encode 时同一字符串有多种切分，必须定优先级：BBPE **严格按训练时产出的 merge rules 顺序**依次合并；Unigram 用 Viterbi 找概率最大的切分。

**4.** SentencePiece 是算法还是库？

> **答：** 是**库**，能跑 BPE 也能跑 Unigram。特点是把空格当普通字符（`▁`）处理，不依赖预分词，所以天然支持中文日文这类无空格语言。

**5.** ⭐ 词表大小影响哪三处？哪处是训练显存大头？

> **答：** Embedding（$Vd$）、LM head（$Vd$）、**logits（$B\times L\times V$）**。第三处是训练显存大头，$B$=4、$L$=8K、$V$=150K、BF16 约 9.8 GB。但从参数**占比**看 emb+head 在大 MoE 里只有 0.1–0.4%。

**6.** 词表变大的好处和代价？

> **答：** 好处：压缩率提高（同样文本更少 token，等于变相扩上下文、降推理成本）、多语言/代码覆盖更好。代价：emb/head 参数变多、logits 显存和 softmax 计算变大、低频 token 训练不充分。

## B. 位置编码（[02](02-position-encoding.md)）

**7.** ⭐ 为什么必须加位置编码？

> **答：** self-attention 是置换等变的：$\text{Attn}(PX)=P\,\text{Attn}(X)$，打乱输入输出只是跟着打乱，模型看不出顺序。

**8.** ⭐⭐ **推导** RoPE 为什么只依赖相对位置。

> **答：** 记位置 $m$ 的旋转矩阵为 $R_m$，旋转矩阵满足 $R_m^\top R_n=R_{n-m}$，所以
> $\langle R_mq, R_nk\rangle=q^\top R_m^\top R_n k=q^\top R_{n-m}k$，只和 $n-m$ 有关。
> 这就是「用绝对位置的操作得到相对位置的效果」。而且它乘在 $q,k$ 上而不是加在 embedding 上，不占 residual stream。

**9.** RoPE 的 $\theta_i$ 怎么变？低维高维分别管什么？

> **答：** $\theta_i=\text{base}^{-2i/d}$，随 $i$ 增大而减小。**低维高频**转得快，刻画近距离精细差别；**高维低频**转得慢，刻画远距离。周期是 $2\pi/\theta_i$。

**10.** ALiBi 和 RoPE 的区别？

> **答：** ALiBi 不加位置编码，直接在 attention score 上按距离减线性偏置 $-m|i-j|$，简单且外推好，但衰减是硬编码的、表达能力受限。RoPE 是旋转 $q,k$，表达能力更强，是现在的主流。

**11.** ⭐ 直接位置插值的问题？NTK 怎么解决？

> **答：** PI 把位置压回训练范围，所有频率一起拉伸，**高频维度被压得太狠**，近距离分辨能力被破坏。
> NTK 不动位置而是**调大 base**（$10000\to10000\cdot s^{d/(d-2)}$），效果是低频拉伸多、高频几乎不动，保住近距离分辨率的同时扩展远距离。

**12.** ⭐ YaRN 比 NTK 多做了什么？

> **答：** ① **按频率分段**：低频（波长 > 上下文）强拉伸、高频尽量不动、中频平滑过渡；② 加 **attention scaling**（$1/\sqrt d\to t/\sqrt d$）补偿上下文变长后 attention 熵的变化。

**13.** M-RoPE 怎么编码多模态位置？

> **答：** 把 RoPE 维度分成几组分别编码 $(t,h,w)$。文本 token 三个分量取同值退化成 1D；图像按行列填 $h,w$；视频再加帧号 $t$。

## C. Normalization（[03](03-normalization.md)）

**14.** 写出 LayerNorm 和 RMSNorm，指出差别。

> **答：** LN：$\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$；RMSNorm：$\gamma\frac{x}{\sqrt{\frac1d\sum x_i^2+\epsilon}}$。
> 两处差别：RMSNorm **不减均值**、**没有 $\beta$**。省一次求均值、一个参数、一次减法，效果基本不掉。

**15.** ⭐⭐ LLM 为什么不用 BatchNorm？

> **答：** ① 推理时 batch 和 seq 长度一直变，BatchNorm 依赖 batch 统计量，batch=1 时方差无意义，训练/推理行为不一致；② 语义不对，BatchNorm 是**跨样本比同一 feature**，LayerNorm 是**样本内比所有 feature**，语言模型每个 token 是独立语义单元；③ 变长序列的 padding 会污染 batch 统计量。

**16.** ⭐⭐ pre-norm 为什么更好训？

> **答：** pre-norm 展开是 $x_L=x_0+\sum_l\text{Sublayer}_l(\text{LN}(x_l))$，**存在一条从输入到输出完全不被 norm 阻断的恒等路径**，梯度能直接回流浅层。post-norm 每层输出都被 LN 重新标定，深了梯度尺度容易失控，需要精细 warmup。
> 代价是 pre-norm 最终效果可能略差一点，折中方案有 sandwich norm。

**17.** residual stream 视角能解释什么？

> **答：** 把网络看成一条"主干道"，每层只往上面**加**贡献而非替换。能解释：layer pruning / early exit 为什么可行（少加几项）、DeepStack 为什么能把视觉特征直接加进 LLM 前几层、LoRA 为什么有效（主干道上加低秩增量）。

**18.** ⭐ QK-norm 解决什么？加在哪？

> **答：** 训练后期 $q,k$ 范数变大导致 $q^\top k/\sqrt{d_h}$ 过大 → softmax 饱和、梯度消失或 loss spike。做法是在算 attention **之前**对 $q,k$ 各做一次 RMSNorm（按 head 维），把 attention logits 尺度锁住。**治的是训练稳定性，不是为了效果。**

**19.** 为什么现代 LLM 去掉 bias？

> **答：** 省一点参数和显存；实验发现不掉点甚至更稳；少一个要同步的张量，分布式实现更简单。

## D. Attention（[04](04-attention.md)）

**20.** ⭐⭐ 为什么除 $\sqrt{d_h}$ 而不是 $d_h$？

> **答：** $q,k$ 各维独立方差为 1 时 $q^\top k$ 的方差是 $d_h$、**标准差**是 $\sqrt{d_h}$。要把尺度拉回 1 就除标准差。不除的话 $d_h$ 一大 logits 尺度就大，softmax 饱和成接近 one-hot，梯度趋近 0。

**21.** ⭐⭐ causal mask 为什么必须在 softmax 之前？

> **答：** 要填 $-\infty$，因为 $e^{-\infty}=0$ 使该位置权重恰好为 0 **且不进入分母**。softmax 之后再乘 0 的话，分母里仍然算了被屏蔽的位置，剩余权重就不归一了。

**22.** ⭐⭐ 算 32K 上下文的 KV cache。

> **答：** $M_{KV}=2\cdot B\cdot L\cdot N_{\text{layer}}\cdot h_{kv}\cdot d_h\cdot\text{bytes}$。
> $B$=1、$L$=32768、32 层、32 头、$d_h$=128、BF16：$2\times32768\times32\times32\times128\times2\approx17.2$ GB。**比 7B 模型的权重还大**，这就是 MQA/GQA/MLA 的动机。

**23.** ⭐ MHA / GQA / MQA 的区别？实现要点？

> **答：** Q 都是 $h$ 个头，区别在 $h_{kv}$：MHA $h_{kv}=h$、GQA $1<h_{kv}<h$、MQA $h_{kv}=1$，KV cache 分别是 $1\times$、$1/g$、$1/h$。
> 实现要点：$W_K,W_V$ 的输出维是 $h_{kv}d_h$，**比 $W_Q$ 窄**；广播回 $h$ 组必须用 `repeat_interleave`（AABBCC）不是 `repeat`（ABCABC），用错 head 和 KV 组就对不上。典型配置 $h$=32、$h_{kv}$=8。

**24.** ⭐⭐ MLA 压的是什么？为什么和 RoPE 冲突？

> **答：** 把 KV 压到低维 latent：$c=xW^{DKV}$，$K=cW^{UK}$、$V=cW^{UV}$，**cache 只存 $c$**，维度 $d_c\ll 2hd_h$。推理时还能把 $W^{UK}$ 吸收进 $W_Q$。
> 冲突原因：旋转和低秩分解不交换，RoPE 不能直接作用在压缩后的 latent 上。解决办法是**拆出一个独立的带 RoPE 的小分支**（decoupled RoPE），这是 MLA 实现上最绕的地方。
> 对比：GQA 减头数，MLA 降维度。

**25.** ⭐ attention sink 是什么？

> **答：** 训练好的模型里几乎所有 head 都会给序列**最开头几个 token** 分配可观注意力，哪怕它们语义无关。
> 解释：softmax 强制权重和为 1，当 query 其实"什么都不想关注"时需要一个地方倾倒概率质量，最靠前、所有 query 都可见的位置成了默认垃圾桶。
> 后果：滑动窗口把开头滑出去模型会崩，所以 StreamingLLM 要永久保留几个 sink token。新模型直接给 softmax 加可学习 sink logit 从根上解决。

**26.** FlashAttention 把什么从 $O(L^2)$ 降到 $O(L)$？

> **答：** **显存**（不把完整 $L\times L$ 矩阵写进 HBM），靠分块计算 + online softmax。**计算复杂度仍是 $O(L^2)$**，它是 IO-aware 优化不是算法复杂度优化。

**27.** 线性注意力和稀疏注意力的思路？

> **答：** 线性（Gated DeltaNet、KDA）用状态矩阵递推代替全局 softmax，降到 $O(L)$；稀疏（DSA、NSA）只算一部分 query-key 对，用 indexer 选；滑动窗口只看最近 $w$ 个。现在常见**混合**：多数层线性/稀疏，少数层保留 full attention 保长程能力（Kimi K3 是 69 层 KDA + 24 层 Gated MLA）。

## E. FFN 与 MoE（[05](05-ffn-moe.md)）

**28.** Attention 和 FFN 的分工？

> **答：** Attention 让 token **之间**交换信息；FFN 对**每个 token 内部**独立做非线性变换（position-wise），把拿到的信息加工。Dense 模型里绝大部分参数在 FFN。

**29.** ⭐⭐ 写出 SwiGLU，推导 $d_{ff}=\frac83d$。

> **答：** $\text{SwiGLU}(x)=(\text{SiLU}(xW_g)\odot(xW_u))W_d$，**只有 gate 分支过 SiLU**。
> 普通 FFN 两个矩阵：$2dd_{ff}=8d^2$（$d_{ff}=4d$）；SwiGLU 三个矩阵：$3dd_{ff}$。令两者相等得 $d_{ff}=\frac83d$。

**30.** ⭐ 写出 MoE 前向。shared expert 是什么？

> **答：** $y=\sum_{j\in\text{TopK}}p_jE_j(x)$，$p=\text{softmax}(\text{router}(x))$。
> routed expert 由 router 动态选中；**shared expert 每个 token 都走**，用来承载通用能力，让 routed expert 专注差异化，避免每个 expert 都重复学基础能力。

**31.** ⭐⭐ 写出 load balancing loss，解释各项。缺点？

> **答：** $\mathcal L_{\text{aux}}=\alpha N\sum_j f_jP_j$，$f_j$ 是实际分到 expert $j$ 的 token 比例，$P_j$ 是 router 给它的平均概率，两者都均匀时最小，乘 $N$ 让最优值与 $N$ 无关。
> 缺点：是和主任务无关的额外 loss，$\alpha$ 太大伤效果、太小不起作用。

**32.** ⭐ aux-loss-free balancing 怎么做？

> **答：** 给每个 expert 的 router 分数加可学习 bias $b_j$，**只用于 top-k 选择**（$\text{TopK}(s_j+b_j)$），**不进入最终加权**（仍用 $s_j$）。训练中过载就调低 $b_j$、欠载就调高。把负载均衡从"额外 loss"变成"路由偏置调节"，不污染主目标。

**33.** ⭐⭐ MoE 的显存按 total 还是 activated？

> **答：** **权重和 optimizer state 按 total，每 token 的 FLOPs 按 activated。** 2.8T 的 MoE 不是"104B 模型的显存"，那 2.8T 参数和 Adam 状态都得存。MoE 换的是「同样算力下更大的容量」，不是省显存。

**34.** expert capacity 超了会怎样？为什么这条重要？

> **答：** 超出容量的 token 被**丢弃**（直接走 residual 跳过 FFN）。重要是因为：EP 本身是数学等价的并行，但 **token dropping 会真的改变训练**，是少数几个真会影响效果的因素之一。

**35.** MoE 的缺点有哪些？

> **答：** ① 显存按 total 算，非常吃显存；② 负载不均导致实际吞吐远低于理论；③ All-to-All 通信重，对网络要求高；④ 训练不稳定需要 balancing 机制；⑤ 小 batch 推理时 expert 利用率低。

## F. LM head 与 MTP（[06](06-lm-head-and-mtp.md)）

**36.** ⭐ weight tying 是什么？经验规律？

> **答：** 输入 embedding 和输出 LM head 共享矩阵（$W_{\text{vocab}}=E^\top$）。好处是省 $Vd$ 参数、小模型上有正则作用；代价是强制绑定输入输出空间，大模型上限制表达。**经验：小模型倾向 tie，大模型倾向不 tie。**
> 算参数量时：tie 是 $Vd$，不 tie 是 $2Vd$。

**37.** ⭐ MTP 的两个作用？

> **答：** ① **训练**：同一位置同时预测未来 $n$ 个 token，是额外监督信号，强迫表示编码更长程信息，实测提效果（主要动机）；② **推理**：MTP 头当**自投机解码的 draft**，一次前向猜几个 token 再用主模型验证，省掉额外的 draft 模型。

**38.** 为什么 logits 是训练显存大头？怎么处理？

> **答：** $[B,L,V]$，$B$=4、$L$=8192、$V$=150K、BF16 约 9.8 GB。框架用 fused cross entropy / chunked loss，算一块释放一块，不长期保存完整 logits。

## G. 串联题

**39.** ⭐⭐ 画出一条完整前向路径。

> **答：** `文本 → tokenizer → token id → embedding (B,S,d) → [pre-norm → attention → residual → pre-norm → FFN/MoE → residual] × N → final norm → lm_head → logits (B,S,V) → softmax → sampling → next token`

**40.** ⭐ 一个 32K 上下文的 7B 模型推理，显存都花在哪？

> **答：** 权重约 14 GB（BF16），**KV cache 约 17 GB**（32 层/32 头/$d_h$=128），加上激活和框架开销。KV cache 比权重还大，所以要 GQA/MLA、量化 KV、PagedAttention 等手段。

**41.** ⭐ 现代 LLM 相比原始 Transformer 改了哪些？

> **答：** ① post-norm → **pre-norm**；② LayerNorm → **RMSNorm**；③ 绝对 PE → **RoPE**；④ GELU FFN → **SwiGLU**（$d_{ff}=\frac83d$）；⑤ MHA → **GQA/MLA**；⑥ 去掉 bias；⑦ 部分模型加 **QK-norm**；⑧ dense FFN → **MoE**；⑨ 可能加 **MTP**。

**42.** 为什么这些改动大多是"为了训练稳定"而不是"为了效果"？

> **答：** 大模型训练最贵的失败是中途发散。pre-norm、RMSNorm、QK-norm、去 bias、load balancing 主要都在治稳定性；真正冲效果的是数据、规模和 post-training。面试能说清「这条治的是稳定性还是效果」会加分。

**43.** ⭐ 哪些结构决定 KV cache 大小？分别怎么优化？

> **答：** $2\cdot B\cdot L\cdot N_{\text{layer}}\cdot h_{kv}\cdot d_h\cdot\text{bytes}$。
> $h_{kv}$ → GQA/MQA；$d_h$ 那一项 → MLA 压成低维 latent；bytes → KV 量化；$L$ → 滑动窗口 / 稀疏注意力；$B\cdot L$ 的碎片 → PagedAttention。

**44.** ⭐ 面试一分钟讲清 attention 家族的演进。

> **答：** MHA 是基准，但 KV cache 正比于头数，32K 上下文就要十几 GB。MQA 把 KV 砍到一个头，cache 省 $h$ 倍但质量明显下降。GQA 折中，KV 分成 $h_{kv}$ 组由多个 query head 共享，省 $g$ 倍而质量几乎不掉，是现在的默认。MLA 换了个方向，把 KV 压成低维 latent 只缓存 latent，压缩比更高、质量不降反升，代价是要为 RoPE 单独拆一个分支。再往前是线性和稀疏注意力，目标是打破 $O(L^2)$，通常和 full attention 混合使用。
