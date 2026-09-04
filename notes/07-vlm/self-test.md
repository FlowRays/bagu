# VLM 自测题库（关掉笔记用）

> 侧边栏顶部有 **「答案」开关**，可以全局显示 / 隐藏所有答案。自测时先隐藏。
> 标 ⭐ 的是高频考点。

## A. 架构（[01](01-architecture.md)）

**1.** $336\times336$、patch=14 得到多少 visual token？

> **答：** $336/14=24$，$24\times24=576$。每个 patch（$14\times14\times3$）先线性投影成 embedding，再过 ViT 得到 $Z_v\in\mathbb R^{576\times d_v}$。

**2.** ⭐ projector 只是把 1024 维变成 4096 维吗？

> **答：** 不是。随机初始化的 projector 维度对上也没用，因为输出向量和 LLM 已学会的语义空间没有对应关系。它做的是 **representation alignment，而不只是 dimension matching**。

**3.** ⭐ 为什么一个两层 MLP 就够？

> **答：** 因为 Vision Encoder 已经很强 —— CLIP ViT 几十层之后输出的已是高度语义化的特征，projector 不需要重新"学会看图"，只需要学 `Vision representation → LLM 能消费的 representation`，这比 `raw pixel → language understanding` 简单太多。
> CLIP 尤其关键：它的训练目标就是 image-text 对齐，视觉空间**本来就被语言监督塑形过**。所以 CLIP 负责第一层 visual-language alignment，projector 只负责 `CLIP representation → LLM representation`。

**4.** projector 的训练目标是不是 $P(z_{\text{dog}})=E(\text{"dog"})$？

> **答：** 不是，通常也不是训练目标。只要求 $P(z)$ 进入 LLM 后，经过若干层 self-attention 能让 LLM 正确预测 "There is a dog in the image."，即成为 **LLM 可解释的 conditioning representation**。

**5.** ⭐ Stage 1 冻结谁、训谁、loss 是什么？

> **答：** Frozen Vision Encoder + **Trainable Projector** + Frozen LLM，数据是 image-caption，loss 就是普通 next-token prediction $\mathcal L=-\sum_t\log p_\theta(y_t|y_{<t},I)$。梯度逼着 projector 学"我要输出什么样的 embedding，才能让这个 frozen LLM 正确生成 caption"。

**6.** ⭐⭐ LLM 冻结了，梯度还能穿过它吗？

> **答：** **能。** 冻结 $\theta_{LLM}$ 只意味着不执行 $\theta_{LLM}\leftarrow\theta_{LLM}-\eta\nabla_{\theta_{LLM}}\mathcal L$ 这一步，但 $\partial\mathcal L/\partial h_{\text{vision}}$ 照样能算出来，所以梯度能继续传到 projector。

**7.** 画出 visual/text token 的四象限 attention，哪一格最关键？

> **答：** 拼成 $H=[H_v;H_t]$ 后是普通 self-attention，四格是 V→V、V→T、**T→V**、T→T。最关键的是 **T→V**：文本 token "dog" 的 query 直接 attend 到相关 visual patch，$q_{\text{dog}}^\top k_{\text{patch}}$ 大 → 从那块区域取信息。这就是 single-stream / unified self-attention。

**8.** 为什么不用一个巨大的 cross-attention 模块？

> **答：** 如果 Vision Encoder 已经产生高度语义化且与语言相关的特征，两个 representation manifold 之间可能只需要一个相对简单的映射。这是 LLaVA 相比 Flamingo / BLIP-2 Q-Former 的简洁之处。

## B. Vision Encoder（[02](02-vision-encoder.md)）

**9.** ⭐ 写出 CLIP 和 SigLIP 的 loss，说明换 sigmoid 解决了什么。

> **答：** CLIP 是 batch 内 softmax contrastive：$L_i=-\log\frac{\exp(s(I_i,T_i)/\tau)}{\sum_j\exp(s(I_i,T_j)/\tau)}$，正确 caption 要和整个 batch 竞争，因此依赖大 batch、batch 内 negative、多卡 all-gather，batch size 直接影响 loss 行为。
> SigLIP 改成 sigmoid pairwise 二分类：$L=-\sum_{ij}\log\sigma(y_{ij}(s_{ij}+b))$，$y_{ij}=+1$ 若 $i=j$ 否则 $-1$。不再需要 softmax over whole batch → **扩展性和 batch size 灵活性更好**。

**9b.** ⭐⭐ image-text pair 怎么构造的？负样本从哪来？会不会有假负样本？

> **答：** **不是人工挑负样本。** 数据本身是配对的 $(I_i,T_i)$，一个 batch 内把所有 image 和 text 两两组合成 $N\times N$，$i=j$ 是 positive、$i\ne j$ 是 negative —— 这叫 **in-batch negatives**。
> **正负极不平衡**：$N=1024$ 时正样本 1024、负样本约 $10^6$，所以 SigLIP 实际 loss 里有可学习的 scale / bias。
> **false negative 确实存在**：$I_1$ 是金毛配 "a golden retriever"，batch 里刚好还有 "a dog playing outside"，后者被当 negative 但其实描述得对。靠数据规模大 + 随机采样在统计上摊平。

**9c.** ⭐ 一句话说清 CLIP 和 SigLIP 的区别，并说明 softmax 带来什么副作用。

> **答：** CLIP 是"**这些句子里哪一句最匹配这张图**"（softmax 单选题），SigLIP 是"**这张图和这句话匹不匹配**"（sigmoid 判断题）。
> softmax 强制 $\sum_j p(T_j|I_i)=1$，候选文本**天生互相竞争**：一张橘猫图配 "a cat" 和 "an orange cat on a sofa" 两个都对，却必须抢概率。sigmoid 各判各的，0.95 和 0.98 同时成立没问题。
> 工程上更关键的是 SigLIP **没有必须跨全 batch 归一化的 softmax 分母**，每个 pair 能独立算，多卡大规模训练不必 all-gather embedding。

**10.** SigLIP2 相比 SigLIP 还加了什么？

> **答：** captioning-based pretraining、self-distillation、masked prediction、online data curation、localization / dense prediction 优化、native aspect ratio / 多分辨率、multilingual。所以 **CLIP 强在 global semantic alignment，SigLIP2 还更重视 dense / localization / multilingual / variable-resolution**。

**11.** ⭐ 写出 DINO 的 loss 和 teacher 更新方式。怎么防 collapse？

> **答：** 两个不同 augmentation 分别给 student 和 teacher，$\mathcal L=-\sum_k p_t(k)\log p_s(k)$。teacher **不反向传播**，是 student 的 EMA：$\theta_t\leftarrow m\theta_t+(1-m)\theta_s$。用 sharpening 和 centering 防止 $p_s=p_t=\text{const}$ 的 representation collapse。名字是 self-**DI**stillation with **NO** labels。

**12.** ⭐ DINO 和 CLIP 学的东西为什么不一样？

> **答：** caption "A red cup on a table." 根本没告诉模型杯子具体在哪、桌子边界、窗户、每个 patch 属于什么、depth、object part，所以 image-text supervision 偏 **global semantics**。DINO 从图像本身学，因此擅长 patch-level representation、local correspondence、segmentation、depth、geometry、object parts、dense perception。
> 记：**CLIP/SigLIP 语言对齐强；DINO 纯视觉 / dense feature 强。**

**13.** DINOv2 / DINOv3 分别多大？

> **答：** DINOv2 最大 ViT-g/14 约 **1.1B**（142M images）；DINOv3（2025）做到 **7B**（1.7B images），并加了 Gram Anchoring 等 dense feature 改进。所以视觉模型本身不是没有 scaling。

**14.** ⭐ Qwen3-VL-8B 的 vision encoder 是什么？

> **答：** **SigLIP-2 架构**，从官方 pretrained checkpoint 初始化，再做 dynamic-resolution 继续训练；8B 用 SigLIP2-So400m。配置：27 层、hidden 1152、MLP 4304（普通 GELU 不是 SwiGLU）、16 heads、patch 16×16、temporal patch 2、spatial merge 2×2、输出 4096、DeepStack 在 8/16/24 层。
> 所以是 **SigLIP2 初始化 + Qwen 特有改造 + 大规模继续训练**，不是 frozen SigLIP。

**15.** ⭐ DeepStack 是什么、加在哪、为什么？

> **答：** 不只用 ViT 最后一层，而是在第 8/16/24 层各取一次 feature，各配一个 merger，然后**加到 LLM 前几层的 hidden state 上**：$h_l\leftarrow h_l+h^{\text{vision}}_l$。这样 intermediate ViT 的 texture / OCR / local / spatial feature 不必等到最后一层全变成高度 semantic 表示才交给 LLM。官方说法是提升 fine-grained detail 与 image-text alignment。

**16.** Qwen3.5-9B 的 vision tower 和 Qwen3-VL 差在哪？

> **答：** core ViT 配置**完全一致**（27L / 1152 / 4304 / 16 heads / patch 16 / merge 2），但当前 checkpoint `deepstack_visual_indexes = []`，**没有 DeepStack**，所以 visual module 从 ~576M 降到 ~456M。而且它已经不叫 Qwen3.5-VL，是从头在 interleaved text/image/video token 上训练的 **native multimodal foundation model**。

**17.** ⭐⭐ Kimi K3 为什么敢不用 SigLIP 初始化？

> **答：** 当 multimodal pretraining 足够大时，**next-token prediction 本身就可以成为 vision encoder 的 supervision** —— $\mathcal L=-\sum_t\log p(y_t|I,y_{<t})$ 的梯度会一路传到 ViT，于是 ViT 直接学"什么视觉表示最有利于 LLM 做 multimodal NTP"，而不是"什么视觉表示最有利于 CLIP image-text similarity"。这两个目标并不一致：$L_{\text{contrastive}}\ne L_{\text{VLM}}$。
> Kimi 报告 SigLIP 初始化版本联合训练时 gradient norm 更高、spike 更多，from-scratch MoonViT-V2 更稳定且最终追平。MoonViT-V2：401M、27L、hidden 1024、FFN 4096、12 heads、patch 14、RMSNorm、无 bias。

**18.** ⭐ 为什么 projector 会有 40M 参数？

> **答：** 因为 $2\times2$ merge 要先 concat：$4\times1152=4608$，然后 $4608\to4608\to4096$，参数 $4608^2+4608\times4096\approx21.2+18.9=40.1$M。Kimi K3 是 $4\times1024=4096\to4096\to7168\approx46.1$M。"projector 就是个小 MLP"是误解。

**19.** ⭐⭐ 有人 scale 过 vision encoder 吗？为什么 6B ViT 没成标配？

> **答：** 有 —— InternVL 的核心卖点就是 Scaling up Vision Foundation Models，直接做了 **InternViT-6B**；DINOv2 1.1B、DINOv3 7B。没成标配的五个原因：
> ① **瓶颈往往首先是 resolution 不是参数量**（OCR 里小字号 resize 后消失，$400M\to40B$ 也救不回来），所以算力先花在 dynamic resolution / AnyRes / tiling / smaller patch / multi-scale / DeepStack；
> ② **ViT compute 随 visual token 爆炸**，self-attention 是 $O(N_v^2d)$，$N_v$ 从 1000 到 10000 是 100×，存在"更大 ViT vs 更高分辨率"的预算竞争；
> ③ **information bottleneck**：10000 个 feature 压成 1000 token 再给 LLM，继续增大 VE 未必等比例提升；
> ④ **很多 benchmark 的最终瓶颈是 LLM reasoning**，$\text{LLM }8B\to70B$ 通常比 $\text{ViT }400M\to6B$ 更值；
> ⑤ **400M 的 ViT 一点都不弱** —— 它只负责 $I\to\text{representation}$，不负责 autoregressive generation、世界知识、long CoT、instruction following，**不能直接拿 ViT 参数规模和 LLM 参数规模比较**。

**20.** 各主流模型的 visual module 参数量？

> **答：** Qwen3-VL-8B ≈416M ViT + 40M 主 merger + 120M（3 个 DeepStack merger）≈ **576M**；Qwen3.5-9B ≈416M + 40M ≈ **456M**；Kimi K3 ≈401M + 46M ≈ **447M**；DeepSeek-V4-Flash-Vision ≈412M + 54.5M ≈ **466M**。都集中在 **0.45B 左右**。

## C. Visual token 压缩（[03](03-visual-token-compression.md)）

**21.** ⭐ $1024\times1024$、patch=14 会有多少 patch？为什么不能直接送 LLM？

> **答：** $1024/14\approx73$，$73\times73\approx5329$ 个 patch。直接送 LLM 会让 context 多 5300 token，而 attention 是 $O((N_t+N_v)^2)$，视频再乘帧数直接爆炸。所以 **ViT 内部可以保留较密的 token，但送进 LLM 前必须压缩**。

**22.** 四类压缩方法分别是什么？

> **答：** ① **spatial merge / pixel shuffle**（Qwen $2\times2$，$N\to N/4$）；② **更激进的 spatial pooling**（DeepSeek $3\times3$，$N\to N/9$）；③ **learned resampler**（Q-Former / Perceiver，固定 $M$ 个 learnable query 做 cross-attention，预算固定但易丢 dense 信息）；④ **token selection / pruning**（按 attention score / saliency / similarity / text-query relevance 留 $K$ 个，视频尤其重要）。

**23.** ⭐ 为什么是 concat + MLP 而不是 average pooling？

> **答：** 平均池化把 4 个 patch 的空间信息抹平了（谁在左上、谁在右下都一样）；concat 保留了 $2\times2$ 内部的相对位置，MLP 可以学怎么融合。代价是 projector 参数涨到 40–50M —— 大家宁愿付这个参数，说明空间信息值这个钱。

**24.** ⭐ DeepSeek-V4-Flash-Vision 的 aligner 怎么算？

> **答：** $9\times1024=9216\to4096\to4096$，$h=W_2\,\text{GELU}(W_1[z_1;\dots;z_9])$，即 **9 个 ViT token → 1 个 LLM token**。参数 $37.75+16.78=54.53$M，加 ViT 411.84M 共约 466M。ViT 是 32 层、hidden 1024、16 heads、FFN 2816、patch 14、2D RoPE，内部是 **full bidirectional attention**（不是 causal）。

**25.** ⭐ 它是先生成海量 ViT token 再压到 384 吗？

> **答：** 不是。有**两层控制**：`vision_max_n_token = 384`，而且**预处理阶段**就根据 patch=14、downsample=3、max=384 反推允许的 resize 分辨率。流程是 `raw image → 按长宽比动态 resize → patch=14 → 32-layer ViT → 3×3 merge → ≤384-token block → LLM`。

**26.** 384 个 image token 遇上 128-token sliding window 会怎样？怎么解决？

> **答：** 按普通 sliding window，第 384 个 image token 看不到第 1 个，**图片会被切碎**。DeepSeek 在 attention mask 里专门识别 `[IMAGE_START, IMAGE_END]` 区间，对同一 image span 扩展可见范围，让图像内部 token 能跨过 128-token window 互相访问（`get_image_visible()` 就是在算当前 token 距 image span 左右边界多远）。

## D. 训练阶段（[04](04-training-stages.md)）

**27.** CLIP / SigLIP / DINO 三者的 loss 和更新对象？

> **答：** CLIP：$L=\frac12(L_{I\to T}+L_{T\to I})$ softmax contrastive，更新 Vision Encoder + Text Encoder；SigLIP：sigmoid pairwise，同样更新 VE + Text Encoder；DINO：$\text{CE}(p_{\text{teacher}},p_{\text{student}})$，student 走 SGD、**teacher 走 EMA**，**没有 text encoder**。

**28.** ⭐ Stage 0 冻结谁、训谁？Qwen3-VL 的 S0–S3 分别改了什么？

> **答：** Stage 0（VL Alignment）：VE frozen、**Projector train**、LLM frozen，loss 是普通 $L_{\text{NTP}}$。Qwen3-VL 的 S0 约 67B tokens，只更新 merger。
> S0 (8K, projector) → S1 (8K, ALL) → S2 (32K, ALL) → S3 (256K, ALL)。**S1/S2/S3 不是换 loss，主要是改 data mixture + context length。**

**29.** ⭐ Pretrain 的 CE 和 SFT 的 CE 差在哪？

> **答：** 数学形式几乎一样，区别是 **loss mask**。Pretrain 几乎全部 token 都算 loss；SFT 是 System→mask、User→mask、Image→没有离散 CE target、**只有 Assistant 算 loss**：$L_{\text{SFT}}=-\sum_{t\in\text{Assistant}}\log p(y_t|I,x,y_{<t})$。

**30.** ⭐⭐ ViT / Projector / LLM 是不是各有一个 loss？

> **答：** **通常不是。** 三个模块共享同一个最终任务 loss（$L_{\text{SFT}}$ 或 $L_{\text{RL}}$），梯度一路 $\mathcal L\to LLM\to Projector\to ViT$，**谁 `requires_grad=True` 谁就更新**。三个都训时三个更新式同时存在：$\theta_V\leftarrow\theta_V-\eta_V\nabla_{\theta_V}L$ 等。
> pretraining 有时加 auxiliary loss（image-text contrastive、grounding/bbox、masked image modeling、DINO self-distillation、detection），但那是**帮助 representation learning**，不是"ViT 必须有专门 loss 才能更新"。

**31.** ⭐⭐ 为什么语言 CE 能更新 ViT？

> **答：** 例：`<image> Which button should I click?` GT 是 "the blue button"，模型 $p(\text{red})=0.6$、$p(\text{blue})=0.1$，$L=-\log0.1$。反向路径 $L\to\text{logits}\to\text{LLM hidden}\to\text{visual embedding}\to\text{Projector}\to\text{ViT feature}$，等于告诉 ViT"你现在产生的视觉表示没让后面区分出 blue"。
> 类比：$\text{image}\to\text{CNN}\to\text{classifier}\to\text{CE}$，输出也不是图片，CE 一样能训 CNN。

**32.** ⭐⭐ 图片不是输出，那它有梯度吗？

> **答：** **有。** loss mask 上 image token 确实是 0（不作为 prediction target），但 assistant 的预测依赖图片，所以 $\partial L/\partial H_v\ne0$，进而
> $\frac{\partial L}{\partial\theta_P}=\frac{\partial L}{\partial H_v}\frac{\partial H_v}{\partial\theta_P}$，$\frac{\partial L}{\partial\theta_V}=\frac{\partial L}{\partial H_v}\frac{\partial H_v}{\partial Z_v}\frac{\partial Z_v}{\partial\theta_V}$。
> 一句话：**image token 是 condition，不是 prediction target；但 condition 仍然在 computation graph 里。** 真正"不需要考虑"的只是给 image token 定义 token-level CE label。

**33.** ⭐ 什么情况下 SFT 要解冻 ViT？

> **答：** 看数据在改变什么。改变"**怎么思考 / 怎么回答**"（math CoT、instruction following、agent planning、输出格式）→ **freeze 就够**，perception 没有 domain shift。改变"**怎么看**"（医学影像、卫星图、GUI 小图标、游戏画面、小目标检测、OCR、spatial grounding、特殊传感器）→ **unfreeze 可能明显更重要**，因为瓶颈就在 $I\to Z_v$。

**34.** ⭐⭐ RL 为什么倾向 freeze ViT？

> **答：** 数学上完全可以解冻，policy gradient 一样能传回 ViT，reward 可以真的改变视觉表示。但 **reward 对 perception 来说太间接**：$R=-1$ 可能因为图没看清、reasoning 错、planning 错、action sampling 错、长程 credit assignment 错。全参更新会让 ViT 为这个 reward 承担部分责任，容易造成 **representation drift**，甚至破坏已有的 OCR / detection / general vision 能力。
> 所以：**SFT 可以较积极地 unfreeze vision，RL 更保守地 freeze vision。**

**35.** 画出完整 pipeline。

> **答：** `Vision Pretraining (L_contrastive/SSL) → VL Alignment (L_NTP, 更新 Projector) → MM Pretraining (L_NTP, 更新 VE+P+LLM) → Instruction SFT (CE on assistant tokens) → Distillation (CE / KL) → RL (PPO/GRPO/DAPO)`。
> Distillation：有 teacher logits 就 $D_{KL}(p_T\|p_S)$，只有 teacher rollout 就退化成 $-\log p_S(y_T)$。Qwen3-VL post-training 明确是 `SFT → Strong-to-Weak Distillation → RL`，distillation 用 text-only 数据增强 LLM reasoning。

**36.** ⭐ 一分钟答"VLM 怎么训练 / 训哪几块"。

> **答：** VLM 的 ViT、projector 和 LLM 并不需要分别定义三个 loss。普通 multimodal SFT 就是 assistant token 上的 autoregressive cross-entropy；只训 LLM 时梯度只更新 LLM，同时训 projector 时 language loss 会通过 LLM 反传到 projector，ViT 也解冻时同一个 language loss 继续反传到 ViT，实现 end-to-end visual-language adaptation。
> RL 同理，GRPO/PPO 的 policy loss 定义在输出 token 的 log probability 上，vision encoder 解冻时 policy gradient 同样能传回 ViT；但实践中 RL 更倾向冻结 vision encoder，因为 reward noisy、credit assignment 间接，直接更新 ViT 容易造成视觉 representation drift。

## E. Patch Embedding（[01b](01b-patch-embedding.md)）

**37.** ⭐⭐ 为什么 `Conv2d(3, d, kernel=P, stride=P)` 等价于 ViT 的 patch embedding？说准确一点。

> **答：** stride 等于 kernel 时**窗口不重叠**，每个输出位置正好对应一个独立 patch，于是每个 patch 都在做同一个 $y_i=Wx_i+b$，$W\in\mathbb R^{d_{out}\times CK_HK_W}$。
> ⚠️ 不是"因为 stride=kernel 卷积才线性" —— 卷积对每个局部窗口一直都是线性的。stride=kernel 带来的是**不重叠**。$K=14,S=7$ 也是 extract patch + Linear，只是 patch 会重叠。

**38.** ⭐ `Conv2d(3, 1280, kernel_size=14, stride=14)` 有几个 kernel、每个多少参数、等价于什么？

> **答：** 1280 个 kernel，每个 $3\times14\times14=588$ 个参数，合起来 $W\in\mathbb R^{1280\times588}$，等价于 $\text{Linear}(588,1280)$ 作用在每个不重叠的 $14\times14$ RGB patch 上。

**39.** ⭐ `out_channels=1280` 时，每个 channel 内部维度是多少？

> **答：** 对**一个 patch 位置**，每个 channel 只有 **1 个标量**，1280 个合起来才是这个 patch 的 1280 维 token（$[3,2,14,14]\to[1280,1,1,1]$）。
> 对整段视频 $[3,8,448,448]\to[1280,4,32,32]$，此时第 $c$ 个 channel 是一张 $4\times32\times32$ 的时空 feature map。两个视角等价：$[1280,4,32,32]\leftrightarrow[4096,1280]$。

**40.** Qwen2.5-VL 为什么用 Conv3D？图片没有时间维怎么办？

> **答：** 为了**图片和视频共用一套 patch embedding**。kernel $(2,14,14)$ 表示一个 tubelet = 连续 2 帧里的 $14\times14$ 区域。图片由 processor 把最后一帧重复补足 $T=2$（概念上 $I\to[I,I]$），不需要两套代码。

**41.** ⭐⭐ $448\times448$ 的图进 Qwen2.5-VL，到 ViT 第一层之前 shape 怎么变？

> **答：** $[3,448,448]$ →（resize/normalize，边长要能被 $14\times2=28$ 整除）→ 复制一帧 $[3,2,448,448]$ → kernel=stride=$(2,14,14)$ 不重叠切块，grid $1\times32\times32$ = **1024 个 patch** → 每 patch $1176$ 个数即 $[1024,1176]$ → 共享 $\text{Linear}(1176,1280)$ → $[1024,1280]$。
> 1024 是 **ViT 内部** token 数，不是给 LLM 的（后面 $2\times2$ merge 压到 256）。

**42.** 为什么 preprocessing 要求边长能被 28 整除？

> **答：** $28=14\times2$：14 是 **patch size**，2 是 **spatial merge size**。两级都要整除，patch grid 才能正好按 $2\times2$ 分组。

## F. Qwen2.5-VL 走查（[05](05-qwen25-vl.md)）

**43.** ⭐ 背出 Qwen2.5-VL ViT 的关键配置。

> **答：** patch $14\times14$、temporal patch 2、hidden 1280、32 层、16 heads（$d_h=80$）、MLP 3420、window $112\times112$、full-attn 在第 7/15/23/31 层，backbone ≈630M。

**44.** ⭐⭐ 32 层的 attention 怎么排？为什么？window 会减少 token 数吗？

> **答：** **28 层 window + 4 层 full**（第 7/15/23/31 层）。因为全 full attention 是 $O(N^2)$，高分辨率下 $N$ 有几千。所以做成"大量 local perception + 周期性 global information exchange"，这也是它能支持 native dynamic resolution 的原因。
> ⚠️ window attention **不减少 token 数**，只限制每个 token 能看到谁。压 token 是 PatchMerger 的事。

**45.** ⭐ Qwen2.5-VL 的 ViT block 和经典 ViT block 差在哪？

> **答：** norm 用 **RMSNorm**；FFN 是 **SwiGLU** 风格 $W_{down}[\text{SiLU}(W_{gate}x)\odot W_{up}x]$，$1280\to3420\to1280$，不是单层 GELU。attention 双向（`is_causal=False`）。整体更像现代 LLM 的 block。

**46.** ⭐⭐ ViT 里的 2D RoPE 和 LLM 里的 MRoPE 是一回事吗？

> **答：** **不是。** ViT 内部按 patch 的 $(h,w)$ 网格坐标构造 position id，在 $Q/K$ 上加二维 rotary，说的是"这个 patch 在第几行第几列"。MRoPE 是 LLM 内部的位置编码。两个不同层级的模块。

**47.** ⭐ PatchMerger 做哪两件事？为什么 concat 不是 pooling？

> **答：** ① $2\times2$ 空间合并，四个 1280 维 token **concat** 成 5120，$N\to N/4$；② $\text{RMSNorm}\to\text{Linear}(5120,5120)\to\text{GELU}\to\text{Linear}(5120,3584)$。
> concat 是为了**先保住信息再让 MLP 学怎么压**，average pooling 会直接抹掉四个 token 的差异。

**48.** ⭐ 一个 LLM visual token 对应原图多大？$448\times448$ 有多少？

> **答：** $14\times14$ patch 再 $2\times2$ merge $\Rightarrow\boxed{28\times28}$ 像素，$N_{\text{LLM-vis}}=\frac H{28}\frac W{28}$。$448\times448\Rightarrow256$（ViT 内部是 1024）。

**49.** ⭐⭐ visual embedding 怎么进 LLM 序列的？有 cross-attention 吗？

> **答：** **没有 cross-attention。** chat template 里先放占位符 `<|image_pad|>`，processor 按 $\frac{T_gH_gW_g}{\text{merge}^2}$ 展开成 256 个；文字走正常 embedding lookup（`<|vision_start|>` 等 special token 也有自己的 embedding）；然后找到所有 `image_token_id` 的位置，用 `masked_scatter` 把这些占位符的 embedding **整个替换成** vision encoder 的输出。之后就是一条普通 $L\times3584$ 序列。
> 这就是 projector 必须输出 3584 的原因 —— 要和 text embedding 同空间同维度。

**50.** ⭐ Stage 1 里 LLM 冻结，ViT 怎么学？

> **答：** 冻结只是**不更新** $\theta_{LLM}$，梯度照样穿过去：$\frac{\partial L}{\partial\theta_V}=\frac{\partial L}{\partial V}\frac{\partial V}{\partial\theta_V}$。等于把 LLM 当固定的语言解释器，逼 ViT 学出"能让它正确预测 caption/OCR"的视觉表示。

**51.** ⭐⭐ Qwen2.5-VL 的 Stage 1 训不训 projector？

> **答：** **报告没说。** 只写了 "only the Vision Transformer is trained" + LLM frozen，没单独交代 merger。可能是字面意思，也可能是表格把 ViT+Merger 统称 vision part。**不要编**，答"报告明确 ViT train / LLM frozen，但没有单独交代 merger 的 freeze 状态"。

**52.** Qwen2.5-VL 的 ViT 是 SigLIP 初始化的吗？

> **答：** **不是。** Qwen 自研并从零训练，先用 DataComp 等做 CLIP 式视觉预训练。"from scratch" 指**没拿现成 SigLIP/CLIP 权重当视觉塔**，不是随机 ViT 直接和 LLM 裸训（那是 Kimi K3 MoonViT-V2 更激进的做法）。Qwen3-VL 才是明确从 SigLIP-2 checkpoint 初始化。

## G. DeepStack 与 Qwen 演进（[06](06-deepstack-and-qwen-evolution.md)）

**53.** ⭐ 三代的 patch / depth / hidden / attention 分别是什么？vision encoder 变大了吗？

> **答：** Qwen2.5-VL：patch 14、32 层、1280、window 为主 + 4 层 global。Qwen3-VL / Qwen3.5：patch 16、27 层、1152、**全局 attention**。
> **没变大，反而变小了**（$1280\to1152$、$32\to27$）。

**54.** Qwen3-VL 的 ViT 位置编码多了什么？

> **答：** 多了一套 **learned absolute PE**：`nn.Embedding(2304, hidden)`，$2304=48\times48$，实际 grid 不同时做 bilinear 插值再 $x_i\leftarrow x_i+p_i$。加上原有的 2D RoPE。
> 分工：absolute PE 说"我大概在哪"，2D RoPE 在 attention 里说"你俩相对空间关系"。

**55.** ⭐⭐ DeepStack 取哪几层？为什么每路都要一个自己的 merger？

> **答：** `[8,16,24]` 加上最终的 $X_{27}$。每路都要自己的 merger 是因为 ViT hidden 1152 和 LLM hidden 4096 的 **token 数和维度两边都对不上**，必须各自过 $2\times2$ merge + MLP：$4\times1152=4608\to4608\to4096$。

**56.** ⭐⭐ DeepStack 是 concat 吗？会撑长 sequence 吗？

> **答：** **不是 concat**，那样 token 数会变 4 倍。是 **element-wise addition**：`hidden_states[visual_pos_masks,:] + visual_embeds`，即 $h_{visual}\leftarrow h_{visual}+D_k$。$\boxed{\text{不增加 sequence length}}$，visual token 仍是 $N/4$。

**57.** ⭐⭐ LLM 输入已经是 ViT 最后一层，为什么 layer 0 后又加更浅的 $X_8$？

> **答：** 不是一一对应式的"ViT 8→LLM 0、ViT final→LLM 3"。实际是 **ViT final 作为正常 LLM input**，$X_8/X_{16}/X_{24}$ 经各自 merger 后在 LLM layer 0/1/2 **之后**作为额外 residual 加到 visual 位置。
> 本质 = final semantic feature + multi-level residual，等于 **ViT → LLM 的 skip connection**。名字里的 "Deep Stack" 是指在 **network depth** 上 stack，不是在 sequence 上。

**58.** ⭐ DeepStack 的动机？为什么 VLM 特别吃这个？

> **答：** ViT 浅层是 edge/texture/小字/OCR 字形，中层是 parts/object/region，深层才是 semantic。只用最后一层，低层细节已被 20 多层重新编码甚至弱化。而 VLM 很吃 OCR / GUI / chart / document / grounding / 小目标这些对局部空间细节敏感的任务。一句话：**不要要求 ViT final feature 一个人承担所有视觉信息**。

**59.** ⭐⭐ Qwen3.5 为什么删掉 DeepStack？

> **答：** 主因**训练范式变了**。Qwen3-VL 是 SigLIP2 + Qwen3 两个各自练好的模块拼起来，vision representation 不是 LLM-native 的，所以既需要 S0 alignment，也需要 DeepStack 这种 architecture-level inductive bias 强制保留 multi-level 特征。Qwen3.5 走 early-fusion native multimodal pretraining，三块从 foundation pretraining 起联合优化，final representation 本身就是为这个 backbone 长出来的。
> 次因：3 个额外 merger ≈120M 参数、四套大 MLP 的 prefill 计算、保留中间 activation + LLM 得知道哪层加哪级的工程复杂度。
> ⚠️ **这是推断，官方没有公开 DeepStack 的删除 ablation。**

**60.** ⭐⭐ "Native multimodal" 是不是没有 ViT / 图片也 autoregressive 生成？

> **答：** **都不是。** Qwen3.5 仍有完整 vision encoder（HF 说 "vision tower reuses the Qwen3-VL encoder"），也仍然只在 text token 上算 loss、ViT 靠文本 CE 反传（$\nabla_{\theta_{ViT}}L\ne0$）。
> "Native" 讲的是**训练方式**：不是先训好 text-only backbone 再外挂视觉，而是从 foundation pretraining 起就用 text/image/video 交错数据联合优化。所以不叫 Qwen3.5-VL，直接叫 Qwen3.5。

**61.** 同分辨率下 Qwen3-VL 的 visual token 比 Qwen2.5-VL 多还是少？

> **答：** **少约 23%**。$28\times28$ 像素/token → $32\times32$ 像素/token，比例 $(28/32)^2\approx0.766$。

**62.** Qwen3.5 真正的大架构变化在哪？

> **答：** **在 LLM 侧。** vision 还是 Qwen3-VL 那套，语言 backbone 换成 3:1 混合：$8\times(3\times\text{Gated DeltaNet}+1\times\text{Full Attention})=32$ 层。这才是它能扛大量 visual token / 长视频 / 长 context 的原因。

**63.** ⭐⭐ 多模态 PT 时混了纯文本数据，ViT 有梯度吗？

> **答：** **纯文本 batch 时没有。** 前向是 `text → LLM → CE`，根本不经过 ViT，所以 $\nabla_{\theta_V}L=0$，只有 LLM 收到梯度；VL batch 才是 ViT+Merger+LLM 都有梯度。
> 之所以要混纯文本，是为了防止 language capability degradation。所以"ViT 是 trainable 的"和"ViT 每个 step 都在更新"是两回事。

**64.** S0 为什么要先冻两端只训 merger？

> **答：** 一开始 merger 是随机的（"ViT 说中文、merger 乱翻、LLM 只懂英文"）。如果直接全参数解冻，两个**已经很好的 pretrained representation** 都可能为了迁就一个很烂的中间接口而乱跑。先固定两端只训桥，把 vision feature space 大致映到 LLM embedding space，即 modality alignment。Qwen3-VL 的 3 个 DeepStack merger 同样在这阶段被同一个 CE 训练，不需要各自的 loss。
