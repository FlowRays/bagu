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
