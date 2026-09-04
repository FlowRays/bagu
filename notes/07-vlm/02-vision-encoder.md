# Vision Encoder：CLIP / SigLIP / DINO，以及主流模型到底用什么

> **"现在主流 VLM 都用 SigLIP"这句话已经不准确了。** 这是本篇最想纠正的一点。

## 1. CLIP 到 SigLIP：为什么换掉 softmax

两者的 vision backbone 都可以是普通 ViT，主要区别是**预训练目标**。

### CLIP：batch 内 softmax contrastive

$$L_i=-\log\frac{\exp(s(I_i,T_i)/\tau)}{\sum_j\exp(s(I_i,T_j)/\tau)}$$

这一张图的正确 caption 要和整个 batch 里其他 caption 竞争，所以很依赖大 batch、batch 内 negative、多卡 all-gather，batch size 直接影响 loss 行为。

### SigLIP：sigmoid pairwise

对每个 pair 直接做二分类，$y_{ij}=+1$ 若 $i=j$ 否则 $-1$：

$$L=-\sum_{ij}\log\sigma\big(y_{ij}(s_{ij}+b)\big)$$

不再需要 softmax over whole batch → 扩展性和 batch size 灵活性更好。

### ⭐ image-text pair 是怎么来的：in-batch negatives

**不是人工给每张图挑负样本。** 训练数据本身就是配对的 $(I_1,T_1),\dots,(I_N,T_N)$，一个 batch 内把所有 image 和所有 text 两两组合成 $N\times N$：

| | $T_1$ 猫 | $T_2$ 汽车 | $T_3$ 苹果 |
|---|---|---|---|
| $I_1$ 猫 | ✅ | ❌ | ❌ |
| $I_2$ 汽车 | ❌ | ✅ | ❌ |
| $I_3$ 苹果 | ❌ | ❌ | ✅ |

$$\boxed{i=j\Rightarrow\text{positive}},\qquad\boxed{i\ne j\Rightarrow\text{negative}}$$

这就叫 **in-batch negatives** —— 一个 batch 里其他图片对应的文本自动成为当前图片的负样本。

⚠️ **正负极度不平衡**：$N=1024$ 时正样本 1024 个，负样本约 $1024^2-1024\approx10^6$ 个。所以 SigLIP 实际 loss 里有可学习的 scale / bias 来处理这个，不是简单把百万个 pair 当等价的二分类。

⚠️ **false negative 是真实存在的**：$I_1$ 是金毛、$T_1=$"a golden retriever"，batch 里刚好还有 $T_2=$"a dog playing outside" —— 按 pairing 规则 $(I_1,T_2)$ 被当成 negative，但它其实描述得挺对。CLIP/SigLIP 这类大规模图文对比学习都有这个问题，只是数据规模大、batch 随机采样，统计上仍能学出好的 representation。

### 一句话记住两者的区别

$$\text{CLIP}:\ \text{这些句子里，哪一句最匹配这张图？（softmax 单选题）}$$
$$\text{SigLIP}:\ \text{这张图和这句话匹不匹配？（sigmoid 判断题）}$$

softmax 强制 $\sum_j p(T_j|I_i)=1$，所以候选文本**天生互相竞争** —— 一张橘猫图配 "a cat" 和 "an orange cat on a sofa"，两个其实都对，但 softmax 下它们必须抢概率。sigmoid 各判各的，$P=0.95$ 和 $P=0.98$ 同时成立完全没问题。

### SigLIP2 加了什么

captioning-based pretraining、self-distillation、masked prediction、online data curation、localization/dense prediction 优化、native aspect ratio / 多分辨率、multilingual。

$$\boxed{\text{CLIP 强在 global semantic alignment}}$$
$$\boxed{\text{SigLIP2 还更重视 dense / localization / multilingual / variable-resolution}}$$

CLIP 不是不能用（LLaVA 就是 CLIP ViT-L/14），只是不再是最强默认选择。

## 2. 现在的两条路线

$$\text{off-the-shelf CLIP}\rightarrow\text{SigLIP2 初始化}\rightarrow\text{自研 ViT}\rightarrow\text{native end-to-end multimodal}$$

| 路线 | 典型模型 | Vision Encoder |
|---|---|---|
| 外部视觉模型 | 早期 LLaVA | CLIP |
| 强预训练 VE | PaliGemma | SigLIP |
| 强预训练 + 大改造 | **Qwen3-VL** | SigLIP2 → Qwen3-ViT |
| 自研 VE | InternVL | InternViT |
| 自研 + from scratch | **Kimi K3** | MoonViT-V2 |
| native multimodal | **Qwen3.5** | Qwen3-VL-style ViT，联合训练 |

## 3. DINO：无标签的自蒸馏

**DINO = self-DIstillation with NO labels**，image-only 自监督：

$$x_1=\text{Aug}_1(I),\quad x_2=\text{Aug}_2(I)$$
$$p_s=f_{\theta_s}(x_1),\quad p_t=f_{\theta_t}(x_2),\qquad \mathcal L=-\sum_k p_t(k)\log p_s(k)$$

teacher **不反向传播**，是 student 的 EMA：

$$\theta_t\leftarrow m\theta_t+(1-m)\theta_s$$

再用 sharpening / centering 防止 $p_s=p_t=\text{const}$ 的 representation collapse。

### 卡点：DINO 和 CLIP 学的东西很不一样

图片是"红杯子放在桌子左边，背景有窗户"，caption 可能只有 "A red cup on a table."。image-text supervision **根本没告诉模型**杯子在哪、桌子边界、窗户、每个 patch 属于什么、depth、object part。所以它偏 **global semantics**。

DINO 从图像本身学，因此特别擅长 patch-level representation、local correspondence、segmentation、depth、geometry、object parts、dense perception。

$$\boxed{\text{CLIP/SigLIP：语言对齐强}}\qquad\boxed{\text{DINO：纯视觉 / dense feature 强}}$$

规模：DINOv2 最大 ViT-g/14 约 **1.1B**（142M images）；DINOv3（2025）做到 **7B**（1.7B images），并加了 Gram Anchoring 等 dense feature 改进。

## 4. 主流模型的实际配置

> 三代 Qwen 的逐项对照（patch/depth/PE/DeepStack）见 [06 第 1 节](06-deepstack-and-qwen-evolution.md#1-三代放一起看)；
> Qwen2.5-VL 的完整走查见 [05](05-qwen25-vl.md)。这里只列各家的 encoder 选型。

### Qwen2.5-VL-7B：自研 ViT，不用 SigLIP

| 参数 | 值 |
|---|---:|
| Vision depth | 32 layers |
| hidden | 1280 |
| MLP intermediate | 3420（SwiGLU） |
| heads | 16 |
| patch size | 14×14（temporal patch 2） |
| attention | 28 层 window(112) + 第 7/15/23/31 层 full |
| spatial merge | 2×2 |
| output hidden | 3584 |
| 参数量 | ≈630M |

$$\boxed{\text{Qwen 自己从零训练的 ViT，没有拿现成 SigLIP/CLIP 权重当视觉塔}}$$

### Qwen3-VL-8B：SigLIP2 路线

技术报告明确：vision encoder 用 **SigLIP-2 架构**，从官方 pretrained checkpoint 初始化，然后做 dynamic-resolution 继续训练。8B 用 SigLIP2-So400m，2B/4B 用约 300M 的 SigLIP2-Large。

| 参数 | 值 |
|---|---:|
| Vision depth | 27 layers |
| hidden | 1152 |
| MLP intermediate | 4304 |
| heads | 16 |
| patch size | 16×16（temporal patch 2） |
| spatial merge | 2×2 |
| output hidden | 4096 |
| DeepStack layers | 8, 16, 24 |

MLP 是普通 GELU，不是 LLM 里常见的 SwiGLU。

$$\boxed{\text{Qwen3-VL ViT}=\text{SigLIP2 初始化}+\text{Qwen 特有改造}+\text{大规模继续训练}}$$

### DeepStack：不只用最后一层

普通做法只取 $ViT_{27}$ 的 feature。Qwen3-VL 额外在第 8/16/24 层各取一次，各配一个自己的 merger，然后在 LLM 前几个 decoder layer 之后**逐元素加到 visual token 的 hidden state 上**（不是 concat，**不增加 sequence length**）：

$$h^{vision}_l\leftarrow h^{vision}_l+D_l$$

动机是浅层的 texture / OCR / 局部空间信息不必等到最后一层全变成高度 semantic 的表示才交给 LLM。

> 完整机制（取哪层、怎么变维、加在哪、为什么 Qwen3.5 又删掉）见 [06 DeepStack 与 Qwen 演进](06-deepstack-and-qwen-evolution.md)。

### Qwen3.5-9B：native multimodal

已经不叫 Qwen3.5-VL 了，官方描述是 "Causal Language Model with Vision Encoder"，从头在 interleaved text/image/video token 上训练的 **native multimodal foundation model**，vision tower 复用 Qwen3-VL encoder 架构。

core ViT 配置和 Qwen3-VL-8B **完全一致**（27L / 1152 / 4304 / 16 heads / patch 16 / merge 2），但当前 checkpoint `deepstack_visual_indexes = []`，**没有** DeepStack。

### Kimi K3：MoonViT-V2 从零训练

| 参数 | MoonViT-V2 |
|---|---:|
| 参数量 | 401M |
| layers | 27 |
| hidden | 1024 |
| intermediate | 4096 |
| heads | 12 |
| patch size | 14 |
| norm | RMSNorm |
| linear/attn bias | False |
| merge kernel | 2×2 |
| projector | PatchMergerV2 |
| LLM hidden | 7168 |

**关键：从零训练，不用 SigLIP contrastive 初始化。** 官方说 MoonViT-V2 直接用 next-token prediction 训练，效果可达到 SigLIP-初始化 baseline，而且训练更稳定。

### 参数量汇总（backbone / projector 拆开）

| 模型 | Vision backbone | Projector / Merger | 合计 |
|---|---:|---:|---:|
| Qwen3-VL-8B | ≈415.9M | 主 merger ≈40.1M + 3 个 DeepStack merger ≈120.4M | **≈576M** |
| Qwen3.5-9B | ≈415.9M | ≈40.1M | **≈456M** |
| Kimi K3 | ≈401M | ≈46.1M | **≈447M** |
| DeepSeek-V4-Flash-Vision | ≈411.8M | Aligner ≈54.5M | **≈466M** |

$$\boxed{\text{都集中在 0.45B 左右，非常有意思}}$$

### 卡点：projector 居然有 40M

"projector 就是个小 MLP"是个误解。Qwen3-VL 因为 $2\times2$ merge，四个 1152 维 token concat 成 4608，然后

$$4608\to4608\to4096:\quad 4608^2+4608\times4096\approx21.2\text{M}+18.9\text{M}=40.1\text{M}$$

Kimi K3：$4\times1024=4096\to4096\to7168$，$\approx16.8+29.4=46.1$M。

## 5. Kimi K3：从零训练的 MoonViT-V2

为什么敢不用 SigLIP？因为当 multimodal pretraining 足够大时：

$$\boxed{\text{next-token prediction 本身就可以成为 vision encoder 的 supervision}}$$

$$\mathcal L=-\sum_t\log p(y_t|I,y_{<t})\ \Longrightarrow\ \text{梯度}\to\text{LLM}\to\text{projector}\to\text{ViT}$$

于是 ViT 直接学"什么视觉表示最有利于 LLM 做 multimodal next-token prediction"，而不是"什么视觉表示最有利于 CLIP image-text similarity"。

这两个目标并不完全一致：

$$L_{\text{contrastive}}\ \neq\ L_{\text{VLM}}$$

Kimi 报告 SigLIP 初始化版本在联合训练里 gradient norm 更高、spike 更多，from-scratch MoonViT-V2 更稳定且最终追平。

$$\boxed{\text{大规模 native multimodal training 下，CLIP/SigLIP 预训练不是必需条件}}$$

## 6. 卡点：为什么大家不 scale vision encoder

**有人试过。** InternVL 的核心卖点就是 Scaling up Vision Foundation Models，直接做了 **InternViT-6B**。DINOv2 1.1B、DINOv3 7B。但 6B ViT 没成为 VLM 标配，五个原因：

1. **瓶颈往往首先是 resolution，不是参数量。** $4096\times4096$、patch=16 就是 65536 个 patch。OCR 里一个小字号 `$` 如果 resize 后直接消失，$400M\to40B$ 也救不回来。所以算力先花在 dynamic resolution / AnyRes / tiling / smaller patch / multi-scale / DeepStack。
2. **ViT compute 随 visual token 爆炸。** self-attention 是 $O(N_v^2d)$，$N_v$ 从 1000 到 10000 是 100×。所以大家一边提分辨率一边疯狂 $2\times2$ merge。存在预算竞争：**更大的 ViT vs 更高的分辨率**，往往后者收益更直接。
3. **information bottleneck。** 就算 7B DINO 看得极细，10000 个 visual feature 经 projector/resampler 压成 1000 token 再给 LLM，中间仍有 visual token bottleneck，继续增大 VE 未必等比例提升。
4. **很多 benchmark 的最终瓶颈是 LLM reasoning。** 几何题里视觉只要读出边长，后面是数学推理，$\text{LLM }8B\to70B$ 通常比 $\text{ViT }400M\to6B$ 更值。
5. **400M 的 ViT 一点都不弱。** 语言里 400M 小，是因为 LLM 既要记知识又要推理生成规划；Vision Encoder 只负责 $I\to\text{representation}$，不负责 autoregressive generation、世界知识、long CoT、instruction following。

$$\boxed{\text{不能直接拿 ViT 参数规模和 LLM 参数规模比较}}$$

不过在 GUI Agent / Game Agent / Embodied / spatial reasoning / world model 这些场景下，问题不是"图片里是什么"，而是"按钮精确在哪、哪个物体移动了、门现在开还是关"，这恰恰需要强 dense visual representation ——**scale vision 未来仍然值得关注**，DINO / JEPA 类表示也是因此值得看。

## 7. 面试答法："现在 VLM 都用 SigLIP 吗"

> **不完全是。** SigLIP2 仍是非常主流的视觉底座，因为相比原始 CLIP，它的 sigmoid contrastive objective 更容易 scale，而且 SigLIP2 加强了 dense feature、localization、multilingual 和多分辨率能力。
>
> **但 frontier VLM 已明显往自定义 vision encoder 和 native multimodal pretraining 演进。** Qwen3-VL 的 ViT 从 SigLIP2-So400m 初始化后继续大规模训练，并加入 dynamic resolution、2D RoPE、2×2 merger 和 DeepStack；Kimi K3 的 MoonViT-V2 完全从零，用 multimodal next-token prediction 直接训练视觉编码器；Qwen3.5 则进一步把 multimodal training 融进 foundation-model pretraining。

## 自测（口述版）

**1.** 写出 CLIP 和 SigLIP 的 loss，说明换成 sigmoid 解决了什么问题。SigLIP2 还加了什么？

> **答：** CLIP 是 batch 内 softmax contrastive：$L_i=-\log\frac{\exp(s(I_i,T_i)/\tau)}{\sum_j\exp(s(I_i,T_j)/\tau)}$，正确 caption 要和整个 batch 竞争，因此依赖大 batch、batch 内 negative、多卡 all-gather。
> SigLIP 改成 sigmoid pairwise 二分类：$L=-\sum_{ij}\log\sigma(y_{ij}(s_{ij}+b))$，$y_{ij}=+1$ 若 $i=j$ 否则 $-1$，**不再需要 softmax over whole batch**，扩展性和 batch size 灵活性更好。
> SigLIP2 还加了：captioning-based pretraining、self-distillation、masked prediction、online data curation、localization/dense prediction 优化、native aspect ratio / 多分辨率、multilingual。

**2.** 写出 DINO 的 loss 和 teacher 更新方式。它为什么能学到 CLIP 学不到的东西？怎么防 collapse？

> **答：** 两个不同 augmentation 分别给 student 和 teacher：$\mathcal L=-\sum_k p_t(k)\log p_s(k)$。teacher **不反向传播**，是 student 的 EMA：$\theta_t\leftarrow m\theta_t+(1-m)\theta_s$。用 sharpening 和 centering 防止 $p_s=p_t=\text{const}$ 的 representation collapse。
> 能学到不同的东西是因为：caption「A red cup on a table.」根本没告诉模型杯子具体在哪、桌子边界、窗户、每个 patch 属于什么、depth、object part，所以 image-text supervision 偏 **global semantics**；DINO 从图像本身学，因此擅长 patch-level representation、local correspondence、segmentation、depth、geometry、dense perception。

**3.** Qwen3-VL-8B 的 vision config 是什么？DeepStack 是什么、加在哪、为什么？

> **答：** **SigLIP-2 架构**，从官方 pretrained checkpoint 初始化后做 dynamic-resolution 继续训练（8B 用 SigLIP2-So400m）。配置：27 层、hidden 1152、MLP 4304（普通 GELU 不是 SwiGLU）、16 heads、patch 16×16、temporal patch 2、spatial merge 2×2、输出 4096。
> **DeepStack**：不只用 ViT 最后一层，而是在第 **8/16/24** 层各取一次 feature、各配一个 merger，然后**加到 LLM 前几层的 hidden state 上**（$h_l\leftarrow h_l+h^{\text{vision}}_l$）。目的是让 intermediate ViT 的 texture / OCR / local / spatial feature 不必等到最后一层全变成高度 semantic 的表示才交给 LLM，官方说法是提升 fine-grained detail 与 image-text alignment。

**4.** Qwen3.5-9B 的 vision tower 和 Qwen3-VL 差在哪？

> **答：** core ViT 配置**完全一致**（27L / 1152 / 4304 / 16 heads / patch 16 / merge 2），但当前 checkpoint `deepstack_visual_indexes = []`，**没有 DeepStack**，所以 visual module 从 ~576M 降到 ~456M。
> 而且它已经不叫 Qwen3.5-VL —— 官方定位是「Causal Language Model with Vision Encoder」，是从头在 interleaved text/image/video token 上训练的 **native multimodal foundation model**。

**5.** Kimi K3 为什么敢不用 SigLIP 初始化？背后的 objective mismatch 是什么？

> **答：** 因为当 multimodal pretraining 足够大时，**next-token prediction 本身就可以成为 vision encoder 的 supervision** —— $\mathcal L=-\sum_t\log p(y_t|I,y_{<t})$ 的梯度会一路传到 ViT。
> **objective mismatch**：SigLIP 优化的是 image-text embedding similarity，而 VLM 最终优化的是 $p(\text{text}|\text{image},\text{context})$，两者**不完全一致**（$L_{\text{contrastive}}\ne L_{\text{VLM}}$）。
> Kimi 报告 SigLIP 初始化版本在联合训练里 gradient norm 更高、spike 更多，from-scratch MoonViT-V2（401M、27L、hidden 1024、FFN 4096、12 heads、patch 14、RMSNorm、无 bias）更稳定且最终追平。

**6.** 为什么 projector 会有 40M 参数？用 Qwen 的 $2\times2$ merge 算一遍。

> **答：** 因为 $2\times2$ merge 要先 concat：$4\times1152=4608$，然后 $4608\to4608\to4096$：
> $$4608^2+4608\times4096\approx21.2\text{M}+18.9\text{M}=\mathbf{40.1M}$$
> Kimi K3 是 $4\times1024=4096\to4096\to7168\approx16.8+29.4=46.1$M。「projector 就是个小 MLP」是误解。

**7.** 有人 scale 过 vision encoder 吗？为什么 6B ViT 没成标配？说出至少三个原因。

> **答：** **有**：InternVL 的核心卖点就是 Scaling up Vision Foundation Models，做了 **InternViT-6B**；DINOv2 1.1B、DINOv3 7B。
> 没成标配的原因：
> ① **瓶颈往往首先是 resolution 不是参数量** —— OCR 里小字号 resize 后直接消失，$400M\to40B$ 也救不回来，所以算力先花在 dynamic resolution / AnyRes / tiling / multi-scale / DeepStack；
> ② **ViT compute 随 visual token 爆炸**（$O(N_v^2d)$，$N_v$ 从 1000 到 10000 是 100×），存在「更大 ViT vs 更高分辨率」的预算竞争；
> ③ **information bottleneck**：10000 个 feature 压成 1000 token 再给 LLM，继续增大 VE 未必等比例提升；
> ④ 很多 benchmark 的最终瓶颈是 **LLM reasoning**，$\text{LLM }8B\to70B$ 通常更值；
> ⑤ **400M 的 ViT 一点都不弱** —— 它只负责 $I\to\text{representation}$，不负责生成、世界知识、long CoT、instruction following，**不能直接拿 ViT 参数规模和 LLM 比**。

