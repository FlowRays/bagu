# DINOv2 完整走查：架构没有魔法，魔法全在训练

> [02](02-vision-encoder.md#3-dino无标签的自蒸馏) 只给了 DINO 的 loss 一句话。这一篇把 DINOv2 从 ViT 结构一直拆到"为什么不坍缩"。
> 读它的动机是具身：[OpenVLA](../../embodied/06-vla/02-gen1-ar-vla.md#openvla-的视觉侧为什么是-dinov2-加-siglip) 的 vision encoder 就是 DINOv2 + SigLIP 并联。

一句话定位：

$$\boxed{\text{DINOv2} = \text{一个普通 ViT} + \text{一整套自监督训练配方}}$$

**backbone 真的没什么特别的。** 值得学的是训练那半边。

## 1. 推理时的架构：就是 ViT

$224\times224$ 输入、patch size 14：

```text
Image [B,3,224,224]
        ↓ Patch Embedding (Conv2d, kernel=14, stride=14)
    [B,256,D]                    224/14=16, 16×16=256
        ↓ 前面拼一个可学习的 CLS token
    [B,257,D]
        ↓ + learned absolute position embedding
        ↓ Transformer × L
        ↓ LayerNorm
   CLS + 256 patch tokens
```

patch embedding 的机制和 [01b](01b-patch-embedding.md) 讲的完全一样，不再重复。

四个规模：

| 模型 | 参数 | Depth | Hidden $D$ | Heads | 每头维度 |
|---|---:|---:|---:|---:|---:|
| ViT-S/14 | ~21M | 12 | 384 | 6 | 64 |
| ViT-B/14 | ~86M | 12 | 768 | 12 | 64 |
| ViT-L/14 | ~300M | 24 | 1024 | 16 | 64 |
| ViT-g/14 | ~1.1B | 40 | 1536 | 24 | 64 |

$d_{\text{head}}=64$ 一路不变，scale 的是 hidden / depth / heads。**和 LLM 的 scaling 习惯一模一样。**

### 位置编码是 learned absolute，不是 RoPE

$$x_i = e_i + p_i$$

$p_i$ 是查表学出来的。换分辨率（$224\to518$）时 patch 数变了，靠**二维插值**把 position embedding 拉到新 grid 上。

⚠️ 这和你刚学的 [Qwen2.5-VL 的 2D RoPE](05-qwen25-vl.md#4-vit-内部是-2d-rope不是-mrope) **完全不是一路设计**。别混。

Transformer block 也是标准 Pre-LN，只多了 LayerScale 和 DropPath 这类稳定化：

$$x' = x + \text{LayerScale}(\text{Attn}(\text{LN}(x))),\qquad x'' = x' + \text{LayerScale}(\text{FFN}(\text{LN}(x')))$$

Attention 本身没有任何 DINO 特有的东西。

## 2. 训练时的架构：多两个 head，多一个 teacher

```text
                      ┌──────────────┐
image ─ augment ────→ │  Student ViT │ ──┬─→ DINO Head ─→ loss
                      └──────────────┘   └─→ iBOT Head ─→ loss
                                                  ↑
                      ┌──────────────┐            │ target
image ─ augment ────→ │  Teacher ViT │ ───────────┘
                      └──────────────┘   （无梯度）
```

$$\text{Student} = \text{ViT} + \text{DINO Head} + \text{iBOT Head}$$

⚠️ **两个 head 训练完就扔掉。** 拿去做 VLM / VLA vision encoder 的只有 ViT backbone。看到 "65536 维 prototype" 时不要以为视觉特征是 65536 维的。

## 3. Multi-crop：teacher 和 student 看到的东西不一样

一张图先切成：

```text
             一张狗的图片
                  │
       ┌──────────┴───────────┐
   2 个 global crop      8 个 local crop
   （整只狗）            （狗头 / 狗腿 / 身子…）
```

分工是关键：

| | 看什么 |
|---|---|
| **Teacher** | 只看 2 个 global crop |
| **Student** | global + local 全看 |

于是训练目标变成：

> Student 只看到狗头，也要产生和 Teacher 看整只狗一致的高层语义。

这就是 view invariance 的来源。

## 4. DINO loss：管 CLS，管"整张图是什么"

$$L_{\text{DINO}} = -\sum_k q_T(k)\log p_S(k)$$

形式就是 soft-label CE（和你在 [蒸馏](../06-post-training/distill/02-sft-and-kd.md) 那边见的一样）。链路：

```text
Teacher: global crop → ViT → CLS → DINO head → q_T
Student: 任意 crop   → ViT → CLS → DINO head → p_S
```

### 卡点：这里的 k 不是类别

DINO head 输出 $K=65536$ 维（ViT-L/g 的大规模配置用 131072）。但是：

$$k=1 \ne \text{"狗"},\qquad k=2 \ne \text{"猫"}$$

**根本没有 label。** 这些是 learned **prototype**，可以理解成视觉 latent space 里的一堆锚点。某张图的输出可能是：

```text
prototype #114    0.31
prototype #9271   0.26
prototype #421    0.11
```

没人给它们起过名字，是训练自己长出来的结构。

head 本身就是个 MLP：

$$D \to 2048 \to 2048 \to 256 \to 65536$$

它唯一的目的是**构造一个适合做自蒸馏的预测空间**，不是最终的视觉表示。

## 5. iBOT loss：管 patch，管"这个位置是什么"

只训 CLS 的话，梯度虽然也会经 attention 流回 patch token，但模型**没有被强制要求**每个 patch 自己有意义。而分割 / 检测 / 机器人恰恰要 patch 特征。

所以加一路 masked image modeling：

```text
Teacher（完整图）:  [狗耳][狗脸][狗眼][背景]
                      ↓ ViT
                     T1    T2    T3    T4

Student（挖掉两块）: [MASK][狗脸][MASK][背景]
                      ↓ ViT
                     S1    S2    S3    S4

要求:  S1 → T1,  S3 → T3
```

$$L_{\text{iBOT}} = -\sum_{i\in M}\sum_k q^T_{i,k}\log p^S_{i,k},\qquad M=\text{被 mask 的位置}$$

**mask 不是把像素涂黑**，是在 patch embedding 之后把那个 token 换成一个 learned `mask_token`。

### 卡点：iBOT 不是 MAE

这是最值得记的一条对比：

| | MAE | iBOT |
|---|---|---|
| 预测目标 | **RGB 像素** | **Teacher 的语义 feature** |
| loss | $\|x-\hat x\|^2$ | soft CE |
| 容易学到 | 纹理、颜色、低级统计 | 局部语义、part-whole 关系 |

$$\boxed{\text{MAE：这块区域长什么像素}\qquad\text{iBOT：这块区域在语义空间里是什么}}$$

iBOT 原文把 Teacher 叫 **online tokenizer**：它动态产生带语义的视觉 target。

Student 在那个位置只看到 `[MASK]`，却要说出"这里应该是狗耳"，只能靠周围 patch + 位置 + 全局上下文推。这就逼出了对局部结构和空间上下文的理解。

结果是 patch 空间会自发形成对应关系，**甚至跨图片**：

```text
Image A            Image B
狗头  ───────────→  狗头
狗腿  ───────────→  狗腿
```

消融很直接：去掉 iBOT，ADE20K 分割从 **47.1 掉到 44.2 mIoU**。

## 6. KoLeo：别让所有图片挤在一起

作用在 CLS feature 上，直觉一句话：

> 不同图片的 embedding 不要全堆在特征空间的一小块里。

$$L_{\text{KoLeo}} = -\frac1B\sum_i \log d(i,\text{NN}(i))$$

$\text{NN}(i)$ 是 batch 内最近邻。最小化这个 loss 等价于**倾向于把最近邻距离拉大**，让表示铺开。

## 7. 总 loss 和一次完整 step

$$\boxed{L = L_{\text{DINO}} + L_{\text{iBOT}} + 0.1\,L_{\text{KoLeo}}}$$

三个尺度分工：

```text
DINO   →  CLS   →  跨 crop 一致   →  "这张图整体是什么"
iBOT   →  patch →  masked 预测    →  "这个位置局部是什么"
KoLeo  →  CLS   →  分布           →  "不同图片别挤在一起"
```

一个 step：

```text
1. augment: x → g1,g2 (global) + l1..l8 (local)
2. Teacher forward: g1,g2 → CLS + patch          （no_grad）
3. Student forward: g1,g2 加 mask；l1..l8 不 mask
4. L_DINO(CLS) + L_iBOT(masked patch) + 0.1·L_KoLeo(CLS)
5. backward → 只更新 Student → optimizer.step()
6. θ_T ← m·θ_T + (1-m)·θ_S                       （EMA）
```

## 8. 最关键的问题：为什么不坍缩

没有 label、没有负样本、teacher 还来自 student。存在一个完美作弊解：

$$\forall x,\quad p(x)=[1,0,0,\dots]$$

猫、狗、汽车全输出同一个 prototype，$p_S=q_T$，loss 极低，**特征全废**。这就是 representation collapse。

### 先分清两种坍缩

假设只有 4 个 prototype。

**坍缩 A：全挤同一个 prototype**

```text
猫   → [0.99, 0.00, 0.00, 0.01]
狗   → [0.99, 0.00, 0.00, 0.01]
汽车 → [0.99, 0.00, 0.00, 0.01]
```

每张图都很自信，但都一样。

**坍缩 B：全都均匀**

```text
猫   → [0.25, 0.25, 0.25, 0.25]
狗   → [0.25, 0.25, 0.25, 0.25]
汽车 → [0.25, 0.25, 0.25, 0.25]
```

同样是 $p(x_1)=p(x_2)=p(x_3)$，只是塌向了另一头。

### DINO 用两个方向相反的力互相制衡

$$\boxed{\text{Sharpening 防坍缩 B}\qquad\text{Centering 防坍缩 A}}$$

**Sharpening** 就是 teacher 用很低的 temperature：

$$q_T = \text{softmax}\left(\frac{z_T-c}{\tau_T}\right),\qquad p_S=\text{softmax}\left(\frac{z_S}{\tau_S}\right)$$

原版设置 $\tau_s\approx0.1$，$\tau_T$ 从 $0.04$ warmup 到 $0.07$。看具体数字：logits $z=[3,2,1]$，

| $\tau$ | softmax |
|---|---|
| $1$ | $[0.67,0.24,0.09]$ |
| $0.1$ | $\approx[0.99995,0.00005,0]$ |

teacher 必须给出一个**明确**的 target，否则 student 学到的只是"什么都差不多"。

**但 sharpening 单独用会引爆坍缩 A。** 因果链很值得记：

```text
prototype 2 稍微占优
      ↓ sharpen 放大这个优势
所有图都指向 prototype 2
      ↓ student 模仿
student 也全指 prototype 2
      ↓ teacher = EMA(student)
teacher 更偏 prototype 2
      ↓ 正反馈
```

**Centering** 就是来打断这个正反馈的。算整个 batch 的 teacher logits 均值：

```text
cat   [1, 8, 2, 1]
dog   [2, 7, 3, 2]
car   [1, 9, 2, 2]
bird  [2, 8, 1, 2]
        ↓
c   = [1.5, 8, 2, 1.75]
```

第二维在全 batch 里都大，减掉 $c$ 之后 cat 变成 $[-0.5,0,0,-0.75]$ ——**那个全局优势被消掉了**。实际用的是 EMA center：

$$c \leftarrow mc + (1-m)\frac1B\sum_i z_i$$

一句话记：

> **哪个 prototype 在 batch 里太热门，就给它降温。**

合起来：

| | 约束对象 | 说的话 |
|---|---|---|
| Sharpening | 单张图 | "你得做出选择" |
| Centering | 整个 batch | "但不能所有人选同一个" |

### DINOv2 换成了 Sinkhorn-Knopp

大规模配方里不再只靠 moving-average centering，而用 Sinkhorn-Knopp（3 次迭代）处理 teacher 的 prototype assignment。它做的事更硬：

```text
原始 score          P1  P2  P3  P4
猫                   8   2   1   1
狗                   7   3   1   1
车                   6   4   2   1
鸟                   9   2   1   1
        直接 softmax → 全去 P1 → 坍缩

Sinkhorn 反复:  prototype 维 normalize → sample 维 normalize → 重复
```

同时要求每行是概率分布（$\sum_k q_{ik}=1$），且**列方向上不能让少数 prototype 垄断整个 batch**。

### EMA teacher 为什么还必须在

有了 Sinkhorn，teacher 直接 = student 行不行？不行，因为 target 会跟着一起乱跑：

```text
student 追 target，而 target 就是 student 自己 → 系统不稳定
```

EMA 让 teacher 成为 student 参数的低通滤波版本：

$$\theta_T \leftarrow m\theta_T+(1-m)\theta_S,\qquad m: 0.994 \to 1.0\ \text{(cosine)}$$

$$\boxed{\text{Teacher = 一个更稳定的 online target generator}}$$

这和 RL 里的 **target network** 是同一个思想。

## 9. register token 是后来的事

有些权重叫 `ViT-L/14-reg`，序列前面多 4 个 learnable register token：

```text
[CLS][REG1][REG2][REG3][REG4][patch 1]...
```

⚠️ 这来自后续的 *Vision Transformers Need Registers*，**不是 DINOv2 原始方法的组成部分**。入门阶段可以完全忽略。

## 10. 和 CLIP / SigLIP 的分工

| | CLIP / SigLIP | DINOv2 |
|---|---|---|
| 数据 | image-text | **image only** |
| 训练 | 对比 / 匹配 | 自蒸馏 + masked patch |
| 强项 | 语义对齐 | **dense / spatial 表示** |
| 文本语义 | 强 | 无 |
| 分割 / 深度 / 对应 | 一般 | **非常强** |

$$\boxed{\text{SigLIP 回答 what}\qquad\text{DINOv2 回答 where / 长什么样}}$$

机器人两个都要："哪个是 red cup"（SigLIP）+"它的杯口在哪、gripper 离它多远"（DINOv2）。所以 [OpenVLA 把两个并联](../../embodied/06-vla/02-gen1-ar-vla.md#openvla-的视觉侧为什么是-dinov2-加-siglip)。

## 脑图

```text
                     DINOv2
                       │
              ┌────────┴────────┐
          Architecture       Training
              │                 │
             ViT          Teacher–Student
        patch size 14       EMA teacher
        CLS + patch          Multi-Crop
        learned abs PE    ┌─────┴─────┐
        LayerScale      DINO         iBOT
                         CLS         patch
                        global       local
                          └─────┬─────┘
                             KoLeo
                                │
                          防坍缩三件套
                    sharpening / centering
                       / Sinkhorn-Knopp
```

## 自测

**1.** DINOv2 的 backbone 有什么特别之处？

> **答：** 基本没有。就是标准 ViT：patch 14、CLS + patch token、**learned absolute PE**（换分辨率靠二维插值，不是 RoPE）、Pre-LN block 加 LayerScale 和 DropPath。四个规模 S/B/L/g 的 $d_{\text{head}}$ 都是 64，scale 的是 hidden/depth/heads。
> DINOv2 的价值全在**训练配方**。

**2.** 训练时的模型和推理时的模型有什么区别？

> **答：** 训练时是 `Student(ViT + DINO head + iBOT head)` + `Teacher(同结构，无梯度)`。**两个 head 训练完就扔**，拿去当 vision encoder 的只有 ViT backbone。所以"65536 维 prototype"不是视觉特征的维度。

**3.** multi-crop 里 teacher 和 student 各看什么？为什么这样分？

> **答：** teacher **只看 2 个 global crop**，student 看 global + 8 个 local crop。
> 这样目标就变成"只看到狗头，也要产生和看整只狗一致的语义"，view invariance 就是这么来的。

**4.** DINO loss 里的 $k$ 是类别吗？

> **答：** **不是**，根本没有 label。是 learned **prototype**，可以理解成视觉 latent space 里的锚点，没人给它们起名字。head 是 $D\to2048\to2048\to256\to65536$ 的 MLP，唯一目的是构造一个适合自蒸馏的预测空间。

**5.** iBOT 和 MAE 的区别？为什么这个区别重要？

> **答：** MAE 预测**RGB 像素**（$\|x-\hat x\|^2$），容易学到纹理颜色这类低级统计；iBOT 预测 **teacher 的语义 feature**（soft CE），学到的是"这块区域在语义空间里是什么"。
> Teacher 在这里扮演 **online tokenizer**。这个区别就是 DINOv2 dense feature 特别好的原因 —— 去掉 iBOT，ADE20K 分割从 **47.1 掉到 44.2 mIoU**。

**6.** mask 是怎么做的？

> **答：** **不是把像素涂黑**，是在 patch embedding 之后把对应位置的 token 替换成一个 learned `mask_token`。

**7.** ⭐ 两种坍缩分别长什么样？

> **答：** **A（全挤一个 prototype）**：每张图都输出 $[0.99,0,0,0.01]$，都很自信但都一样。
> **B（全都均匀）**：每张图都输出 $[0.25,0.25,0.25,0.25]$。
> 共同点都是 $p(x_1)=p(x_2)=p(x_3)$，只是塌向两个相反的方向。

**8.** ⭐ sharpening 和 centering 各防哪一种坍缩？为什么必须同时用？

> **答：** **sharpening**（teacher 用低 temperature，$\tau_T\approx0.04\to0.07$）防**均匀坍缩 B**：teacher 必须给出明确 target，否则 student 只学到"什么都差不多"。
> **centering**（减 batch 内 logits 的 EMA 均值）防**独占坍缩 A**：哪个 prototype 在 batch 里太热门就给它降温。
> 必须同时用是因为 **sharpening 单独用会正反馈引爆 A**：某个 prototype 稍占优 → sharpen 放大 → student 模仿 → teacher EMA 跟随 → 更占优。centering 就是来打断这个环的。
> 一句话：**sharpening 对单张图说"你得选一个"，centering 对整个 batch 说"但不能都选同一个"**。

**9.** DINOv2 相对原版 DINO 在防坍缩上改了什么？

> **答：** 大规模配方用 **Sinkhorn-Knopp**（3 次迭代）代替单纯的 moving-average centering，交替在 prototype 维和 sample 维归一化，更硬地要求 batch 内的 assignment 均衡。

**10.** 有了 Sinkhorn，teacher 直接等于 student 行不行？

> **答：** 不行。target 会跟着 student 一起乱跑，系统不稳定。EMA（$m:0.994\to1.0$ cosine）让 teacher 成为 student 参数的低通滤波版本，是一个**更稳定的 online target generator** —— 和 RL 的 **target network** 同一个思想。

**11.** 总 loss 是什么？三项各管什么尺度？

> **答：** $L = L_{\text{DINO}} + L_{\text{iBOT}} + 0.1L_{\text{KoLeo}}$。
> **DINO** 管 CLS / 全局语义 / 跨 crop 一致；**iBOT** 管 patch / 局部语义 / 空间结构；**KoLeo** 管 CLS 特征分布，别让不同图片挤在一起（$-\frac1B\sum_i\log d(i,\text{NN}(i))$，最小化它等于把最近邻距离拉大）。

**12.** `ViT-L/14-reg` 里的 register token 是 DINOv2 的一部分吗？

> **答：** 不是。来自后续的 *Vision Transformers Need Registers*，在 CLS 后加 4 个 learnable token。入门阶段可以忽略。

**13.** 为什么具身模型喜欢 SigLIP + DINO 并联？

> **答：** 机器人既要知道"哪个是 red cup"（语言对齐，SigLIP 强），又要知道"杯口在哪、gripper 离它多远、边缘在哪"（dense spatial，DINOv2 强）。
> 记：**SigLIP 回答 what，DINOv2 回答 where**。
