# 第一代 VLA：RT-1 → RT-2 → RT-X → OpenVLA

> [01](01-vla-models.md) 是一页纸的演进线。这一篇把**第一代（动作离散化 + token 预测）**逐 tensor 拆开。
> 前置：[动作空间从零](../05-action/01b-action-space-from-zero.md)、[VLM 架构](../../notes/07-vlm/00-map.md)。

为什么要按历史顺序学：**后面每一种设计，都是在修前一代暴露的问题。** 直接跳 π0 / G0.5 会把动机全吞掉。

```text
RT-1     一个机器人学会很多任务
  ↓  机器人数据装不下世界知识
RT-2     把预训练 VLM 直接变成 policy，动作变成语言 token
  ↓  只有单一本体
RT-X     多本体数据混合训练（粗对齐）
  ↓  不开源、模型巨大
OpenVLA  开源 7B，现代 VLM 架构
  ↓  仍然：单帧、无 chunk、逐标量 256-bin
第二代   Diffusion Policy / π0（连续动作 + flow matching）
```

## 1. RT-1：还不算 VLA，但是起点

⚠️ **RT-1 严格说不是今天意义上的 VLA**，它是 language-conditioned robot policy。"VLA"这个词是 RT-2 才正式立起来的。但 RT-2 几乎就是在回答 RT-1 提出的问题，所以从这里开始。

数据（2022，Google）：约 **130k+ episodes**、**700+ tasks**、13 台机器人、采了 17 个月，人类遥操作 + 每条 episode 标一句自然语言。训练方式是纯 **behavior cloning**，没有 RL。

### 输入不是一张图，是 6 帧

$$(I_{t-5:t},\ l)\ \longrightarrow\ a_t$$

每张 $300\times300$。⚠️ **RT-1 的主输入里没有显式 proprioception**，靠 6 帧 RGB 历史 + 语言指令决定动作。

### 语言：整句一个向量，不是 token 序列

和现代 VLM 完全不同：

```text
"pick up the coke can"
        ↓  Universal Sentence Encoder (USE)
   一个句子向量 e_l ∈ R^d
```

不是 `pick / up / the / coke / can` 五个 token。**语言根本不在 Transformer 的 token stream 里。**

### 语言怎么影响视觉：FiLM

**FiLM = Feature-wise Linear Modulation**。CNN 的某层特征 $F$，用语言算两个量再去调制它：

$$F' = \gamma(e_l)\odot F + \beta(e_l)$$

直觉：同一张图里有红杯子、可乐罐、抽屉、薯片袋。

```text
instruction = "pick up the coke can"  →  可乐罐相关的 feature 重要
instruction = "open the drawer"       →  抽屉相关的 feature 重要
```

所以不是 $\text{image}\to\text{feature}$，而是

$$(\text{image},\text{language})\longrightarrow \text{task-conditioned visual feature}$$

### 视觉：CNN 不是 ViT

```text
300×300×3
    ↓ FiLM-conditioned EfficientNet-B3
  9×9×512
    ↓ flatten（9×9=81 个空间位置）
 81 × 512   ← 81 个 visual token
```

形式上很像 ViT 的 $N\times d$，只是这些 token 来自 CNN feature map 的 flatten，不是 patch embedding。

### TokenLearner：81 → 8

6 帧 × 81 = **486 个 token**，而 2022 年还要求约 3 Hz 实时，太贵。所以做 learned pooling：

$$z_k=\sum_{i=1}^{81}\alpha_{ki}x_i,\qquad \sum_i\alpha_{ki}=1$$

**不是随便抽 8 个位置**，是学 8 组 attention 权重。第 1 个可能盯杯子附近、第 2 个盯 gripper、第 3 个盯桌面。

于是 $6\times8=48$ 个 token 进 Transformer，再加 position encoding 区分新旧帧。

### 规模：今天看非常小

| 部分 | 参数 |
|---|---|
| FiLM-EfficientNet | ~16M |
| Transformer（8 层 decoder-only） | ~19M |
| **合计** | **~35M** |

### 动作：11 维，因为有移动底盘

RT-1 的机器人是「移动底盘 + 7-DoF 臂 + 夹爪」：

| 部分 | 维度 |
|---|---:|
| mode（terminate 等） | 1 |
| arm translation | 3 |
| arm rotation | 3 |
| gripper | 1 |
| base rotation | 1 |
| base translation | 2 |
| **合计** | **11** |

每一维离散成 **256 个均匀 bin**。$\Delta x=0.137$ 变成 bin 145，于是 $\mathbb R^{11}\to\{0..255\}^{11}$。

### 卡点：RT-1 不是 LLM 式的自回归

很容易顺手说成"第一代 VLA 就是逐 token 自回归生成动作"，**对 RT-1 不成立**。它默认是**并行预测 11 个维度**：

$$h\ \longrightarrow\ \{p(a_1),p(a_2),\dots,p(a_{11})\},\qquad \text{每个都是 256-way 分类}$$

官方实现默认 `include_prev_timesteps_actions=False`；原论文的消融里才有"autoregressively conditioning on actions"这个变体。

$$\boxed{\text{discrete action}\ \neq\ \text{LLM-style autoregressive action}}$$

RT-1 只迈出了**把连续控制变成离散分类**这一步。把动作真正塞进 VLM 的 vocabulary 是 RT-2 干的。

loss 因此特别朴素：

$$L=-\sum_{d=1}^{11}\log p_\theta(y_d\mid I_{t-5:t},l)$$

$$\boxed{\text{RT-1} = \text{Behavior Cloning} + \text{Cross Entropy}}$$

没有 value model、advantage、PPO、GRPO。

## 2. RT-2：把动作伪装成语言

$$\boxed{\text{RT-2} = \text{把一个预训练 VLM 直接训练成 robot policy}}$$

### 动机：130k episodes 装不下世界

指令："pick up the object that can be used as a hammer"。这需要两种知识：

| | 内容 | 谁教得了 |
|---|---|---|
| **A. 世界知识** | 石头是硬的 → 可以当锤子 | 只有 web 数据 |
| **B. 机器人控制** | 找到石头 → 移过去 → 抓起来 | robot demonstration |

RT-1 的 130k 条能教 B，但永远覆盖不了人类世界的所有物体、符号、数字、关系、常识。而当时的 VLM 早就在海量图文上学过这些。

于是问题变成：**能不能把 VLM 的 web knowledge 迁移到 robot action？** 这就是论文标题里的 *Transfer Web Knowledge to Robotic Control*。

### 激进的地方：不要中间那层文本

直觉方案是分层：

```text
camera + instruction → VLM → "pick up the rock" → 另一个 robot policy → 动作
```

RT-2 直接砍掉中间：

```text
camera + instruction → VLM → 动作
```

$$\boxed{\text{VLM 本身就是 policy}}$$

### 动作是 8 个分量

$$a_t=[\text{terminate},\ \Delta x,\Delta y,\Delta z,\ \Delta r_x,\Delta r_y,\Delta r_z,\ g]$$

即 1 terminate + 6D EE delta + 1 gripper。

⚠️ 常见的说法"RT-2 沿用 RT-1 的 action discretization"，指的是**继承 256-bin 这个方法**，不是说两者的动作向量每一维都相同（RT-1 是 11 维、含底盘）。

### 关键一步：把 bin 变成文本字符串

$$[1,145,92,129,5,101,127,217]\ \longrightarrow\ \texttt{"1 145 92 129 5 101 127 217"}$$

于是一条 robot demonstration 被改写成：

```text
Image:    <robot camera>
Question: What action should the robot take to "pick up the coke can"?
Answer:   1 145 92 129 5 101 127 217
```

对照一条普通 VQA：

```text
Image:    [一张驴的图片]
Question: What animal is this?
Answer:   donkey
```

**对 Transformer 来说没有本质区别**，都是 $(\text{image},\text{prompt})\to\text{token sequence}$。

RT-2 最漂亮的地方就在这：它没有发明复杂的 robotics 架构，只是改了 **action representation**，让机器人控制变成 VLM 本来就会做的 sequence generation。

### 没有 action head

```text
              VLM
               ↓
          same decoder
               ↓
          same LM head
               ↓
       vocabulary logits
          /          \
        text        action
```

$$\boxed{\text{text 和 action 共用全部参数}}$$

### 两个 backbone，两种塞法

RT-2 不是一个固定架构，是一个 **training recipe**，作者用两个 VLM 验证：

| 版本 | backbone | action token 怎么来 |
|---|---|---|
| RT-2-PaLI-X | PaLI-X | tokenizer 本来就给 $0..1000$ 每个整数独立 token，**直接拿来用** |
| RT-2-PaLM-E | PaLM-E | 没有方便的整数 token，**找出词表里最不常用的 256 个覆盖掉** |

后者就是今天 `add special tokens` 的暴力版。

### 这次是真正的自回归

$$p(a_t\mid o_t,l)=\prod_{i=1}^{8}p(x_i\mid o_t,l,x_{<i})$$

loss 彻底就是 VLM 的 NTP：

$$L_{\text{robot}}=-\sum_{i=1}^{8}\log p_\theta(x_i\mid I,l,x_{<i})$$

### Co-Fine-Tuning：第二个核心创新

只拿 robot data 微调会怎样？**灾难性遗忘**。机器人数据很窄，训几十万步后模型越来越只会输出 `128 91 241 5 ...`，而原来的识物、数数、比较能力退化 —— 可 RT-2 想要的恰恰就是那些能力。

所以 batch 是混的：

| 版本 | robot data 占比 | 其余 |
|---|---|---|
| RT-2-PaLI-X | ~50% | 原始 vision-language 数据 |
| RT-2-PaLM-E | ~66% | 同上 |

一个 batch 里可能同时有 "What animal?"→"donkey"、caption、和 robot image→"1 128 91 ..."。

而且**不需要**两套 objective：

$$L_{\text{VQA}}=-\sum_t\log p(y_t|x,y_{<t}),\qquad L_{\text{robot}}=-\sum_t\log p(a_t|x,a_{<t})$$

从 Transformer 角度这就是同一个 token CE，只是不同样本的 target token 不一样。**不用写成 $L_{\text{text}}+\lambda L_{\text{action}}$。**

这正是 [co-training 防遗忘](01-vla-models.md#3-vlm-backbone-怎么处理) 的原始出处。

### 推理时为什么不会突然吐出 "banana"

因为做了 **output vocabulary constraint**：

| 任务 | 允许的输出 |
|---|---|
| 普通 VQA | 全词表 |
| robot control | **只允许合法 action token** |

思路和今天的 grammar-constrained / structured generation 一样。

### web knowledge 到底怎么传到 action 的

这个必须理解，否则会觉得"混点 VQA 数据有什么用"：

```text
Web data 学:    image        → semantic     （看到 "3" 知道这是数字 3）
Robot data 学:  semantic target → action    （定位目标 → 靠近 → 抓取）
                        ↓
              两者共享同一套 VLM 参数
                        ↓
    robot data 里从来没有 "move banana to the number 3"
    但模型可以组合出来
```

$$\boxed{\text{semantic generalization 来自参数共享，不是来自某个模块}}$$

### RT-2 已经试过 CoT

```text
User:   I need to hammer a nail. What object might be useful?
Plan:   Use the rock.
Action: 1 129 138 ...
```

同一个模型自回归地生成 reasoning token + action token。

⚠️ 所以 **G0.5 的 reasoning + action unified stream 不是凭空出现的**，RT-2 在 2023 年就有过早期原型。

### RT-1 → RT-2 的差别

| | RT-1 | RT-2 |
|---|---|---|
| backbone | 专用 robot Transformer (35M) | 预训练 VLM (PaLI-X / PaLM-E) |
| vision | EfficientNet + TokenLearner | VLM 的 vision encoder |
| language | USE 句向量 + FiLM 调制 | VLM 原生 language token |
| action | 256-bin **内部类别** | 256-bin **VLM 词表 token** |
| 解码 | 11 维**并行**分类 | 8 个 token **自回归** |
| web 预训练 | ❌ | ✅ |
| robot + web 联合训练 | ❌ | ✅ |
| 核心 | scale robot data | **transfer web knowledge → action** |

最关键的变化不是 $35M\to55B$，而是

$$\boxed{\text{robot policy}\ \longrightarrow\ \text{general VLM} + \text{robot action language}}$$

### RT-2 的四个先天问题

后面所有 VLA 基本都在修这四条：

1. **量化粗糙**：连续动作切 256 bin，精细控制有 quantization error
2. **每维一个 token**：8 维就 8 个 token，将来 27 维人形怎么办
3. **自回归解码慢**：还要预测 $H=50$ 的 chunk 的话就是几百个 token
4. **本体单一**：语义泛化解决了，**cross-embodiment 没解决**

## 3. RT-X：多本体数据能不能混

Open X-Embodiment（2023）是一次大规模社区协作：

| | 数量 |
|---|---|
| 真实机器人 trajectory | **1M+** |
| robot embodiment | **22 种** |
| 汇集的已有数据集 | 60 个 |
| 参与实验室 | 34 个 |

⚠️ **仓库 ≠ 训练配比。** 论文的 RT-X 实验实际只用了其中 **9 种 manipulator embodiment** 组成的 mixture，不是把 22 种全塞进去。读论文很容易看混。

### 不同数据集哪里不一样

| | 差异 |
|---|---|
| **相机** | wrist / third-person、分辨率、有没有 depth |
| **state** | 关节角 / 末端位姿 + 四元数 / 还带底盘 |
| **action** | EE delta / 关节目标 / 速度 / 双臂 …… 维度和语义都不同 |

所以不能 `torch.cat(all_robot_datasets)`。

### 四步粗对齐

**① 格式统一**：全部转成 **RLDS**（标准 episode 格式：step → observation / action / metadata / language）。不同实验室原来是 `.npy / .pkl / rosbag / hdf5 / tfrecord` 各写各的 pipeline。

⚠️ 但**统一文件格式和统一机器人语义是两回事**，难的是后者。

**② 观测粗统一**：每个数据集选一个 canonical camera，统一分辨率，只保留 `RGB + language`。论文自己叫 **coarsely aligned observation space**。

**③ action schema 粗统一**：各种 manipulator 动作转成共同的 7D 末端动作

$$a=[x,y,z,\text{roll},\text{pitch},\text{yaw},\text{gripper}]$$

某个机器人没有的自由度就**补 0**：

```text
Robot A:  [x y z roll pitch yaw gripper]
Robot B:  [x y z  0     0   yaw gripper]
```

**④ 每个数据集自己归一化**（见下）。

### RT-X 的 cross-embodiment 是粗对齐

这是这一节最该记住的、也最反直觉的一条。

你会自然以为：既然都是 $[x,y,z,r,p,y,g]$，那 token 128 在所有机器人上意味着同样的物理动作吧？**不是。**

论文明确说：**没有统一各数据集的末端坐标系，也允许不同数据集保持自己的 absolute / relative / position / velocity 控制方案。**

```text
Dataset A 的 +x = 机器人正前方
Dataset B 的 +x = 可能是右方
        ↓
同一个 [+0.1, 0, 0]
        ↓
A: 手向前          B: 手向右
```

更夸张的是连绝对 / 增量都没统一：$x=0.5$ 在 A 可能是"去世界坐标 0.5 m"，在 B 是"相对当前移动一点"，在 C 甚至是某个速度。

$$\boxed{\text{unified slot 只表示"这个数据集里和 EE x 方向控制有关的量"}}$$

$$\boxed{\text{coarse alignment}\ \neq\ \text{true universal action representation}}$$

### 那还能一起训？靠 per-dataset normalization

```text
Dataset A:  Δx ∈ [-0.02, 0.02] m
Dataset B:  Δx ∈ [-0.10, 0.10] m
```

直接混的话，A 的 0.02 是最大动作、B 的 0.02 是很小的动作，语义完全不一致。所以各自归一化，然后再切 bin：

```text
Robot A:  0.015 m  →  normalized 0.75  →  token 224
Robot B:  0.080 m  →  normalized 0.75  →  token 224
```

推理时再用对应本体的规则反归一化：

```text
token 224 → 0.75 → Robot A 的 de-normalizer → 0.015 m
                 → Robot B 的 de-normalizer → 0.080 m
```

所以模型学到的**不是** "224 = 移动 1.5 cm"，而是

$$\boxed{\text{224} = \text{对当前本体而言一个较大的正向动作}}$$

$$\boxed{\text{shared token}\ \neq\ \text{shared physical magnitude}}$$

### 模型怎么知道现在是哪台机器人

RT-X 没有 G0.5 那种显式的 embodiment token / 统一 action codec。**embodiment 信息很大程度隐含在图像里**：

```text
看到 Franka 长这样   → 进入 Franka 的 behavior manifold
看到 WidowX 长这样   → 进入 WidowX 的 behavior manifold
```

和 VLM"看到猫就自动调用猫的表示"是同一回事。

### 为什么居然有正迁移

Franka 和 WidowX 的关节结构不同，但执行 "pick up the coke can" 时高层是共享的：

```text
视觉：找到 coke can
空间：判断目标相对 gripper 的方向
行为结构：approach → descend → close → lift
```

不同的只是"具体电机怎么完成"。而末端表示已经消掉了一部分关节层面的差异。

结果：RT-1-X 在 small-data domain 的平均成功率比只训目标机器人的基线高约 **50%**；RT-2-X 在 emergent skill 评测上约 **3×** 于原 RT-2，能处理 "move apple near cloth / on cloth / between can and orange" 这类空间语义。

$$\boxed{\text{不同机器人的数据放在一起，确实会正迁移}}$$

⚠️ **RT-X 不是新架构。** 就是拿 RT-1 和 RT-2 用多本体 mixture 重训，得到 RT-1-X 和 RT-2-X。论文自己强调贡献是验证 **X-embodiment data scaling** 可行。

### 留下的问题

1. **不是真正统一的 action semantics**（坐标系、绝对/增量、物理量、尺度都没统一）
2. **装不下人形**：双臂就 14D，加下半身 >20D，$[x,y,z,r,p,y,g]$ 明显不够
3. **仍然是朴素离散化**：27 维 × chunk 50 步 = 1350 个值

第 3 条直接导向后来的 **FAST tokenizer** 和 G0.5 的 **ActionCodec**。

## 4. OpenVLA：第一代的集大成 + 开源

$$\boxed{\text{OpenVLA} = \text{Prismatic VLM} + \text{Open-X robot data} + \text{RT-2 式 action token}}$$

动作范式没有革命性变化，推进的是视觉能力、开源性、数据规模和跨机器人泛化。7B 模型，约 **970k** 条 Open-X trajectory。

```text
                    robot RGB (224×224)
                    ┌──────┴──────┐
                DINOv2         SigLIP
                    └──── concat ─┘        ← channel 维拼接
                          ↓
                      Projector
                          ↓
                   256 visual tokens
instruction → tokenizer ──┤
                          ↓
                    Llama-2 7B
                          ↓
                     7 action tokens
                          ↓
                  detokenize + de-normalize
                          ↓
                    7D 连续动作
```

### OpenVLA 的视觉侧为什么是 DINOv2 加 SigLIP

两个 encoder **并联**看同一张图（不是串联）：

| | 输出 shape | 学到什么 |
|---|---|---|
| DINOv2 ViT-L/14 | $256\times1024$ | 形状、边缘、局部位置、空间对应 |
| SigLIP | $256\times1152$ | 语义、"这是什么" |

$224/14=16$，$16\times16=256$ 个 patch。

⚠️ **融合是 channel 拼接，不是变成 512 个 token**：

$$X_i=[X_i^{\text{DINO}};X_i^{\text{SigLIP}}],\qquad 1024+1152=2176$$

```text
DINO:    d1 d2 d3 ... d256
SigLIP:  s1 s2 s3 ... s256
              ↓
        [d1;s1] [d2;s2] ... [d256;s256]     ← 序列长度还是 256
```

$$\boxed{X_{\text{vision}}\in\mathbb R^{256\times2176}}$$

序列长度没变，变的是 feature dimension。（同样的"加宽而不是加长"思路，在 [DeepStack](../../notes/07-vlm/06-deepstack-and-qwen-evolution.md#4--最关键的一点不是-concat是逐元素相加) 那里是逐元素相加。）

为什么要两个：指令 "pick up the red cup"，模型要先靠 SigLIP 回答"哪个是 red cup"，再靠 DINOv2 回答"它具体在哪、边缘在哪、gripper 相对它在哪"。

$$\boxed{\text{SigLIP：what}\qquad\text{DINOv2：where / geometry}}$$

DINOv2 为什么擅长 dense feature，见 [DINOv2 走查](../../notes/07-vlm/07-dinov2.md)。

### Projector 和序列拼接

Llama-2 7B 的 hidden 是 4096，所以 MLP projector 做 $2176\to4096$：

$$[256,2176]\ \longrightarrow\ [256,4096]$$

然后和 text embedding 拼成标准多模态序列：

$$X=[v_1,\dots,v_{256},e_1,\dots,e_N]\in\mathbb R^{(256+N)\times4096}$$

**到这里为止一点 robotics 特有的东西都没有，就是一个 VLM。** 把它变成 VLA 的是 ActionTokenizer。

### 归一化用 1%–99% 分位数

动作 7 维：$[\Delta x,\Delta y,\Delta z,\Delta r_x,\Delta r_y,\Delta r_z,g]$。

为什么不用 min/max：假设绝大多数 $\Delta x\in[-0.03,0.03]$，但有一个离群值 $0.5$。按 $[-0.5,0.5]$ 切 256 bin 的话，正常动作全挤在中间几个 bin 里，有效精度极差。所以用

$$\tilde a=2\cdot\frac{a-q_{01}}{q_{99}-q_{01}}-1\ \in[-1,1]$$

推理时反过来：

$$a=\tfrac12(\tilde a+1)(q_{99}-q_{01})+q_{01}$$

⚠️ gripper 往往本来就是 0/1，配置里有 normalization mask 决定这一维走不走同样的归一化。**别以为 7 维全都是连续量、全都一样处理。**

### 卡点：只有 256 个 action token，不是 7×256

不是每一维各有自己的 256 个 token（那就是 1792 个词表项）。**所有维度共用同一套 256 bin**：

```text
Δx → bin 170 → ACTION_170
Δy → bin 100 → ACTION_100
Δz → bin 170 → ACTION_170     ← 和 Δx 是同一个 token
```

那模型怎么知道 `ACTION_170` 是 x 还是 z？**靠位置。** 输出序列的顺序永远固定：

```text
第 1 个 → Δx    第 2 个 → Δy    ...    第 7 个 → gripper
```

和结构化输出一个道理，不需要 `<X_ACTION_170>` 这种带维度前缀的 token。

这 256 个 token 就是 Llama 词表里**最后 / 最少用的 256 项**被征用，和 RT-2-PaLM-E 的做法一样。**没有新建 action head。**

### 一个 training sample 长什么样

```text
Human:      What action should the robot take to pick up the red cup?
Assistant:  <A166><A115><A134><A128><A129><A120><A255>
```

加上 $I_t$ 那张图。在模型眼里就是一条标准的多模态 instruction-tuning 样本。

label mask：

```text
Human prompt / instruction   → -100（不算 loss）
action tokens                → 正常 label
```

$$L=-\sum_{i\in\text{action}}\log p_\theta(x_i\mid x_{<i},I)$$

### 图片 token 不算 loss，vision encoder 怎么学

和 [VLM 那边](../../notes/07-vlm/04-training-stages.md#7-为什么语言-ce-能更新-vit) 是同一个问题、同一个答案。梯度链：

```text
L_action → LM head → Llama → visual tokens → Projector → DINO + SigLIP
```

$$\boxed{\text{不用给 image token 单独定义 loss，vision encoder 照样通过 action CE 学习}}$$

### 全参微调，而且 vision encoder 必须解冻

```text
freeze_vision_backbone = False
freeze_llm_backbone    = False
```

DINOv2 / SigLIP / Projector / Llama / embedding **全部更新**。

⚠️ 这和很多 VLM 训练"冻 ViT 保泛化"的直觉相反。作者做了消融：**冻结 vision encoder 明显降低机器人性能**。

原因是需求不一样：

```text
普通 VQA 需要：  "这是杯子"
机器人需要：     gripper 离杯口还差 8mm，该往左还是往右，手腕角度对齐没有
```

$$\boxed{\text{robot control 对 spatial precision 的要求远高于 VQA}}$$

这也正是 DINOv2 在这里格外有价值的原因。

### 和 RT-2 一个重要区别：OpenVLA 没有混 web 数据

| | RT-2 | OpenVLA |
|---|---|---|
| 训练数据 | robot + web VL **co-finetune** | **只有 robot data** |
| 起点 | 预训练 VLM | Prismatic VLM（已做过 ~1M 图文 instruction tuning） |

论文自己指出：**这可能就是 OpenVLA 语义泛化不如 RT-2-X 的原因之一** —— RT-2-X 一直在混 Internet data，OpenVLA 的 robot training 阶段没有。

```text
DINO + SigLIP + Llama 预训练
        ↓
   Prismatic VLM（~1M LLaVA-1.5 mixture）
        ↓
   970k robot trajectories（只有机器人数据）
        ↓
      OpenVLA
```

### 数据：970k 是筛出来的，不是有多少用多少

Open-X 原始池子 >70 个数据集、>2M 条 trajectory。OpenVLA 筛掉不好统一的，只留：

```text
manipulation
至少一个 third-person RGB camera
single-arm
end-effector control
```

$$\boxed{\text{OpenVLA 的 cross-embodiment 前提是：先把大家裁剪到"长得足够像"}}$$

它**没有**解决任意 embodiment。这正是 G0.5 后来不接受的前提：人形、双臂、移动底盘不该被裁成单臂 7D。

配比也不按 trajectory 数量采样，借用了 Octo 的 heuristics：**低多样性的数据集 down-weight，高任务 / 场景多样性的 up-weight**。

$$\boxed{\text{data quality/diversity}\ \neq\ \text{trajectory count}}$$

**DROID 是个有意思的失败案例**：以 10% 权重加进来，但训练中它的 action-token accuracy 一直上不去，最后**训练的最后三分之一直接把它拿掉了**。说明更多 robot data 不一定更好，能不能 fit、action convention 是否一致都很关键。

### 一条 trajectory 拆成 T 个样本

$$\boxed{\text{970k trajectories}\ \neq\ \text{970k training samples}}$$

官方 dataloader 默认 `window_size=1`、`future_action_window_size=0`，也就是：

```text
Sample 0:  (I_0, l) → a_0
Sample 1:  (I_1, l) → a_1
Sample 2:  (I_2, l) → a_2
```

### 卡点：OpenVLA 是单帧、无本体状态、无 chunk

$$\boxed{\pi_\theta(a_t\mid I_t,l)}$$

而不是更现代的 $\pi_\theta(a_{t:t+H}\mid I_{\le t},s_t,l)$。三样都没有：

- 没有 proprioception $s_t$
- 没有历史 $I_{t-k:t}$
- 没有 action chunk $a_{t:t+H}$

这是一个很强的 **Markov 假设**：当前这张图已经包含足够信息告诉我下一步怎么动。

对 reach / pick / place / push 这类任务够用（"gripper 在杯子上方"→"往下"）。但如果任务需要"我刚才打开的是哪个抽屉""我已经放了几个物体""上一次抓取失败了吗"，单帧就不够了。这正是后来 VLA 要加 observation history / visual memory 的原因。

### 训练配置里两个反直觉的数字

| | 值 |
|---|---|
| epochs | **27** |
| lr | $2\times10^{-5}$ 恒定，**无 warmup** |
| global batch | 2048 |
| 算力 | 64×A100 约 14 天 |

LLM 预训练通常 1~2 个 epoch，为什么这里要 27？因为作者发现 action token accuracy 到 **95%+** 之后，真机性能还在涨。

机器人不是"大概说对意思就行"：预测 bin 130 和 bin 150 可能让机械臂往不同方向走。**控制对 imitation accuracy 的要求非常高。**

### 和 RT-2 的对照

| | RT-2 | OpenVLA |
|---|---|---|
| VLM | PaLI-X / PaLM-E | Prismatic |
| LLM | 最大几十 B | Llama-2 **7B** |
| vision | 原 VLM encoder | **DINOv2 + SigLIP** |
| robot data | Google 为主 | **970k Open-X** |
| web co-training | ✅ | ❌ |
| 开源 | ❌ | **模型 + 代码 + pipeline** |
| action | 256-bin token | 256-bin token |
| loss | token CE | token CE |

7B 的 OpenVLA 在 29 个任务上比 55B 的 RT-2-X 平均成功率高 **16.5 个百分点**，参数量小约 7 倍。

## 5. 第一代闭环，以及它留下的四个问题

$$\boxed{RGB\xrightarrow{\text{DINO+SigLIP}}256\times2176\xrightarrow{\text{Proj}}256\times4096\xrightarrow{\text{Llama2 7B}}7\text{ tokens}\xrightarrow{\text{detok}}7D\text{ action}}$$

四个问题，每一个都对应第二代的一项设计：

| 问题 | 谁来解决 |
|---|---|
| 1. 单帧，无 history | visual memory / 多帧输入 |
| 2. 无 proprioception | 状态投影成 token 拼进序列 |
| 3. 一次只出一个动作，无 chunk | [action chunking](../05-action/03-chunking-and-realtime.md)（ACT） |
| 4. 逐标量 256-bin + AR，精度和速度都受限 | 连续动作（diffusion / flow matching）或更好的 tokenizer（FAST / ActionCodec） |

第 4 条还有一层：逐标量离散化**完全没有利用 trajectory 级别的冗余**。一段"抓杯子"的 $A\in\mathbb R^{50\times7}$ 里全是结构（连续向前、连续向下、闭合、上抬），但 tokenizer 只是把 $0.010,0.012,0.011$ 一个一个编码。

### 通往第二代的两条路

```text
连续 action
     │
     ├──────────────┐
     ▼              ▼
先离散化        直接建模连续分布
     │              │
action token    regression / diffusion / flow matching
     │              │
RT-1/RT-2       Diffusion Policy
OpenVLA         π0 / π0.5 / G0
```

⚠️ 而 **G0.5 又绕回了第一条路**，但不是朴素 binning，而是学一个 ActionCodec（RVQ）把整段 $[H,D]$ 压成少量 code token。它的判断是：

$$\boxed{\text{第一代 AR VLA 的问题不是 AR 本身，是 action tokenizer 太低效}}$$

## 6. 面试一分钟版

> 第一代 VLA 的核心是**把连续动作离散化成 token，用 next-token prediction 训**。
>
> RT-1 迈出第一步：动作切 256 bin 变成分类问题，但它是专用 robot Transformer，语言只是 USE 句向量经 FiLM 调制视觉，而且**11 个维度是并行预测的，不是自回归**。
>
> RT-2 才真正立起 VLA：把动作 bin 写成文本字符串，机器人数据伪装成 VQA，**和 text 共用同一个 LM head**，于是 web knowledge 能通过参数共享迁移到动作上。为了不遗忘，它 co-finetune 机器人数据和网络图文数据。
>
> RT-X 验证多本体数据能正迁移，但它的 cross-embodiment 是**粗对齐**：只统一了 7D 末端 schema 和 per-dataset 归一化，坐标系和绝对/增量都没统一，所以同一个 action token 在不同机器人上是不同的物理位移。
>
> OpenVLA 是开源集大成：DINOv2 + SigLIP 在 channel 维拼成 $256\times2176$、projector 到 4096、Llama-2 7B 出 7 个 action token，全参微调（**vision encoder 必须解冻，冻了掉点**），970k 条筛过的 Open-X 数据训 27 个 epoch。
>
> 它留下四个问题：单帧无历史、无本体状态、无 action chunk、逐标量量化太粗 —— 这四条就是第二代 π0 那一代的全部动机。

## 自测

**1.** 为什么说 RT-1 严格来讲不算 VLA？

> **答：** 它是 language-conditioned robot policy：专用 robot Transformer，语言只是一个 USE 句向量经 FiLM 调制 CNN 特征，**不在 token stream 里**，也没有 web 规模的视觉语言预训练。"VLA"这个词是 RT-2 才正式立起来的。

**2.** FiLM 是什么？在 RT-1 里解决什么？

> **答：** Feature-wise Linear Modulation：$F'=\gamma(e_l)\odot F+\beta(e_l)$，用语言向量算缩放和偏移去调制 CNN 特征。
> 解决的是：同一张图里有杯子、可乐罐、抽屉、薯片袋，指令不同该关注的 feature 就不同。所以不是 $\text{image}\to\text{feature}$，而是 $(\text{image},\text{language})\to\text{task-conditioned feature}$。

**3.** TokenLearner 做什么？为什么需要它？

> **答：** 把每帧 81 个 CNN feature token 压成 8 个：$z_k=\sum_i\alpha_{ki}x_i$，学 8 组 attention 权重做 learned pooling（**不是随便抽 8 个位置**）。
> 因为 6 帧 × 81 = 486 个 token，而 2022 年要求约 3 Hz 实时，太贵。压完 $6\times8=48$。

**4.** ⭐ RT-1 的动作是怎么预测的？这和"第一代 VLA 逐 token 自回归"的说法冲突吗？

> **答：** RT-1 是**并行预测 11 个维度**，每维一个 256-way 分类，官方实现默认 `include_prev_timesteps_actions=False`。
> 所以那个说法对 RT-1 **不成立**。记：$\boxed{\text{discrete action}\neq\text{LLM-style autoregressive action}}$。RT-1 只做到"把连续控制变成离散分类"，把动作真正塞进 VLM 词表并自回归生成是 RT-2 干的。

**5.** ⭐ RT-2 最核心的一步是什么？为什么说它很漂亮？

> **答：** 把离散化后的动作写成**文本字符串** `"1 145 92 129 5 101 127 217"`，于是 robot demonstration 变成一条 VQA 样本，对 Transformer 来说和 "What animal?"→"donkey" 没有本质区别。
> 漂亮在于：**没有发明任何复杂的 robotics 架构，只改了 action representation**，让机器人控制变成 VLM 本来就会做的 sequence generation。同一个 LM head 既输出 "banana" 也输出 "128"。

**6.** RT-2 的两个 backbone 各自怎么获得 action token？

> **答：** **PaLI-X** 的 tokenizer 本来就给 $0..1000$ 每个整数独立 token，直接拿来当 bin；**PaLM-E** 没有，于是**找出词表里最不常用的 256 个 token 覆盖掉**，赋予 action bin 的语义。

**7.** ⭐ co-fine-tuning 解决什么？为什么不需要两套 loss？

> **答：** 解决**灾难性遗忘** —— 只用窄的机器人数据微调，模型会越来越只会输出动作数字，而 web knowledge 恰恰是 RT-2 想利用的东西。所以 batch 里混着 VQA / caption / robot（robot 占 ~50%（PaLI-X）或 ~66%（PaLM-E)）。
> 不需要 $L_{\text{text}}+\lambda L_{\text{action}}$，因为从 Transformer 角度**两者就是同一个 token CE**，只是不同样本的 target token 不同。

**8.** 推理时怎么保证不输出 "banana"？

> **答：** **output vocabulary constraint**：机器人任务下只允许从合法 action token 里选，普通 VQA 才用全词表。思路和今天的 grammar-constrained / structured generation 一样。

**9.** web knowledge 到底通过什么机制传到 action？

> **答：** **参数共享**。web data 学 `image → semantic`（看到 "3" 知道是数字 3），robot data 学 `semantic target → action`（定位 → 靠近 → 抓取），两者共享同一套 VLM 参数，于是可以组合出 robot data 里从没出现过的 "move banana to the number 3"。

**10.** RT-X 的四步粗对齐是什么？

> **答：** ① 格式统一到 **RLDS**；② 观测粗统一（每数据集选一个 canonical camera，统一到 RGB + language）；③ action schema 粗统一到 7D 末端 $[x,y,z,r,p,y,g]$，缺的自由度**补 0**；④ **每个数据集自己做归一化**。

**11.** ⭐ 为什么说 RT-X 的 cross-embodiment 只是"粗对齐"？

> **答：** 只是 **slot 名字一样**。论文明确说没有统一末端坐标系，也允许各数据集保持自己的 absolute / relative / position / velocity 方案。所以 A 的 $+x$ 是正前方、B 的可能是右方；$x=0.5$ 在 A 是绝对坐标、在 B 是增量、在 C 可能是速度。
> $\boxed{\text{coarse alignment}\neq\text{true universal action representation}}$

**12.** ⭐ 同一个 action token 在两台机器人上是同一个物理动作吗？

> **答：** **不是。** 因为每个数据集独立归一化：Robot A 的 0.015 m 和 Robot B 的 0.080 m 都可能归一化到 0.75、量化成 token 224，推理时再各自反归一化回去。
> 模型学到的是"**对当前本体而言一个较大的正向动作**"，不是"移动 1.5 cm"。记：$\boxed{\text{shared token}\neq\text{shared physical magnitude}}$。

**13.** 模型怎么知道当前是哪台机器人？

> **答：** RT-X 没有显式的 embodiment token，**信息隐含在图像里** —— 看到 Franka 的样子就进入 Franka 的 behavior manifold。和 VLM 看到猫就调用猫的表示同理。

**14.** 不同机器人为什么会有正迁移？

> **答：** 高层是共享的：找到目标、判断目标相对 gripper 的方向、approach → descend → close → lift 这个行为结构。不同的只是具体电机怎么完成，而末端表示已经消掉了一部分关节层面的差异。
> 结果：RT-1-X 在 small-data domain 平均成功率高约 **50%**；RT-2-X 在 emergent skill 上约 **3×**。

**15.** ⭐ OpenVLA 的两个 vision encoder 怎么融合？为什么不是 512 个 token？

> **答：** **在 channel 维拼接**：$X_i=[X_i^{\text{DINO}};X_i^{\text{SigLIP}}]$，$1024+1152=2176$，得到 $256\times2176$。序列长度还是 **256**，增加的是 feature dimension。
> 为什么要两个：SigLIP 回答"哪个是 red cup"（what），DINOv2 回答"它在哪、边缘在哪、gripper 离它多远"（where/geometry）。

**16.** OpenVLA 从 pixel 到动作的完整 shape 链？

> **答：** $I\in\mathbb R^{B\times3\times224\times224}$ → DINO $B\times256\times1024$ + SigLIP $B\times256\times1152$ → concat $B\times256\times2176$ → projector $B\times256\times4096$ → 拼 text $B\times(256+N)\times4096$ → Llama-2 7B → 7 个 action token → detokenize + 反归一化 → $a_t\in\mathbb R^7$。

**17.** 为什么用 1%–99% 分位数而不是 min/max 归一化？

> **答：** 有离群值时 min/max 会毁掉有效精度。绝大多数 $\Delta x\in[-0.03,0.03]$ 但有一个 $0.5$ 的话，按 $[-0.5,0.5]$ 切 256 bin 会让正常动作全挤在中间几个 bin 里。

**18.** ⭐ OpenVLA 有多少个 action token？模型怎么区分 x 和 z？

> **答：** **只有 256 个**，所有维度共用同一套 bin（不是 $7\times256=1792$）。$\Delta x$ 和 $\Delta z$ 完全可能是同一个 `ACTION_170`。
> 靠**位置**区分：输出序列顺序永远固定，第 1 个是 $\Delta x$、第 7 个是 gripper。和结构化输出一个道理。这 256 个 token 是征用 Llama 词表里最少用的 256 项，**没有新建 action head**。

**19.** loss 只算 action token，vision encoder 怎么拿到梯度？

> **答：** 梯度链 `L_action → LM head → Llama → visual tokens → Projector → DINO + SigLIP`。**不需要给 image token 单独定义 loss**。和 VLM 那边"语言 CE 凭什么能训 ViT"是同一个问题。

**20.** ⭐ OpenVLA 冻结 vision encoder 了吗？为什么？

> **答：** **没有，全参微调**（`freeze_vision_backbone=False`）。消融显示**冻结明显掉点**。
> 原因是需求不同：VQA 只要"这是杯子"，机器人要"gripper 离杯口还差 8mm、该往左还是右、手腕角度对齐没有"。$\boxed{\text{robot control 对 spatial precision 的要求远高于 VQA}}$。

**21.** OpenVLA 和 RT-2 在训练数据上的关键区别？后果是什么？

> **答：** RT-2 全程 **co-finetune robot + web VL 数据**；OpenVLA 从已经做过图文 instruction tuning 的 Prismatic 初始化后，**只用 robot data**。
> 后果：论文自己指出这可能就是 OpenVLA **语义泛化不如 RT-2-X** 的原因之一。

**22.** 970k 是怎么筛出来的？OpenVLA 的 cross-embodiment 有什么隐含前提？

> **答：** 从 >70 个数据集、>2M 条里筛出 manipulation + 至少一个 third-person RGB + **single-arm** + **end-effector control**。
> 隐含前提是：**先把大家裁剪到"长得足够像"的单臂 7D 问题，再混合训练**。它没有解决任意 embodiment —— 这正是 G0.5 不接受的前提。

**23.** 数据配比是按 trajectory 数量采的吗？DROID 发生了什么？

> **答：** 不是，用 Octo 的 heuristics：**低多样性 down-weight，高任务/场景多样性 up-weight**。$\boxed{\text{data quality/diversity}\neq\text{trajectory count}}$
> **DROID** 以 10% 权重加进来后 action-token accuracy 一直上不去，**训练的最后三分之一直接拿掉了**。说明更多 robot data 不一定更好。

**24.** OpenVLA 的 policy 形式是什么？缺了哪三样？

> **答：** $\pi_\theta(a_t\mid I_t,l)$ —— 一个很强的 **Markov 假设**。
> 缺：**proprioception $s_t$**、**历史 $I_{t-k:t}$**、**action chunk $a_{t:t+H}$**。
> 对 reach/pick/place/push 够用，但"我刚才开的是哪个抽屉""上次抓取失败了吗"这类就不行了。

**25.** 为什么 robot data 要训 27 个 epoch？

> **答：** 作者发现 action token accuracy 到 **95%+** 之后真机性能还在涨。机器人不是"大概说对意思就行"，预测 bin 130 和 bin 150 会让机械臂往不同方向走，控制对 imitation accuracy 要求极高。

**26.** ⭐ 第一代留下的四个问题，各自被谁解决？

> **答：** ① 单帧无历史 → visual memory / 多帧；② 无本体状态 → 状态投影成 token；③ 无 chunk → **action chunking**（ACT）；④ 逐标量 256-bin + AR 慢且粗 → 连续动作（diffusion / flow matching）或更好的 tokenizer（FAST / ActionCodec）。
> 第 4 条深一层：逐标量编码**完全没利用 trajectory 级冗余**，一段抓取的 $50\times7$ 里全是结构，却被当成 350 个独立数字编码。

**27.** G0.5 又回到 action token，是"倒退"吗？

> **答：** 不是。它的判断是 $\boxed{\text{第一代 AR VLA 的问题不是 AR 本身，是 action tokenizer 太低效}}$，所以保留 AR 的统一性（reasoning 和 action 同一个 stream、同一个 CE），用 learned ActionCodec（RVQ）把整段 $[H,D]$ 压成少量 code token 来解决 token efficiency。
