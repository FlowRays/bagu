# G0.5：从多模态观测到机器人动作

> 基于 2026-09-08 阅读的本地 `GalaxeaVLA` 仓库及 [G0.5 论文](https://arxiv.org/html/2608.11739v1)。
> 前置：[VLM 架构](../../notes/07-vlm/01-architecture.md)、[动作空间从零](../05-action/01b-action-space-from-zero.md)。

**G0.5 的核心：把一段机器人动作压缩成 token，让同一个视觉语言模型生成推理文字和动作 token，再把动作 token 解码成连续控制。**

阅读时先区分两个层次：论文强调统一的自回归（AR）动作生成；这个代码仓库还支持独立 Action Expert 的 Flow Matching（FM）路径。具体运行哪一条，由任务配置决定。不能只看到 `action_expert` 类，就把 G0.5 的主线理解成 FM。

## 1. 先跟着一次“拿起杯子”走完整条链

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 640" role="img" aria-label="G0.5 自回归架构：图像、语言和状态组成输入，经 Qwen3.5 生成可选推理及动作码，由 ActionCodec 还原动作并闭环执行" style="width:100%;max-width:840px;height:auto">
<defs><marker id="g05-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#73726c"/></marker></defs>
<g fill="none" stroke="#73726c" stroke-width="2" marker-end="url(#g05-arrow)"><path d="M145 100 V145 H420 V180"/><path d="M420 100 V180"/><path d="M695 100 V145 H420"/><path d="M420 235 V265"/><path d="M420 325 V355"/><path d="M420 415 V445"/><path d="M420 505 V535"/><path d="M190 565 H30 V60 H40"/></g>
<g stroke="#d0c9bb" fill="#f0eee6"><rect x="40" y="20" width="210" height="80" rx="12"/><rect x="315" y="20" width="210" height="80" rx="12"/><rect x="590" y="20" width="210" height="80" rx="12"/><rect x="190" y="180" width="460" height="55" rx="12"/><rect x="190" y="445" width="460" height="60" rx="12"/><rect x="190" y="535" width="460" height="60" rx="12"/></g>
<g stroke="#c2410c" fill="#fff0e5"><rect x="190" y="265" width="460" height="60" rx="12"/><rect x="190" y="355" width="460" height="60" rx="12"/></g>
<g fill="#29261b" font-family="sans-serif" font-size="18" text-anchor="middle"><text x="145" y="51">多视角图像 / 历史帧</text><text x="145" y="80">视觉编码器</text><text x="420" y="51">任务指令 / 本体标识</text><text x="420" y="80">文本 Embedding</text><text x="695" y="51">机器人状态</text><text x="695" y="80">状态 MLP</text><text x="420" y="215">统一输入序列：B × S × 2048</text><text x="420" y="302">Qwen3.5：24 层自回归主干</text><text x="420" y="392">可选推理 token → 动作 token</text><text x="420" y="482">ActionCodec：动作码 → 连续动作块</text><text x="420" y="572">执行动作，获得新观测</text></g>
<text x="45" y="625" fill="#73726c" font-family="sans-serif" font-size="16">闭环：执行后重新观察；预测长度与实际执行步数分别由配置和部署端决定。</text>
</svg>

把“拿起桌上的杯子”交给机器人时，模型需要同时知道：杯子在哪里、要求做什么、自己是什么机器人、手臂目前在哪里。最后输出的是未来一段时间的控制序列。

## 2. 图像、语言和状态怎样变成同一种输入？

| 输入 | 例子 | 进入主干前的处理 |
|---|---|---|
| 多视角 RGB | 外部相机、腕部相机画面 | 视觉编码器 → 视觉 embedding |
| 任务指令 | “拿起桌上的杯子” | 文本 tokenizer → embedding |
| 本体标识 | 当前机器人类型 | 序列中的条件信息 |
| 本体状态（proprioception） | 关节、末端、夹爪状态 | MLP → 状态 embedding |

当前 `configs/model/g05.yaml` 使用宽度为 2048 的语言主干。视觉特征和状态特征也转换到这个宽度，组成：

$$X\in\mathbb{R}^{B\times S\times2048}$$

$B$ 是 batch size，$S$ 是组合后的序列长度。这里“统一”指向量宽度相同，可以放进同一个序列；并不表示三种输入使用同一个编码器。

代码在 `G05Model._forward_embed()` 中先获取文本 embedding，再将图像占位符替换成视觉特征，将状态占位符替换成 MLP 输出。图像并没有先被翻译成一段文字。

状态 MLP 的结构是：

```text
proprio [B, T, D]
    → Linear → GELU → LayerNorm → Linear
    → state embedding [B, T, 2048]
```

对应源码：`src/g05/models/g05/g05_model.py` 的 `_forward_embed()`，以及 `helpers/proprio_helper.py` 的 `ProprioEmbedder`。

## 3. Qwen3.5 主干到底做什么？

本地基础配置的语言主干共 24 层，隐藏宽度 2048，按下面模式重复六次：

```text
[GatedDeltaNet 线性注意力 × 3 → 完整注意力 × 1] × 6
```

线性注意力通过递推状态整合序列信息；完整注意力通过 query 与此前位置的 key/value 做注意力计算。这里不是 24 层都保存普通 Transformer 那样的完整 KV 序列，代码使用 `SparseKVCache` 处理混合结构。

每个位置最终得到一个 hidden vector，经 LM head 投影到词表：

$$h_i\in\mathbb{R}^{2048}\quad\longrightarrow\quad
\mathrm{logits}_i\in\mathbb{R}^{V}$$

词表中既有文字，也有动作相关 token。所以主干仍然在做 next-token prediction，只是“下一个 token”可能已经进入动作部分。

对应源码：`src/g05/models/g05/qwen35/mixture_qwen35.py`。类名中的 `Mixture` 是仓库对这类模型模块的命名，不能仅凭这个名字就理解成带专家路由的 MoE。

## 4. ActionCodec：连续动作为什么能变成 token？

设一个连续动作块为：

$$A\in\mathbb{R}^{H\times D}$$

$H$ 是预测的动作步数，$D$ 是每步动作维数。逐时间步、逐标量生成 token 会产生很长的序列；ActionCodec 学习把动作块压缩成一组离散码。

```text
构造 VLA 训练目标：
真实动作块 → 按运动部件分组 → Codec Encoder → RVQ → 动作 token

机器人推理：
VLM 生成动作 token → RVQ 码本还原 latent → Codec Decoder → 连续动作块
```

**RVQ 是残差向量量化。** 第一轮选择一个码本向量表达主要信息，下一轮编码剩余误差，后续轮继续补充细节。一个动作码因此是压缩运动表示的索引，不必对应某个关节在某一时刻的值。

当前实现的 `encode()` 接收 `{部件名: [B,T,D_i]}`，把各组补齐到共同部件宽度，批量编码并量化，再按组返回 code。启用配置中的 block DCT 时，编码前还会做时间块 DCT，解码后做逆变换。

### 为什么要分组？

对齐预训练布局的 `libero_ar.yaml` 使用 27 维动作空间：

| 语义组 | 维度 |
|---|---|
| left_control | 9 |
| left_gripper | 1 |
| right_control | 9 |
| right_gripper | 1 |
| lower_body | 7 |

这些是共享表示的槽位，不表示每种机器人都有 27 个实际关节。不同机器人把动作映射到相应语义组，解码后再转换为自身控制接口使用的形式。

动作序列可以只包含活跃组。只有右臂执行任务时，可以省略不活动组的动作码，减少生成长度。夹爪在当前 tokenizer 配置中使用规则式二值处理，不能把所有组都理解成完全相同的 RVQ 编码过程。

**这里有两个不同的“decoder”：**

- Qwen3.5 decoder 根据观测、指令和上下文决定输出哪些 token。
- ActionCodec decoder 把生成的动作码还原成连续运动。

对应源码：`src/g05/tokenizer/models/actioncodec2_v2/modeling_actioncodec2v2.py` 的 `encode()` / `decode()`；参数见 `configs/tokenizer/actioncodec.yaml`，实际残差轮数等还受任务配置覆盖。

## 5. CoT 为什么能直接参与动作生成？

下面是示意序列，省略了完整 chat 模板和特殊 token：

```text
条件： [图像] [本体标识] 拿起杯子 [机器人状态]
生成： Subtask: 接近杯子
       BBox: ...
       Action: [右臂分组标记] [动作码] [动作码] ...
```

动作 token 排在推理 token 后面，自回归模型预测动作时就能使用前面生成的推理。两类输出共用主干、词表与 next-token 训练目标。

推理字段可以包含 `Subtask`（子任务）、`BBox`（目标框）、`Trace`（轨迹）和 `ActionHint`（动作提示）。这不要求每次把全部字段生成一遍：模板决定监督内容，推理配置决定是否启用。

本地基础配置的 `predict_cot: false`。因此，“模型支持 CoT”和“这次运行生成 CoT”是两件事。模板构造见 `src/g05/data_processor/processor/samples_builder.py`。[README 模型概览](https://github.com/OpenGalaxea/GalaxeaVLA/blob/main/README_zh.md)

## 6. 视觉记忆放在哪里？

视觉记忆在视觉编码器里融合历史帧。如果杯子暂时被手臂挡住，历史观测还能提供之前的位置和运动信息。

```text
历史帧 + 当前帧
    → 视觉编码器中的空间 / 时间信息交换
    → 丢弃历史帧 token，保留融合过历史的当前帧表示
    → 语言主干
```

它让语言主干获得历史信息，同时控制视觉序列长度。论文采用分解的空间、时间注意力，并在训练时随机丢弃历史帧。[论文 §3.3](https://arxiv.org/html/2608.11739v1#S3.SS3)

源码在 `src/g05/models/g05/qwen35/vision.py`。基础 `g05.yaml` 的 `temporal_freq: 0`，没有启用这条时间注意力路径；实际是否使用记忆，需要查看最终配置与输入帧数。

## 7. 训练和推理分别在做什么？

### AR 训练：用正确的历史预测下一个 token

真实动作通过 ActionCodec 编码为目标动作 token，与可选推理内容一起组成监督序列。训练使用 teacher forcing：预测第 $i$ 个目标时，前面提供正确 token。

$$\mathcal{L}_{\mathrm{AR}}=-\sum_{i\in\mathcal{T}}
\log p_\theta(y_i\mid c,y_{<i})$$

$c$ 是观测、指令与状态；$\mathcal{T}$ 是参与监督的目标位置。代码通过 `labels == -100` 排除无需计算损失的位置。推理文字和动作码都可以使用这个交叉熵目标。

对应 `src/g05/models/g05/helpers/ar_helper.py` 的 `cal_ce_loss()`。这是 VLA 的训练目标；ActionCodec 自己学习压缩与重建动作的训练是另外一件事。

### AR 推理：使用自己刚生成的 token 继续生成

1. `prefill()`：编码观测和指令，建立主干缓存。
2. `generate_text()`：按配置生成可选推理，更新上下文。
3. `generate_action()`：继续生成动作码，调用 Codec 解码。
4. 部署端执行动作，获得新观测，再次规划。

这些方法位于 `src/g05/models/g05/g05_policy.py`。分成三个 API 是推理流程的组织方式，不代表有三个独立的 AR 网络。

预测长度 $H$ 不等于必须连续执行 $H$ 步后才能观察。执行多少步、是否异步、如何处理重叠动作块，由部署端决定。

## 8. 仓库里的 Action Expert 又是什么？

它用于另一条 FM 路径。它和 ActionCodec decoder 是不同模块。

```text
观测 / 指令 / 状态 → VLM → 条件缓存 ──────┐
                                        ↓
随机动作噪声 + 时间 t → Action Expert → 速度场 → 积分 → 连续动作块
```

基础配置中的 Action Expert 为 24 层、隐藏宽度 1024，使用 AdaLN 注入时间条件，默认推理积分 10 步。FM helper 管理时间采样、损失和积分循环。

以配置中的 `pi_convention` 为例，真实动作是 $A$，噪声是 $\epsilon$：

$$x_t=(1-t)A+t\epsilon,\qquad v^*=\epsilon-A$$

训练让网络预测这个速度场；推理从 $t=1$ 的噪声出发，向 $t=0$ 积分得到动作。实现见 `src/g05/models/g05/helpers/fm_helper.py`。

### 用具体配置判断当前运行路径

| 本地配置 | discrete_action | continuous_action | 主要动作目标 |
|---|---|---|---|
| `configs/task/libero_ar.yaml` | true | false | AR 交叉熵，FM 权重为 0 |
| `configs/task/libero.yaml` | false | true | FM，CE 权重为 0 |
| `configs/model/g05.yaml` 基础配置 | true | true | 两条路径均开启，可被任务配置覆盖 |

这里的 `continuous_action` 是“启用 FM 分支”的配置名。**AR 最终也会解码出连续动作**，不能按这个字段的字面意思认为 AR 无法输出浮点控制。

`fm.joint_training` 则控制 FM 损失是否通过条件缓存回传给 VLM。基础配置为 false，缓存会 detach；`libero.yaml` 覆盖为 true。它不表示整个模型是否处于 train 模式。

## 9. 建议的源码阅读顺序

以下路径相对于本地 `GalaxeaVLA/`。网页不使用服务器绝对文件路径作为链接；在终端或编辑器里搜索方法名即可。

| 顺序 | 文件 / 方法 | 这一步要回答的问题 |
|---|---|---|
| 1 | `configs/task/libero_ar.yaml`，再对照 `configs/model/g05.yaml` | 当前到底走 AR 还是 FM？输入和动作维度是什么？ |
| 2 | `src/g05/models/g05/g05_model.py`：`_forward_embed()` | 图像、文字、状态怎么排进一个序列？ |
| 3 | `src/g05/models/g05/qwen35/mixture_qwen35.py`：`MixtureQwen35` | embedding、混合注意力层、LM head 如何连接？ |
| 4 | `src/g05/models/g05/g05_policy.py`：`generate_action()` | AR 输出如何进入动作解码？FM 分支在哪里？ |
| 5 | `src/g05/tokenizer/models/actioncodec2_v2/modeling_actioncodec2v2.py`：`decode()` | code 如何还原成动作张量？ |
| 6 | `src/g05/models/g05/helpers/ar_helper.py`：`cal_ce_loss()` | 哪些 token 被监督？ |

仓库中的 `docs/architecture/g05_architecture_zh.md` 有不少历史通用框架描述，例如 18 层、PaliGemma / SigLIP 示例。阅读当前 Qwen3.5 版本时，以当前任务配置、Qwen3.5 子类和实际调用路径为准。

继续学习可对照：[动作 head](../05-action/02-action-head.md)、[Flow Matching](../02-visual-generation/03-flow-matching.md)、[chunk 与实时控制](../05-action/03-chunking-and-realtime.md)。
