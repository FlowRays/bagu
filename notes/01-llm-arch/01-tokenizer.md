# Tokenizer 与词表

> 高频考点：BPE / BBPE / WordPiece / Unigram 的区别，encode 为什么比 decode 难，词表大小影响什么。

## 1. 四种子词算法

都是为了在「字符级太长」和「词级会 OOV」之间取平衡。

| 算法 | 方向 | 合并/删除依据 | 用在哪 |
|---|---|---|---|
| **BPE** | 自底向上合并 | 相邻 pair 的**频次**最高 | GPT-2 早期 |
| **BBPE** | 自底向上合并 | 同 BPE，但基本单元是 **byte** | 现在几乎全用这个 |
| **WordPiece** | 自底向上合并 | **互信息式打分** $\frac{f(ab)}{f(a)\,f(b)}$ | BERT |
| **Unigram** | 自顶向下删除 | 从大词表开始，删掉**删了之后似然损失最小**的 token | SentencePiece / T5 |

### BPE

1. 先按字符切分
2. 统计所有相邻 pair 的频次
3. 把最高频的 pair 合并成一个新 token，加进词表
4. 重复到词表达到目标大小

产出两样东西：**词表** 和 **merge rules（有序的合并规则表）**。

### BBPE：为什么现在都用它

BPE 按 Unicode 字符切分，字符集本身就有十几万个，而且遇到没见过的字符就 OOV。

BBPE 改成按 **UTF-8 的 256 个 byte** 切分：

$$\boxed{\text{任何字符串都是 byte 序列}\ \Rightarrow\ \text{永远不会 OOV}}$$

代价是一个中文字符通常占 3 个 byte，起点更碎，需要更多 merge 才能合成有意义的单元。

### WordPiece

合并依据不是纯频次，而是类似互信息的打分：

$$\text{score}(a,b)=\frac{f(ab)}{f(a)\cdot f(b)}$$

含义是「这两个片段**捆绑出现**的倾向有多强」，而不是「一起出现得多不多」。高频但各自也高频的组合（比如 `the` + `s`）分数不会很高。

### Unigram

反着来：先建一个很大的候选词表，然后基于 unigram 语言模型算每个 token 的贡献，**迭代删掉价值最低的**，直到目标大小。

好处是它对同一个字符串保留了多种切分可能，可以做 subword regularization（训练时随机采样不同切分，当数据增强）。

> **SentencePiece 是一个库，不是一个算法。** 它能跑 BPE 也能跑 Unigram。它的特点是把空格当成普通字符（`▁`）来处理，因此不依赖预分词，天然支持中文日文这种没有空格的语言。

## 2. encode 比 decode 难

**decode 很简单**：token id → 查表 → 拼起来 → byte 序列解成 UTF-8。纯查表。

**encode 有优先级问题**：同一个字符串可能有多种切分方式，必须定义用哪一种。

BBPE 的做法是**严格按照训练时产出的 merge rules 顺序**依次尝试合并：

```text
输入   l o w e s t
按 merge 表顺序:
  第 12 条 (e,s)  -> l o w es t
  第 30 条 (es,t) -> l o w est
  第 47 条 (l,o)  -> lo w est
  ...
```

$$\boxed{\text{merge rules 的顺序就是编码的优先级，不能乱}}$$

Unigram 则不同，它用 Viterbi 找**概率最大的切分**。

## 3. Special tokens 与 chat template

特殊 token 是词表里被预留的、不由 merge 产生的 id：

```text
<|endoftext|>  <|im_start|>  <|im_end|>  <pad>  <unk>  <s>  </s>
```

**chat template** 是把结构化对话渲染成一个字符串的规则，通常用 **Jinja2** 写在 `tokenizer_config.json` 里：

```jinja
{% for m in messages %}<|im_start|>{{ m.role }}
{{ m.content }}<|im_end|>
{% endfor %}<|im_start|>assistant
```

渲染出来是：

```text
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
```

**面试常问**：训练和推理的 template 必须完全一致，差一个换行都会掉点，因为模型见到的 token 序列变了。SFT 时 loss mask 也是按 template 的边界来划的（只在 assistant 段算 loss，见 [SFT 与 KD](../06-post-training/distill/02-sft-and-kd.md)）。

## 4. 词表大小影响什么

$V$ 直接出现在三个地方：

| 位置 | 规模 | 说明 |
|---|---|---|
| Embedding | $V\times d$ | 参数量 |
| LM head | $V\times d$ | 参数量（不 tie 的话是另一份） |
| **logits** | $B\times L\times V$ | **激活值，训练时的大头** |

$V=150\text{K}$、$d=4096$ 时 embedding 就是 0.6B 参数。logits 更吓人：$B=4,L=8192$、BF16 下约 9.8 GB（见 [显存账本](../03-training-fundamentals/01-memory-accounting.md#7-容易忽略的大头logits)）。

但从**占比**看，大 MoE 里 embedding + head 通常只占总参数的 0.1%–0.4%（见 [参数构成](../04-distributed-infra/03-model-param-breakdown.md)）。

### 词表变大的取舍

| 变大的好处 | 变大的代价 |
|---|---|
| **压缩率**提高：同样文本用更少 token，等于变相扩了上下文、降了推理成本 | embedding / head 参数变多 |
| **多语言**覆盖更好：中文、代码不会被切得太碎 | logits 显存和 softmax 计算变大 |
| | 低频 token 训练不充分 |

**压缩率**（每个 token 平均多少字节）是评估 tokenizer 的核心指标。中文在只为英文优化的词表下可能一个字要 2–3 个 token，换成中文友好的词表可以接近 1 token 1 字。

## 自测

1. BPE / BBPE / WordPiece / Unigram 各自的合并或删除依据是什么？方向有什么不同？
2. BBPE 为什么永远不会 OOV？代价是什么？
3. 为什么说 decode 简单而 encode 有优先级问题？BBPE 靠什么定优先级？
4. SentencePiece 是算法还是库？它怎么处理空格，为什么这样对中文友好？
5. chat template 用什么写？训练和推理不一致会怎样？
6. 词表大小影响哪三处？哪一处是训练显存的大头？
7. 词表变大的好处和代价各是什么？压缩率指的是什么？
