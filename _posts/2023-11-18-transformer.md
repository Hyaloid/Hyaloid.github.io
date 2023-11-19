---
layout: post
title: Transformer 模型结构不完全详解
subtitle: Transformer 究竟是什么？
tags: [llm, Transformer]
comments: true
thumbnail-img: /assets/img/20231118/transformer-architecture.png
author: SeaMount
---

2017 年 Google Brain 在 NeurIPS 上面发表了论文 Attention Is All You Need，文章中提出了一种新的 seq2seq 模型 —— Transformer。Transformer 舍弃了先前的 RNN/CNN 结构，采用 encoder-decoder 的结构，只使用注意力模型来进行序列建模，解决了之前 RNN 固有顺序不能并行化的缺点，Transformer 对于长序列语义的捕获能力也比先前的 RNN 结构更强。

## Attention

RNN 结构能够很好地处理输入为一个 vector 的任务，但 RNN 却不能很好地处理输入为多个 vector 的任务，然而，注意力机制可以很好处理输入为多个 vector 的任务，并且可以并行计算这些输入的注意力。

注意力，顾名思义，就是一种能让模型对重要信息重点关注并充分学习和吸收的技术，即把注意力集中放在重要的点上，而忽略其他不重要的因素。注意力机制是人工神经网络中一种模仿认知注意力的技术。这种机制可以增强神经网络输入数据中某些部分的权重，同时减弱其他部分的权重，以此将网络的关注点聚焦于数据中最小的一部分。数据中哪些部分比其他部分更重要取决于上下文。可以通过梯度下降法对注意力机制进行训练。

假设我们有一个以索引 $$i$$ 排列的 token 序列。对于每一个 token $$i$$，神经网络计算出一个相应的满足 $$\sum_i w_i=1$$ 的非负软权重 $$w_i$$。每个标记都对应一个由词嵌入得到的向量 $$v_i$$。加权平均 $$\sum_i w_i v_i$$ 即是注意力机制的输出结果。

注意力机制有许多变体：点注意力（dot-product attention）、QKV 注意力（query-key-value attention）、自注意力（self-attention）、交叉注意力（cross attention）等。Transformer 是第一个完全依靠自注意力来计算输入和输出表示的转换模型，避免了循环和重复。

### self-attention

RNN/LSTM 在计算时需要考虑前序信息，所以不能并行，导致训练时间较长。对于远距离的相互依赖的特征，要经过若干时间步步骤的信息累积才能将两者联系起来，而距离越远，有效捕获的可能性越小。self-attention 可以很好地处理 RNN/LSTM 中存在的问题。

self-attention 通过位置编码保证序列关系，计算上不依赖序列关系，所以可以实现完全的并行；在计算相关性时，任何一个点都会与整个序列中的所有输入做相关性计算，避免了长距离依赖的问题。

self-attention 简而言之就是一个句子内的单词，互相看其他单词对自己的影响有多大。

{: .box-note}
**Attention 和 self-attention 的区别：** \
以 encoder-decoder 框架为例，输入 source 和输出 target 内容是不一样的。比如对于英-中机器翻译来说，source 是英文句子，target 是对应的翻译出来的中文句子。Attention 发生在 target 的元素 query 和 source 中的所有元素之间。\
Self-attention，指的不是 target 和 source 之间的 attention 机制，而是 source 内部元素之间或者 target 内部元素之间发生的 attention 机制，也可以理解为 target = source 这种特殊情况下的 attention。\
两者具体计算过程是一样的，只是计算对象发生了变化而已。

[On the Relationship between Self-Attention and Convolutional Layers](https://arxiv.org/abs/1911.03584) 中证明了 CNN 是最简单的 self-attention。当数据量小时，CNN 的效果比 self-attention 更好。

#### self-attention 的计算过程

![self-attention](/assets/img/20231118/self-attention.png){: .mx-auto.d-block :}

1. 初始化 Q，K，V

    {: .box-note}
    Q(Query): 查询是你想要了解的信息或者从你想要的文本中提取的特征。它类似于你对文本中的某个词语提出的问题或者你想要了解的内容； \
    K(Key): 键是文本中每个词语的表示。它类似于每个词语的标识符或者关键信息，用于帮助计算查询与其他词语之间的关联程度；\
    V(Value): 值是与每个词语相关的具体信息或者特征。它类似于每个词语的具体含义或者特征向量。\
    别看 Q、K、V 定义上花里胡哨的，每个字母意思都说得有模有样，但计算 Q、K、V 本质上还是在计算向量之间的相似度。

    以上图为例，有四个输入向量，其中，I 是整个输入序列 $$I=\begin{matrix}[a^1, \dots , a^4]
    \end{matrix}$$，$$q_i=W^qa^i$$，$$k_i=W^ka^i$$，$$v_i=W^va^i$$。

    $$Q=W^qI$$

    $$K=W^kI$$

    $$V=W^vI$$

    其中 $$W^q$$，$$W^k$$，$$W^v$$ 均为需要学习的参数矩阵。

2. 计算 self-attention score

    ![attention score](/assets/img/20231118/attention-score.png){: .mx-auto.d-block :}

    $$\alpha_{i,j}=(k^i)^Tq^j$$

    Attention 矩阵:

    $$A=
    \begin{bmatrix}
    \alpha_{1, 1}&\alpha_{2, 1}&\alpha_{3, 1} \\
    \alpha_{1, 2}&\alpha_{2, 2}&\alpha_{3, 2} \\
    \alpha_{1, 3}&\alpha_{2, 3}&\alpha_{3, 3} \\
    \alpha_{1, 4}&\alpha_{2, 4}&\alpha_{3, 4} \\
    \end{bmatrix}
    $$

3. 对 self-attention score 进行缩放和归一化，得到 softmax score

    ![soft self-attention](/assets/img/20231118/soft-self-attention.png){: .mx-auto.d-block :}

    $$\alpha^{'}_{i, j}=exp(\alpha_{i, j})/\sum_j exp(\alpha_{i, j})$$

    $$A'=softmax(A)$$

4. softmax score 乘以 value 向量，求和得到 attention value

    ![self-attention-compute-process](/assets/img/20231118/self-attention-process.png){: .mx-auto.d-block :}

    $$b^i=\sum_j \alpha^{'}_{i, j}v^j$$

    attention score 最大的向量支配最后的结果 b，即为最重要的信息。

    output 矩阵：
    $$ 
    O = VA' 
    $$
    ，其中$$V=\begin{matrix}[v^1, \dots , v^4]
    \end{matrix}$$。Self-attention 可以同时计算不同的输入向量，实现并行计算，提高计算速度。

### dot-product attention

![dot product attention](/assets/img/20231118/scaled-dot-product-attention.png){: .mx-auto.d-block :}

Transformer 中的 attention 采用了 scaled dot-product attention，Transformer 中 attention 的计算公式如下：

$$
Attention(Q,K,V) = softmax(\frac {QK^T}{\sqrt{d_k}})V
$$

$$\sqrt{d_k}$$
是一个的超参数，避免内积的结果过大。softmax 是为了归一化，没什么特殊的道理，可以换成其它的激活函数，有人尝试将 softmax 换成 relu 之后，发现 relu 的结果甚至优于 softmax 的结果。

dot-product attention 在计算过程中和 self-attention 的差别在于 scale 放缩参数。Transformer 中采用 $$\sqrt{d_k}$$ 作为 scale 参数，避免内积结果过大。Mask 为可选项。其余计算步骤和 self-attention 一样。

### multi-head attention

为了提取更多的交互信息，使用多头注意力（Multi-Head Self-Attention），即在多个不同的投影空间中捕获不同的交互信息。类比卷积神经网络理解，其中一个卷积核通常用于捕获某一类 pattern 的信息，故采用多个卷积核。多头注意力机制采用多个 head，便可以捕捉到不同的相关性。在实践中，首先通过 $$m$$ 个 head 生成 $$m$$ 个不同的输出 $$B_1,B_2,\dots,B_M$$，将其合并后再通过一层全连接层进行线性变换。

![multi-head-attention2](/assets/img/20231118/multi-head-attention2.png){: .mx-auto.d-block :}

假设使用 $$m$$ 个 head，矩阵运算如下：

$$A_m = \frac{K^T_m Q_M}{\sqrt{D_k}}$$

$$B_m=V_msoftmax(A_m)$$

$$B=W^o
\begin{bmatrix}
B_1 \\
\dots \\
B_M \\
\end{bmatrix}
$$

![multi-head attention](/assets/img/20231118/multi-head-attention.png){: .mx-auto.d-block :}

上图中 $$h$$ 为注意力头的个数，采用多个 head, 每个 head 有独立的 Q、K、V 的权值矩阵，不同的 head 为 Attention 层提供了不同的表示子空间，不同的表示关注不同位置。

Transformer 中 Multi-Head attention 的计算公式如下:

$$
head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
$$

$$
Multi-Head(Q,K,V)=Concat(head_1,\dots,head_h)W^O
$$

一个简单的 Multi-Head Attention 实现如下：
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v) # q==k==v==x(input)

        # split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        return out
```

Linear 层做的事情主要是将 Q、K、V 分成 h 个 head。


实现多头注意力时，有两种常用的形式：
1. Narrow Self-attention：把输入词向量切割成 h 块，每一块使用一次自注意力运算；这种方法速度快，节省内存，但是效果不好；
2. Wide Self-attention：对输入词向量独立地使用 h 次自注意力运算；这种方法效果好，但会花费更多地时间和内存。

### cross attention

交叉注意力是一种再 NLP 任务的架构中使用的机制，其思想是使一个序列能够关注另一个序列。Transformer 中的交叉注意力机制与 self-attention 的计算方式非常相似，只不过在 cross attention 中 $$K$$、$$V$$ 都来自 encoder，$$Q$$ 来自 decoder，而在 self-attention 中 $$Q = K = V$$。

## Encoder

![pos-layernorm](/assets/img/20231118/pos-layernorm.png){: .mx-auto.d-block :}
<center style="font-size:14px;color:#C0C0C0;">
(a) Post-LN Transformer layer; (b) Pre-LN Transformer layer
</center>

包含 Multi-Head Self-Attention 和 Feed Forward Network。在 Attention Is All You Need 中，encoder 按照 input、Multi-Head Attention、addtion、Layer Norm、FFN、addition、Layer Norm、output 的顺序组成。但是这种原始的结构并不是最好的设计，[On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) 证明了将 Layer Norm 前置效果更好。

## Decoder

包含了两个 Multi-Head Attention，第一个 Multi-Head Attention 采用了 Masked 操作，第二个 Multi-Head Attention 的 K、V 矩阵使用 encoder 的编码信息矩阵 C 进行计算，而 Q 使用上一个 decoder block 的输出计算，之后有一个 Feed Forward Network，专注于处理某个具体位置的信息，最后 softmax 计算下一个输出的概率。

包含 2 个 Multi-Head Attention、Multi-Head Cross Attention 和 Feed Forward Network。decoder 是一个自回归的结构。decoder 在训练模式和推理模式下的输入是不同的，在训练模式下使用 teacher forcing，decoder 输入是 ground truth，不管输出是什么，都会将正确答案作为输入；在测试模式下，将之前的输出作为新一轮的输入，可对照上 transformer decoder input 为 shifted right 的结构。所以在训练的时候 decoder 也是可以并行训练输入序列的，将得到的矩阵整体进行一次 mask 即可；而在推理的时候不能并行处理输入，需要将后面的序列 mask 起来，多次计算得到结果。

相较于 encoder，decoder 中的 attention 矩阵是一个满秩矩阵，而 encoder 的 attention 矩阵非满秩矩阵。因此，decoder 的表达能力高于 encoder 的表达能力。

## Transformer

![transformer-architechture](/assets/img/20231118/transformer-architecture.png){: .mx-auto.d-block :}
<center style="font-size:14px;color:#C0C0C0;">
Transformer 模型结构
</center>

Transformer 的输入由 input embedding、positional embedding、output embedding 组成。
Input embedding 和 output embedding 没有本质的区别，都是对输入序列进行编码，即 Word Embedding，将文本空间中的某个单词通过一定的方法映射或者嵌入到另一个数值向量空间中，可以使用 Word2Vec、GloVe 算法实现。

Positional embedding 用于表示输入向量在序列中的位置，因为 Transformer 不采用 RNN 结构，而是使用全局信息，不能利用单词的顺序信息，所以 positional embedding 对 Transformer 结构来说非常重要。Transformer 使用预定义函数，通过函数计算出位置信息的方式实现 positional embedding。Transformer 中计算位置信息的函数计算公式如下：

$$PE_{(pos,2i)}=sin(\frac{pos}{10000^{2i/d}})$$

$$PE_{(pos,2i+1)}=cos(\frac{pos}{10000^{2i/d}})$$

pos 代表的是词在句子中的位置，d 是词向量的维度，2i 代表 d 中的偶数维度，2i + 1 代表的是技术维度。这种计算方式使得每一维都对应一个正弦曲线，并且每一维 i 都对应不同周期的正余弦曲线。

在 transformer 中 $$input = input\_embedding + positional\_embedding$$。

Transformer 由 encoder 和 decoder 两个部分组成。在 Attention Is All You Need 中，分别堆叠了 6 个 encoder block 和 6 个 decoder block，最后一个 encoder 的输出与 decoder 部分通过交叉注意力机制连接起来。encoder 先处理 sequence，处理好之后，通过交叉注意力将处理好的结果输入到 decoder，decoder 再得到输出的结果。decoder 每次的输出都会与先前 decoder 产生的输出一起再作为 decoder 的输入。

Transformer 使用了 attention 机制，将序列中的任意两个位置的距离缩小为一个常量，Transformer 具有更好的并行性，符合现有的 GPU 框架。
## 参考连接

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [wiki: 注意力机制](https://zh.wikipedia.org/zh-hans/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6)
3. [知乎专栏：Positional Encoding](https://zhuanlan.zhihu.com/p/454482273)
4. [知乎专栏：Transformer模型详解（图解最完整版）](https://zhuanlan.zhihu.com/p/338817680)
5. [Machine-learning-notes](https://luweikxy.gitbook.io/machine-learning-notes/self-attention-and-transformer#Self-Attention%E6%9C%BA%E5%88%B6)
6. [详解Self-Attention和Multi-Head Attention](https://imzhanghao.com/2021/09/15/self-attention-multi-head-attention/)