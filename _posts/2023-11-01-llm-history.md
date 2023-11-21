---
layout: post
title: Google & OpenAI 大模型之战
tags: [llm, Google, OpenAI]
comments: true
thumbnail-img: /assets/img/20231101/llm-evolution.png
author: SeaMount
---

![llm-evolution-tree](/assets/img/20231101/llm-evolution.png)
<center style="font-size:14px;color:#C0C0C0;">
图源：<a href="https://arxiv.org/abs/2304.13712">
Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond
</a>
</center>

这是一个百家争鸣，百花齐放的大模型时代，多家科技公司都卷入了这场大模型战争之中，相继推出了各种各样的模型。但今天的主角是 Google 和 OpenAI 两家科技巨头。

## 缘起

故事要从 2011 年开始讲起。2011 年，本着将深度学习集成到谷歌的基础设计之中的目的， Google Brain 在 Jeff Dean、Greg Corrado、Andrew Ng 的带领之下成立。它本身为 Google X 实验室的项目之一，但奈何 Google X 在一项人工智能项目上取得了惊人的效益和成功，该团队一年赚到的钱超过了整个 Google X 部门的成本，谷歌便将该团队分离出来成为单独的部门，Google Brain 初步成型。2013 年 3 月，Google Brain 聘用深度学习之父 Geoffrey Hinton，并且收购了他所领导的 DNNResearch Inc. 公司。Hinton 随即表示会将他的精力分为两部分，分别给到在大学的科研工作和在谷歌的工作之中。2014 年 1 月 26 日，谷歌收购 DeepMind。2023 年 4 月，Alphabet 表示将合并 DeepMind 和谷歌大脑。

2015 年 12 月，Elon Musk、Sam Altman 等人秉持着推动人工智能的安全性和受益性，以确保人工智能技术对人类的利益产生积极的影响的初衷，在旧金山创办了 OpenAI。OpenAI 在成立初期主要进行人工智能研发，并且创始人承诺将所有的研究成果都公之于众，是一个非营利性组织。但是毕竟 AI 企业的发展成本非常高昂，训练大型 AI 模型需要巨额的投入，与此同时还需要持续的投资来维持研发，光用爱发电是不行滴。OpenAI 在 2019 年决定转变为一个“有盈利上限”的公司。转向盈利模式可以为 OpenAI 带来更多的资金支持，促进其技术和研究的发展。

## 初露锋芒

2017 年 6 月，Google Brain 在 NeurIPS 上发表了被称为大语言模型开山之作的文章 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，推出了 Transformer 模型结构。相较于之前的 RNN 结构，Transformer 能支持并行计算，可以更好地捕获长距离依赖，并且有很强的可扩展性，由此大模型时代的序幕正式揭开。

2018 年 6 月，OpenAI 基于 Transformer 架构发布了 117M 参数量的 GPT-1，同时发表了论文 [Improving Language Understanding by Generative Pre-training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)。GPT-1 堆叠了 12 个 decoder，并且采取 Pretrain + FineTuning 的模式。

2018 年 10 月，Google 发表了 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](http://arxiv.org/abs/1810.04805)，推出 340M 参数量的模型 BERT。BERT 也使用 Pretrain + FineTuning 的 NLP 范式，但与 GPT 不同的是，BERT 是 encoder-only 结构。继 BERT 之后，NLP 范式彻底转变，进入到 Pretrain + FineTuning 阶段。

此时，BERT 在各方面指标上的表现都吊打 GPT-1，谷歌先胜一局。

## 再度厮杀

2019 年 2 月，OpenAI 发表论文 [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)，继续采用 decoder-only 的结构，并推出最大模型为 48 层，参数量 1.5B 的 GPT-2。GPT-2 证明了 decoder-only 的结构有 zero-shot 的能力，实现 AGI 不是梦！GPT-2 在原始 Transformer decoder 的基础上对 LN 层的位置做了调整，并且更改了下游的范式为 Prompt。

2019 年 10 月，Google 发表论文 [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](http://arxiv.org/abs/1910.10683)，并提出了一个新的预训练模型 T5 (Transfer Text-to-Text Transformer)，采用 encoder-decoder 的结构，T5 将所有的任务，比如分类、相似度计算、文本生成等都用一个 Text-to-Text 框架解决，采用 teacher forcing 和交叉熵损失进行训练。T5 的最大模型参数量达到 11B，成为了全新的 NLP SOTA 预训练模型。

T5 模型碾压 GPT-2，以绝对优势胜出，谷歌继续领先。

## 三度较量

2020 年 5 月，OpenAI 坚守 decoder-only 架构，发表论文 [Language Models are Few-Shot Learner](http://arxiv.org/abs/2005.14165)，并且发布了最大参数量为 175B 的模型 GPT-3，在训练数据级别上，GPT-3 相较于之前的训练数据集大小也是有了惊人的增长，直接来到 TB 级别，达到 45TB。


| 模型 | 参数量 | 预训练数据量 |
| ---- | ---- | ---- |
| GPT-1 | 117M | 5GB |
| GPT-2 | 1.5B | 40GB |
| GPT-3 | 175B | 45TB |

2021 年 1 月，Google 发表论文 [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](http://arxiv.org/abs/2101.03961)，重磅推出 Switch Transformer，最大参数量为 1.6T，是世界上首个万亿级参数的模型。Switch Transformer 采用 encoder-decoder 架构，并且改进了 MoE 路由算法，降低了通信和计算成本，首次展示了用较低精度 (bf16) 训练大型稀疏模型。

2021 年 5 月，Google 展示了对话应用语言模型 LaMDA，并在后续发表文章 [LaMDA: Language Models for Dialog Applications](http://arxiv.org/abs/2201.08239)。LaMDA 有 137B 参数量，为 decoder-only 结构。

2021 年 6 月，OpenAI 联合 Github 推出了自动代码补全工具 Github Copilot，推出 12B 参数量的 CodeX，并发表论文 [Evaluating Large Language Models Trained on Code](http://arxiv.org/abs/2107.03374)。CodeX 是 GPT-3 的后代，基于 GPT-3 使用代码数据对模型进行了微调。

2021 年 10 月，Google 继续推出 137B 参数量的 decoder-only 结构模型 FLAN，并发表文章 [Finetuned Language Models Are Zero-Shot Learners](http://arxiv.org/abs/2109.01652)。采用了 Instruction tuning 的方式。

2022 年 3 月，OpenAI 发表论文 [Training language models to follow instructions with human feedback](http://arxiv.org/abs/2203.02155)，引入了 RLHF 机制，推出 InstructGPT。相较于 175B 参数的 GPT-3，人们似乎更喜欢 13B 参数的 InstructGPT 生成的回复。古话说得好“氪不救非，玄不改命”，并不是模型规模越大就越好的，还是需要添加出色的方法。

OpenAI 越挫越勇，一路猛追，但此时业界仍旧更加认可 Google 的大模型。

## 巅峰对决

2022 年 4 月，Google 发布基于 Pathways 架构的模型 [PaLM: Scaling Language Modeling with Pathways](http://arxiv.org/abs/2204.02311)，在 PaLM 中，提到了 CoT 技术。PaLM 的最大模型参数量达到 540B，采用 decoder-only 架构。

2022 年 9 月，Google 发布 70B 参数的 Sparrow，采用 encoder-decoder 结构，加入了 RLHF 和 Retrival 技术，但却反响平平。

2022 年 11 月 30 日，OpenAI 发布了约 200B 参数量的 ChatGPT，轰动全球。ChatGPT 是无心插柳得到的产物，OpenAI 团队起初只是想改进 GPT 语言模型，但是考虑到想要产出用户想要的东西，就必须使用强化学习，而聊天机器人就是理想候选人。ChatGPT 借鉴了前人所有出色的点（CodeX的代码能力、Instruction tuning、RLHF、CoT），在 GPT-3 的基础上做了调整，以人类对话的形式不断提供反馈机制，使得人工智能软件知道何时做得很好，以及哪里需要改进。

2023 年 2 月，Google 基于 LaMDA 发布了新一代对话系统 Bard，参数量为 137B。不同于 LaMDA，Bard 是 encoder-decoder 架构。与 GPT 比较起来，Bard 时效性更高，但给出的回复却不尽如人意。

2023 年 3 月，OpenAI 发布 GPT-4，参数量多达 1.76T，支持多模态输入，站在了大模型时代的山巅。

至此，OpenAI 手持 GPT-4 遥遥领先。

## 总结与思考

所有大卫战胜歌莉娅的故事都值得我们思考。当 Google 拿出效果极好的 encoder-only 的 BERT 时，OpenAI 有没有进行过灵魂拷问，有没有怀疑过自己选择的路？想必也是有的，但 OpenAI 仍然拿出了坚定的信念和惊人的魄力，坚持使用 decoder-only 的 GPT，践行者“暴力美学”，以堆叠模型的路径，实现 AGI。与此同时，OpenAI 还善于站在巨人的肩膀上解决问题，无论是 Transformer、RLHF、Instruction tuning 还是 CoT，OpenAI 都取其精华，默默为自己的大模型添砖加瓦。

Google 成立之初可谓是无懈可击，但最后仍被 OpenAI 反超。一方面可能是由于自身举棋不定，在 encoder-only、encoder-decoder 和 decoder-only 之间反复横跳，没有进行深耕，当然也就意料之中没有惊人的收获；另一方面是对于优秀的创新点不够敏感，2017 年 Google 就提出了 RLHF，但却是 OpenAI 在 2020 年首先将 RLHF 思想用到 GPT-3 之中，Google 直到 2022 年 9 月才用于 Sparrow 之中。但从另一个角度来看，Google 敢于做出不同的尝试，即使在已知 decoder 相较于 encoder 表达能力更强的情况下，仍然尝试去挖掘 encoder 中的理解能力，这种创新的精神也是十分难能可贵的。

Google 和 OpenAI 还将相继推出 Gemini 和 Gobi，这场大模型之战还未到握手言和之际，甚至可以说一切才刚刚开始。
