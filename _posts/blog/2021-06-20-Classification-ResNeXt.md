---
layout: post
title: ResNeXt
categories: [Classification]
description: ResNeXt
keywords: Classification
---


分类模型 ResNeXt
---


## 背景
传统提高模型准确率的方法都是加深或加宽网络，但是随着超参数数量的增加（比如 $channels$ 数，$filter \ size$ 等等），网络设计的难度和计算开销也会增加。因此本文提出的 $ResNeXt$ 结构可以在不增加参数复杂度的前提下提高准确率，同时还减少了超参数的数量。

## 思想来源
$ResNeXt$ 的思想来源于基于一下两点：
1. $VGG$ 网络采用堆叠的方式实现，之前的 $ResNeXt$ 也借用了这样的思想
2. $Inception$ 系列网络采用了 $split-transform-merge$ 的策略，但其存在一个问题：网络的超参数设定的针对性比较强，当应用在别的数据集上需要修改许多参数，因此可扩展性不是很强。

作者在这篇论文中提出网络 $ResNeXt$，同时采用 $VGG$ 堆叠的思想和 $Inception$的$split-transform-merge$ 思想，但是可扩展性比较强，可以认为是在增加准确率的同时基本不改变或降低模型的复杂度。

作者在 $ResNeXt$ 网路中提出了 $cardinality$ 概念，即分组的数量。如下图所示，左边为 $ResNet$ 中的 $block$；右边为 $ResNeXt$ 改进后的 $cardinality=32$ 的 $blokc$，其中每个被聚合的拓扑结构都是一样的，这样做减轻了设计的负担，这是和 $Inception$ 的差别。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/ResNeXt/resnext-block.jpg?raw=true"
    width="520" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">ResNeXt block</div>
</center>

并且作者指出增加 $cardinality$ 比增加模型的深度和宽度更有助于提高模型的精度。

## 全连接网络
全连接网络中每个神经元可以看作是通过 $aggregating \ transformation$ 变换的方式生成。公式如下：

$$
\sum_{i=1}^D{w_ix_i}
$$
其中，$x=\left[ x  _  1,x  _  2,...,x  _  D \right] $是一个$D$维的输入向量；$w  _  i$是第$i$个$channel$的$filter$权重。

内积计算可以看作是$splitting-transforming-aggregating$的组合： 

1. $Splitting$：输入向量 $x$ 被分为低维 $embedding$，即单维空间的$x_  i$；
2. $Transforming$：变换得到低维表示，即：$w  _  ix  _  i$；
3. $Aggregating$： 通过相加将所有的 $embeddings$ 变换聚合，即 $\sum  _  {i=1}^D{}$.

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/ResNeXt/neuron.jpg?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">neuron</div>
</center>

## 聚合变换
通过上面的分析，可以将逐元素变换$w  _  ix  _  i$替换为函数或网络，此时聚合变换表示为：
$$
\mathcal{F}\left( x \right) =\sum_{i=1}^C{\mathcal{T}\left( x \right)}
$$
其中，$\mathcal{T}\left( x \right)$ 可以是任意函数，其将x投影到一个嵌入空间(一般是低维空间)，并进行变换；$C$ 是待聚合的变换集的大小，即 $Cardinality$，$Cardinality$ 的维度决定了更复杂变换的数目，类似于 $D$；对于变换函数的设计，采用策略是：所有的 $\mathcal{T}  _  i$ 拓扑结构相同。

基于 $residual$ 函数的 $aggregated \ transformation$：

$$
y=x+\sum_{i=1}^C{\mathcal{T}_ i\left( x \right)}
$$

如下图所示，作者展示了 $3$ 种不同的 $ResNeXt \ blocks$：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/ResNeXt/Equivalent-building-blocks-of-ResNeXt..jpg?raw=true"
    width="720" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Equivalent building blocks of ResNeXt.</div>
</center>

* $(a)$ 就是前面的 $aggregated \ residual \ transformations$
* $(b)$ 则是采用两层卷积后 $concatenate$，再卷积，有点类似 $Inception-ResNet$，只不过这里的路径都是相同的拓扑结构
* $(c)$ 分组卷积

作者在文中明确说明这三种结构是严格等价的，并且用这三个结构做出来的结果一模一样，在本文中展示的是 $(c)$ 的结果，因为 $(c)$ 的结构比较简洁而且速度更快。

## 分组卷积
$Group \ convolution$，最早在 $AlexNet$ 中出现，由于当时硬件资源有限，训练 $AlexNet$ 时卷积操作不能全部放在同一个 $GPU$ 处理，因此作者把 $feature \ maps$ 分给多个 $GPU$ 分别进行处理，最后把多个 $GPU$ 的结果进行融合。

分组卷积的思想影响比较深远，当前一些轻量级的 $SOTA$（$State \ Of \ The \ Art$）网络，都用到了分组卷积的操作，以节省计算量。比如当 $input-channel=256$, $output-channel=256$, $kernel \ size=3 \times 3$:
* 不做分组卷积的时候，分组卷积的参数为 $256 \times 256 \times 3 \times 3$
* 当分组卷积的时候，比如说 $group=2$，每个 $group$ 的 $input-channel$、$output-channel=128$，参数数量为 $2 \times 12 \times 128 \times 3 \times 3$，为原来的 $1/2$。

最后输出的 $feature \ maps$ 通过 $concatenate$ 的方式组合，而不是 $elementwise \ add$。 如果放到两张 $GPU$ 上运算，那么速度就提升了 $4$ 倍。

## 参考
* [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf)
* [ResNeXt论文阅读笔记](https://www.twblogs.net/a/5bf3511bbd9eee37a0606231/zh-cn)
* [论文阅读 - ResNeXt - Aggregated Residual Transformations for DNN](https://www.aiuai.cn/aifarm126.html)