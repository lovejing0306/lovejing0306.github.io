---
layout: post
title: Inception-v1
categories: [Classification]
description: Inception-v1
keywords: Classification
---


分类模型 Inception-v1
---


## 背景
&emsp;&emsp;$inception$（也称 $GoogLeNet$ ）是 $2014$ 年 $Christian \ Szegedy$ 提出的一种全新的深度学习结构，在这之前的 $AlexNet$、$VGG$ 等结构都是通过增大网络的深度（层数）来获得更好的训练效果，但层数的增加会带来很多负作用，比如 $overfit$、梯度消失、梯度爆炸等。

&emsp;&emsp;$inception$ 的提出从另一种角度来提升训练结果：能更高效的利用计算资源，在相同的计算量下能提取到更多的特征，从而提升训练结果。

## 网络结构

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-V1/inception4.jpg?raw=true"
    width="512" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">incpetion4</div>
</center>

说明：
* $Inception-v1$ 采用了模块化的结构（ $Inception$ 结构），方便增添和修改；
* 网络最后采用了$average \ pooling$（平均池化）来代替全连接层，该想法来自 $NIN$（$Network \ in \ Network$），事实证明这样可以将准确率提高 0.6%。但是，实际在最后还是加了一个全连接层，主要是为了方便对输出进行灵活调整；
* 虽然移除了全连接，但是网络中依然使用了 $Dropout$; 
* 为了避免梯度消失，网络额外增加了 $2$ 个辅助的 $softmax$ 用于向前传导梯度（辅助分类器）。辅助分类器是将中间某一层的输出用作分类，并按一个较小的权重（$0.3$）加到最终分类结果中，这样相当于做了模型融合，同时给网络增加了反向传播的梯度信号，也提供了额外的正则化，对于整个网络的训练很有裨益（事实上在较低的层级上这样处理基本没作用，作者在后来的 $inception \ v3$ 论文中做了澄清）。而在实际测试的时候，两个额外的 $softmax$ 会被去掉。

## 网络参数

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-V1/inception3.jpg?raw=true"
    width="720" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception3</div>
</center>

## 网络技巧
### inception模块
&emsp;&emsp;通过设计一个稀疏网络结构，不仅能够产生稠密的数据，而且既能增加神经网络表现，又能保证计算资源的使用效率。谷歌提出了最原始 $Inception$ 的基本结构：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-V1/inception1.jpg?raw=true"
    width="512" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception1</div>
</center>

该结构将 $CNN$ 中常用的卷积（$1 \times 1$，$3 \times 3$，$5 \times 5$）、池化操作（$3 \times 3$）堆叠在一起（卷积、池化后的尺寸相同，将通道相加），一方面增加了网络的宽度，另一方面也增加了网络对尺度的适应性。

&emsp;&emsp;网络卷积层中的网络能够提取输入的每一个细节信息，同时 $5 \times 5$ 的滤波器也能够覆盖大部分接受层的的输入。还可以进行一个池化操作，以减少空间大小，降低过度拟合。在这些层之上，在每一个卷积层后都要做一个 $ReLU$ 操作，以增加网络的非线性特征。

&emsp;&emsp;原始版本的 $Inception$ 中，所有的卷积核都在上一层的所有输出上来做，而 $5 \times 5$ 的卷积核所需的计算量太大了，造成了特征图的厚度很大，为了避免这种情况，在 $3 \times 3$ 前、$5 \times 5$ 前、$max \ pooling$ 后分别加上了 $1 \times 1$ 的卷积核，起到了降低特征图厚度的作用，这也就形成了 $Inception \ v1$ 的网络结构，如下图所示：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-V1/inception2.jpg?raw=true"
    width="512" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception2</div>
</center>

&emsp;&emsp;$inception$ 结构的主要贡献有两个：一是使用 $1 \times 1$ 的卷积 来进行升降维；二是在多个尺寸上同时进行卷积再聚合。

### 1x1卷积
* 在相同尺寸的感受野中叠加更多的卷积，能提取到更丰富的特征。这个观点来自于 $Network \ in \ Network$
* 使用 $1 \times 1$ 卷积进行降维，降低了计算复杂度。比如，上一层的输出为 $100 \times 100 \times 128$，经过具有 $256$ 个通道的 $5 \times 5$ 卷积层之后( $stride=1$，$pad=2$ )，输出数据为 $100 \times 100 \times 256$，其中，卷积层的参数为 $128 \times 5 \times 5 \times 256= 819200$。而假如上一层输出先经过具有 $32$ 个通道的 $1 \times 1$ 卷积层，再经过具有 $256$ 个输出的 $5 \times 5$ 卷积层，那么输出数据仍为为 $100 \times 100 \times 256$，但卷积参数量已经减少为 $128 \times 1 \times 1 \times 32 + 32 \times 5 \times 5 \times 256= 204800$，大约减少了 $4$ 倍。
* 有人会问，用 $1 \times 1$ 卷积降到 $32$ 个特征后特征数不就减少了么，会影响最后训练的效果么？

    答案是否定的，只要最后输出的特征数不变（$256$组），中间的降维类似于压缩的效果，并不影响最终训练的结果。

### 多个尺寸上进行卷积再聚合
* 直观感觉上在多个尺度上同时进行卷积，能提取到不同尺度的特征。特征更为丰富也意味着最后分类判断时更加准确。
* 利用稀疏矩阵分解成密集矩阵计算的原理来加快收敛速度。
* $Hebbin$ 赫布原理。$Hebbin$ 原理是神经科学上的一个理论，解释了在学习的过程中脑中的神经元所发生的变化，用一句话概括就是 $fire \ togethter \ wire \ together$。赫布认为“两个神经元或者神经元系统，如果总是同时兴奋，就会形成一种‘组合’，其中一个神经元的兴奋会促进另一个的兴奋”。

    比如狗看到肉会流口水，反复刺激后，脑中识别肉的神经元会和掌管唾液分泌的神经元会相互促进，“缠绕”在一起，以后再看到肉就会更快流出口水。用在 $inception$ 结构中就是要把相关性强的特征汇聚到一起。这有点类似上面的解释 $2$，把 $1 \times 1$，$3 \times 3$，$5 \times 5$ 的特征分开。因为训练收敛的最终目的就是要提取出独立的特征，所以预先把相关性强的特征汇聚，就能起到加速收敛的作用。
* 在 $inception$ 模块中有一个分支使用了 $max \ pooling$，作者认为 $pooling$ 也能起到提取特征的作用，所以也加入到模块中。注意这个 $pooling$ 的 $stride=1$，$pooling$ 后没有减少数据的尺寸。

### Global Average Pooling
* $Network \ in \ Network$ 最早提出了用 $Global \ Average \ Pooling$（$GAP$）层来代替全连接层的方法，具体方法就是对每一个 $feature$ 上的所有点做平均，有 $n$ 个 $feature$ 就输出 $n$ 个平均值作为最后的 $softmax$ 的输入。
* 对数据在整个 $feature$ 上作正则化，防止了过拟合；
* 不再需要全连接层，减少了参数的数目（一般全连接层是整个结构中参数最多的层），过拟合的可能性降低；
* 不用再关注输入图像的尺寸，因为不管是怎样的输入都是一样的平均方法，传统的全连接层要根据尺寸来选择参数数目，不具有通用性。

### 层级分支分类器
&emsp;&emsp;$inception$ 结构在某些层级上加了分支分类器，输出的 $loss$ 乘以一个系数再加到总的 $loss$ 上，作者认为可以防止梯度消失问题（事实上在较低的层级上这样处理基本没作用，作者在后来的 $inception \ v3$ 论文中做了澄清）。