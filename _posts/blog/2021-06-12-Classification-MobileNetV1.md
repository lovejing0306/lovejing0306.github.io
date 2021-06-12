---
layout: post
title: MobileNetV1
categories: [Classification]
description: MobileNetV1
keywords: Classification
---


分类模型 MobileNetV1
---


## 背景
有一些应用场景（如移动设备、嵌入式设备）对模型大小、计算量都有限制。本文的目标就是构建小且高效的网络结构。本文提出的方法在模型大小显著降低的情况下，在 $ImageNet$ 等数据集上得到了很好的效果。

## Depthwise Separable Convolution

$Depthwise \ Separable \ Convolution$ 与普通卷积的输入与输出均相同，中间过程不同。

计算量：$D  _  K\cdot D  _  K\cdot M\cdot D  _  F\cdot D  _  F+M\cdot N\cdot D  _  F\cdot D  _  F$

### 问题描述

输入： $D  _  F\cdot D  _  F\cdot M $ ，其中 $D  _  F$ 为原始图片尺寸， $M$ 为输入通道数量。

输出： $D  _  F\cdot D  _  F\cdot N $ ，其中 $D  _  F$ 为输出图片尺寸， $N$ 为输出通道数量。

### 普通卷积

卷积核： $D  _  K\cdot D  _  K\cdot M\cdot N $

计算量：$D  _  K\cdot D  _  K\cdot M\cdot N\cdot D  _  F\cdot D  _  F$

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNet-V1/%20StandardConvolution.jpg?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Standard Convolution</div>
</center>

### Depthwise Separable Convolution

$Depthwise \ Separable \ Convolution$ 将标准卷积拆分为了两个操作：深度卷积($depthwise \ convolution$) 和 逐点卷积($pointwise \ convolution$)。

#### depthwise convolution

卷积核：$M$ 个 $D  _  k\cdot D  _  k\cdot 1\cdot 1$

输入： $D  _  F\cdot D  _  F\cdot M $

输出： $D  _  F\cdot D  _  F\cdot M $

作用：负责滤波作用

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNet-V1/DepthwiseConvolutional.jpg?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Depthwise Convolutional</div>
</center>

$depthwise \ convolution$ 针对每个输入通道采用不同的卷积核，也就是说一个卷积核对应一个输入通道，所以说 $depthwise \ convolution$ 是 $depth$ 级别的操作。

#### pointwise convolution

卷积核：$1\cdot 1 \cdot M \cdot N$

输入： $D  _  F\cdot D  _  F\cdot M $

输出： $D  _  F\cdot D  _  F\cdot N $

作用：负责转换通道

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNet-V1/PointwiseConvolution.jpg?raw=true"
    width="480" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Pointwise Convolution</div>
</center>

$pointwise \ convolution$ 其实就是普通的卷积，只不过其采用 $1 \times 1$ 的卷积核。

### 计算量比较

$$
\frac{D_K\cdot D_K\cdot M\cdot D_F\cdot D_F+M\cdot N\cdot D_F\cdot D_F}{D_K\cdot D_K\cdot M\cdot N\cdot D_F\cdot D_F}=\frac{1}{N}+\frac{1}{D_{K}^{2}}
$$

一般情况下 $N$ 取值比较大，那么如果采用 $3 \times 3$ 卷积核的话，$depthwise \ separable \ convolution$ 相较标准卷积可以降低大约 $9$ 倍的计算量。

## MobileNet-V1 结构
### 标准卷积结构和Depthwise Separable Convolution结构比对

在实际使用深度级可分离卷积核时，加了 $BN$ 和 $ReLU$，其和标准卷积结构对比如下：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNet-V1/Figure3.jpg?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Figure 3</div>
</center>

### 网络总体结构

一共由 $28$ 层构成(不包括 $Avg \ Pool$ 和 $FC$ 层，且把深度卷积和逐点卷积分开算)，其除了第一层采用的是标准卷积核之外，剩下的卷积层都用的是 $Depthwise \ Separable \ Convolution$。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNet-V1/Table1.jpg?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Table 1</div>
</center>

### 网络介绍
网络参数和计算量分布：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNet-V1/Table2.jpg?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Table 2</div>
</center>

* $MobileNet$ 中大多数计算量和参数都在 $1 \times 1$ 的卷积中，可以使用高度优化的。
* 由于模型较小，所以可以减少使用正则化，因为模型小不容易过拟合。

### 控制MobileNet模型大小的两个超参数
#### Width Multiplier
该参数用于控制特征图的通道数量，即维度。对于 $depthwise$ 卷积操作，其计算量为：

$$
D_K\cdot D_K\cdot \alpha M\cdot D_F\cdot D_F+\alpha M\cdot \alpha N\cdot D_F\cdot D_F
$$

通常 $\alpha$ 在 $(0, 1]$ 之间，比较典型的值有 $1, 0.75, 0.5$ 和 $0.25$。计算量和参数数量减少程度与未使用宽度因子之前提高了 $\frac{1}{\alpha ^2}$ 倍。

#### Resolution Multiplier
该参数用于控制特征图的宽和高。对于 $depthwise$ 卷积操作，其计算量为：

$$
D_K\cdot D_K\cdot \alpha M\cdot \rho D_F\cdot \rho D_F+\alpha M\cdot \alpha N\cdot \rho D_F\cdot \rho D_F
$$

通常 $\rho$在$(0, 1]$ 之间，比较典型的输入分辨为 $224$、$192$、$160$、$128$。计算量减少程度与未使用宽度因子之前提高了 $\frac{1}{\alpha ^2\times \rho ^2}$ 倍，参数量没有影响。

## 缺点
$Depthwise \ Convolution$ 有潜在问题，部分卷积核的参数会变为 $0$。出现这种情况可能和使用 $relu$ 激活函数有关。

## 参考
[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)

[MobileNet v1模型详读](https://zhuanlan.zhihu.com/p/58554116)

[精读深度学习论文(8) MobileNet V1](https://zhuanlan.zhihu.com/p/33634489)