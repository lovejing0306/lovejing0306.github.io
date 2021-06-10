---
layout: post
title: Xception
categories: [Classification]
description: Xception
keywords: Classification
---


分类模型 Xception
---


## 背景
&emsp;&emsp;$Xception$ 是 $google$ 继 $inception$ 后提出的对 $inception \ v3$ 的另一种改进，主要是采用 $depthwise \ separable \ convolution$ (深度可分离卷积)来替换原来 $inception \ v3$ 中的卷积操作。

## 网络结构
&emsp;&emsp;$xception$ 的结构基于 $resnet$，但是将其中的卷积层换成了 $depthwise \ separable \ conv$。如下图所示，整个网络被分为了三个部分：$Entry$，$Middle$ 和 $Exit$：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Xception/xception1.jpg?raw=true"
    width="720" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">xception1</div>
</center>

## 网络技巧
### xception block
&emsp;&emsp;在 $inception v3$ 中，一个典型的 $inception$ 模块长下面这个样子：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Xception/xception2.jpg?raw=true"
    width="400" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">xception2</div>
</center>

&emsp;&emsp;对于一个卷积层来说，它要学习的是一个三维的 $filter$，包括两个空间维度（$spatial \ dimension$），即 $width$ 和 $height$；以及一个通道维度（$channel \ dimension$）。这个 $filter$ 和输入，在 $3$ 个维度上进行卷积操作，得到最终的输出。可以用伪代码表示如下：

```
// 对于第i个filter
// 计算输入中心点(x, y)对应的卷积结果
sum = 0
for c in 1:C
  for h in 1:K
    for w in 1:K
      sum += in[c, y-K/2+h, x-K/2+w] * filter _ i[c, h, w]
out[i, y, x] = sum
```

可以看到，在三维卷积中，$channel$ 这个维度和 $spatial$ 的两个维度并无不同。

&emsp;&emsp;在 $Inception$ 中，卷积操作更加轻量级。输入首先被 $1 \times 1$ 的卷积核处理，得到了跨 $channel$ 的组合($cross-channel \  correlation$)，同时将输入的 $channel \  dimension$ 减少了 $3-4$ 倍（一会$4$个支路要做 $concat$ 操作）。这个结果被后续的 $3 \times 3$ 卷积和 $5 \times 5$ 卷积核处理，处理方法和普通的卷积一样。

&emsp;&emsp;由此作者想到，$inception$ 能够 $work$ 证明：卷积的通道（$channel$）相关性和空间（$spatial$）相关性是可以解耦的，没必要把它们一起完成。

&emsp;&emsp;作者将 $Inception$ 结构做简化，保留了主要结构，去掉了 $avg \ pooling$ 操作，如下所示：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Xception/xception3.jpg?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">xception3</div>
</center>

将底层的 $3$ 个 $1 \times 1$ 的卷积核组合起来，其实图 $1$ 和图 $2$ 是等价的。一个“大的” $1 \times 1$ 的卷积核（$channels$ 数目变多），其输出结果在 $channels$ 上被分为若干组（$group$），每组分别和不同的 $3 \times 3$ 卷积核做卷积，再将这 $3$ 份输出 $concat$ 起来，得到最后的输出，如下图所示：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Xception/xception4.jpg?raw=true"
    width="400" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">xception4</div>
</center>

那么，如果把分组数目继续调大呢？极限情况，可以使得 $group \ number = channel \ number$，即先进行普通卷积操作，再对 $1 \times 1$ 卷积后的每个 $channel$ 分别进行 $3 \times 3$ 卷积操作，最后将结果 $concat$，如下所示：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Xception/xception5.jpg?raw=true"
    width="400" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">xception5</div>
</center>

### Depthwise Separable Convolution
&emsp;&emsp;$xception$ 中的这种结构和一种名为 $Depthwise \ Separable \ Convolution$ 的技术很相似，$Depthwise \ Separable \ Convolution$ 首先使用 $group \ conv$ 在 $spatial \ dimension$ 上卷积，然后使用 $1 \times 1$ 的卷积核做 $cross \ channel$ 的卷积（又叫做 $pointwise \ conv$）。主要有两点不同：

1. 操作循序不一致：$Depthwise \ Separable \ Convolution$ 先进行 $3 \times 3$卷积，再进行 $1 \times 1$ 卷积；$inception$ 先进行 $1 \times 1$ 卷积，再进行 $3 \times 3$ 卷积。
2. 是否使用非线性激活操作：$inception$中，两次卷积后都使用 $Relu$；$Depthwise \ Separable \ Convolution$ 中，在 $depthwise$ 卷积后一般不添加 $Relu$。

### 基于的假设
$cross \ channel$ 的相关和 $spatial$ 的相关可以完全解耦。

## 优势
1. 使分类效果进一步提升