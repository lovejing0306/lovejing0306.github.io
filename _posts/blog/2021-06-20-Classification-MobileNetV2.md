---
layout: post
title: MobileNetV2
categories: [Classification]
description: MobileNetV2
keywords: Classification
---


分类模型 MobileNetV2
---


## 背景
$MobileNet \ V2$ 是在 $V1$ 的基础上做了一些结构上的调整，其主要通过添加 $Inverted \ Residual$ 和 $Linear \ Bottleneck$，使得$MobileNet \ V2$ 的精度进一步提高，结构进一步合理。

$MobileNet \ V1$ 在设计的时候使用 $Deepwise \ Separable \ Convolution$ 代替传统的卷积，大大降低了模型的计算量和复杂度，但是其仍然存在以下两个缺陷：
* 直筒型的结构影响网络性能，后续的网络如 $ResNet$ 等，在网络中重复使用图像特征能够提高网络的性能。（引入 $Inverted \ Residual$）
* $Depthwise \ Convolution$ 导致特征退化问题：由于 $Depthwise \ Convolution$ 使用很小的卷积核（$1 \times 1$），经过 $BN$ 归一化，以及 $relu$ 激活之后很容易变为 $0$，即变成死节点,导致特征退化。（引入 $Linear \ Bottleneck$）

## 网络结构
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNet-V2/Table2.jpg?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Table 2 </div>
</center>

## MobileNetV2 改进
当单独去看特征图上每个通道的像素值时，其实这些值代表的特征可以映射到一个低维子空间的流形区域上。在完成卷积操作之后往往会接一层激活函数来增加特征的非线性性，一个最常见的激活函数便是 $relu$。

由在残差网络中介绍的数据处理不等式($DPI$)，$relu$ 一定会带来信息损耗，而且这种损耗是没有办法恢复的，$relu$ 的信息损耗是当通道数非常少的时候更为明显。

为什么这么说呢？看图 $1$ 中的例子，其输入是一个表示流形数据的矩阵，和卷机操作类似，它会经过 $n$ 个 $relu$ 的操作得到 $n$ 个通道的特征图，然后试图通过这 $n$ 个特征图还原输入数据，还原的越像说明信息损耗的越少。从图 $1$ 中可以看出，当 $n$ 的值比较小时，$relu$ 的信息损耗非常严重，但是当 $n$ 的值比较大的时候，输入流形就能还原的很好了。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNet-V2/Figure1.jpg?raw=true"
    width="720" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Figure 1 </div>
</center>

对于上面提到的信息损耗问题分析，有两种解决方案：
1. 既然是 $relu$ 导致的信息损耗，那么我们就将 $relu$ 替换成线性激活函数；
2. 如果比较多的通道数能减少信息损耗，那么我们就使用更多的通道。


### Linear Bottlenecks
当然不能把 $relu$ 全部换成线性激活函数，不然网络将会退化为单层神经网络，一个折中方案是在输出特征图的通道数较少的时候也就是 $bottleneck$ 部分使用线性激活函数，其它时候使用 $relu$。代码片段如下：

```python
def  _ bottleneck(inputs, nb _ filters, t):
    x = Conv2D(filters=nb _ filters * t, kernel _ size=(1,1), padding='same')(inputs)
    x = Activation(relu6)(x)
    x = DepthwiseConv2D(kernel _ size=(3,3), padding='same')(x)
    x = Activation(relu6)(x)
    x = Conv2D(filters=nb _ filters, kernel _ size=(1,1), padding='same')(x)
    # do not use activation function
    if not K.get _ variable _ shape(inputs)[3] == nb _ filters:
        inputs = Conv2D(filters=nb _ filters, kernel _ size=(1,1), padding='same')(inputs)
    outputs = add([x, inputs])
    return outputs
```

这里使用了 $MobileNet$ 中介绍的 $relu6$ 激活函数，它是对 $relu$ 在 $6$上 的截断，数学形式为：

$$
\text{Re}LU\left( 6 \right) =\min \left( \max \left( 0,x \right) ,6 \right) 
$$

结构如下图所示：
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNet-V2/LinearBottlenecks.jpg?raw=true"
    width="720" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Linear Bottlenecks </div>
</center>

### Inverted Residual
当激活函数使用 $relu$ 时，可以通过增加通道数来减少信息的损耗，使用参数 $t$ 来控制，该层的通道数是输入特征图的 $t$ 倍。传统的残差块的 $t$ 一般取小于 $1$ 的小数，常见的取值为 $0.1$，而在 $v2$ 中这个值一般是介于 $5～10$ 之间的数，在作者的实验中 $t=6$。

由于残差网络和 $v2$ 的 $t$ 的不同取值范围，于是分别形成了沙漏形（两头大中间小）和锥子形（两头小中间大）的结构，如图 $3$ 所示，其中斜线特征图表示使用的是线性激活函数。这也就是为什么这种形式的卷积 $block$ 被叫做 $Interved \ Residual \ block$，由于把 $short-cut$ 转移到了 $bottleneck$ 层。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNet-V2/Figure3.jpg?raw=true"
    width="720" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Figure 3 </div>
</center>

## V2 和 V1 对比

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNet-V2/Figure4.jpg?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Figure 4 </div>
</center>

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNet-V2/Figure2.jpg?raw=true"
    width="540" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Figure 2 </div>
</center>

如图 $(b)$ 所示，$MobileNet \ v1$ 最主要的贡献是使用了 $Depthwise \ Separable \ Convolution$，它又可以拆分成 $Depthwise \ Convolution$ 和 $Pointwise \ Convolution$。$MobileNet \ v2$ 主要是将残差网络和 $Depthwise \ Separable$ 卷积进行了结合，通过分析单通道的流形特征对残差块进行了改进，包括对中间层的扩展 $(d)$ 以及 $bottleneck$ 层的线性激活 $(c)$。$Depthwise \ Separable \ Convolution$ 的分离式设计直接将模型压缩了 $8$ 倍左右，但是精度并没有损失非常严重。

$Depthwise \ Separable \ Convolution$ 的设计非常精彩但遗憾的是目前 $cudnn$ 对其的支持并不好，导致在使用 $GPU$ 训练网络过程中无法从算法中获益，但是使用串行 $CPU$ 并没有这个问题，这也就给了 $MobileNet$ 很大的市场空间，尤其是在嵌入式平台。

## 参考
[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)

[MobileNet v1 和 MobileNet v2](https://zhuanlan.zhihu.com/p/50045821)

[MobileNet V2 详解](https://perper.site/2019/03/04/MobileNetV2-%E8%AF%A6%E8%A7%A3/)

[MobileNet V2 论文初读](https://zhuanlan.zhihu.com/p/33075914)