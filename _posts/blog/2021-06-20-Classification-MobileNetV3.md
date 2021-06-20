---
layout: post
title: MobileNetV3
categories: [Classification]
description: MobileNetV3
keywords: Classification
---


分类模型 MobileNetV3
---


## 背景
$MobileNetV3$ 提供了两个版本，分别为 $MobileNetV3 \ Large$ 以及 $MobileNetV3 \ Small$，分别适用于对资源不同要求的情况，论文中提到，$MobileNetV3 \ Small$ 在 $ImageNet$ 分类任务上，较 $MobileNetV2$，精度提高了大约 3.2%，时间却减少了 15%，$MobileNetV3 \ Large$ 在 $ImageNet$ 分类任务上，较 $MobileNetV2$，精度提高了大约 4.6%，时间减少了 5%，$MobileNetV3 \ Large$ 与 $MobileNetV2$ 相比，在 $COCO$ 上达到相同的精度，速度快了 25%，同时在分割算法上也有一定的提高。

本文另一个亮点在于，网络的设计利用了 $NAS$（$network \ architecture \ search$）算法以及 $NetAdapt$ 算法。

## 减少网络计算量的方法
* 基于轻量化网络设计：比如 $MobileNet$ 系列，$ShuffleNet$ 系列， $Xception$ 等，使用 $Group$ 卷积、$1\times1$ 卷积等技术减少网络计算量的同时，尽可能的保证网络的精度。
* 模型剪枝： 大网络往往存在一定的冗余，通过剪去冗余部分，减少网络计算量。
* 量化：利用 $TensorRT$ 量化，一般在 $GPU$ 上可以提速几倍。
* 知识蒸馏：利用大模型（$teacher \ model$）来帮助小模型（$student \ model$）学习，提高 $student \ model$ 的精度。


## 相关工作
* $SqueezeNet$：提出 $Fire \ Module$ 设计，主要思想是先通过 $1\times1$ 卷积压缩通道数（$Squeeze$），再通过并行使用 $1\times1$ 卷积和 $3\times3$ 卷积来抽取特征（$Expand$），通过延迟下采样阶段来保证精度。
通过减少 $MAdds$ 来加速的轻量模型：
* $MobileNetV1$：提出深度可分离卷积；
* $MobileNetV2$：提出反转残差线性瓶颈块；
* $ShuffleNet$：结合使用分组卷积和通道混洗操作，进一步减少 $MAdds$；
* $CondenseNet$：在训练阶段学习组卷积，以保持层与层之间有用的紧密连接，以便功能重用；
* $ShiftNet$：提出了与点向卷积交织的移位操作，以取代昂贵的空间卷积；


## 网络结构
* $NAS$ 搜索全局结构
* $NetAdapt$ 搜索层结构

$MobileNetV3$ 通过应用平台感知 $NAS$ 和 $NetAdapt$ 进行网络搜索，并结合本节中定义的网络改进所提出。如下图所示，分 $Large$、$Small$ 两款，分别针对高资源和低资源使用情况。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNet-V3/Table1.jpg?raw=true"
    width="480" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> MobileNetV3-Large </div>
</center>

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNet-V3/Table2.jpg?raw=true"
    width="480" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> MobileNetV3-Small </div>
</center>

## MobileNetV3 改进
### 更改末端层结构
$MobileNetV2$ 的 $avg \ pooling$ 之前，存在一个 $1\times1$ 的卷积层，目的是提高特征图的维度，更有利于结构的预测，但是会带来一定的计算量。

为了减少延迟并保留高维特性，$MobileNetV3$ 将 $MobileNetV2$ 中的 $1\times1$ 卷积层移到 $avg \ pooling$ 之后。首先利用 $avg \ pooling$ 将特征图大小由 $7\times7$ 降到了 $1 \times 1$，然后再利用 $1\times1$ 提高维度，这样就减少了 $7 \times 7=49$ 倍的计算量。

为了进一步的降低计算量，作者直接去掉了前面纺锤型卷积中的 $3\times3$和$1\times1$ 卷积，进一步减少了计算量，如下图第二行所示。作者将其中的 $3\times3$ 以及 $1\times1$ 去掉后，精度并没有得到损失。这里降低了大约 $15ms$ 的速度。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNet-V3/Figure5.jpg?raw=true"
    width="560" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Figure 5 </div>
</center>

### 更改初始卷积核个数
修改头部卷积核数量，$MobileNetV2$ 中使用 $32$ 个 $3\times3$ 的卷积来构建初始滤波器的边缘检测，作者发现，很多特征图都是彼此的镜像。所以 $MobileNetV3$ 中改成了 $16$，在保证了精度的前提下，降低了 $3ms$ 的速度。

### H-Swish 激活函数
引入新的非线性激活函数 $h-swish$，它是 $swish$ 非线性的改进版本，计算速度比 $swish$ 更快（比 $relu$ 慢），更易于量化，精度上没有差异。网络越深越能够有效的减少网络参数量。

$swish$ 相比 $relu$ 提高了精度，但因 $sigmoid$ 函数而计算量大，公式：

$$
swish\left( x \right) =x\cdot \sigma \left( x \right) 
$$

$h-swish$ 将 $sigmoid$ 函数替换为分段线性硬模拟，使用的 $relu6$ 在众多软硬件框架中都可以实现，量化时有避免了数值精度的损失。所以 $swish$ 的硬版本也变成了：

$$
h-swish\left( x \right) =x\frac{\text{Re}LU6\left( x+3 \right)}{6}
$$

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNet-V3/Figure6.jpg?raw=true"
    width="560" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Figure 6 </div>
</center>

### 引入SE模块
$SE$ 模块能够让网络自动学习到了每个特征通道的重要程度。作者在 $bottlenet$ 结构中加入了 $SE$ 结构，将其放在了 $depthwise \ filter$ 之后，如下图。
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNet-V3/Figure4.jpg?raw=true"
    width="560" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Figure 4 </div>
</center>

由于 $SE$ 结构会消耗一定的时间，所以作者在含有 $SE$ 的结构中，将 $expansion \ layer$ 的 $channel$ 变为原来的 $1/4$，这样做提高精度的同时并没有增加时间消耗。


## 分类表现
在 $ImageNet$ 上的精度表现，以及在 $Pixel \ Phone$ 上的耗时统计。

### 浮点模式

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNet-V3/Table3..jpg?raw=true"
    width="480" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Table 3 </div>
</center>

### 量化模式
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNet-V3/Table4.jpg?raw=true"
    width="480" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Table 4 </div>
</center>

## 参考
[Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf)

[论文学习笔记-MobileNet v3](https://blog.csdn.net/sinat_37532065/article/details/90813655)

[mobilenet系列之又一新成员---mobilenetV3](https://blog.csdn.net/Chunfengyanyulove/article/details/91358187)

[CNN模型合集 MobileNet V3](https://zhuanlan.zhihu.com/p/69315156)