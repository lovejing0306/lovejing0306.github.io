---
layout: post
title: MobileNeXt
categories: [Classification]
description: MobileNeXt
keywords: Classification
---


分类模型 MobileNeXt
---

# MobileNeXt

## 简介
依图研发团队通过分析 $MobileNetV2$ 中倒残差模块的优势与劣势，提出了一种改良版本的 $MobileNeXt$，大大提升了移动端神经网络的性能。在不使用 $SE$ 等性能提升模块的条件下，相比于主流的 $MobileNetV2$，$MobileNeXt$ 在相同计算量下可达到 1.7% 的分类精度提升。

## 分析 MobileNetV1 和 MobileNetV2
$MobileNetV1$ 中引入深度可分离卷积，大大降低了网络的计算量，虽然速度提高了，但是低维信息映射到高维的维度较低，经过 $ReLU$ 后再映射回低维时损失比较大。因此，$Google$ 研究人员提出 $MobileNetV2$，其在 $MobileNetV1$  的基础上引入了倒残差和线性瓶颈两个模块。

$MobileNetV2$ 降低时延的同时精度也有所提升，在高维度使用深度可分离卷积，倒残差网络可以有效降低计算开销，保证模型性能。同时，瓶颈结构（这里指的是 $block$ 与 $block$ 之间的连接，$block$ 内的连接是纺锤型的）的连接方式可以有效降低点操作的数量、减少所需要的内存访问，进而进一步减小硬件上的读取延时，提升硬件执行效率。

但 $MobileNetV2$ 中的倒残差结构可能会造成优化过程中梯度回传抖动，进而影响模型收敛趋势，导致模型性能降低。现有的研究表明：
* 更宽的网络可以缓解梯度混淆问题并有助于提升模型性能；
* 倒残差模块中的短连接可能会影响梯度回传；

## 改进
为了解决 $MobileNetV2$ 中瓶颈结构导致的优化问题，依图团队重新思考了由 $ResNet$ 提出的传统瓶颈结构的链接方式，这种连接方式把梯度主要集中在较高维度的网络层，可以减少梯度抖动、加速网络收敛。

通过上述分析依图团队提出了一种新的网络模块 $Sandglass \ Block$，既能保留高维度网络加速收敛和训练的优势，又能利用深度卷积带来的计算开销收益，减少高维特征图的内存访问需求，提高硬件执行效率。 

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNeXt/Fig2.jpg?raw=true"
    width="720" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Fig 2 </div>
</center>

依图团队把短链接放置在高维度神经网络层，并使用深度卷积来降低计算开销，然后使用两连续层 $1 \times 1$ 卷积来进一步降低计算开销的效果。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNeXt/Fig3.jpg?raw=true"
    width="720" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Fig 3 (Sandglass Block) </div>
</center>

$Sandglass \ Block$ 可以保证更多的信息从 $bottom$ 层传递给 $top$ 层，进而有助于梯度回传；执行了两次深度卷积以编码更多的空间信息。

### 降维和升维的位置
$MobileNetV2$ 的倒残差模块中先进行升维再进行降维，$MobileNeXt$ 中为确保高维度特征的短连接，依图团队对两个 $1 \times 1$ 卷积的顺序进行了调整，将两个 $1 \times 1$ 卷积放到了两个深度卷积的中间，如图 $Fig3$ 所示。

### 高维度短连接
依图团队没有在瓶颈层（$block$ 和 $block$ 连接处）间构建短连接，而是在更高维特征之间构建短连接。更宽的短连接有助于更多信息从输入传递给输出，从而有更多的梯度回传，如图 $Fig3$ 所示。

> 此时，会出现一个问题，高维度的短链接会导致更多的点加操作、需求更多的内存读取访问，直接连接高维度短链接会降低硬件执行效率。

为了解决该问题，依图团队提出只使用一部分信息通道进行跳跃链接，这一操作可直接减少点加操作和特征图大小，进而直接提升硬件执行效率。实验结果显示，仅使用一半的信息通道进行跳跃链接不会造成精度损失。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNeXt/short _ link.png?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> short link </div>
</center>


### 引入更丰富的空间信息
倒残差模块的深度卷积在两个 $1 \times 1$ 卷积之间，而 $1 \times 1$ 卷积会降低空间信息编码，因此依图团队将深度卷积置于两个 $1 \times 1$ 卷积之外，确保深度卷积在高维空间得到处理并获得更丰富的特征表达。

### 激活函数的位置
线性瓶颈层有助于避免特征出现零化现象，从而导致信息损失。因此，在降维的 $1 \times 1$ 卷积后不再添加激活函数。同时最后一个深度卷积后也不添加激活函数，激活函数只添加在第一个深度卷积与最后一个 $1 \times 1$ 卷积之后。
 
## 结构
### Sandglass Block 结构
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNeXt/Table1.jpg?raw=true"
    width="720" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Table 1 </div>
</center>

### 网络结构
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNeXt/Table2.jpg?raw=true"
    width="720" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Table 2 </div>
</center>

## 实验结果
### 分类
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNeXt/Table3.jpg?raw=true"
    width="380" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Table 3 </div>
</center>

### 检测
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNeXt/Table9.jpg?raw=true"
    width="720" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Table 9 </div>
</center>

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNeXt/Table10.jpg?raw=true"
    width="720" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Table 10 </div>
</center>

### 神经网络搜索

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/MobileNeXt/Table13.jpg?raw=true"
    width="720" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Table 13 </div>
</center>

## 参考
[Rethinking Bottleneck Structure for Efficient Mobile Network Design](https://arxiv.org/abs/2007.02269)

[超越MobileNet！依图提出新一代移动端网络MobileNeXt ，代码刚刚开源！](https://mp.weixin.qq.com/s?__biz=MzU4OTg3Nzc3MA==&mid=2247484066&idx=1&sn=0a3c8992086480242331a02773e64761&chksm=fdc785c7cab00cd121961e0199584404a8e3531d129f86e04414df92efe44c47ef0ef4a253d5&mpshare=1&scene=1&srcid=1123ouJjaRPAhWWjNqopk3dW&sharer_sharetime=1606119590105&sharer_shareid=fd37339e47e28142caa0dc8462230525&version=3.0.36.2330&platform=mac#rd)

[依图科技MobileNeXt ---一种新颖的移动端卷积神经网络](https://www.jianshu.com/p/22860f4ce793)

[yitu-opensource/MobileNeXt](https://github.com/yitu-opensource/MobileNeXt)

[Andrew-Qibin/ssdlite-pytorch-mobilenext](https://github.com/Andrew-Qibin/ssdlite-pytorch-mobilenext)