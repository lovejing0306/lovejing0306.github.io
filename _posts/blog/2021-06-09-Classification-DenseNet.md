---
layout: post
title: DenseNet
categories: [Classification]
description: DenseNet
keywords: Classification
---


分类模型 DenseNet
---


## 简介
&emsp;&emsp;$ResNet$ 模型的核心是通过建立前面层与后面层之间的 “短路连接”（$shortcuts \ skip \ connection$），这有助于训练过程中梯度的反向传播，从而能训练出更深的 $CNN$ 网络。

&emsp;&emsp;$DenseNet$ 模型的基本思路与 $ResNet$ 一致，但是 $DenseNet$ 建立的是前面所有层与后面层的密集连接（$dense \ connection$）。$DenseNet$ 的另一大特色是通过 $featrue \
map$ 在 $channel$ 上的合并来实现特征重用（$feature reuse$）。这些特点让 $DenseNet$ 在参数和计算成本更少的情形下实现比 $ResNet$ 更优的性能，$DenseNet$ 也因此斩获 $CVPR 2017$ 的最佳论文奖。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/DenseNet/densenet2.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图1 ResNet网络的短路连接机制（其中+代表的是元素级相加操作）</div>
</center>

## 网络结果
### 基本结构
&emsp;&emsp;$DenseNet$ 的网络结构主要由 $DenseBlock$ 和 $Transition$ 组成，如下图所示：

![densenet3](https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/DenseNet/densenet3.jpg?raw=true)

### 非ImageNet数据集
* 使用 $3$ 个 $Dense \ Block$
* 每个 $Block$ 都有相同的层数
* 模型为 $DenseNet$，配置为 $\left\{ L=40,k=12 \right\} $，$\left\{ L=100,k=12 \right\} $，$\left\{ L=100,k=24 \right\} $，其中，$L$ 为层的总数，$k$ 为每层输出的通道数
* 模型为 $DenseNet-BC$，配置为 $\left\{ L=100,k=12 \right\}$，$\left\{ L=250,k=24 \right\} $，$\left\{ L=190,k=40 \right\} $
* 在送入第一个 $Dense \ Block$ 前，会先送到一个 $16$ 通道的卷积层
* 使用 $3 \times 3$ 的小卷积，采用 $zero-padding$ 保持 $feature \ map$ 尺寸
* 最后一个 $Dense \ Block$ 后接一个 $global \ average \ pooling$，再跟 $softmax$ 分类。

### ImageNet数据集
* 使用的是 $DenseNet-BC$
* 使用 $4$ 个 $Dense Block$
* 在送入第一个 $Dense Block$ 前，会先送到一个 $7×7×2k$ 的 $stride=2$ 的卷积层
* 所有的 $layers$ 的 $feature \ map$ 都设置为 $k$

&emsp;&emsp;在 $ImageNet$ 上，具体的 $DenseNet-BC$ 如下图：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/DenseNet/densenet1.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图1 ResNet网络的短路连接机制（其中+代表的是元素级相加操作）</div>
</center>

## 网络技巧
### Dense Connection
&emsp;&emsp;相比 $ResNet$，$DenseNet$ 提出了一个更激进的密集连接机制：即互相连接所有的层，具体来说就是每个层都会接受其前面所有层作为其额外的输入。图 $1$ 为 $ResNet$ 网络的连接机制，作为对比，图 $2$ 为 $DenseNet$ 的密集连接机制。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/DenseNet/densenet4.jpeg?raw=true"
    width="512" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图1 ResNet网络的短路连接机制（其中+代表的是元素级相加操作）</div>
</center>

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/DenseNet/densenet5.jpeg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图2 DenseNet网络的密集连接机制（其中c代表的是channel级连接操作）</div>
</center>

可以看到，$ResNet$ 是每个层与前面的某些层（一般是2-3层）短路连接在一起，连接方式是通过元素级相加。而在 $DenseNet$ 中，每个层都会与前面所有层 在 $channel$ 维度上合并（$concat$）在一起（这里各个层的特征图大小是相同的），并作为下一层的输入。


&emsp;&emsp;对于一个 $L$ 层的网络，$DenseNet$ 共包含 $\frac { L\left( L+1 \right)  }{ 2 }$ 个连接，相比 $ResNet$，这是一种密集连接。而且 $DenseNet$ 是直接 $concat$ 来自不同层的特征图，这可以实现特征重用，提升效率，这一特点是 $DenseNet$ 与 $ResNet$ 最主要的区别。

&emsp;&emsp;使用公式表示，传统的网络在 $l$ 层的输出为：

$$
{ x }_ { l }={ H }_ { l }\left( { x }_ { l-1 } \right) 
$$

$ResNet$ 增加了来自上一层输入的 $identity$ 函数：

$$
{ x }_ { l }={ H }_ { l }\left( { x }_ { l-1 } \right) +{ x }_ { l-1 }
$$

$DenseNet$ 会连接前面所有层作为输入：

$$
{ x }_ { l }={ H }_ { l }\left( \left[ { x }_ { 0 },{ x }_ { 1 },...,{ x }_ { l-1 } \right]  \right) 
$$

其中，${ H }  _  { l }\left( \cdot  \right) $ 代表是非线性转化函数，它是一个组合操作，其可能包括一系列的 $BN$($Batch Normalization$)，$ReLU$，$Pooling$ 及 $Conv$ 操作。注意这里 $l$ 层与 $l-1$ 层之间可能实际上包含多个卷积层。

&emsp;&emsp;为了更直观地理解其密集连接方式，下面给出 $DenseNet$ 的前向过程：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/DenseNet/densenet6.png?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">..</div>
</center>

从上图可以看出，${ h } _ { 3 }$ 的输入不仅包括来自 ${ h }  _  { 2 }$ 的 ${ x }  _  { 2 }$，还包括前面两层的 ${ x }  _  { 1 }$ 和$ { x }  _  { 2 }$，它们是在 $channel$ 维度上连接在一起的。


&emsp;&emsp;$DenseNet$ 的密集连接方式需要特征图大小保持一致。为了解决这个问题，$DenseNet$ 网络中使用 $DenseBlock+Transition$ 的结构，其中 $DenseBlock$ 是包含很多层的模块，每个层的特征图大小相同，层与层之间采用密集连接方式。而 $Transition$ 模块是连接两个相邻的 $DenseBlock$，并且通过 $Pooling$ 使特征图大小降低。下面给出 $DenseNet$ 的网络结构，它共包含 $4$ 个 $DenseBlock$，各个 $DenseBlock$ 之间通过 $Transition$ 连接在一起。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/DenseNet/densenet7.png?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">..</div>
</center>

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/DenseNet/densenet8.png?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">..</div>
</center>

### DenseBlock
&emsp;&emsp;在 $DenseBlock$ 中，各个层的特征图大小一致，可以在 $channel$ 维度上连接。$DenseBlock$ 中的非线性组合函数 $H\left( \cdot  \right) $ 采用的是 $BN + ReLU + 3\times3 Conv$ 的结构，如下图所示：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/DenseNet/densenet9.png?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">..</div>
</center>

与 $ResNet$ 不同，所有 $DenseBlock$ 中各个层卷积之后均输出 $k$ 个特征图，即得到的特征图的 $channel$ 数为 $k$，或者说采用 $k$ 个卷积核。$k$ 在 $DenseNet$ 称为 $growth \ rate$，是一个超参数。一般情况下使用较小的 $k$（比如 $12$），就可以得到较佳的性能。假定输入层的特征图的 $channel$ 数为 ${ k }  _  { 0 }$，那么 $l$ 层输入的 $channel$ 数为 ${ k }  _  { 0 }+k\left( l-1 \right)$，因此随着层数增加，尽管 $k$设 定得较小，$DenseBlock$ 的输入会非常多，不过这是由于特征重用所造成的，每个层仅有 $k$ 个特征是自己独有的。

&emsp;&emsp;由于后面层的输入会非常大，$DenseBlock$ 内部可以采用 $bottleneck$ 层来减少计算量，主要是在原有的结构中增加 $1 \times 1$ $Conv$，即 $BN+ReLU+1\times1 \ Conv+BN+ReLU+3\times3 \ Conv$，称为 $DenseNet-B$ 结构，如下图所示：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/DenseNet/densenet10.png?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">..</div>
</center>

其中 $1 \times 1$ $Conv$ 得到 $4k$ 个特征图它起到的作用是降低特征数量，从而提升计算效率。

### Transition
&emsp;&emsp;$Transition$ 层主要是连接两个相邻的 $DenseBlock$，并且降低特征图大小。$Transition$ 层包括一个 $1\times1$ 的卷积和 $2\times2$ 的 $AvgPooling$，结构为 $BN+ReLU+1\times1 \ Conv+2\times2 \AvgPooling$。此外，$Transition$ 层可以起到压缩模型的作用。假定 $Transition$ 的上接 $DenseBlock$ 得到的特征图 $channels$ 数为 $m$，$Transition$ 层可以产生 $\left\lfloor \theta m \right\rfloor$ 个特征（通过卷积层），其中 $\theta \in (0,1]$ 是压缩系数（$compression rate$）。当 $\theta =1$ 时，特征个数经过 $Transition$ 层没有变化，即无压缩，而当压缩系数小于 $1$ 时，这种结构称为 &DenseNet-C$，文中使用 $\theta = 0.5$。对于使用 $bottleneck$ 层的 $DenseBlock$ 结构和压缩系数小于 $1$ 的 $Transition$ 组合结构称为 $DenseNet-BC$。

# 优点
* 减轻了梯度消失问题
* 加强了 $feature$ 的传递 
* 更有效地利用了 $feature$
* 一定程度上较少了参数数量

# 注意
&emsp;&emsp;如果实现方式不当的话，$DenseNet$ 可能耗费很多 $GPU$ 显存，一种高效的实现如下图所示：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/DenseNet/densenet11.png?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">..</div>
</center>
