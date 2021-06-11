---
layout: post
title: ResNet
categories: [Classification]
description: ResNet
keywords: Classification
---


分类模型 ResNet
---


## 背景
&emsp;&emsp;$2015$ 年，微软亚洲研究院的何凯明等人使用残差网络 $ResNet[4]$ 参加了当年的 $ILSVRC$，在图像分类、目标检测等任务中的表现大幅超越前一年的比赛的性能水准，并最终取得冠军。

&emsp;&emsp;残差网络的明显特征是有着相当深的深度，从 $32$ 层到 $152$ 层，其深度远远超过了之前提出的深度网络结构，而后又针对小数据设计了 $1001$ 层的网络结构。残差网络 $ResNet$ 的深度惊人，极其深的深度使得网络拥有极强的表达能力。

## 网络结构

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/ResNet/resnet1.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">resnet1</div>
</center>

&emsp;&emsp;简单网络的基准（上图中间）主要受到 $VGG$ 网络（上图左边）的启发。卷积层主要有 $3 \times 3$ 的滤波器，并遵循两个简单的设计规则：
1. 对于相同的输出特征图尺寸，层具有相同数量的滤波器；
2. 如果特征图尺寸减半，则滤波器数量加倍，以便保持每层的时间复杂度。

通过步长为 $2$ 的卷积层直接执行下采样。网络以全局平均池化层和具有 $softmax$ 的 $1000$ 维全连接层结束。上图中间的 $resnet$ 加权层总数为 $34$。

&emsp;&emsp;值得注意的是与 $VGG$ 网络（上图左边）相比，$34$ 层的 $resent$ 有更少的滤波器和更低的复杂度。$34$ 层基准有 $36$ 亿 $FLOP$ (乘加)，仅是 $VGG-19$（$196$ 亿 $FLOP$）的 $18$%。

&emsp;&emsp;残差网络。 基于上述的简单网络，插入快捷连接（$shortcut \ connections$）（上图右边），将网络转换为其对应的残差版本。当输入和输出具有相同的维度时（上图中的实线快捷连接）时，可以直接使用恒等快捷连接（方程$(1)$）。当维度增加（图 $3$ 中的虚线快捷连接）时，考虑两个选项：
1. 快捷连接仍然执行恒等映射，额外填充零输入以增加维度。此选项不会引入额外的参数；
2. 方程 $y=F\left( x,{ W }  _  { i } \right) +{ W }  _  { s }x$ 中的投影快捷连接用于匹配维度（由 $1 \times 1$ 卷积完成）。

&emsp;&emsp;对于这两个选项，当快捷连接跨越两种尺寸的特征图时，它们执行时步长为 $2$。

&emsp;&emsp;下图中展示了更多的细节和其他变种：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/ResNet/resnet2.jpg?raw=true"
    width="640" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">resnet2</div>
</center>

## 网络技巧
### residual block
&emsp;&emsp;当更深的网络能够开始收敛时，暴露了一个退化问题：随着网络深度的增加，准确率达到饱和（这可能并不奇怪）然后迅速下降。意外的是，这种下降不是由过拟合引起的，并且在适当的深度模型上添加更多的层会导致更高的训练误差。下图显示了一个典型的例子：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/ResNet/resnet3.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">error</div>
</center>

上图显示了 $20$ 层和 $56$ 层的“简单”网络在 $CIFAR-10$ 上的训练误差（左）和测试误差（右）。更深的网络有更高的训练误差和测试误差。$ImageNet$ 上的类似现象如下图所示：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/ResNet/resnet4.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">error</div>
</center>

&emsp;&emsp;为了解决网络退化问题，$resnet$ 引入了深度残差学习框架。$resnet$ 明确地让这些层拟合残差映射，而不是希望每几个堆叠的层直接拟合期望的基础映射。形式上，将期望的基础映射表示为 $H(x)$，将堆叠的非线性层拟合另一个映射 $F\left( x \right) =H\left( x \right) -x$。原始的映射重写为 $F(x)+x$。$resnet$ 假设残差映射比原始的、未参考的映射更容易优化。在极端情况下，如果一个恒等映射是最优的，那么将残差置为零比通过一堆非线性层来拟合恒等映射更容易。

&emsp;&emsp;公式 $F(x)+x$ 可以通过带有“快捷连接”的前向神经网络来实现，如下图所示：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/ResNet/resnet5.jpg?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">resnet5</div>
</center>

$resnet$ 中快捷连接简单地执行恒等映射，并将其输出添加到堆叠层的输出。恒等快捷连接既不增加额外的参数也不增加计算复杂度。整个网络仍然可以由带有反向传播的 $SGD$ 进行端到端的训练。

### Identity Mapping by Shortcuts (快捷恒等映射)
&emsp;&emsp;残差块的正式定义为：
$$
y=F\left( x,{ W }_ { i } \right) +x \quad (1)
$$

$x$ 和 $y$ 是层的输入和输出向量。$F\left( x,{ W }  _  { i } \right)$ 表示要学习的残差映射。上图中的例子有两层，$F={ W }  _  { 2 }\sigma \left( { W }  _  { 1 }x \right) $ 中 $\sigma$ 表示 $ReLU$，为了简化写法忽略偏置项。$F+x$ 操作通过快捷连接和各个元素相加来执行。在相加之后我们采纳了第二种非线性（即 $\sigma \left( y \right) $）。

&emsp;&emsp;方程 $(1)$ 中的快捷连接既没有引入外部参数又没有增加计算复杂度。这不仅在实践中有吸引力，而且在简单网络和残差网络的比较中也很重要。这样可以公平地比较同时具有相同数量的参数，相同深度，宽度和计算成本的简单/残差网络（除了不可忽略的元素加法之外）。

&emsp;&emsp;方程 $(1)$ 中 $x$ 和 $F$ 的维度必须是相等的。如果不是这种情况（例如，当更改输入/输出通道时），可以通过快捷连接执行线性投影 ${ W }  _  { s }$ 来匹配维度：

$$
y=F\left( x,{ W }_{ i } \right) +{ W }_{ s }x \quad (2)
$$

&emsp;&emsp;残差函数 $F$ 的形式是可变的。本文中的实验包括有两层或三层的函数 $F$:

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/ResNet/resnet6.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">resnet6</div>
</center>

同时可能有更多的层。但如果 $F$ 只有一层，方程 $(1)$ 类似于线性层：$y={ W }  _  { 1 }x + x$，没有看到优势。

## 网络特点
* 网络较瘦，控制了参数数量；
* 存在明显层级，特征图个数逐层递进，保证输出特征表达能力；
* 使用了较少的池化层，大量使用下采样，提高传播效率；
* 没有使用 $Dropout$，利用$BN$和全局平均池化进行正则化，加快了训练速度；
* 层数较高时减少了 $3 \times 3$ 卷积个数，并用 $1 \times 1$ 卷积控制了 $3 \times 3$ 卷积的输入输出特征图数量，称这种结构为“瓶颈”($bottleneck$)。

## 网路改进
&emsp;&emsp;原始残差网络由很多“残差单元”组成。每一个单元( $Fig.1(a)$ )可以表示为： 

$$
\begin{aligned}
&{ y }_ { l }=h\left( { x }_ { l } \right) +F\left( { x }_ { l },{ W }_ { l } \right) \quad (3)\\ &{ x }_ { l+1 }=f\left( { y }_ { l } \right) \quad (4)
\end{aligned}
$$

其中，${ x }  _  { l }$ 和 ${ x }  _  { l+1 }$ 是第 $l$ 个单元的输入和输出，$F$ 表示一个残差函数。在 $He2015$ 中， $h\left( { x }  _  { l } \right) ={ x }  _  { l }$ 代表一个恒等映射，$f$ 代表 $ReLU$。 

&emsp;&emsp;在“$Identity \ Mappings \ in \ Deep \ Residual \ Networks$”文章中，作者提出了新的网络结构，和原有结构比较如下：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/ResNet/resnet7.jpg?raw=true"
    width="256" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">resnet7</div>
</center>

作者提出，将 $BN$层 和 $ReLU$ 看做是后面带参数的卷积层的 “前激活”（$pre-activation$），取代原先的“后激活”（$post-activation$）。这样就得到了上图右侧的新结构。

&emsp;&emsp;从上图右可以看到，本单元的输入 ${ x }  _  { l }$ 首先经过了$BN$层和$ReLU$层的处理，然后才通过卷积层，之后又是 $BN-ReLU-conv$ 的结构。这些处理过后得到的 ${ y }  _  { l }$，直接和 ${ x }  _  { l }$ 相加，得到该层的输出 ${ x }  _  { l+1 }$。

&emsp;&emsp;利用这种新的残差单元结构，作者构建了 $1001$ 层的残差网络，在 $CIFAR-10/100$ 上进行了测试，验证了新结构能够更容易地训练（收敛速度快），并且拥有更好的泛化能力（测试集合 $error$ 低）。

## ~~Identity Mapping~~
&emsp;&emsp;如果 $f$ 也是一个恒等映射：${ x }  _  { l+1 }={ y }  _  { l }$，可以将公式 $(4)$ 代入公式 $(3)$ 得到公式 $(5)$：

$$
{ x }_ { l+1 }={ x }_ { l }+F\left( { x }_ { l },{ W }_ { l } \right) \quad (5)
$$

通过递归，对于任意深的单元 $L$ 和任意浅的单元 $l$，可以得到公式 $(6)$：

$$
{ x }_ { L }={ x }_ { l }+\sum _ { i=l }^{ L-1 }{ F\left( { x }_ { i },{ W }_ { i } \right)  } \quad (6)
$$

&emsp;&emsp;公式 $(6)$ 展现了一些良好的特性：
1. 对于任意深的单元 $L$ 的特征 ${ x }  _  { L }$ 可以表达为浅层单元 $l$ 的特征 ${ x }  _  { l }$ 加上一个形如 $\sum   _  { i=l }^{ L-1 }{ F }$ 的残差函数，这表明了任意单元 $L$ 和 $l$ 之间都具有残差特性。
2. 对于任意深的单元 $L$，它的特征 ${ x }  _  { L }={ x }  _  { 0 }+\sum   _  { i=0 }^{ L-1 }{ F\left( { x }  _  { i },{ W }  _  { i } \right)  } $，即为之前所有残差函数输出的总和(加上 ${ x }  _  { 0 }$ )。而正好相反的是，“plain network”中的特征 ${ x }  _  { L }$ 是一系列矩阵向量的乘积，也就是 $\prod   _  { i=0 }^{ L-1 }{ { W }  _  { i }{ x }  _  { 0 } } $ (忽略了 $BN$ 和 $ReLU$ )。

&emsp;&emsp;公式 $(6)$ 也具有良好的反向传播特性。 假设损失函数为 $\varepsilon$，从反向传播的链式法则可以得到公式 $(7)$：

$$
\frac { \partial \varepsilon  }{ \partial { x }_ { l } } =\frac { \partial \varepsilon  }{ \partial { x }_ { L } } \frac { \partial { x }_ { L } }{ \partial { x }_ { l } } =\frac { \partial \varepsilon  }{ \partial { x }_ { L } } \left( 1+\frac { \partial \sum _ { i=l }^{ L-1 }{ F\left( { x }_ { l },{ W }_ { l } \right)  }  }{ \partial { x }_ { l } }  \right) \quad (7)
$$

公式 $(7)$ 表明了梯度 $\frac { \partial \varepsilon  }{ \partial { x }  _  { l } } $ 可以被分解成两个部分：其中 $\frac { \partial \varepsilon  }{ \partial { x }  _  { L } } $ 直接传递信息而不涉及任何权重层，而另一部分 $\frac { \partial \varepsilon  }{ \partial { x }  _  { L } } \frac { \partial \sum   _  { i=l }^{ L-1 }{ F\left( { x }  _  { l },{ W }  _  { l } \right)  }  }{ \partial { x }  _  { l } } $ 表示通过权重层的传递。$\frac { \partial \varepsilon  }{ \partial { x }  _  { l } } $ 可以被分解成两个部分：其中 $\frac { \partial \varepsilon  }{ \partial { x }  _  { L } } $ 保证了信息能够直接传回任意浅层 $l$。

&emsp;&emsp;公式 $(7)$ 同样表明了在一个 $mini-batch$ 中梯度 $\frac { \partial \varepsilon  }{ \partial { x }  _  { L } } $ 不可能出现消失的情况，因为通常 $\frac { \partial \sum   _  { i=l }^{ L-1 }{ F\left( { x }  _  { l },{ W }  _  { l } \right)  }  }{ \partial { x }  _  { l } } $ 对于一个 $mini-batch$ 总的全部样本不可能都为 $-1$。这意味着，哪怕权重是任意小的，也不可能出现梯度消失的情况。

&emsp;&emsp;公式 $(6)$ 和公式 $(7)$ 表明了，在前向和反向阶段，信号都能够直接的从一个单元传递到其他任意一个单元。公式 $(6)$ 的条件基础是两个恒等映射：
1. 恒等跳跃连接 $h\left( { x }  _  { l } \right) ={ x }  _  { l }$
2. $f$ 也是一个恒等映射。

## ~~Identity Mapping 的重要性~~
&emsp;&emsp;设计一个简单的修改，使用 $h\left( { x }  _  { l } \right) ={ \lambda  }  _  { l }{ x }  _  { l }$ 来替代恒等捷径,如公式 $(8)$：

$$
{ x }_ { l+1 }={ \lambda  }_ { l }{ x }_ { l }+F\left( { x }_ { l },{ W }_ { l } \right) 
$$

其中 ${ \lambda  }  _  { l }$ 是一个调节标量(简单起见，仍然假设f是恒等映射)。通过方程的递归，可以得到类似于公式 $(8)$ 的等式：

$$
{ x }_{ L }=\left( \prod _{ i=l }^{ L-1 }{ { \lambda  }_{ i } }  \right) { x }_{ l }+\sum _{ i=l }^{ L-1 }{ \left( \prod _{ j=i+1 }^{ L-1 }{ { \lambda  }_{ i } }  \right)  } F\left( { x }_{ i },{ W }_{ i } \right) 
$$

或者公式 $(9)$：

$$
{ x }_ { L }=\left( \prod _{ i=l }^{ L-1 }{ { \lambda  }_ { i } }  \right) { x }_ { l }+\sum _ { i=l }^{ L-1 }{ \hat { F } \left( { x }_ { i },{ W }_ { i } \right)  } 
$$

&emsp;&emsp;类似于公式 $(7)$，有以下形式的反向传播过程，公式 $(10)$：

$$
\frac { \partial \varepsilon  }{ \partial { x }_{ l } } =\frac { \partial \varepsilon  }{ \partial { x }_{ L } } \left( \prod _{ i=l }^{ L-1 }{ { \lambda  }_{ i } } +\frac { \partial \sum _{ i=l }^{ L-1 }{ \hat { F } \left( { x }_{ i },{ W }_{ i } \right)  }  }{ \partial { x }_{ l } }  \right) 
$$

在公式 $(10)$ 中，第一项由因子 $\prod   _  { i=l }^{ L-1 }{ { \lambda  }  _  { i } } $ 进行调节。对于一个极深的网络( $L$ 很大)，如果对于所有的 $i$ 都有 ${ \lambda  }  _  { i }>1$，那么这个因子将会是指数型的放大，此时将造成梯度爆炸；如果 ${ \lambda  }  _  { i }<1$ ，那么这个因子将会是指数型的缩小或者是消失，从而阻断从捷径反向传来的信号，并迫使它流向权重层，此时将造成梯度消失。以上情况都会对优化造成困难。