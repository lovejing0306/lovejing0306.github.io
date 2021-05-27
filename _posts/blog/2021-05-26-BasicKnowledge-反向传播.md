---
layout: post
title: 反向传播
categories: [BasicKnowledge]
description: 反向传播
keywords: BasicKnowledge
---


深度学习基础知识点反向传播
---


## 简介
&emsp;&emsp;反向传播的核心是 代价函数 $C$ 关于任何权重 $w$ 或者偏置 $b$ 的偏导数 $\frac { \partial C }{ \partial w }$，这个表达式告诉我们在改变权重和偏置时，代价函数变化的快慢。实际上它解释了如何通过改变权重和偏置来改变整个网络的行为。因此，这也是学习反向传播细节的重要价值所在。

## 使用矩阵快速计算输出
&emsp;&emsp;首先给出网络中权重的定义：使用 ${ w }  _  { jk }^{ l }$ 表示从 ${ \left( l-1 \right)  }^{ th }$ 层的 ${ k }^{ th }$ 个神经元到 ${ \left( l \right)  }^{ th }$ 层的 ${ j }^{ th }$ 个神经元链接上的权重。例如，下图给出了网络中第二层的第四个神经元到第三层的 第二个神经元的链接上的权重:

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Backpropagation/bp1.png?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">bp1</div>
</center>

这样的表示一开始看起来感觉比较奇怪，需要花点去时间消化。但是，到后面会发现这样的表示会比较方便并且也很自然。

&emsp;&emsp;对网络的偏置和激活值也会使用类似的表示，使用 ${ b }  _  { j }^{ l }$ 表示在 ${ l }^{ th }$ 层第 ${ j}^{ th }$ 个神经元的偏置，使用 ${ a }  _  { j }^{ l }$ 表示 ${ l }^{ th }$ 层第 ${ j}^{ th }$ 个神经元的激活值。下图给出了这样表示的含义:

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Backpropagation/bp2.png?raw=true"
    width="256" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">bp2</div>
</center>

有了这些表示，${ l }^{ th }$ 层的 ${ j}^{ th }$ 个神经元的激活值 ${ a }  _  { j }^{ l }$ 就和 ${ l-1 }^{ th }$ 层的激活值 ${ a }^{ l-1 }$ 通过方程关联起来：

$$
{ a }_ { j }^{ l }=\sigma \left( \sum _ { k }^{  }{ { w }_ { jk }^{ l } } { a }_ { k }^{ l-1 }\quad +\quad { b }_ { j }^{ l } \right)  (1)
$$

其中求和是在 ${ l-1 }^{ th }$ 层的所有 $k$ 个神经元上进行。为了方便用矩阵表示，我们对每一层 $l$ 都定义一个权重矩阵 ${ w }^{ l }$。权重矩阵 ${ w }^{ l }$ 的元素是连接到 ${ l }^{ th }$ 层神经元的权重，确切地说，在第 ${ j}^{ th }$ 行第 ${ k}^{ th }$ 列的元素是 ${ w }  _  { jk }^{ l }$。类似的，对每一层$l$都定义一个偏置向量 ${ b }^{ l }$，偏置向量的每个元素就是前面给出的 ${ b }  _  { j }^{ l }$，每个元素对应于 ${ l }^{ th }$ 层的每个神经元。最后，定义激活向量 ${ a }^{ l }$，其元素是激活值 ${ a }  _  { j }^{ l }$。

&emsp;&emsp;了解了上述表示后可以将公式$(1)$写成下面简约的表达形式：
$$
{ a }^{ l }=\sigma \left( { w }^{ l }{ a }^{ l-1 }+{ b }^{ l } \right) (2)
$$

在使用公式 $(2)$ 计算 ${ a }^{ l }$ 的过程中，计算了中间量 ${ z }^{ l }={ w }^{ l }{ a }^{ l-1 }+{ b }^{ l }$，我们称 ${ z }^{ l }$ 为 ${ l }^{ th }$ 层神经元的带权输入。公式 $(2)$ 有时候会以带权输入的形式写作 ${ a }^{ l }=\sigma \left( { z }^{ l } \right)$。同时要指出的是 ${ z }^{ l }$ 的每个元素是 ${ z }  _  { j }^{ l }=\sum   _  { k }{ { w }  _  { jk }^{ l } } { a }  _  { k }^{ l-1 }\quad +\quad { b }  _  { j }^{ l }$，其实 ${ z }  _  { j }^{ l }$ 就是 ${ l}^{ th }$ 层 ${ j}^{ th }$ 个神经元的激活函数的带权输入。

## 关于代价函数的两个假设
&emsp;&emsp;反向传播的目标是计算代价函数 $C$ 关于 $w$ 和 $b$ 的偏导数 $\frac { \partial C }{ \partial w } $ 和$\frac { \partial C }{ \partial b }$。为了让反向传播可行，需要两个主要假设。在给出假设之前，先看一个具体的代价函数，二次代价函数,该函数有下列形式:

$$
C=\frac { 1 }{ 2n } \sum _ { x }{ { \left\| y\left( x \right) -{ a }^{ L }\left( x \right)  \right\|  }^{ 2 } } (3)
$$

其中，$n$ 是训练样本的总数；求和运算遍历了每个训练样本 $x$；$y = y(x)$ 是对应的目标输出；$L$ 表示网络的层数；${ { a }^{ L }=a }^{ L }\left( x \right)$ 是当输入$x$时网络输出的激活值向量。

&emsp;&emsp;第⼀个假设是代价函数 $C$ 可以被写成在每个训练样本 $x$ 上代价函数 ${ C }  _  { x }$ 的均值，表示形式为 $C=\frac { 1 }{ n } { C }  _  { x }$，对每个独⽴的训练样本其代价是 ${ C }  _  { x }={ \left\| { y }  _  { x }-{ a }  _  { x }^{ L } \right\|  }^{ 2 }$。

&emsp;&emsp;需要这个假设是由于反向传播实际上是对⼀个独⽴的训练样本计算了 $\frac { \partial { C }  _  { x } }{ \partial w } $和$\frac { \partial { C }  _  { x } }{ \partial b }$。 然后在所有训练样本上求均值获取 $\frac { \partial { C } }{ \partial w }$ 和 $\frac { \partial { C } }{ \partial b }$。实际上，有了这个假设，我们会认为训练样本 $x$ 已经被固定，丢掉其下标，将代价函数 ${ C }  _  { x }$ 看做 $C$。最终我们会把下标加上，现在只是为了简化表⽰。

&emsp;&emsp;第⼆个假设是代价函数可以写成关于神经⽹络输出的函数：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Backpropagation/bp3.png?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">bp3</div>
</center>

例如，⼆次代价函数满⾜这个要求，因为对于⼀个单独的训练样本$x$其⼆次代价函数可以写作：
$$
C=\frac { 1 }{ 2 } { \left\| y-{ a }^{ L } \right\|  }^{ 2 }=\frac { 1 }{ 2 } \sum _ { j }{ { \left( { y }_ { j }-{ a }_ { j }^{ L } \right)  }^{ 2 } } (4)
$$

这是输出层的关于激活值的函数，将 $C$ 看成仅有输出激活值 ${ a }^{ L }$ 的函数。

## Hadamard 乘积
&emsp;&emsp;假设 $s$ 和 $t$ 是两个同样维度的向量。那么我们使⽤ $s\odot t$ 来表⽰按元素的乘积。所以 $s\odot t$ 的元素就是 ${ \left( s\odot t \right)  }  _  { j }={ s }  _  { j }{ t }  _  { j }$。举例来说，

$$
\left[ \begin{matrix} 1 \\ 2 \end{matrix} \right] \odot \left[ \begin{matrix} 3 \\ 4 \end{matrix} \right] =\left[ \begin{matrix} 1\ast 3 \\ 2\ast 4 \end{matrix} \right] =\left[ \begin{matrix} 3 \\ 8 \end{matrix} \right] (5)
$$

这种按元素的乘法有时候被称为 $Hadamard$ 乘积，或者 $Schur$ 乘积。

## 四个基本⽅程
&emsp;&emsp;反向传播其实就是计算偏导数 $\frac { \partial C }{ \partial { w }  _  { jk } } $和$\frac { \partial C }{ \partial b  _  { j } } $。为了计算这些值，我们引⼊⼀个 中间量 $\delta   _  { j }^{ l }$ ，称其为在 ${ l }^{ th }$ 层 ${ j }^{ th }$ 个神经元上的误差(度量)。
反向传播给出计算误差 $\delta   _  { j }^{ l }$ 的流程，然后将其关联到计算 $\frac { \partial C }{ \partial { w }  _  { jk } } $ 和 $\frac { \partial C }{ \partial b  _  { j } } $ 上。定义 ${ l }^{ th }$ 层的第 ${ j }^{ th }$ 个神经元上的误差 $\delta   _  { j }^{ l }$ 为：

$$
{ \delta  }_{ j }^{ l }=\frac { \partial C }{ \partial { z }_{ j }^{ l } } (6)
$$

&emsp;&emsp;使⽤ ${ \delta  }^{ l }$ 表⽰ ${ l }^{ th }$ 层的误差向量。反向传播会提供⼀种计算每层的 ${ \delta  }^{ l }$ 的⽅法，然后将这些误差与最终我们需要的量 $\frac { \partial C }{ \partial { w }  _  { jk } }$ 和 $\frac { \partial C }{ \partial b  _  { j } } $ 联系起来。

&emsp;&emsp;输出层误差的⽅程，${ \delta  }^{ L }$ ： 每个元素定义如下：

$$
{ \delta  }_ { j }^{ L }=\frac { \partial C }{ \partial { a }_{ j }^{ L } } \sigma \left( { z }_{ j }^{ L } \right) (BP1)
$$

其中，$\frac { \partial C }{ \partial { a }  _  { j }^{ L } }$ 表⽰代价随着 ${ j }^{ th }$ 个输出激活值的变化⽽改变的速度。$\sigma \left( { z }  _  { j }^{ L } \right)$ 刻画了在 ${ z }  _  { j }^{ L }$ 处激活函数 $ \sigma$ 的变化速度。

&emsp;&emsp;以矩阵形式重写公式 $(BP1)$，将其简约为：

$$
{ \delta  }^{ L }={ \nabla  }_ { a }C\odot { \sigma  }^{ \prime  }\left( { z }^{ L } \right) (BP1.1)
$$

这⾥ ${ \nabla  }  _  { a }C$ 被定义成⼀个向量，其元素是偏导数 $\frac { \partial C }{ \partial { a }  _  { j }^{ L } }$。可以将 ${ \nabla  }  _  { a }C$ 看成是 $C$ 关于输出激活值的改变速度。


&emsp;&emsp;当代价函数为⼆次代价函数时，有 ${ \nabla  }  _  { a }C=\left( { a }^{ L }-y \right)$，所以 $(BP1)$ 的整个矩阵形式为：

$$
{ \delta  }^{ L }=\left( { a }^{ L }-y \right) \odot { \sigma  }^{ \prime  }\left( { z }^{ L } \right) (7)
$$

&emsp;&emsp;使⽤下⼀层的误差 ${ \delta  }^{ l+1 }$ 来表⽰当前层的误差 ${ \delta  }^{ l }$：

$$
{ \delta  }^{ l }=\left( { \left( { w }^{ l+1 } \right)  }^{ T }{ \delta  }^{ l+1 } \right) \odot { \sigma  }^{ \prime  }\left( { z }^{ l } \right) (BP2)
$$

其中 ${ \left( { w }^{ l+1 } \right)  }^{ T }$ 是 ${ \left( l+1 \right)  }^{ th }$ 层权重矩阵 $\left( { w }^{ l+1 } \right)$ 的转置。

&emsp;&emsp;通过组合 $(BP1)$ 和 $ (BP2)$，可以计算任何层的误差 ${ \delta  }^{ l }$。⾸先使⽤ $(BP1)$ 计算 ${ \delta  }^{ L }$，然后应⽤⽅程 $ (BP2)$ 来计算 ${ \delta  }^{ L-1 }$，然后再次⽤⽅程 $(BP2)$ 来计算 ${ \delta  }^{ L-2 }$ ，如此⼀步⼀步地反向传播完整个⽹络。

&emsp;&emsp;代价函数关于⽹络中任意偏置的改变率：

$$
\frac { \partial C }{ \partial { b }_ { j }^{ l } } ={ \delta  }_ { j }^{ l } (BP3)
$$

其实，误差 ${ \delta  }  _  { j }^{ l }$ 和偏导数值 $\frac { \partial C }{ \partial { b }  _  { j }^{ l } }$ 完全⼀致。因为 $(BP1)$ 和 $(BP2)$ 已经告诉如何计算 ${ \delta  }  _  { j }^{ l }$，所以就可以将 $(BP3)$ 简记为：

$$
\frac { \partial C }{ \partial b } ={ \delta  }
$$

其中，$\delta$ 和偏置 $b$ 都是针对同⼀个神经元。

&emsp;&emsp;代价函数关于任何⼀个权重的改变率：

$$
\frac { \partial C }{ \partial { w }_ { jk }^{ l } } ={ a }_ { k }^{ l-1 }{ \delta  }_ { j }^{ l } (BP4)
$$

这告诉我们如何计算偏导数 $\frac { \partial C }{ \partial { w }  _  { jk }^{ l } }$，公式也可以写成下⾯更简介的方式：

$$
\frac { \partial C }{ \partial { w } } ={ a }_ { in }{ \delta  }_ { out } (8)
$$

其中，${ a }  _  { in }$ 是输⼊给权重 $w$ 的神经元的激活值，${ \delta  }  _  { out }$ 是输出⾃权重 $w$ 的神经元的误差。下面给出具体的演示图：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Backpropagation/bp4.png?raw=true"
    width="160" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">bp4</div>
</center>

&emsp;&emsp;由 $(8)$ 可看出当激活值 ${ a }  _  { in }$ 很⼩，${ a }  _  { in }\approx 0$，梯度 $\frac { \partial C }{ \partial { w } }$ 也会趋向很⼩。这 样，权重会学习的很缓慢，表⽰在梯度下降的时候，这个权重不会改变太多。换⾔之，$(BP4)$ 的⼀个结果就是来⾃低激活值神经元的权重学习会⾮常缓慢。

&emsp;&emsp;此外，当 $\sigma \left( { z }  _  { j }^{ l } \right)$ 近似为 $0$ 或者 $1$ 的时候 $\sigma$ 函数变得⾮常平。这时 $\sigma \left( { z }  _  { j }^{ l } \right) \approx 0$。所以如果输出神经元处于低激活值 $\approx 0$ 或者⾼激活值 $\approx 1$ 时，最终层的权重学习缓慢。这样的情形称输出神经元已经饱和，并且，权重也会终⽌学习（或者学习⾮常缓慢）。类似的结果对于输出神经元的偏置也是成⽴的。

&emsp;&emsp;总之，如果输⼊神经元激活值很低，或者输出神经元已饱和（过 ⾼或者过低的激活值），那么权重学习会很缓慢。

## 四个基本⽅程总结

$$
\begin{aligned}
&{ \delta  }^{ L }={ \nabla  }_{ a }C\odot { \sigma  }^{ \prime  }\left( { z }^{ L } \right) 
\\
&{ \delta  }^{ l }=\left( { \left( { w }^{ l+1 } \right)  }^{ T }{ \delta  }^{ l+1 } \right) \odot { \sigma  }^{ \prime  }\left( { z }^{ l } \right)
\\
&\frac { \partial C }{ \partial { b }_{ j }^{ l } } ={ \delta  }_{ j }^{ l }
\\
&\frac { \partial C }{ \partial { w }_{ jk }^{ l } } ={ a }_{ k }^{ l-1 }{ \delta  }_{ j }^{ l }
\end{aligned}
$$

## 四个基本⽅程的证明
&emsp;&emsp;这四个基本的⽅程 $(BP1)-(BP4)$，都是多元微积分的链式法则的推论。从⽅程 $(BP1)$ 开始，它给出了输出误差 ${ \delta  }^{ L }$ 的表达式。为了证明这个⽅程，回忆误差的定义：

$$
{ \delta  }_ { j }^{ l }=\frac { \partial C }{ \partial { z }_ { j }^{ l } } (9)
$$

应⽤链式法则，可以就输出激活值的偏导数的形式重新表⽰上⾯的偏导数：

$$
{ \delta  }_{ j }^{ L }=\frac { \partial C }{ \partial { a }_{ j }^{ L } } \frac { \partial { a }_{ j }^{ L } }{ \partial { z }_{ j }^{ L } } (10)
$$

由于 ${ a }  _  { j }^{ L }=\sigma \left( { z }  _  { j }^{ L } \right)$，右边的第⼆项可以写为 ${ \sigma  }^{ \prime  }\left( { z }  _  { j }^{ L } \right)$，⽅程变成：

$$
{ \delta  }_ { j }^{ L }=\frac { \partial C }{ \partial { a }_ { j }^{ L } } \sigma \left( { z }_ { j }^{ L } \right) (11)
$$

这正是 $(BP1)$ 的分量形式。

&emsp;&emsp;接下来证明 $(BP2)$，它给出了以下⼀层误差 ${ \delta  }  _  { j }^{ L+1 }$的形式表⽰误差 ${ \delta  }  _  { j }^{ L }$。为此，可以使用 ${ \delta  }  _  { k }^{ l+1 }=\frac { \partial C }{ \partial { z }  _  { k }^{ l+1 } }$ 的形式重写 ${ \delta  }  _  { j }^{ l }=\frac { \partial C }{ \partial { z }  _  { j }^{ l } }$。使⽤链式法则：

$$
\begin{aligned}
{ \delta  }_{ j }^{ l }&=\frac { \partial C }{ \partial { z }_{ j }^{ l } } \\ &=\sum _{ k }{ \frac { \partial C }{ \partial { z }_{ k }^{ l+1 } }  } \frac { \partial { z }_{ k }^{ l+1 } }{ \partial { z }_{ j }^{ l } } \\ &=\sum _{ k }{ \frac { \partial { z }_{ k }^{ l+1 } }{ \partial { z }_{ j }^{ l } } { \delta  }_{ k }^{ l+1 } }
\end{aligned} (12)
$$

最后⼀⾏交换了右边的两项，并⽤ ${ \delta  }  _  { k }^{ l+1 }$ 的定义代⼊。为了对最后⼀⾏的第⼀项求值，需要使用以下推导：

$$
{ z }_{ k }^{ l+1 }=\sum _{ j }{ { w }_{ kj }^{ l+1 }{ a }_{ j }^{ l } } +{ b }_{ k }^{ l+1 }=\sum _{ j }{ { w }_{ kj }^{ l+1 }\sigma \left( { z }_{ j }^{ l } \right)  } +{ b }_{ k }^{ l+1 } (13)
$$

求偏导数可得到：

$$
\frac { \partial { z }_{ k }^{ l+1 } }{ \partial { z }_{ j }^{ l } } ={ w }_{ kj }^{ l+1 }{ \sigma  }^{ \prime  }\left( { z }_{ j }^{ l } \right) (14)
$$

把它代⼊ $(12)$ 得到：

$$
{ \delta  }_{ j }^{ l }=\sum _{ k }{ { w }_{ kj }^{ l+1 }{ \delta  }_{ k }^{ l+1 }{ \sigma  }^{ \prime  }\left( { z }_{ j }^{ l } \right)  } 
$$

这正是 $(BP2)$ 的分量形式。
