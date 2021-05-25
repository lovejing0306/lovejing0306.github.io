---
layout: post
title: 梯度消失和梯度爆炸
categories: [BasicKnowledge]
description: 梯度消失和梯度爆炸
keywords: BasicKnowledge
---


深度学习基础知识点梯度消失和梯度爆炸
---


## 梯度消失
### 概念
&emsp;&emsp;从深层网络角度来讲，不同的层学习的速度差异很大，表现为网络中靠近输出层的学习情况良好，靠近输入层的学习很慢，有时甚至训练了很久，前几层的权值和刚开始随机初始化的值差不多，这种现象称为“梯度消失”。梯度消失根本原因在于反向传播训练法则。

### 演示
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/GradientProblem/gp1.png?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">gp1</div>
</center>

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/GradientProblem/gp2.png?raw=true"
    width="400" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">gp2</div>
</center>

### 什么导致了梯度消失
&emsp;&emsp;为了弄清楚为何出现梯度消失现象，下图先给出一个只有三个隐藏层，每个隐藏层只有一个神经元的神经⽹络：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/GradientProblem/gp3.png?raw=true"
    width="400" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">gp3</div>
</center>

其中，$\mathop w\nolimits  _  {1,} \mathop w\nolimits  _  {2,...}$ 是权重，⽽ ${ b }  _  { 1 },{ b }  _  { 2 }...$ 是偏置，$C$ 则是某个代价函数。第 $j$ 个神经元的输出${ a }  _  { j }=\sigma \left( { z }  _  { j } \right) $，其中 $\sigma$ 是通常的$S$型激活函数，⽽ ${ z }  _  { j }={ w }  _  { j }\ast { a }  _  { j-1 }+{ b }  _  { j }$ 是神经元的带权输⼊。

&emsp;&emsp;为了理解梯度消失发⽣的原因，我们研究一下代价函数关于第一个神经元的梯度 $\frac { \partial C }{ \partial { b }  _  { 1 } } $，下面给出具体的表达式：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/GradientProblem/gp4.png?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">gp4</div>
</center>

表达式结构如下：每个神经元有⼀个 ${ \sigma  }^{ \prime  }\left( { z }  _  { j } \right) $ 项；每个权重有⼀个 ${ w }  _  { j }$ 项；还有⼀个 $\frac { \partial C }{ \partial { a }  _  { 4 } } $ 项，表⽰最后的代价函数。

&emsp;&emsp;接下来给出上面表达式的由来，假设对偏置 ${ b }  _  { 1 }$ 进⾏了微⼩的调整 $\Delta { b }  _  { 1 }$ ，这会导致⽹络中剩余元素发生⼀系列的变化。⾸先会对第⼀个隐藏元输出产⽣⼀个 $\Delta { a }  _  { 1 }$ 的变化，接下来会导致第⼆个神经元的带权输⼊产⽣ $\Delta { z }  _  { 2 }$ 的变化。从第⼆个神经元输出随之发⽣ $\Delta { a }  _  { 2 }$ 的变化。以此类推，最终会对代价函数产⽣ $\Delta C$ 的变化。这⾥有：

$$
\frac { \partial C }{ \partial { b }_ { 1 } } \approx \frac { \Delta C }{ \Delta { b }_ { 1 } } 
$$

&emsp;&emsp;现在看看 $\Delta { b }  _  { 1 }$ 如何影响第⼀个神经元的输出 $\Delta { a }  _  { 1 }$的。因为有${ a }  _  { 1 }=\sigma \left( { z }  _  { 1 } \right) =\sigma \left( { w }  _  { 1 }{ a }  _  { 0 }+{ b }  _  { 1 } \right) $， 所以有：

$$
\begin{aligned}
{ \Delta a }_{ 1 }&\approx \frac { \partial \sigma \left( { w }_{ 1 }{ a }_{ 0 }+{ b }_{ 1 } \right)  }{ \partial { b }_{ 1 } } \Delta { b }_{ 1 }\\ &={ \sigma  }^{ \prime  }\left( { z }_{ 1 } \right) \Delta { b }_{ 1 }
\end{aligned}
$$

其中，${ \sigma  }^{ \prime  }\left( { z }  _  { 1 } \right) $ 看起很熟悉：其实是上⾯关于 $\frac { \partial C }{ \partial { b }  _  { 1 } }$的表达式的第⼀项。直觉上看，这项将偏置的改变 $\frac { \partial C }{ \partial { b }  _  { 1 } }$ 转化成了输出的变化${ \Delta a }  _  { 1 }$。${ \Delta a }  _  { 1 }$随之⼜影响了带权输⼊${ z }  _  { 2 }={ w }  _  { 2 }{ a }  _  { 1 }+{ b }  _  { 2 }$：

$$
\begin{aligned}
\Delta { z }_{ 2 }&\approx \frac { \partial { z }_{ 2 } }{ \partial { a }_{ 1 } } \Delta { a }_{ 1 }\\ &={ w }_{ 2 }\Delta { a }_{ 1 }
\end{aligned}
$$

&emsp;&emsp;将 $\Delta { z }  _  { 2 }$ 和 $\Delta { a }  _  { 1 }$ 的表达式组合起来，我们可以看到偏置 ${ b }  _  { 1 }$ 中的改变如何通过⽹络传输影响到$ { z }  _  { 2 }$ 的：

$$
\Delta { z }_{ 2 }\approx { \sigma  }^{ \prime  }\left( { z }_{ 1 } \right) { w }_{ 2 }\Delta { b }_{ 1 }
$$

以此类推下去，跟踪传播改变的路径就可以完成。在每个神经元，我们都会选择⼀个 ${ \sigma  }^{ \prime  }\left( { z }  _  { j } \right) $ 的项，然后在每个权重选择出⼀个 ${ w }  _  { j }$ 项。最终的结果就是代价函数中变化 $\Delta C$ 的相关于偏置 $\Delta { b }  _  { 1 }$ 的表达式：

$$
\Delta C={ \sigma  }^{ \prime  }\left( { z }_ { 1 } \right) { w }_ { 2 }{ \sigma  }^{ \prime  }\left( { z }_ { 2 } \right) { w }_ { 3 }{ \sigma  }^{ \prime  }\left( { z }_ { 3 } \right) { w }_ { 4 }{ \sigma  }^{ \prime  }\left( { z }_ { 4 } \right) \frac { \partial C }{ \partial { a }_ { 4 } } \Delta { b }_ { 1 }
$$

除以$\Delta { b }  _  { 1 }$，变得到了梯度的表达式：

$$
\frac { \partial C }{ \partial { b }_{ 1 } } ={ \sigma  }^{ \prime  }\left( { z }_{ 1 } \right) { w }_{ 2 }{ \sigma  }^{ \prime  }\left( { z }_{ 2 } \right) { w }_{ 3 }{ \sigma  }^{ \prime  }\left( { z }_{ 3 } \right) { w }_ { 4 }{ \sigma  }^{ \prime  }\left( { z }_{ 4 } \right) \frac { \partial C }{ \partial { a }_{ 4 } } 
$$

&emsp;&emsp;为何出现梯度消失： 整个梯度的表达式为：
$$
\frac { \partial C }{ \partial { b }_ { 1 } } ={ \sigma  }^{ \prime  }\left( { z }_{ 1 } \right) { w }_{ 2 }{ \sigma  }^{ \prime  }\left( { z }_{ 2 } \right) { w }_{ 3 }{ \sigma  }^{ \prime  }\left( { z }_{ 3 } \right) { w }_{ 4 }{ \sigma  }^{ \prime  }\left( { z }_{ 4 } \right) \frac { \partial C }{ \partial { a }_{ 4 } } 
$$

除了最后⼀项，该表达式是⼀系列形如 ${ w }  _  { j }{ \sigma  }^{ \prime  }\left( { z }  _  { j } \right) $ 的乘积。为了理解每个项的⾏为，先看看下⾯的 $sigmoid$ 函数导数的图像：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/GradientProblem/gp5.jpg?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">gp5</div>
</center>

&emsp;&emsp;该导数在 ${ \sigma  }^{ \prime  }\left( 0 \right) =\frac { 1 }{ 4 }$ 时达到最⾼。现在，如果使⽤标准⽅法来初始化⽹络中的权重，那么会使⽤⼀个均值为 $0$ 标准差为 $1$ 的⾼斯分布。因此所有的权重通常会满⾜ $\|{ w }  _  { j }\| < \frac { 1 }{ 4 }$。有了这些信息，会发现有 ${ w }  _  { j }{ \sigma  }^{ \prime  }\left( { z }  _  { j } \right) < \frac { 1 }{ 4 }$。当所有这些项的乘积时，最终结果肯定会指数级下降：项越多，乘积的下降的越快。这便是梯度消失的合理解释。

&emsp;&emsp;为了得到更清楚的解释，⽐较⼀下$\frac { \partial C }{ \partial { b }  _  { 1 } }$和⼀个更后⾯的偏置的梯度$\frac { \partial C }{ \partial { b }  _  { 3 } }$。以下是具体的计算公式：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/GradientProblem/gp6.png?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">gp6</div>
</center>

&emsp;&emsp;两个表⽰式有很多相同的项。但是 $\frac { \partial C }{ \partial { b }  _  { 1 } }$ 还多包含了两个项。由于这些项都是 $<\frac { 1 }{ 4 } $ 的。 所以 $\frac { \partial C }{ \partial { b }  _  { 1 } }$ 会是 $\frac { \partial C }{ \partial { b }  _  { 3 } }$ 的 $\frac { 1 }{ 16 } $ 或者更⼩。这其实就是消失的梯度出现的本质原因了。

&emsp;&emsp;当然，这⾥并⾮是一个关于梯度消失严格的的证明⽽是⼀个不太正式的论断。还有可能有⼀些其它产⽣的原因。特别地，我们想要知道权重 ${ w }  _  { j }$ 在训练中是否会增⻓。如果会，那么 ${ w }  _  { j }{ \sigma  }^{ \prime  }\left( { z }  _  { j } \right) $ 会不会不在满⾜之前 ${ w }  _  { j }{ \sigma  }^{ \prime  }\left( { z }  _  { j } \right) <\frac { 1 }{ 4 } $的约束。事实上，如果项变得很⼤——超过$1$，那么将不再 遇到消失的梯度问题。实际上，这时候梯度会在反向传播的时候发⽣指数级地增⻓。也就是说，遇到了梯度爆炸的问题。

## 梯度爆炸
### 概念
&emsp;&emsp;在深层网络或递归神经网络中，误差梯度在更新中累积得到一个非常大的梯度，这样的梯度会大幅度更新网络参数，进而导致网络不稳定。在极端情况下，权重的值变得特别大，以至于结果会溢出。当梯度爆炸发生时，网络层之间反复乘以大于 $1.0$ 的梯度值使得梯度值成倍增长。

### 什么导致了梯度爆炸
&emsp;&emsp;⾸先，将⽹络的权重设置得很⼤，⽐如 ${ w }  _  { 1 }={ w }  _  { 2 }={ w }  _  { 3 }={ w }  _  { 4 }=100$。然后，选择偏置使得 ${ \sigma  }^{ \prime  }\left( { z }  _  { j } \right) $ 项不会太⼩。这是很容易实现的：选择偏置来保证每个神经 元的带权输⼊是 ${ z }  _  { j }=0$ (这样 ${ \sigma  }^{ \prime  }\left( { z }  _  { j } \right) =\frac { 1 }{ 4 } $ )。⽐如说，希望 ${ z }  _  { 1 }={ w }  _  { 1 }\ast { a }  _  { 0 }+{ b }  _  { 1 }$。只要把 ${ b }  _  { 1 }=-100\ast { a }  _  { 0 }$ 即可。使⽤同样的⽅法来获得其他的偏置。这样可以发现所有的项 ${ w }  _  { j }\ast { \sigma  }^{ \prime  }\left( { z }  _  { j } \right) $ 都等于 $100\ast \frac { 1 }{ 4 } =25$。最终，便获得了梯度的爆炸。

### 梯度爆炸会引发哪些问题？
* 在深度多层感知机网络中，梯度爆炸会导致网络不稳定，最好的结果是无法从训练数据中学习，最坏的结果是由于权重值为 $NaN$ 而无法更新权重。
* 在循环神经网络（$RNN$）中，梯度爆炸会导致网络不稳定，使得网络无法从训练数据中得到很好的学习，最好的结果是网络不能在长输入数据序列上学习。

### 如何知道网络中是否有梯度爆炸问题？
* 模型无法在训练数据上收敛（比如，损失函数值非常差）
* 模型不稳定，在更新的时候损失有较大的变化
* 模型的损失函数值在训练过程中变成$NaN$值
* 模型在训练过程中，权重变化非常大
* 模型在训练过程中，权重变成NaN值
* 每层的每个节点在训练时，其误差梯度值一直是大于 $1.0$

### 如何解决梯度爆炸问题？
* 重新设计网络模型
  > 在深层神经网络中，梯度爆炸问题可以通过将网络模型的层数变少来解决。
* 使用修正线性激活函数
  > 在深度多层感知机中，当激活函数选择为一些之前常用的Sigmoid或Tanh时，网络模型会发生梯度爆炸问题。而使用修正线性激活函数（ReLU）能够减少梯度爆炸发生的概率，对于隐藏层而言，使用修正线性激活函数（ReLU）是一个比较合适的激活函数。
* 使用长短周期记忆网络
  > 由于循环神经网络中存在的固有不稳定性，梯度爆炸可能会发生。比如，通过时间反向传播，其本质是将循环网络转变为深度多层感知神经网络。通过使用长短期记忆单元（LSTM）或相关的门控神经结构能够减少梯度爆炸发生的概率。
* 使用梯度裁剪
  > 在深度多层感知网络中，当有大批量数据以及LSTM是用于很长时间序列时，梯度爆炸仍然会发生。当梯度爆炸发生时，可以在网络训练时检查并限制梯度的大小，这被称作梯度裁剪。

  > 梯度裁剪是处理梯度爆炸问题的一个简单但非常有效的解决方案，如果梯度值大于某个阈值，我们就进行梯度裁剪。

* 使用权重正则化
  > 如果梯度爆炸问题仍然发生，另外一个方法是对网络权重的大小进行校验，并对大权重的损失函数增添一项惩罚项，这也被称作权重正则化，常用的有L1（权重的绝对值和）正则化与L2（权重的绝对值平方和再开方）正则化。



## 打赏

如果文章对您有帮助，欢迎丢香蕉抛硬币。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/Reward/wechat.JPG?raw=true"
    width="300" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">微信</div>
</center>

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/Reward/zhifubao.JPG?raw=true"
    width="300" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">支付宝</div>
</center>



<script type="text/x-mathjax-config">
  MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/latest.js?config=TeX-MML-AM_CHTML">
</script>