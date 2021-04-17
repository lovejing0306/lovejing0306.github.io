---
layout: post
title: 激活函数
categories: [BasicKnowledge]
description: 激活函数
keywords: BasicKnowledge
---


深度学习基础知识点激活函数
---

## 什么是激活函数
* 激活函数是人工神经网络的一个极其重要的特征。
* 激活函数决定一个神经元是否应该被激活，激活代表神经元接收的信息与给定的信息有关。
* 激活函数对输入信息进行非线性变换，然后将变换后的输出信息作为输入信息传给下一层神经元。

## 激活函数作用
* 如果不用激活函数，每一层输出都是上层输入的线性函数，无论神经网络有多少层，输出都是输入的线性组合。
* 激活函数给神经元引入了非线性因素，使得神经网络可以任意逼近任何非线性函数。

## 如何选择激活函数
* 浅层网络在分类器时，$Sigmoid$ 函数及其组合通常效果更好。
* 由于梯度消失问题，有时要避免使用 $sigmoid$ 和 $tanh$ 函数。
* $ReLU$函数是一个通用的激活函数，目前在大多数情况下使用。
* 如果神经网络中出现死神经元，那么 $PReLU$ 函数就是最好的选择。
* 注意，$ReLU$ 函数只能在隐藏层中使用。
* 通常，可以从 $ReLU$ 函数开始，如果 $ReLU$ 函数没有提供最优结果，再尝试其他激活函数。

## 激活函数种类
  
### identity
* 函数定义
 $$
 f(x)=x
 $$
* 导数
 $$
 { f }^{ ' }(x)=1
 $$
* 函数图形

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/identity.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">identity</div>
</center>

* 优点
    * 适合于潜在行为是线性（与线性回归相似）的任务
* 缺点
    * 无法提供非线性映射
   
### step(单位阶跃函数)
* 函数定义
 $$
 { f }(x)=\begin{cases} \begin{matrix} 0 & x<0 \end{matrix} \\ \begin{matrix} 1 & x\ge 0 \end{matrix} \end{cases}
 $$
* 导数
 $$
 { f }^{ ' }(x)=\begin{cases} \begin{matrix} 0 & x\neq 0 \end{matrix} \\ \begin{matrix} ? & x=0 \end{matrix} \end{cases}
 $$
* 函数图形

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/step.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">step</div>
</center>

* 优点
    * 激活函数 $Step$ 更倾向于理论而不是实际，它模仿了生物神经元要么全有要么全无的属性。
* 缺点
    * 它无法应用于神经网络，因为其导数是 $0$（除了零点导数无定义以外），这意味着基于梯度的优化方法并不可行     

### sigmoid
* 函数定义
 $$
 { f }(x)=\sigma (x)=\frac { 1 }{ 1+{ e }^{ -x } } 
 $$
* 导数
 $$
 { f }^{ ' }(x)=f(x)(1-f(x))
 $$
* 函数图形

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/sigmoid.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">sigmoid</div>
</center>

* 优点
    * $Sigmoid$ 函数的输出映射在 $(0,1)$ 之间，单调连续，输出范围有限，优化稳定，可以用作输出层。
    * 求导容易。
* 缺点
    * 由于其软饱和性，容易产生梯度消失，导致训练出现问题。
    * 其输出并不是以 $0$ 为中心的。

### tanh(双曲正切函数)
* 函数定义
 $$
 { f }(x)=tanh(x)=\frac { { e }^{ x }-{ e }^{ -x } }{ { e }^{ x }+{ e }^{ -x } } 
 $$
* 导数
 $$
 { f }^{ ' }(x)=1-f(x)^{ 2 }
 $$
* 函数图形

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/tanh.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">tanh</div>
</center>

* 优点
    * 比 $Sigmoid$ 函数收敛速度更快。
    * 相比 $Sigmoid$ 函数，$tanh$ 是 $0$ 均值的。
* 缺点
    * 与$Sigmoid$函数相同，由于饱和性容易产生的梯度消失。

### relu(修正线性单元)
* 函数定义
 $$
 f(x)=\begin{cases} \begin{matrix} 0 & x<0 \end{matrix} \\ \begin{matrix} x & x\ge 0 \end{matrix} \end{cases}
 $$
* 导数
 $$
 { { f }(x) }^{ ' }=\begin{cases} \begin{matrix} 0 & x<0 \end{matrix} \\ \begin{matrix} 1 & x\ge 0 \end{matrix} \end{cases}
 $$
* 函数图形

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/relu.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">relu</div>
</center>

* 优点
    * 收敛速度快。
    * $Sigmoid$ 和 $tanh$ 涉及了很多很 $expensive$ 的操作（比如指数），$ReLU$ 可以更加简单的实现。
    * 当输入 $x>=0$ 时，$ReLU$ 的导数为常数，这样可有效缓解梯度消失问题。
    * 当 $x<0$ 时，$ReLU$ 的梯度总是 $0$，提供了神经网络的稀疏表达能力。
* 缺点
    * $ReLU$ 的输出不是 $zero-centered$。
    * 神经元坏死现象，某些神经元可能永远不会被激活，导致相应参数永远不会被更新（在负数部分，梯度为 $0$ ）。产生这种现象的两个原因：参数初始化问题；$learning \ rate$ 太高导致在训练过程中参数更新太大（会导致更新后的参数为负值，负值在下一次更新时会导致梯度为 $0$ ）。 
    * 不能避免梯度爆炸问题。
* 为什么 $ReLU$ 不是全程可微/可导也能用于基于梯度的学习？
    * 从数学的角度看 $ReLU$ 在 $0$ 点不可导，因为它的左导数和右导数不相等；但在实现时通常会返回左导数或右导数的其中一个，而不是报告一个导数不存在的错误。从而避免了这个问题

### lrelu(带泄露修正线性单元)
* 函数定义
 $$
 f(x)=\begin{cases} \begin{matrix} 0.01x & x<0 \end{matrix} \\ \begin{matrix} x & x\ge 0 \end{matrix} \end{cases}
 $$
* 导数
 $$
 { { f }(x) }^{ ' }=\begin{cases} \begin{matrix} 0.01 & x<0 \end{matrix} \\ \begin{matrix} 1 & x\ge 0 \end{matrix} \end{cases}
 $$
* 函数图形

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/lrelu.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">lrelu</div>
</center>

* 优点
    * 避免梯度消失
    * 由于导数总是不为零，因此可减少死神经元的出现
* 缺点
    * 其表现并不一定比 $ReLU$ 好
    * 无法避免梯度爆炸问题

### prelu(参数化修正线性单元)
* 函数定义
 $$
 f(\alpha ,x)=\begin{cases} \begin{matrix} \alpha x  & x<0 \end{matrix} \\ \begin{matrix} x & x\ge 0 \end{matrix} \end{cases}
 $$
* 导数
 $$
 { { f }(\alpha ,x) }^{ ' }=\begin{cases} \begin{matrix} \alpha  & x<0 \end{matrix} \\ \begin{matrix} 1 & x\ge 0 \end{matrix} \end{cases}
 $$
* 函数图形

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/prelu.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">prelu</div>
</center>

* 优点
    * $PReLU$ 是 $LReLU$ 的改进，可以自适应地从数据中学习参数
    * 收敛速度快、错误率低
    * $PReLU$ 可以用于反向传播的训练，可以与其他层同时优化
* 缺点
 
### rrelu(随机带泄露的修正线性单元)
* 函数定义
 $$
 f(\alpha ,x)=\begin{cases} \begin{matrix} \alpha  & x<0 \end{matrix} \\ \begin{matrix} x & x\ge 0 \end{matrix} \end{cases}
 $$
* 导数
 $$
 { { f }(\alpha ,x) }^{ ' }=\begin{cases} \begin{matrix} \alpha  & x<0 \end{matrix} \\ \begin{matrix} 1 & x\ge 0 \end{matrix} \end{cases}
 $$
* 函数图形

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/rrelu.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">rrelu</div>
</center>

* 优点
    * 为负值输入添加了一个线性项，这个线性项的斜率在每一个节点上都是随机分配的（通常服从均匀分布）
* 缺点

### elu(指数线性单元)
* 函数定义
 $$
 f(\alpha ,x)=\begin{cases} \begin{matrix} \alpha \left( { e }^{ x }-1 \right)  & x<0 \end{matrix} \\ \begin{matrix} x & x\ge 0 \end{matrix} \end{cases}
 $$
* 导数
 $$
 { { f }(\alpha ,x) }^{ ' }=\begin{cases} \begin{matrix} f(\alpha ,x)+\alpha  & x<0 \end{matrix} \\ \begin{matrix} 1 & x\ge 0 \end{matrix} \end{cases}
 $$
* 函数图形

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/elu.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">elu</div>
</center>

* 优点
    * 导数收敛为零，从而提高学习效率。
    * 能得到负值输出，这能帮助网络向正确的方向推动权重和偏置变化。
    * 防止死神经元出现。
* 缺点
    * 计算量大，其表现并不一定比 $ReLU$ 好。
    * 无法避免梯度爆炸问题。

### selu(扩展指数线性单元)
* 函数定义
 $$
 f(\alpha ,x)=\lambda \begin{cases} \begin{matrix} \alpha \left( { e }^{ x }-1 \right)  & x<0 \end{matrix} \\ \begin{matrix} x & x\ge 0 \end{matrix} \end{cases}
 $$
* 导数
 $$
 { { f }(\alpha ,x) }^{ ' }=\lambda \begin{cases} \begin{matrix} \alpha \left( { e }^{ x } \right)  & x<0 \end{matrix} \\ \begin{matrix} 1 & x\ge 0 \end{matrix} \end{cases}
 $$
* 函数图形

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/selu.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">selu</div>
</center>

* 优点
    * 是$ELU$的一个变种。其中 λ 和 α 是固定数值（分别为 $1.0507$ 和 $1.6726$ ）。
    * 经过该激活函数后使得样本分布自动归一化到 $0$ 均值和单位方差。
    * 不可能出现梯度消失或爆炸问题。
* 缺点
    * 这个激活函数相对较新——需要更多论文比较性地探索其在 CNN 和 RNN 等架构中应用。

### GELU
高斯误差线性单元激活函数在最近的 Transformer 模型（谷歌的 BERT 和 OpenAI 的 GPT-2）中得到了应用。GELU 的论文来自 2016 年，但直到最近才引起关注。

这种激活函数的形式为：
$$
GELU\left( x \right) =0.5x\left( 1+\tan\text{h}\left( \sqrt{\frac{2}{\pi}}\left( x+0.044715x^3 \right) \right) \right) 
$$
看得出来，这就是某些函数（比如双曲正切函数 tanh）与近似数值的组合。

* 优点
    * 似乎是 NLP 领域的当前最佳；尤其在 Transformer 模型中表现最好；
    * 能避免梯度消失问题。
* 缺点
    * 尽管是 2016 年提出的，但在实际应用中还是一个相当新颖的激活函数。

### ~~softsign~~
* 函数定义
 $$
 f(x)=\frac { x }{ \left| x \right| +1 } 
 $$
* 导数
 $$
 { f }^{ ' }(x)=\frac { 1 }{ { (1+\left| x \right| ) }^{ 2 } } 
 $$
* 函数图形

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/softsign.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">softsign</div>
</center>

* 优点
    * $Softsign$ 是 $Tanh$ 激活函数的另一个替代选择
    * $Softsign$ 是反对称、去中心、可微分，并返回 $-1$ 和 $1$ 之间的值。
    * $Softsign$ 更平坦的曲线与更慢的下降导数表明它可以更高效地学习
* 缺点
    * 导数的计算比$Tanh$更麻烦。

### ~~softplus~~
* 函数定义
 $$
 f(x)=\ln { (1+{ e }^{ x }) } 
 $$
* 导数
 $$
 { f }^{ ' }(x)=\frac { 1 }{ 1+{ e }^{ -x } } 
 $$
* 函数图形

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/softplus.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">softplus</div>
</center>

* 优点
    * 作为 $ReLU$ 的一个不错的替代选择，$SoftPlus$ 能够返回任何大于 $0$ 的值。
    * 与 $ReLU$ 不同，$SoftPlus$ 的导数是连续的、非零的，无处不在，从而防止出现死神经元。
* 缺点
    * 导数常常小于 $1$，也可能出现梯度消失的问题。
    * $SoftPlus$ 另一个不同于 $ReLU$ 的地方在于其不对称性，不以零为中心，可能会妨碍学习。

### softmax
* 函数定义
 $$
 { f }_{ i }(\overset { \rightarrow  }{ x } )=\frac { { e }^{ { x }_{ i } } }{ \sum _{ j=1 }^{ J }{ { e }^{ { x }_{ j } } }  } 
 $$
* 导数
 $$
 \frac { \partial { f }_{ i }(\vec { x } ) }{ \partial { x }_{ j } } ={ f }_{ i }(\vec { x } )({ \delta  }_{ ij }-{ f }_{ j }(\vec { x } ))
 $$
* 函数图形

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/softmax.png?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">softmax</div>
</center>

* 优点
 
* 缺点

## 激活函数比较
### Sigmoid 和 Softmax
* 二分类问题时 $sigmoid$ 和 $softmax$ 是一样的，都是求 $cross \ entropy \ loss$，而 $softmax$ 可以用于多分类问题。
* $softmax$ 是 $sigmoid$ 的扩展，因为，当类别数 $k=2$ 时，$softmax$ 回归退化为 $logistic$ 回归。
* $softmax$ 建模使用的分布是多项式分布，而 $logistic$ 则基于伯努利分布。
* 多个 $logistic$ 回归通过叠加也同样可以实现多分类的效果，但是 $softmax$ 回归进行的多分类，类与类之间是互斥的，即一个输入只能被归为一类；多个 $logistic$ 回归进行多分类，输出的类别并不是互斥的，即"苹果"这个词语既属于"水果"类也属于"$3C$"类别。



<script type="text/x-mathjax-config">
  MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/latest.js?config=TeX-MML-AM_CHTML">
</script>