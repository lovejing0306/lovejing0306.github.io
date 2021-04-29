---
layout: post
title: L1 和 L2 正则化
categories: [BasicKnowledge]
description: L1 和 L2 正则化
keywords: BasicKnowledge
---


深度学习基础知识点 L1 和 L2 正则化 
---

## 简介
正则化（$Regularization$）是机器学习中一种常用的技术，其主要目的是控制模型复杂度，减小过拟合。最基本的正则化方法是在原目标（代价）函数中添加惩罚项，对复杂度高的模型进行“惩罚”。所谓 『惩罚』 是指对损失函数中的某些参数做一些限制。其数学表达形式为：

$$
\tilde { J } \left( w;X,y \right) =J\left( w;X,y \right) +\lambda \Omega \left( w \right) 
$$

式中 $X$，$y$ 为训练样本和相应标签，$w$ 为权重系数向量；$J\left( \right)$为目标函数，$\Omega\left( w \right)$ 即为惩罚项，可理解为模型“规模”的某种度量；参数$\lambda$制正则化强弱。

&emsp;&emsp;不同的$\Omega$ 函数对权重$w$的最优解有不同的偏好，因而会产生不同的正则化效果。最常用的$\lambda$函数有两种，即 ${l}_ 1$ 范数和 $l_ 2$ 范数，相应称之为$l_ 1$正则化和$l_ 2$正则化。

$$
\begin{aligned} \Omega \left(w \right) &={\left\| w \right\|}_ 1=\sum_{ i }{\left| w_ i\right|}  \\ \Omega \left(w \right) &=\left\| w \right\|_2^{ 2 }=\sum _ { i }{ { w }_ { i }^{ 2 } } \end{aligned}
$$

&emsp;&emsp;从上式可以看到正则化项是对系数做了处理（限制）。$L_1$ 正则化和 $L_2$ 正则化的说明如下：
* $l_1$ 正则化是指权值向量$w$中各个元素的绝对值之和，通常表示为 $\left\| w \right\|_ 1$
* $l_2$ 正则化是指权值向量$w$中各个元素的平方和然后再求平方根，通常表示为 $\left\|w\right\|_ 2$

## L1正则化和L2正则化来源

### 正则化理解之基于约束条件的最优化
&emsp;&emsp;对于模型权重系数$w$求解是通过最小化目标函数实现的，即求解：

$$
\min_{ w }{ J\left( w;X,y \right)} 
$$

我们知道，模型的复杂度可用$VC$维来衡量。通常情况下，模型 $VC$ 维与系数 $w$ 的个数成线性关系：即 $w$ 数量越多，$VC$ 维越大，模型越复杂。因此，为了限制模型的复杂度，很自然的思路是减少系数 $w$ 的个数，即让 $w$ 向量中一些元素为 $0$ 或者说限制 $w$ 中非零元素的个数。为此，我们可在原优化问题中加入一个约束条件：

$$
\begin{matrix} \min _{ w }{ J\left( w;X,y \right)  }  & s.t.{ \left\| w \right\|  }_{ 0 }\le C \end{matrix}
$$

$\left\|  \right\|_ 0$ 范数表示向量中非零元素的个数。

&emsp;&emsp;但由于该问题是一个 $NP$ 问题，不易求解，为此我们需要稍微“放松”一下约束条件。为了达到近似效果，我们不严格要求某些权重 $w$ 为 $0$，而是要求权重 $w$ 应接近于 $0$，即尽量小。从而用 ${ l }_ { 1 }$ 、 ${ l }_ { 2 }$ 范数来近似 ${ l }_ { 0 }$ 范数，即：

$$
\begin{matrix} \min _{ w }{ J\left( w;X,y \right) \quad s.t.{ \left\| w \right\|  }_{ 1 }\le C }  \\ \min _{ w }{ J\left( w;X,y \right) \quad s.t.{ \left\| w \right\|  }_{ 2 }\le C }  \end{matrix}
$$

使用 ${l}_ {2}$ 范数时，为方便后续处理，可对 $\left\| w \right\|_ {2}$ 进行平方，此时只需调整 $C$ 的取值即可。

&emsp;&emsp;利用拉格朗日算子法，可将上述带约束条件的最优化问题转换为不带约束项的优化问题，构造拉格朗日函数：

$$
\begin{matrix} L(w,\lambda )=J\left( w;X,y \right) +\lambda ({ \left\| w \right\|  }_{ 1 }-C) \\ L(w,\lambda )=J\left( w;X,y \right) +\lambda ({ \left\| w \right\|  }_{ 2 }^{ 2 }-C) \end{matrix}
$$

其中 $\lambda >0$，我们假设 $\lambda$ 的最优解为 ${ \lambda  }^{ \ast  }$，则对拉格朗日函数求最小化等价于：

$$
\begin{matrix} \min _{ w }{ J\left( w;X,y \right) +{ \lambda  }^{ * }{ \left\| w \right\|  }_{ 1 }\quad  }  \\ \min _{ w }{ J\left( w;X,y \right) +{ \lambda  }^{ * }{ \left\| w \right\|  }_{ 2 }^{ 2 }\quad  }  \end{matrix}
$$

可以看出，上式与 $\min_ { w }{ \tilde { J } (w;X,y) }$ 等价。

&emsp;&emsp;故此，我们得到对 ${ l }_ { 1 }$ 、 ${ l }_ { 2 }$ 正则化的第一种理解：
* ${ l }_ { 1 }$ 正则化等价于在原优化目标函数中增加约束条件 ${ \left\| w \right\|  }_ { 1 }<C$ 
* ${ l }_ { 2 }$ 正则化等价于在原优化目标函数中增加约束条件 ${ \left\| w \right\|  }_ { 2 }^{ 2 }<C$

## L1和L2正则化的直观理解
### L1正则化和特征选择

&emsp;&emsp;假设有如下带 ${ l }_ { 1 }$ 正则化的损失函数：

$$
J={ J }_ { 0 }(w;X,y)+\lambda { \left\| w \right\|}_ { 1 }
$$

其中 ${ J }_ { 0 }$ 是原始的损失函数，加号后面的一项是 ${ l }_ { 1 }$ 正则化项，$\lambda $ 是正则化系数。注意到 ${ l }_ { 1 }$ 正则化是权值的绝对值之和，$J$是带有绝对值符号的函数，因此$J$是不完全可微的。机器学习的任务就是要通过一些方法（比如梯度下降）求出损失函数的最小值。

&emsp;&emsp;当我们在原始损失函数 ${ J }_ { 0 }$ 后添加 ${ l }_ { 1 }$ 正则化项时，相当于对 ${ J }_ { 0 }$ 做了一个约束。令$L=\lambda { \left\| w \right\|  }_ { 1 }$，则，$J={ J }_ { 0 }+L$,此时我们的任务变成在 $L$约束下求出 ${ J }_ { 0 }$ 取最小值的解。

&emsp;&emsp;考虑二维的情况，即只有两个权值 ${ w }^{ 1 }$ 和 ${ w }^{ 2 }$ 此时 $L=\left| { w }^{ 1 } \right| +\left| { w }^{ 2 } \right| $，求解${ J }_ { 0 }$ 的过程可以画出等值线，同时 ${ l }_ { 1 }$ 正则化的函数 $L$ 也可以在 ${ w }^{ 1 }{ w }^{ 2 }$ 的二维平面上画出来。如下图：

![l1](https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Regularization/L1.png?raw=true)

图中等值线是 ${ J }_ { 0 }$ 的等值线，黑色方形是 $L$ 函数的图形。在图中，${ J }_ { 0 }$ 等值线与$L$图形首次相交的地方就是最优解。

&emsp;&emsp;上图中 ${ J }_ { 0 }$ 与 $L$ 在 $L$ 的一个顶点处相交，**这个顶点就是最优解**。注意到这个顶点的值是 $({ w }^{ 1 },{ w }^{ 2 })=(0,w)$。
可以直观想象，因为 $L$ 函数有很多『突出的角』（二维情况下四个，多维情况下更多），${ J }_ { 0 }$ 与这些角接触的机率会远大于与 $L$ 其它部位接触的机率，而在这些角上，会有很多权值等于0，这就是为什么 ${ l }_ { 1 }$ 正则化可以产生稀疏模型，进而可以用于特征选择。

### L2正则化和过拟合
#### 为什么 ${ l }_ { 2 }$ 正则化不具有稀疏性
&emsp;&emsp;假设有如下带 $L_2$ 正则化的损失函数： 

$$
J={ J }_{ 0 }(w;X,y)+\lambda { \left\| w \right\|  }_{ 2 }^{ 2 }
$$

同样可以画出他们在二维平面上的图形，如下：

![l2](https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Regularization/L2.png?raw=true)

二维平面下 ${ l }_ { 2 }$ 正则化的函数图形是个圆，与方形相比，被磨去了棱角。因此 ${ J }_ { 0 }$ 与 $L$ 相交时 ，使得 ${ w }^{ 1 }$ 或 ${ w }^{ 2 }$ 等于零的机率小了许多，这就是为什么 ${ l }_ { 2 }$ 正则化不具有稀疏性的原因。

#### 为什么 ${ l }_ { 2 }$ 正则化可以获得值很小的参数
&emsp;&emsp;以线性回归中的梯度下降法为例。假设要求的参数为 $\theta $，${ h }_ { \theta  }(x)$ 是假设函数，那么线性回归的代价函数如下：

$$
J(\theta )=\frac { 1 }{ 2m } \sum _{ i=1 }^{ m }{ { \left( { h }_{ \theta  }({ x }^{ (i) })-y \right)  }^{ 2 } } 
$$

那么在梯度下降法中，最终用于迭代计算参数 $\theta$ 的迭代式为：

$$
{ \theta  }_{ j }:={ \theta  }_{ j }-\eta \frac { 1 }{ m } \sum _{ i=1 }^{ m }{ ({ h }_{ \theta  }({ x }^{ (i) })-{ y }^{ (i) }){ x }_ { j }^{ (i) } } 
$$

其中 $\eta$ 是 $learning \ rate$. 上式是没有添加 ${ l }_ { 2 }$ 正则化项的迭代公式，如果在原始代价函数之后添加 ${ l }_ { 2 }$ 正则化，则迭代公式会变成下面的样子：

$$
{ \theta  }_{ j }:={ \theta  }_ { j }(1-\eta \frac { \lambda  }{ m } )-\eta \frac { 1 }{ m } \sum _{ i=1 }^{ m }{ ({ h }_{ \theta  }({ x }^{ (i) })-{ y }^{ (i) }){ x }_ { j }^{ (i) } } 
$$

其中 $\lambda$ 就是正则化参数。从上式可以看到，与未添加 ${ l }_ { 2 }$ 正则化的迭代公式相比，每一次迭代， ${ \theta  }_ { j }$ 都要先乘以一个小于 $1$ 的因子 ，从而使得 ${ \theta  }_ { j }$ 不断减小，因此总得来看，$\theta$ 是不断减小的。

## 为什么参数大的模型容易过拟合
&emsp;&emsp;拟合过程中通常都倾向于让权值尽可能小，最后构造一个所有参数都比较小的模型。因为一般认为参数值小的模型比较简单，能适应不同的数据集，也在一定程度上避免了过拟合现象。可以设想一下对于一个线性回归方程，如果参数很大，只要数据偏移一点点，就会对结果造成很大的影响；但如果参数足够小，数据偏移得多一点也不会对结果造成什么影响，专业一点的说法是『抗扰动能力强』，即能够抵抗训练数据中的噪声。

## 作用
* ${ l }_ { 1 }$ 正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择，一定程度上，$L_ 1$ 也可以防止过拟合
* ${ l }_ { 2 }$ 正则化可以防止模型过拟合（$overfitting$）

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