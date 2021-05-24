---
layout: post
title: 权重初始化
categories: [BasicKnowledge]
description: 权重初始化
keywords: BasicKnowledge
---


深度学习基础知识点卷积之权重初始化
---


## 简介
&emsp;&emsp;在深度学习中，初始化权重的方法对模型的收敛速度和性能有着至关重要的影响。说白了，神经网络其实就是对权重参数$w$的不停迭代更新，以期达到较好的性能。

## 权重初始化满足条件
* 各层激活值不会出现饱和现象
* 各层激活值不为 $0$

## 权重初始化的方式

### 权重初始化为0
&emsp;&emsp;在深度学习中把权重初始化为 $0$ 是不可以的。如果所有的参数都是 $0$，那么所有神经元的输出都将是相同的，在反向传播时同一层内的所有神经元的行为也是相同的 --- $gradient$ 相同，$weight \ update$ 也相同。这种现象称为“网络对称”。

&emsp;&emsp;此外，一般只在训练逻辑回归模型时才使用 $0$ 初始化所有参数。

### 权重随机初始化
#### 初始化方式
1. 初始化为均值为 $0$， 标准差为 $0.01$ 的正态分布 $W=0.01\times np.random.randn(node\  _   in,\quad node\  _   out)$
2. 初始化为均值为 $0$， 标准差为 $1$ 的正态分布 $W=np.random.randn(node\  _   in,\quad node\  _   out)$

#### 缺点
1. 随机分布选择不当，就会导致网络陷入优化困境。
2. 以采用$tanh$激活函数为例，创建了一个 $10$ 层的神经网络，每一层的参数都是随机正态分布，均值为 $0$，标准差为 $0.01$。随着层数的增加，输出值迅速向 $0$ 靠拢，此时函数类似于线性的，导致神经网络失去非线性功能。此外，在进行 $back \ propagation$ 时会导致梯度很小，使得参数难以被更新。
3. 以采用 $tanh$ 激活函数为例，创建了一个 $10$ 层的神经网络，每一层的参数都是随机正态分布，均值为 $0$，标准差为 $1$。几乎所有的值集中在 $-1$ 或 $1$ 附近，神经元饱和了！注意到 $tanh$ 在 $-1$ 和 $1$ 附近的 $gradient$ 都接近 $0$，这同样导致了 $gradient$太 小，参数难以被更新。

### Xavier初始化
&emsp;&emsp;$Xavier$ 初始化是为了解决随机初始化的问题，其思想是尽可能的让输入和输出服从相同的分布，能够避免后面层的激活函数的输出值 趋向于 $0$。

#### 初始化方式
$W=np.random.randn(node\  _   in,\quad node\  _   out)/np.sqrt(node\  _   in)$

除以输入节点数的平方根就是为了让分布保持一致

#### 优点
1. 能够很好的适应 $tanh$ 激活函数，

#### 缺点
2. 对 $ReLU$ 激活函数无能为力，当达到 $5$，$6$层后几乎又开始趋向于 $0$，更深层的话很明显又会趋向于 $0$。

### He初始化
&emsp;&emsp;为了解决 $Xavier$ 初始化的问题，何恺明大神提出了一种针对 $ReLU$ 的初始化方法，一般称作 “$He \ initialization$”。其思想是在 $ReLU$ 网络中，假定每一层有一半的神经元被激活，另一半为 $0$，所以，要保持方差不变，只需要在 $Xavier$ 的基础上再除以 $2$。
#### 初始化方式

$$W=np.random.randn(node\  _   in,\quad node\  _   out)/np.sqrt(node\  _   in/2)$$

#### 优点
1. 完美解决 $Xavier$ 初始化在 $ReLU$ 上遇到的问题。

### 偏置初始化
* 初始化方式
  * 通常偏置项初始化为 $0$，或比较小的数，如：$0.01$。

## 注意
* 参数初始值不能取的太小，因为小的参数在反向传播时会导致小的梯度，对于深度网络来说，也会产生梯度弥散问题，降低参数的收敛速度。


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