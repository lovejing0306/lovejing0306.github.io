---
layout: post
title: 逻辑回归
categories: [MachineLearning]
description: 逻辑回归
keywords: MachineLearning
---


逻辑回归
---


$logistic$ 回归是一种二分类算法，直接为样本估计出它属于正负类样本的概率。其基本思路是先将向量进行线性加权，然后计算 $logistic$ 函数得到 $[0,1]$ 之间的概率值，

## 二分类模型

### 基本思想
对数几率回归就是用线性回归模型的预测结果去逼近真实标记的对数几率。

### 描述
&emsp;&emsp;给定数据集 $D=\{ \left( { x }  _  { 1 },{ \tilde { y }  }  _  { 1 } \right) ,\left( { x }  _  { 2 },{ \tilde { y }  }  _  { 2 } \right) ,\cdots ,\left( { x }  _  { N },{ \tilde { y }  }  _  { N } \right)  \} $，其中 ${ x }  _  { i }={ \left( { x }  _  { i,1 },{ x }  _  { i,2 },\cdots ,{ x }  _  { i,n } \right)  }^{ T }\in X\subseteq { R }^{ n }$; ${ \tilde { y }  }  _  { i }\in Y={ 0,1 } $；$i=1,2,\cdots ,N$。求解二分类问题，最理想的是单位阶跃函数：

$$
p\left( y=1|x \right) =\begin{cases} 0,&z<0 \\ 0.5,&z=0 \\ 1,&z>0 \end{cases},z={ w }^{ T }x+b
$$

但是阶跃函数不满足单调可微的性质，不能直接用作 $g(\cdot)$。

&emsp;&emsp;对数几率函数($logistic \ function$)就是这样的一个替代函数：

$$
p\left( y=1|x \right) =\frac { 1 }{ 1+{ e }^{ -z } } ,z={ w }^{ T }x+b
$$

这样的模型称作对数几率回归($logistic \ regression$或$logit \ regression$）模型。

&emsp;&emsp;由于$p\left( y=0 \| x \right) =1-p\left( y=1 \| x \right) $，则有：

$$
\ln { \frac { p\left( y=1|x \right)  }{ p\left( y=0|x \right)  }  } ={ w }^{ T }x+b
$$

其中，比值 $\frac { p\left( y=1 \| x \right)  }{ p\left( y=0 \| x \right)  } $ 称为几率，反映了样本作为正例的相对可能性。几率的对数称作对数几率($log odds$，也称作 $logit$)。

&emsp;&emsp;虽然对数几率回归名字带有回归，但是它是一种分类的学习方法。其优点：
1. 直接对分类的可能性进行建模，无需事先假设数据分布，这就避免了因为假设分布不准确带来的问题。
2. 不仅预测出来类别，还得到了近似概率的预测，这对许多需要利用概率辅助决策的任务有用。
3. 对数函数是任意阶可导的凸函数，有很好的数学性质，很多数值优化算法都能直接用于求取最优解。

### 参数估计
&emsp;&emsp;求解对数几率回归，可以用极大似然估计法估计模型参数。令：

$$
\begin{aligned}
p\left( y=1|x \right) &=\pi \left( x \right) \\ p\left( y=0|x \right) &=1-\pi \left( x \right) 
\end{aligned} 
$$

对于每一个样本 $x$，其分类概率为：

$$
p\left( y|x \right) ={ \pi \left( x \right)  }^{ y }{ \left( 1-\pi \left( x \right)  \right)  }^{ 1-y }
$$

由于各个观测样本之间相互独立，它们的联合分布为各边缘分布的乘积。于是似然函数为：

$$
L\left( w \right) =\prod _{ i=1 }^{ m }{ { \left[ \pi \left( { x }_{ i } \right)  \right]  }^{ { y }_{ i } } } { \left[ 1-\pi \left( { x }_{ i } \right)  \right]  }^{ 1-y_{ i } }
$$

其对数似然函数为：

$$
\begin{aligned}
\ln { L\left( w \right) }&= \sum _ { i=1 }^{ m }{ \left\{ { y }_{ i }\ln { \left[ \pi \left( { x }_{ i } \right)  \right]  } +\left( 1-{ y }_{ i } \right) \ln { \left[ 1-\pi \left( { x }_{ i } \right)  \right]  }  \right\}  } \\ &=\sum _{ i=1 }^{ m }{ \ln { \left[ 1-\pi \left( { x }_{ i } \right)  \right]  }  } +\sum _{ i=1 }^{ m }{ { y }_{ i }\ln { \frac { \pi \left( { x }_{ i } \right)  }{ 1-\pi \left( { x }_{ i } \right)  }  }  } \\ &=\sum _{ i=1 }^{ m }{ \ln { \left[ 1-\pi \left( { x }_{ i } \right)  \right]  }  } +\sum _{ i=1 }^{ m }{ { y }_{ i }\left( { w }^{ T }{ x }_{ i }+b \right)  } \\ &=\sum _{ i=1 }^{ m }{ -\ln { \left[ 1+{ e }^{ { w }^{ T }{ x }_{ i }+b } \right]  }  } +\sum _{ i=1 }^{ m }{ { y }_{ i }\left( { w }^{ T }{ x }_ { i }+b \right)  } 
\end{aligned}
$$

对对数似然函数 $lnL(W)$ 求偏导：

$$
\begin{aligned}
\frac { \partial \ln { L\left( w \right)  }  }{ \partial { w }_{ k } } &=-\sum _{ i=1 }^{ m }{ \frac { { e }^{ { w }^{ T }{ x }_{ i }+b } }{ 1+{ e }^{ { w }^{ T }{ x }_{ i }+b } }  } { x }_{ ik }+\sum _{ i=1 }^{ m }{ { y }_{ i }{ x }_{ ik } } \\ &=\sum _{ i=1 }^{ m }{ \left[ { y }_{ i }-\pi \left( { x }_{ i } \right)  \right]  } { x }_{ ik }
\end{aligned}
$$

参数更新：

$$
{ w }_{ k }={ w }_{ k }+\eta \sum _{ i=1 }^{ m }{ \left[ { y }_{ i }-\pi \left( { x }_{ i } \right)  \right]  } { x }_{ ik }
$$

可以看到与线性回归类似，只是 ${ w }^{ T }{ x }  _  { i }$ 换成了 $\pi \left( { x }  _  { i } \right) $，而$\pi \left( { x }  _  { i } \right) $ 实际上就 ${ w }^{ T }{ x }  _  { i }$ 经过 $\pi \left( z \right) $ 映射过来的。

&emsp;&emsp;最终 $logistic$ 回归模型为：

$$
\begin{aligned}
p\left( y=1|x \right) =\frac { { e }^{ { w }^{ T }{ x }_ { i }+b } }{ 1+{ e }^{ { w }^{ T }{ x }_{ i }+b } } \\ p\left( y=0|x \right) =\frac { 1 }{ 1+{ e }^{ { w }^{ T }{ x }_ { i }+b } } 
\end{aligned}
$$

### 优缺点
#### 优点
1. 模型简单，易于理解和实现
2. 分类时的计算量小，速度快

#### 缺点
1. 只能处理二分类问题，且问题必须是线性可分，若要处理非线性问题则需要进行函数转换
2. 容易欠拟合，一般准确度不是很高
3. 对于非线性特征，需要进行转换
4. 当特征空间很大时，逻辑回归的性能不是很好

### 面点
1. 逻辑回归极大似然估计时，在目标函数上加一个高斯分布假设会是什么样的？<br>
   在目标函数中加一个高斯函数，取对数后，实际上相当于在对数似然函数上加了正则化项
2. 逻辑回归中取对数是由于将求似然函数的最大值转换成求对数似然函数的最大值，将问题转换便于求解。
3. 为什么逻辑回归把特征离散化后的效果更好？
   1. 使模型的收敛速度加快
   2. 离散化后数据变的稀疏，稀疏向量的内积运算速度更快
   3. 离散化后的特征对异常数据有更强的鲁棒性
   4. 特征离散化后简化了逻辑回归模型，降低了过拟合的风险
4. 为什么使用逻辑函数？

    因为逻辑函数是单调增的，并且值域在 $(0, 1)$ 之间。

## 多分类模型
&emsp;&emsp;二分类的 $logistic$ 回归模型可以推广到多分类问题。设离散型随机变量 $ y $ 的取值集合为：\{ 1,2,\cdots ,K \} $，则多元 $logistic$ 回归模型为：

$$
\begin{aligned}
p\left( y=k|x \right) &=\frac { exp\left( { w }^{ T }{ x }_ { k }+b \right)  }{ 1+\sum _{ j=1 }^{ K-1 }{ exp\left( { w }^{ T }{ x }_{ j }+b \right)  }  } ,k=1,2,\cdots ,K-1\\ p\left( y=K|x \right) &=\frac { 1 }{ 1+\sum _{ j=1 }^{ K-1 }{ exp\left( { w }^{ T }{ x }_{ j }+b \right)  }  } 
\end{aligned}
$$

其参数估计方法类似二项 $logistic$ 回归模型。
    
