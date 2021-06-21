---
layout: post
title: 线性回归
categories: [MachineLearning]
description: 线性回归
keywords: MachineLearning
---


线性模型
---


1. 给定样本 $x$，其中 $x={ \left( { x }  _  { 1 },{ x }  _  { 2 },\cdots ,{ x }  _  { n } \right)  }^{ T }$，${ x }  _  { i }$ 为样本 $X$ 的第 $i$ 个特征，特征有 $n$ 种。线性模型 ($linear \ model$) 的形式为：
   $$
   f\left( x \right) =w\cdot x+b
   $$
   其中 $w={ \left( { w }  _  { 1 },{ w }  _  { 2 },\cdots ,{ w }  _  { n } \right)  }^{ T }$ 为每个特征对应的权重生成的权重向量。
    
2. 线性模型的优点
   1. 模型简单
   2. 可解释性强，权重向量 $w$ 直观地表达了各个特征在预测中的重要性。
3. 很多功能强大的非线性模型可以在线性模型的基础上通过引入层级结构或者非线性映射得到。

## 线性回归(Linear Regression)

### 描述
&emsp;&emsp;给定一个点集 $D$，用一个函数去拟合该点集，使得点集与拟合函数间的误差最小，如果拟合后的函数描述的是一条直线，被称为线性回归，如果拟合后的函数描述的是一条曲线，被称为非线性回归。

### 问题
&emsp;&emsp;给定数据集 $D=\{ \left( { x }  _  { 1 },{ \tilde { y }  }  _  { 1 } \right) ,\left( { x }  _  { 2 },{ \tilde { y }  }  _  { 2 } \right) ,\cdots ,\left( { x }  _  { N },{ \tilde { y }  }  _  { N } \right)  \}$，其中 ${ x }  _  { i }={ \left( { x }  _  { i,1 },{ x }  _  { i,2 },\cdots ,{ x }  _  { i,n } \right)  }^{ T }\in X\subseteq { R }^{ n }$;${ \tilde { y }  }  _  { i }\in Y\subseteq R$。线性回归问题试图学习模型：

$$
f\left( x \right) =w\cdot x+b
$$

该问题也被称作多元线性回归(每个样本由多个特征组成)。
    
&emsp;&emsp;对于每个 ${ x }  _  { i }$，其预测值为 ${ \hat { y }  }  _  { i }=f\left( { x }  _  { i } \right) =w\cdot { x }  _  { i }+b$。采用平方损失函数，则在训练集 $D$ 上，模型的损失函数为：
   
$$
L\left( f \right) =\sum _ { i=1 }^{ N }{ { \left( { \hat { y }  }_ { i }-{ \tilde { y }  }_ { i } \right)  }^{ 2 } } =\sum _ { i=1 }^{ N }{ { \left( w\cdot { x }_ { i }+b-{ \tilde { y }  }_ { i } \right)  }^{ 2 } } 
$$

优化目标是损失函数最小化，即：

$$
\left( { w }^{ * },{ b }^{ * } \right) =arg\min _ { w,b }{ \sum _ { i=1 }^{ N }{ { \left( w\cdot { x }_ { i }+b-{ \tilde { y }  }_ { i } \right)  }^{ 2 } }  } 
$$

### 求解
&emsp;&emsp;采用梯度下降法来求解上述最优化问题得到的解称为数值解，采用最小二乘法求解上述最优化问题得到的解称为解析解。最小二乘法的作用求出使损失函数最小权重。
    
令：

$$
\begin{aligned} \theta &={ \left( { w }_{ 1 },{ w }_{ 2 },\cdots ,{ w }_{ n },b \right)  }^{ T } \\ x&={ \left( { x }_{ 1 },{ x }_{ 2 },\cdots ,{ x }_{ n },1 \right)  }^{ T } \end{aligned}
$$

则有：

$$
{ h }_ { \theta  }\left( x \right) ={ \theta  }_ { 1 }{ x }_ { 1 }+{ \theta  }_ { 2 }{ x }_ { 2 }+\cdots +{ \theta  }_ { n }{ x }_ { n }=\sum _ { i=1 }^{ n }{ { \theta  }_ { i }{ x }_ { i } } 
$$

写成矩阵的形式：

$$
{ h }_ { \theta  }\left( x \right) ={ \theta  }^{ T }x
$$

损失函数为：

$$
L\left( \theta  \right) =\frac { 1 }{ 2 } \sum _{ i=1 }^{ m }{ { \left( { \hat { y }  }_{ i }-{ \tilde { y }  }_{ i } \right)  }^{ 2 } } =\frac { 1 }{ 2 } \sum _{ i=1 }^{ m }{ { \left( { { h }_{ \theta  }\left( { x }_{ i } \right)  }-{ \tilde { y }  }_{ i } \right)  }^{ 2 } } 
$$

写成矩阵的形式：

$$
\begin{aligned}
L\left( \theta  \right) &=\frac { 1 }{ 2 } { \left( X\theta -Y \right)  }^{ T }{ \left( X \theta -Y \right)  } \\ &=\frac { 1 }{ 2 } \left( { \theta  }^{ T }{ X }^{ T }-{ Y }^{ T } \right) \left( X\theta -Y \right) \\ &=\frac { 1 }{ 2 } \left( { \theta  }^{ T }{ X }^{ T }X \theta -{ \theta  }^{ T }{ X }^{ T }Y -{ Y }^{ T }X \theta -{ Y }^{ T }Y \right) 
\end{aligned}
$$

其中，$X$ 为 $m$ 行 $n$ 列的矩阵。

&emsp;&emsp;为求得 $L\left( \theta  \right) $ 的极小值，可以通过对 $\theta $ 求导，并令导数为零，从而得到解析解：

$$
\begin{aligned}
\frac { \partial L\left( \theta  \right)  }{ \partial \theta  } &=\frac { 1 }{ 2 } \left( 2{ X }^{ T }X\theta -{ X }^{ T }Y-{ X }^{ T }Y-0 \right) \\ &={ X }^{ T }X\theta -{ X }^{ T }Y
\end{aligned}
$$

令 $\frac { \partial L\left( \theta  \right)  }{ \partial \theta  } =0$，则有：

$$
\theta ={ \left( { X }^{ T }{ X } \right)  }^{ -1 }{ X }^{ T }Y
$$

> 令函数的一阶导数为 $0$，可求得函数的极小值。

1. 当 ${ X }^{ T }{ X } $ 为满秩矩阵时，可得 $\theta ={ \left( { X }^{ T }{ X } \right)  }^{ -1 }{ X }^{ T }Y$。最终学得的多元线性回归模型为：$f\left( { x }  _  { i } \right) ={ w }^{ { * }^{ T } }{ x }  _  { i }+{ b }^{ * }$。
    
2. 当 ${ X }^{ T }{ X } $ 不是满秩矩阵。此时存在多个解析解，他们都能使得均方误差最小化。究竟选择哪个解作为输出，由算法的偏好决定。

### 目标函数得出方式
1. 极大似然估
2. 最小二乘法

### 目标函数求解方式
1. 数值解：梯度下降法
2. 解析解：正规方程，直接求解

### 优缺点
1. 优点
   1. 算法易于理解，计算简单
2. 缺点
   1. 对非线性数据拟合不好
   2. 可能出现欠拟合现象

## 岭回归(Ridge Regression)
### 描述
&emsp;&emsp;岭回归是对最小二乘法的一种改良，通过放弃最小二乘法的无偏性，以损失部分信息、降低精度为代价获得更符合实际、更可靠的回归系数，对病态数据的拟合要强于最小二乘法。

### 问题
&emsp;&emsp;在使用普通最小二乘法求解回归系数 $w$ 时，当矩阵$X$不是满秩或某些列之间的线性相关性比较大时，${ X }^{ T }{ X } $ 的行列式接近于 $0$，即 ${ X }^{ T }{ X } $ 接近于奇异，上述问题变为一个不适定问题。此时，计算 ${ \left( { X }^{ T }X \right)  }^{ -1 }$ 误差会很大，所以传统的最小二乘法缺乏稳定性与可靠性。

&emsp;&emsp;为了解决上述问题我们可以使用岭回归方法，对普通最小二乘法的回归系数增加如下约束：

$$
\sum _{ j=1 }^{ n }{ { \theta  }_{ j }^{ 2 } } \le \lambda 
$$

### 求解
&emsp;&emsp;在平方误差的基础上增加 $L2$ 正则化项（使用拉格朗日简化后的结果）：

$$
L\left( \theta  \right) =\frac { 1 }{ 2 } \sum _{ i=1 }^{ m }{ { \left( { { h }_{ \theta  }\left( { x }_{ i } \right)  }-{ \tilde { y }  }_{ i } \right)  }^{ 2 } } +\lambda \sum _{ j=1 }^{ n }{ { \theta  }_{ j }^{ 2 } } ,\lambda >0
$$

通过调节惩罚系数$\lambda$ 的值可以使得在方差和偏差之间达到平衡：随着 $\lambda$ 的增大，模型方差减小而偏差增大。

### 优缺点
1. 优点
   1. 可以有效的解决过拟合问题
2. 缺点
   1. R平方值会稍低于普通回归分析

### 应用
1. 用于处理特征数多于样本数的情况
2. 维数缩减，通过引入偏差（惩罚项），使不重要的参数趋于0
3. 解决过拟合问题

## 最小收缩和选择算子（Least absolute shrinkage and selection operator, LASSO）
### 描述
&emsp;&emsp;$LASSO$ 回归和岭回归类似，都是一种对模型参数的长度有约束的线性回归模型。岭回归约束参数使用的是 $L2$ 范数，$LASSO$ 回归约束参数使用的是 $L1$ 范数。

### 问题
&emsp;&emsp;对普通最小二乘法的回归系数增加如下约束：

$$
\sum _{ j=1 }^{ n }{ \left| { \theta  }_{ j } \right|  } \le \lambda 
$$

### 求解
&emsp;&emsp;在平方误差的基础上增加 $L1$ 正则化项（使用拉格朗日简化后的结果）：

$$
L\left( \theta  \right) =\frac { 1 }{ 2 } \sum _ { i=1 }^{ m }{ { \left( { { h }_ { \theta  }\left( { x }_ { i } \right)  }-{ \tilde { y }  }_ { i } \right)  }^{ 2 } } +\lambda \sum _ { j=1 }^{ n }{ \left| { \theta  }_ { j } \right|  } ,\lambda >0
$$

### 应用

1. 防止过拟合
2. 维数缩减，去除不重要的参数，使不重要的参数变为 $0$
3. 特征选择，选出重要的特征