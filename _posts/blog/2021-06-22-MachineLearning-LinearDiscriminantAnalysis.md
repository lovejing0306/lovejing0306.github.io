---
layout: post
title: 线性判别分析
categories: [MachineLearning]
description: 线性判别分析
keywords: MachineLearning
---


线性判别分析(LDA, Linear Discriminant Analysis)
---


## 基本思想
1. 训练时：给定训练样本集，找出一条直线，其满足将所有样例投影到该直线后，能够使同类样例的投影点尽可能接近、异类样例的投影点尽可能远离。
2. 预测时：对新样本进行分类时，将其投影到学到的直线上，在根据投影点的位置来确定新样本的类别。

## 二分类模型
### 描述
&emsp;&emsp;假设数据集为：$D=\{ \left( { x }  _  { 1 },{ \tilde { y }  }  _  { 1 } \right) ,\left( { x }  _  { 2 },{ \tilde { y }  }  _  { 2 } \right) ,\cdots ,\left( { x }  _  { N },{ \tilde { y }  }  _  { N } \right)  \} $，其中 ${ x }  _  { i }={ \left( { x }  _  { i,1 },{ x }  _  { i,2 },\cdots ,{ x }  _  { i,n } \right)  }^{ T }\in X\subseteq { R }^{ n }$;${ \tilde { y }  }  _  { i }\in Y={ 0,1 } $；$i=1,2,\cdots ,N$。设 ${ D }  _  { 0 }$ 表示类别为$0$的样例的集合，这些样例的均值向量为 ${ \mu  }  _  { 0 }={ \left( { \mu  }  _  { 1 }^{ 0 },{ \mu  }  _  { 2 }^{ 0 },\cdots ,{ \mu  }  _  { n }^{ 0 } \right)  }^{ T }$，这些样例的特征之间协方差矩阵为 ${ \Sigma  }  _  { 0 }$（协方差矩阵大小为 $n\times n$ ）。设 ${ D }  _  { 1 }$ 表示类别为 $1$ 的样例的集合，这些样例的均值向量为 ${ \mu  }  _  { 1 }={ \left( { \mu  }  _  { 1 }^{ 1 },{ \mu  }  _  { 2 }^{ 1 },\cdots ,{ \mu  }  _  { n }^{ 1 } \right)  }^{ T }$，这些样例的特征之间协方差矩阵为 ${ \Sigma  }  _  { 1 }$（协方差矩阵大小为 $n\times n$ ）。

&emsp;&emsp;假定直线为： $y={ w }^{ T }x$，其中 $w={ \left( { w }  _  { 1 },{ w }  _  { 2 },\cdots ,{ w }  _  { n } \right)  }^{ T }$,$x={ \left( { x }  _  { 1 }{ ,x }  _  { 2 },\cdots ,{ x }  _  { n } \right)  }^{ T }$。这里省略了常量 $b$ ，因为考察的是样本点在直线上的投影，总可以平行移动直线到原点而保持投影不变，此时 $b=0$。

&emsp;&emsp;将数据投影到直线上，则：两类样本的中心在直线上的投影分别为 ${ w }^{ T }{ \mu  }  _  { 0 }$ 和${ w }^{ T }{ \mu  }  _  { 1 }$ ；两类样本投影的方差分别为 ${ w }^{ T }{ \Sigma  }  _  { 0 }w$ 和 ${ w }^{ T }{ \Sigma  }  _  { 1 }w$ (由于直线是一维空间，因此上面四个值均为实数)。投影结果如下图所示：

![LDA](http://www.huaxiaozhuan.com/%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0/imgs/linear/lda.png)
 
&emsp;&emsp;根据线性判别分析的思想：
1. 要使同类样例的投影点尽可能接近，则可以使同类样例投影点的方差尽可能小，即 ${ w }^{ T }{ \Sigma  }  _  { 0 }w+{ w }^{ T }{ \Sigma  }  _  { 1 }w$ 尽可能小；
2. 要使异类样例的投影点尽可能远，则可以使异类样例的中心投影点尽可能远，即 ${ \left\| { w }^{ T }{ \mu  }  _  { 0 }-{ w }^{ T }{ \mu  }  _  { 1 } \right\|  }^{ 2 }$ 尽可能大；
3. 同时考虑两者，则得到最大化的目标：
   $$
   J=\frac { { \left\| { w }^{ T }{ \mu  }_ { 0 }-{ w }^{ T }{ \mu  }_ { 1 } \right\|  }^{ 2 } }{ { w }^{ T }{ \Sigma  }_ { 0 }w+{ w }^{ T }{ \Sigma  }_ { 1 }w } =\frac { { w }^{ T }\left( { \mu  }_ { 0 }-{ \mu  }_ { 1 } \right) { \left( { \mu  }_ { 0 }-{ \mu  }_ { 1 } \right)  }^{ T }w }{ { w }^{ T }\left( { \Sigma  }_ { 0 }+{ \Sigma  }_ { 1 } \right) w } 
   $$
    
### 求解
&emsp;&emsp;类内散度矩阵($within-class \ scatter \ matrix$)：

$$
{ S }_ { w }={ \Sigma  }_ { 0 }+{ \Sigma  }_ { 1 }=\sum _ { x\in { D }_ { 0 } }{ \left( x-{ \mu  }_ { 0 } \right)  } { \left( x-{ \mu  }_ { 0 } \right)  }^{ T }+\sum _ { x\in { D }_ { 1 } }{ \left( x-{ \mu  }_ { 1 } \right)  } { \left( x-{ \mu  }_ { 1 } \right)  }^{ T }
$$

它是每个类的散度矩阵之和。

&emsp;&emsp;类间散度矩阵($between-class \ scatter \ matrix$)：

$$
{ S }_ { b }=\left( { \mu  }_ { 0 }-{ \mu  }_ { 1 } \right) { \left( { \mu  }_ { 0 }-{ \mu  }_ { 1 } \right)  }^{ T }
$$

它是向量 $\left( { \mu  }  _  { 0 }-{ \mu  }  _  { 1 } \right) $ 与它自身的外积。

&emsp;&emsp;利用类内散度矩阵和类间散度矩阵，线性判别分析的最优化目标为：

$$
J=\frac { { w }^{ T }{ S }_ { b }w }{ { w }^{ T }{ S }_ { w }w } 
$$

$J$也称作${ S }  _  { b }$与${ S }  _  { w }$的广义瑞利商 。

&emsp;&emsp;现在求解最优化问题：

$$
L\left( { w }^{ * } \right) =arg\max _ { w }{ \frac { { w }^{ T }{ S }_ { b }w }{ { w }^{ T }{ S }_ { w }w }  } 
$$

考虑到分子与分母都是关于 $w$ 的二次项，因此上式的解与$w$的长度无关，只与$w$的方向有关。令 ${ w }^{ T }{ S }  _  { w }w=1$，则最优化问题改写为：

$$
L\left( { w }^{ * } \right) =arg\min _ { w }{ -{ w }^{ T }{ S }_ { b }w } \quad s.t.{ w }^{ T }{ S }_ { w }w=1
$$

&emsp;&emsp;应用拉格朗日乘子法，上式等价于 ${ \left( { \mu  }  _  { 0 }-{ \mu  }  _  { 1 } \right)  }^{ T }w={ \lambda  }  _  { w }$，其中 ${ \lambda  }  _  { w }$ 为实数。则 ${ S }  _  { b }w=\left( { \mu  }  _  { 0 }-{ \mu  }  _  { 1 } \right) { \left( { \mu  }  _  { 0 }-{ \mu  }  _  { 1 } \right)  }^{ T }w={ \lambda  }  _  { w }\left( { \mu  }  _  { 0 }-{ \mu  }  _  { 1 } \right) $。代入上式有：

$$
{ S }_ { b }w={ \lambda  }_ { w }\left( { \mu  }_ { 0 }-{ \mu  }_ { 1 } \right) =\lambda { S }_ { w }w
$$

由于与 $w$ 的长度无关，可以令${ \lambda  }  _  { w }=\lambda $则有：

$$
\left( { \mu  }_ { 0 }-{ \mu  }_ { 1 } \right) ={ S }_ { w }w\Rightarrow w={ S }_ { w }^{ -1 }\left( { \mu  }_ { 0 }-{ \mu  }_ { 1 } \right) 
$$

&emsp;&emsp;考虑数值解的稳定性，在实践中通常是对 ${ S }  _  { w }$ 进行奇异值分解：${ S }  _  { w }=U\Sigma { V }^{ T }$，其中 $\Sigma $ 为实对角矩阵，对角线上的元素为 ${ S }  _  { w }$ 的奇异值；$U$,$V$ 均为酉矩阵，它们的列向量分别构成了标准正交基。然后 ${ S }  _  { w }^{ -1 }=V{ \Sigma  }^{ -1 }{ U }^{ T }$。
            
## 多分类模型
&emsp;&emsp;线性判别分析可以推广到多分类任务中。假定存在 $C$ 个类，属于第 $i$ 个类的样本的集合为 ${ D }  _  { i }$，${ D }  _  { i }$ 中的样例数为 ${ m }  _  { i }$。其中：$\sum   _  { i=1 }^{ C }{ { m }  _  { i } } =m$，$m$ 为样本总数。

&emsp;&emsp;定义类别$i$的均值向量为所有该类别样本的均值：

$$
{ \mu  }_ { i }={ \left( { \mu  }_ { 1 }^{ i },{ \mu  }_ { 2 }^{ i },\cdots ,{ \mu  }_ { n }^{ i } \right)  }^{ T }=\frac { 1 }{ { m }_ { i } } \sum _ { { x }_ { i }\in { D }_ { i } }{ { x }_ { i } } 
$$

类别 $i$ 的样例，特征之间协方差矩阵为 ${ \Sigma  }  _  { i }$（协方差矩阵大小为 $n\times n$）。

&emsp;&emsp;定义所有样例的均值向量：

$$
\mu ={ \left( { \mu  }_ { 1 },{ \mu  }_ { 2 },\cdots ,{ \mu  }_ { n } \right)  }^{ T }=\frac { 1 }{ m } \sum _ { i=1 }^{ m }{ { x }_ { i } } 
$$

&emsp;&emsp;类别 $i$ 的类内散度矩阵为：

$$
{ S }_ { wi }=\sum _{ { x }_ { i }\in { D }_ { i } }{ \left( x-{ \mu  }_ { i } \right) { \left( x-{ \mu  }_ { i } \right)  }^{ T } } 
$$

实际上它等于样本集 ${ D }  _  { i }$ 的协方差矩阵 ${ \Sigma  }  _  { i }$，刻画了同类样例投影点的方差。

&emsp;&emsp;总的类内散度矩阵为：

$$
{ S }_ { w }=\sum _ { i=1 }^{ C }{ { S }_ { wi } } 
$$

它刻画了所有类别的同类样例投影点的方差。

&emsp;&emsp;总的类间散度矩阵为：

$$
{ S }_ { b }=\sum _{ i=1 }^{ C }{ { m }_ { i }\left( { \mu  }_ { i }-\mu  \right) { \left( { \mu  }_ { i }-\mu  \right)  }^{ T } } 
$$

它刻画了异类样例的中心的投影点的相互距离。注意：$\left( { \mu  }  _  { i }-\mu  \right) { \left( { \mu  }  _  { i }-\mu  \right)  }^{ T }$ 也是一个协方差矩阵，它刻画的是第 $i$ 类与总体之间的关系。
> 由于这里有不止两个中心点，因此不能简单的套用二分类线性判别分析的做法。这里用每一类样本集的中心点与总的中心点的距离作为度量。

> 考虑到每一类样本集的大小可能不同（密度分布不均），对这个距离施加权重，权重为每类样本集的大小。
            
&emsp;&emsp;根据线性判别分析的思想，设 $W\in { R }^{ n\times \left( C-1 \right)  }$ 是投影矩阵。经过推导可以得到最大化的目标：

$$
J=\frac { tr\left( { W }^{ T }{ S }_ { b }W \right)  }{ tr\left( { W }^{ T }{ S }_ { w }W \right)  } 
$$

其中，$tr(.)$ 表示矩阵的迹。
> 一个矩阵的迹是矩阵对角线的元素之和，它是一个矩阵不变量，也等于所有特征值之和。

> 还有一个常用的矩阵不变量就是矩阵的行列式，它等于矩阵的所有特征值之积。

&emsp;&emsp;与二分类线性判别分析不同，在多分类线性判别分析中投影方向是多维的，因此使用投影矩阵$W$。二分类线性判别分析的投影方向是一维的（只有一条直线），所以使用投影向量$w$。

&emsp;&emsp;上述最优化问题可以通过广义特征值问题求解：

$$
{ S }_ { b }W=\lambda { S }_ { w }W
$$

$W$ 的解析解为 ${ S }  _  { w }^{ -1 }{ S }  _  { b }$ 的 $C-1$ 个最大广义特征值所对应的特征向量组成的矩阵；多分类线性判别分析将样本投影到 $C-1$维空间；通常 $C-1$ 远小于数据原有的特征数，$LDA$ 因此也被视作一种经典的监督降维技术。

