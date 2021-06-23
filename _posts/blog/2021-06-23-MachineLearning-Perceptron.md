---
layout: post
title: 感知机
categories: [MachineLearning]
description: 感知机
keywords: MachineLearning
---


感知机
---


## 定义
&emsp;&emsp;感知机是二分类的线性分类模型，属于判别模型。模型的输入为实例的特征向量，模型的输出为实例的类别：正类取值 $+1$, 负类取值 $-1$。感知机的物理意义：将输入空间（特征空间）划分为正、负两类的分离超平面。

&emsp;&emsp;设输入空间（特征空间）为 $\mathcal X \subseteq { \mathbb R }^{ n }$；输出空间为 $\mathcal Y =\{+1,-1\}$；输入$ x\in \mathcal X $ 为特征空间的点；输出 $y \in \mathcal Y$ 为实例的类别。定义函数 $f\left( x \right) =sign\left( { w }^{ T }x+b \right) $ 为感知机。其中：$w \subseteq { \mathbb R }^{ n }$ 为权值向量，$b \subseteq { \mathbb R }$ 为偏置。$sign$ 为符号函数：

$$
sign\left( x \right) =\begin{cases} +1,x\ge 0 \\ -1,x<0 \end{cases}
$$

&emsp;&emsp;感知机的几何解释：${ w }^{ T }x+b=0$ 对应特征空间 $\mathbb R^{n}$ 上的一个超平面 $S$，称作分离超平面。$w$ 是超平面 $S$ 的法向量，$b$ 是超平面的截距。超平面 $S$ 将特征空间划分为两个部分：超平面 $S$ 上方的点判别为正类；超平面 $S$ 下方的点判别为负类。

## 感知器收敛定理
&emsp;&emsp;如果训练数据集线性可分，那么感知器算法可以保证在有限步内找到分类超平面。

## 思想
&emsp;&emsp;通过最小化误分类点到超平面的总距离，来求取分离超平面。

## 损失函数
&emsp;&emsp;给定数据集 $D=\{ \left( { x }  _  { 1 },{ \tilde { y }  }  _  { 1 } \right) ,\left( { x }  _  { 2 },{ \tilde { y }  }  _  { 2 } \right) ,\cdots ,\left( { x }  _  { N },{ \tilde { y }  }  _  { N } \right)  \}$，其中 ${ x }  _  { i }={ \left( { x }  _  { i,1 },{ x }  _  { i,2 },\cdots ,{ x }  _  { i,n } \right)  }^{ T }\in X\subseteq { R }^{ n }$; ${ \tilde { y }  }  _  { i }\in Y={ +1,-1} $；$i=1,2,\cdots ,N$。若存在某个超平面 $S$：${ w }^{ T }x+b=0$，能够将数据集中的正实例点与负实例点完全正确地划分到超平面的两侧，则称数据集 $D$ 为线性可分数据集；否则称数据集 $D$ 线性不可分。划分到超平面两侧，用数学语言描述为：$\left( { w }^{ T }{ x }  _  { i }+b \right) { y }  _  { i }>0$。

&emsp;&emsp;根据感知机的定义：对分类正确的点 $\left( { x }  _  { i },{ y }  _  { i } \right)$，有 $\left( { w }^{ T }{ x }  _  { i }+b \right) { y }  _  { i }>0$。对分类错误的点 $\left( { x }  _  { i },{ y }  _  { i } \right) $，有 $\left( { w }^{ T }{ x }  _  { i }+b \right) { y }  _  { i }<0$。

损失函数一个自然的选择是误分类点的总数。但是这样的损失函数不是参数 $w$，$b$的连续可导函数，不易优化。这里选择的损失函数是误分类点到超平面 $S$ 的总距离。

&emsp;&emsp;输入空间${ \mathbb R }^{ n }$中任一点${ x }  _  { i }$到超平面$S$的距离为：

$$
\frac { \left| { w }^{ T }{ x }_ { i }+b \right|  }{ { \left\| w \right\|  }_ { 2 } } 
$$

对于误分类的数据点 $\left( { x }  _  { i },{ y }  _  { i } \right) $来说：

$$
-{ y }_{ i }\left( { w }^{ T }{ x }_{ i }+b \right) >0
$$

成立。因为当 ${ w }^{ T }{ x }  _  { i }+b>0$，${ y }  _  { i }=-1$，而当 ${ w }^{ T }{ x }  _  { i }+b<0$ 时，${ y }  _  { i }=+1$。因此误分类点 ${ x }  _  { i }$ 到超平面 $S$ 的距离是：

$$
-\frac { { y }_ { i }\left( { w }^{ T }{ x }_ { i }+b \right)  }{ { \left\| w \right\|  }_ { 2 } } 
$$

&emsp;&emsp;假设超平面 $S$ 的误分类点集合为 $M$，那么所有误分类点到超平面 $S$ 的总距离为：

$$
-\frac { 1 }{ { \left\| w \right\|  }_ { 2 } } \sum _ { { x }_ { i }\in M }{ { y }_ { i }\left( { w }^{ T }{ x }_ { i }+b \right)  } 
$$  

不考虑$\frac { 1 }{ { \left\| w \right\|  }  _  { 2 } } $，就得到感知机学习的损失函数：

$$
L\left( w,b \right) =-\sum _ { { x }_ { i }\in M }{ { y }_ { i }\left( { w }^{ T }{ x }_ { i }+b \right)  } 
$$

之所以不考虑$\frac { 1 }{ { \left\| w \right\|  }  _  { 2 } } $，因为感知机要求训练集线性可分，最终误分类点数量为零，此时损失函数为零。即使考虑分母，也是零。

&emsp;&emsp;若训练集线性不可分，则感知机算法无法收敛。误分类点越少或者误分类点距离超平面越近 $S$， 则损失函数 $L$ 越小。

## 最优化损失函数   
&emsp;&emsp;给定训练集 $D=\{ \left( { x }  _  { 1 },{ \tilde { y }  }  _  { 1 } \right) ,\left( { x }  _  { 2 },{ \tilde { y }  }  _  { 2 } \right) ,\cdots ,\left( { x }  _  { N },{ \tilde { y }  }  _  { N } \right)  \} $，其中 ${ x }  _  { i }={ \left( { x }  _  { i,1 },{ x }  _  { i,2 },\cdots ,{ x }  _  { i,n } \right)  }^{ T }\in X\subseteq { R }^{ n }$;${ \tilde { y }  }  _  { i }\in Y={ +1,-1} $；$i=1,2,\cdots ,N$。求参数 $w$，$b$，使其为以下损失函数极小化问题的解：

$$
\min _ { w,b }{ L\left( w,b \right) =-\sum _ { { x }_ { i }\in M }{ { y }_ { i }\left( { w }^{ T }{ x }_ { i }+b \right)  }  } 
$$

其中，$M$ 为误分类点的集合。


&emsp;&emsp;假设误分类点集合 $M$ 是固定的，那么损失函数 $L(w,b)$ 的梯度为：

$$
\begin{aligned}
{ \nabla  }_ { w }L\left( w,b \right) &=-\sum _ { { x }_ { i }\in M }{ { y }_ { i }{ x }_ { i } } \\ { \nabla  }_  { b }L\left( w,b \right) &=-\sum _ { { x }_ { i }\in M }{ { y }_ { i } } 
\end{aligned}
$$

通过梯度下降法，随机选取一个误分类点 $\left( { x }  _  { i },{ y }  _  { i } \right) $，对 $w$ 和 $b$ 进行更新：

$$
\begin{aligned}
w&\leftarrow w+\eta { y }_ { i }{ x }_ { i }\\ b&\leftarrow b+\eta { y }_ { i }
\end{aligned}
$$

其中，$\eta \in (0,1]$是步长，即学习率。通过迭代可以使得损失函数 $L(w,b)$ 不断减小直到 $0$。
  
## 伪码

输入：
1. 线性可分训练集$D=、\{ \left( { x }  _  { 1 },{ \tilde { y }  }  _  { 1 } \right) ,\left( { x }  _  { 2 },{ \tilde { y }  }  _  { 2 } \right) ,\cdots ,\left( { x }  _  { N },{ \tilde { y }  }  _  { N } \right)  \} $，其中 ${ x }  _  { i }={ \left( { x }  _  { i,1 },{ x }  _  { i,2 },\cdots ,{ x }  _  { i,n } \right)  }^{ T }\in X\subseteq { R }^{ n }$;${ \tilde { y }  }  _  { i }\in Y={ +1,-1} $；$i=1,2,\cdots ,N$；
2. 学习率$\eta \in (0,1]$

输出：
1. ${ w }^{ * }$ 和 ${ b }^{ * }$ 
2. 感知机模型：$f\left( x \right) =sign\left( { w }^{ * }\cdot x+{ b }^{ * } \right) $

步骤：
1. 选取初始值 ${ w }  _  { 0 }$ 和 ${ b }  _  { 0 }$
2. 在训练集中选取数据 $\left( { x }  _  { i },{ y }  _  { i } \right) $。若 ${ y }  _  { i }\left( { w }^{ T }{ x }  _  { i }+b \right) \le 0$ 则：

$$
\begin{aligned}
w&\leftarrow w+\eta { y }_ { i }{ x }_ { i }\\ b&\leftarrow b+\eta { y }_ { i }
\end{aligned}
$$

3. 在训练集中重复选取数据来更新 $w$ 和 $b$ 直到训练集中没有误分类点。
            

## 性质
1. 对于某个误分类点 $\left( { x }  _  { i },{ y }  _  { i } \right) $，假设它被选中用于更新参数。假设迭代之前，分类超平面为 $S$，该误分类点距超平面的距离为 $d$；假设迭代之后，分类超平面为 ${ S }^{ \prime  }$，该误分类点距超平面的距离为 ${ d }^{ \prime  }$。则：

$$
\begin{aligned}
\Delta d&={ d }^{ \prime  }-d\\ &=\frac { \left| { w }^{ T }{ x }_ { i }+{ b }^{ \prime  } \right|  }{ { \left\| w \right\|  }_ { 2 } } -\frac { \left| { w }^{ T }{ x }_ { i }+b \right|  }{ { \left\| w \right\|  }_ { 2 } } \\ &=-\frac { { y }_ { i }\left( { w }^{ T }{ x }_{ i }+{ b }^{ \prime  } \right)  }{ { \left\| w \right\|  }_{ 2 } } +\frac { { y }_{ i }\left( { w }^{ T }{ x }_{ i }+{ b } \right)  }{ { \left\| w \right\|  }_ { 2 } } \\ &\simeq -\frac { { y }_{ i } }{ { \left\| w \right\|  }_ { 2 } } \left[ \left( { w }^{ \prime  }-w \right) \cdot { x }_{ i }+\left( { b }^{ \prime  }-b \right)  \right] \\ &=-\frac { { y }_{  i } }{ { \left\| w \right\|  }_{ 2 } } \left[ \eta { y }_{ i }{ x }_ { i }\cdot { x }_ { i }+\eta { y }_ { i } \right] \\ &=-\frac { { \eta y }_ { i }^{ 2 } }{ { \left\| w \right\|  }_ { 2 } } \left( { x }_ { i }\cdot { x }_ { i }+1 \right) <0
\end{aligned}
$$

因此有 ${ d }^{ \prime  } < d$，这里要求 ${ w }^{ \prime  }\simeq w$，因此步长 $\eta$ 要相当小。
    
2. 几何解释：当一个实例点被误分类时，调整 $w$，$b$ 使得分离平面向该误分类点的一侧移动，以减少该误分类点与超平面间的距离，直至超平面越过所有的误分类点以正确分类。
3. 感知机学习算法由于采用不同的初值或者误分类点选取顺序的不同，最终解可以不同。
    

## 收敛性
&emsp;&emsp;感知机收敛性定理说明：
1. 当训练集线性可分时，感知机学习算法原始形式迭代是收敛的。
   1. 此时算法存在许多解，既依赖于初值，又依赖于误分类点的选择顺序。
   2. 为了得出唯一超平面，需要对分离超平面增加约束条件。
2. 当训练集线性不可分时，感知机学习算法不收敛，迭代结果会发生震荡。
        

## 对偶形式
### 推导
&emsp;&emsp;根据原始迭代形式 ：

$$
\begin{aligned}
w&\leftarrow w+\eta { y }_ { i }{ x }_ { i }\\ b&\leftarrow b+\eta { y }_ { i }
\end{aligned}
$$

取初始值 ${ w }  _  { 0 }$,${ b }  _  { 0 }$ 均为 $0$。则 $w$，$b$ 可以改写为：

$$
\begin{aligned}
w&=\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }{ y }_ { i }{ x }_ { i } } \\ b&=\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }{ y }_ { i } } 
\end{aligned}
$$

这就是感知机学习算法的对偶形式

### 伪码
输入：
1. 线性可分训练集 $D=\{ \left( { x }  _  { 1 },{ \tilde { y }  }  _  { 1 } \right) ,\left( { x }  _  { 2 },{ \tilde { y }  }  _  { 2 } \right) ,\cdots ,\left( { x }  _  { N },{ \tilde { y }  }  _  { N } \right)  \} $，其中 ${ x }  _  { i }={ \left( { x }  _  { i,1 },{ x }  _  { i,2 },\cdots ,{ x }  _  { i,n } \right)  }^{ T }\in X\subseteq { R }^{ n }$;${ \tilde { y }  }  _  { i }\in Y={ +1,-1} $；$i=1,2,\cdots ,N$；
2. 学习率$\eta \in (0,1]$

输出：
1. $\alpha $，$b$，其中 $\alpha ={ \left( { \alpha  }  _  { 1 },{ \alpha  }  _  { 2 },\cdots ,{ \alpha  }  _  { N } \right)  }^{ T }$。
2. 感知机模型 $f\left( x \right) =sign\left( \sum   _  { j=1 }^{ N }{ { \alpha  }  _  { j }{ y }  _  { j }{ x }  _  { j }\cdot x } +b \right) \le 0$。

步骤：
1. 初始化：$\alpha \leftarrow 0$,$b\leftarrow 0$。
2. 在训练集中随机选取数据 $\left( { x }  _  { i },{ y }  _  { i } \right) $，若 ${ y }  _  { i }\left( \sum   _  { j=1 }^{ N }{ { \alpha  }  _  { j }{ y }  _  { j }{ x }  _  { j }\cdot x } +b \right) \le 0$则更新：
   $$
   \begin{aligned}
   { \alpha  }_ { i }&\leftarrow { \alpha  }_ { i }+\eta \\ b&\leftarrow b+\eta { y }_ { i }
   \end{aligned}
   $$
3. 在训练集中重复选取数据来更新 $\alpha $，$b$直到训练集中没有误分类点。

&emsp;&emsp;在对偶形式中，训练集 $D$ 仅仅以内积的形式出现，因为算法只需要内积信息。可以预先将 $D$ 中的实例间的内积计算出来，并以矩阵形式存储。即 $Gram$ 矩阵：

$$
G={ \left[ { x }_ { i }\cdot { x }_ { j } \right]  }_ { N\times N }
$$

与原始形式一样，感知机学习算法的对偶形式也是收敛的，且存在多个解。

## 优缺点
1. 优点
   1. 简单、易于实现
2. 缺点
   1. 要求数据集必须是线性可分的