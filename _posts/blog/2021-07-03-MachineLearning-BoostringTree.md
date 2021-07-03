---
layout: post
title: 提升树
categories: [MachineLearning]
description: 提升树
keywords: MachineLearning
---


提升树
---


## 简述
&emsp;&emsp;提升树是以决策树为基学习器的提升方法，它被认为是统计学习中性能最好的方法之一。对分类问题，提升树中的决策树是二叉决策树；对回归问题，提升树中的决策树是二叉回归树。

&emsp;&emsp;提升树模型可以表示为以决策树为基学习器的加法模型：

$$
f(x)=f_M(x)=\sum_{m=1}^{M}h_m(x;\Theta_m) 
$$

其中，$h  _  m(x;\Theta  _  m)$ 表示第 $m$ 个决策树；$\Theta  _  m$ 为第 $m$ 个决策树的参数；$M$ 为决策树的数量。

&emsp;&emsp;提升树算法采用前向分步算法，首先确定初始提升树 $f  _  0(x)=0$，第 $m$ 步模型为：

$$
f_m(x)=f_{m-1}(x)+h_m(x;\Theta_m) 
$$

其中，$h  _  m(\cdot)$ 为待求的第 $m$ 个决策树。通过经验风险极小化确定第$m$个决策树的参数 $\Theta  _  m$：

$$
\hat\Theta_m=\arg\min_{\Theta_m}\sum_{i=1}^{N}L(\tilde y_i,f_m(x_i)) 
$$

这里没有引入正则化，而在 $xgboost$ 中会引入正则化。

&emsp;&emsp;不同问题的提升树学习算法主要区别在于使用的损失函数不同（设预测值为 $\hat y$，真实值为 $\tilde y$)：
1. 回归问题中，通常使用平方误差损失函数：$L(\tilde y,\hat y)=(\tilde y-\hat y)^{2}$；
2. 分类问题中，通常使用指数损失函数：$L(\tilde y,\hat y)=e^{-\tilde y\hat y}$。

### 算法
&emsp;&emsp;给定训练数据集 $\mathbb D=\{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\},\quad x  _  i \in \mathcal X \subseteq \mathbb R^{n},\tilde y  _  i \in \mathcal Y \subseteq \mathbb R$，其中 $\mathcal X$ 为输入空间，$\mathcal Y$ 为输出空间。如果将输入空间 $\mathcal X $ 划分为 $J$ 个互不相交的区域 $\mathbf R  _  1,\mathbf R  _  2,\cdots,\mathbf R  _  J$，并且在每个区域上确定输出的常量 $c  _  j$， 则决策树可以表示为：

$$
h(x;\Theta)=\sum_{j=1}^{J}c_jI(x \in \mathbf R_j)
$$

其中，参数 $\Theta=\{(\mathbf R  _  1,c  _  1),(\mathbf R  _  2,c  _  2),\cdots,(\mathbf R  _  J,c  _  J)\}$ 表示决策树的划分区域和各区域上的输出；$J$ 是决策树的复杂度，即叶结点个数。

&emsp;&emsp;回归问题中，提升树采用平方误差损失函数。此时：

$$
\begin{aligned} L\left( \tilde { y } ,{ f }_ { m }\left( x \right)  \right) &=L\left( \tilde { y } ,{ f }_ { m-1 }\left( x \right) +{ h }_ { m }\left( x;{ \Theta  }_ { m } \right)  \right)  \\ &={ \left( \tilde { y } -{ f }_ { m-1 }\left( x \right) -{ h }_ { m }\left( x;{ \Theta  }_ { m } \right)  \right)  }^{ 2 } \\ &={ \left( r-{ h }_{ m }\left( x;{ \Theta  }_{ m } \right)  \right)  }^{ 2 } \end{aligned}
$$   

其中 $r=\tilde y-f  _  {m-1}(x)$ 为当前模型拟合数据的残差。所以对回归问题的提升树算法，第 $m$ 个决策树 $h  _  m(\cdot)$ 只需要简单拟合当前模型的残差。

&emsp;&emsp;不仅是回归提升树算法，其它的 $boosting$ 回归算法也是拟合当前模型的残差。

#### 回归提升树伪码
输入：训练数据集 $\mathbb D=\{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\},\quad x  _  i \in \mathcal X \subseteq \mathbb R^{n},\tilde y  _  i \in \mathcal Y \subseteq \mathbb R$

输出：提升树 $f  _  M(x)$

算法步骤：
1. 初始化 $f  _  0(x)=0$
2. 对于 $m=1,2,\cdots,M$
    1. 计算残差：$r  _  {m,i}=\tilde y  _  i-f  _  {m-1}(x  _  i),i=1,2,\cdots,N$。构建训练残差：
     $$
     \mathbf r_m=\{(x_1,r_{m,1}),(x_2,r_{m,2}),\cdots,(x_N,r_{m,N})\} 
     $$
    2. 通过学习一个回归树来拟合残差 $\mathbf r  _  m$，得到 $h  _  m(x;\Theta  _  m)$。
    3. 更新 $f  _  m(x)=f  _  {m-1}(x)+h  _  m(x;\Theta  _  m)$
3. 得到回归问题提升树：
  $$
  f_M(x)=\sum_{m=1}^{M}h_m(x;\Theta_m) 
  $$


            


