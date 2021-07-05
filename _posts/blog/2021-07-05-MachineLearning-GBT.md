---
layout: post
title: 梯度提升树
categories: [MachineLearning]
description: 梯度提升树
keywords: MachineLearning
---


梯度提升树(GBT)
---


&emsp;&emsp;提升树中，当损失函数是平方损失和指数损失时，每一步优化都很简单。因为平方损失函数和指数损失函数的求导非常简单。当损失函数是一般函数时，往往每一步优化不是很容易。针对这个问题，($Freidman$) 提出了梯度提升算法。

&emsp;&emsp;梯度提升树 ($GBT$) 利用最速下降法的近似方法。其关键是利用损失函数的负梯度在当前模型的值作为残差的近似值，从而拟合一个回归树。根据泰勒展开式：

$$
\begin{aligned} L\left( \tilde { y } ,{ f }_ { m }\left( x \right)  \right) &=L\left( \tilde { y } ,{ f }_ { m-1 }\left( x \right) +{ h }_ { m }\left( x;{ \Theta  }_ { m } \right)  \right)  \\ &=L\left( \tilde { y } ,{ f }_ { m-1 }\left( x \right)  \right) +\frac { \partial L\left( \tilde { y } ,{ f }_ { m-1 }\left( x \right)  \right)  }{ \partial { f }_ { m-1 }\left( x \right)  } { h }_ { m }\left( x;{ \Theta  }_ { m } \right)  \end{aligned}
$$

则有：

$$
\begin{aligned} \Delta L&=L\left( \tilde { y } ,{ f }_ { m }\left( x \right)  \right) -L\left( \tilde { y } ,{ f }_ { m-1 }\left( x \right)  \right)  \\ &=\frac { \partial L\left( \tilde { y } ,{ f }_ { m-1 }\left( x \right)  \right)  }{ \partial { f }_ { m-1 }\left( x \right)  } { h }_ { m }\left( x;{ \Theta  }_ { m } \right)  \end{aligned}
$$

要使损失函数降低，一个可选的方案是：$h  _  m(x;\Theta  _  m)=-\frac{\partial L(\tilde y,f  _  {m-1}(x))}{\partial f  _  {m-1}(x)}$ 。对于平方损失函数，它就是通常意义上的残差；对于一般损失函数，它就是残差的近似。

&emsp;&emsp;梯度提升树用于分类模型时，是梯度提升决策树 ($GBDT$)；用于回归模型时，是梯度提升回归树 ($GBRT$)。

## 梯度提升回归树伪码
1. 输入：
   1. 训练数据集 $\mathbb D=\{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\},\quad x  _  i \in \mathcal X \subseteq \mathbb R^{n},\tilde y  _  i \in \mathcal Y \subseteq \mathbb R$
   2. 损失函数 $L(\tilde y, \hat y)$
2. 输出：回归树 $f  _  M(x)$
3. 算法步骤：
   1. 初始化：$f  _  0(x)=\arg\min  _  c\sum  _  {i=1}^{N}L(\tilde y  _  i,c)$。它是一颗只有根结点的树，根结点的输出值为：使得损失函数最小的值。
   2. 对于 $m=1,2,\cdots,M$
      1. 对于样本 $i=1,2,\cdots,N$，计算：
         $$
         { r }_ { m,i }={ -\left[ \frac { \partial L\left( \tilde { y } ,f\left( { x }_ { i } \right)  \right)  }{ \partial f\left( { x }_ { i } \right)  }  \right]  }_ { f\left( x \right) ={ f }_ { m-1 }\left( x \right)  }
         $$
      2. 对 $r  _  {m,i}$ 拟合一棵回归树，得到第 $m$ 棵树的叶结点区域 $R  _  {m,j},j=1,2,\cdots,J$
      3. 对 $j=1,2,\cdots,J$ 计算每个区域 $\mathbf R  _  {m,j}$ 上的输出值：
         $$
         c_{mj}=arg\min_c\sum_{x_i\in R_{mj}}{L\left( y_i,f_{m-1}\left( x_i \right) +c \right)}
         $$
      4. 更新 $f  _  m(x)=f  _  {m-1}(x)+\sum  _  {j=1}^{J}c  _  {mj}I(x \in \mathbf R  _  {m,j})$
                
   3. 最终得到回归树：$f  _  M(x)=\sum  _  {m=1}^{M}\sum  _  {j=1}^{J}c  _  {m,j}I(x \in \mathbf R  _  {m,j})$。

梯度提升决策树算法 ($GBDT$) 与 ($GBRT$) 类似，主要区别是 ($GBDT$) 的损失函数与 ($GBRT$) 的损失函数不同。

## 正则化技巧
### 增加学习率
&emsp;&emsp;在工程应用中，通常利用下列公式来更新模型： 

$$
f_m(x)=f_{m-1}(x)+\nu h_m(x;\Theta_m),\quad 0\lt \nu \le1
$$

其中 $\nu$ 称作学习率。学习率是正则化的一部分，它可以降低模型更新的速度（需要更多的迭代）。经验表明，一个小的学习率 ($\nu\lt 0.1$) 可以显著提高模型的泛化能力（相比较于 $\nu=1$) 。如果学习率较大会导致预测性能出现较大波动。

### 随机采样
&emsp;&emsp;$Freidman$ 从 $bagging$ 策略受到启发，采用随机梯度提升来修改了原始的梯度提升树算法。每一轮迭代中，新的决策树拟合的是原始训练集的一个子集（而并不是原始训练集）的残差。这个子集是通过对原始训练集的无放回随机采样而来。子集的占比$f$是一个超参数，并且在每轮迭代中保持不变。如果 $f=1$，则与原始的梯度提升树算法相同；较小的 $f$ 会引入随机性，有助于改善过拟合，因此可以视作一定程度上的正则化；工程经验表明，$0.5 \le f\le 0.8$ 会带来一个较好的结果。

&emsp;&emsp;这种方法除了改善过拟合之外，另一个好处是：未被采样的另一部分子集可以用来计算包外估计误差。因此可以避免额外给出一个独立的验证集。

### 控制叶子节点数
&emsp;&emsp;梯度提升树要求每棵树的叶子结点至少包含$m$个样本，其中 $m$ 为超参数。在训练过程中，一旦划分结点会导致子结点的样本数少于 $m $，则终止划分。这也是一种正则化策略，它会改善叶结点的预测方差。

## RF vs GBT
从模型框架的角度来看：
1. 梯度提升树 ($GBT$) 为 $boosting$ 模型。
2. 随机森林 ($RF$) 为 $bagging$ 模型。

从偏差分解的角度来看：
1. 梯度提升树采用弱分类器（高偏差，低方差）。梯度提升树综合了这些弱分类器，在每一步的过程中降低了偏差，但是保持低方差。
2. 随机森林采用完全成长的子决策树（低偏差，高方差）。随机森林要求这些子树之间尽可能无关，从而综合之后能降低方差，但是保持低偏差。

如果在梯度提升树和随机森林之间二选一，几乎总是建议选择梯度提升树。
1. 随机森林的优点：天然的支持并行计算，因为每个子树都是独立的计算。
2. 梯度提升树的优点：
   1. 梯度提升树采用更少的子树来获得更好的精度。因为在每轮迭代中，梯度提升树会完全接受现有树（投票权为 $1$）。而随机森林中每棵树都是同等重要的（无论它们表现的好坏），它们的投票权都是 $\frac 1N$，因此不是完全接受的。
   2. 梯度提升树也可以修改从而实现并行化。
   3. 梯度提升树有一个明确的数学模型。因此任何能写出梯度的任务，都可以应用梯度提升树（比如 $ranking$ 任务）。而随机森林并没有一个明确的数学模型。

注：由于 $GBDT$ 很容易出现过拟合的问题，所以推荐的 $GBDT$ 深度不要超过 $6$，而随机森林可以在 $15$以上。