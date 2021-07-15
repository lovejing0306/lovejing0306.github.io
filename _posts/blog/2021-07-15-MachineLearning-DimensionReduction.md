---
layout: post
title: 数据降维
categories: [MachineLearning]
description: 数据降维
keywords: MachineLearning
---


数据降维
---


## 维度灾难
&emsp;&emsp;$k$近邻法要求样本点比较密集。给定测试样本点 $x$，理论上希望在$x$附近距离 $\delta \gt 0$ 范围内总能找到一个训练样本 $x^{\prime}$，其中 $\delta$ 是个充分小的正数。即：要求训练样本的采样密度足够大，也称作“密采样”。  
1. 假设 $\delta=0.001$， 且假设样本点只有一个特征，且该特征归一化后范围是 $[0,1]$ ，则需要 $1000$ 个样本点平均分布在 $[0,1]$之间。此时任何测试样本在其附近 $ 0.001 $ 距离范围内总能找到一个训练样本。

2. 假设 $\delta=0.001$，且假设样本点只有十个特征，且该特征归一化后范围是 $[0,1]$，则需要 $10^{30}$ 个样本点平均分布 $[0,1]$ 之间。此时任何测试样本在其附近 $ 0.001 $ 距离范围内总能找到一个训练样本。

&emsp;&emsp;现实应用中特征维度经常成千上万，要满足密采样所需的样本数目是个天文数字。另外许多学习方法都涉及距离计算，而高维空间会给距离计算带来很大的麻烦（高维空间中计算内积都麻烦）。

&emsp;&emsp;在高维情形下出现的数据样本稀疏、距离计算困难等问题是所有机器学习方法共同面临的严重障碍，称作“维度灾难” ($Curse \ of Dimensionality$)。

&emsp;&emsp;缓解维度灾难的一个重要途径是降维 ($Dimension Reduction$)。降维之所以有效的原因是：人们观测或者收集到的样本数据虽然是高维的，但是与学习任务密切相关的也许仅仅是某个低维分布，即高维空间中的一个低维“嵌入”。

常见的降维方法：
1. 监督降维算法。如：线性判别分析 ($Linear Discriminant Analysis:LDA$)。
2. 无监督降维算法。如：主成分分析 ($PCA$)。

&emsp;&emsp;对于降维效果的评估，通常是比较降维前后学习器的性能。如果性能有所提高，则认为降维起了作用。也可以将维数降至二维或者三维，然后通过可视化技术来直观地判断降维效果。
    
&emsp;&emsp;对于常见的降维算法，无论是 $PCA$ 还是流形学习，都是基于距离来计算重构误差。此时建议对特征进行标准化，由于距离的计算依赖于特征的量纲。如身高特征：
1. 如果采用 $m$ 量纲，则取值范围通常在 $1-2$之间。
2. 如果采用 $cm$ 量纲，则取值范围通常在 $100-200$之间。
采用不同的量纲会导致不同的重构误差。

## 主成分分析 PCA
主成分分析 ($Principal \ Component \ Analysis:PCA$) 是最常用的一种降维方法。

### PCA 原理
通过构造新的低维坐标空间，使得原始数据集中的样本点投影到新的低维坐标空间后的方差最大。通过最小化重构误差实现。

#### 坐标变换
&emsp;&emsp;给定数据集 $\mathbb D=\{x  _  1,\cdots,x  _  N\}$，其中 $x  _  i\in \mathbb R^n$。假定样本经过了中心化，即：

$$
x_i \leftarrow x_i- \frac 1N \sum_ {j=1}^{N}x_j
$$

$\bar{x} =\frac 1N \sum  _  {j=1}^{N}x  _  j$ 称作数据集 $\mathbb D$ 的中心向量，它的各元素就是各个特征的均值。之所以进行中心化，是因为经过中心化之后常规的线性变换就是绕原点的旋转变换，也就是坐标变换。

&emsp;&emsp;令 $z=(z  _  1,\cdots,z  _  d)^T,d\lt n$，它表示样本 $x$ 降低到 $d$ 维度。令 $W=(w  _  1,w  _  2,\cdots,w  _  d)$，则有：

$$
z=W^Tx
$$

根据坐标变换矩阵的性质，有：
1. $\|\|w  _  i\|\|  _  2=1$， $i=1,2,\cdots,d$
2. $w  _  i\cdot w  _  j=0,i\ne j$
3. $W^T W=I  _  {d\times d}$。

&emsp;&emsp;对数据集 $\mathbb D$ 中的样本 $x  _  i$，降维后的数据为 $z  _  i$。令：

$$
\begin{matrix} X=\left[ \begin{matrix} { x }_ { 1 }^{ T } \\ \vdots  \\ { x }_ { N }^{ T } \end{matrix} \right] =\begin{bmatrix} { x }_ { 1,1 } & { x }_ { 1,2 } & \cdots  & { x }_ { 1,n } \\ { x }_ { 2,1 } & { x }_ { 2,2 } & \cdots  & { x }_ { 2,n } \\ \vdots  & \vdots  & \ddots  & \vdots  \\ { x }_ { N,1 } & { x }_ { N,2 } & \cdots  & { x }_ { N,n } \end{bmatrix} & Z=\left[ \begin{matrix} { z }_ { 1 }^{ T } \\ \vdots  \\ { z }_ { N }^{ T } \end{matrix} \right] =\begin{bmatrix} { z }_ { 1,1 } & { z }_ { 1,2 } & \cdots  & { z }_ { 1,n } \\ { z }_ { 2,1 } & { z }_ { 2,2 } & \cdots  & { z }_ { 2,n } \\ \vdots  & \vdots  & \ddots  & \vdots  \\ { z }_ { N,1 } & { z }_ { N,2 } & \cdots  & { z }_ { N,n } \end{bmatrix} \end{matrix}
$$

即 $X$ 的第 $i$ 行就是样本 $x  _  i^T$，$Z$ 的第 $i$ 行就是降维后的数据 $z  _  i^T$。
1. 令 $u  _  j=(x  _  {1,j},\cdots,x  _  {N,j})^T$，它表示 $X$ 的第 $j$ 列，也就是原始的第 $j$ 个特征。
2. 令 $v  _  j=(z  _  {1,j},\cdots,z  _  {N,j})^T$，它表示 $Z$ 的第 $j$ 列，也就是降维之后的第 $j$ 个特征。

则根据 $z  _  {i,j}=w  _  j\cdot x  _  i=\sum  _  {k=1}^n w  _  {j,k}\times x  _  {i,k}$，有：

$$
 v_j = \sum_ {k=1}^n w_ {j,k} u_j
$$

因此降维的物理意义为：通过线性组合原始特征，从而去掉一些冗余的或者不重要的特征、保留重要的特征。


#### 重构误差
&emsp;&emsp;考虑对$z$后的样本为： 

$$
\hat{x} = W z
$$

对整个数据集 $\mathbb D$ 所有重建样本与原始样本的误差为：

$$
\sum_ {i=1}^{N} \|\|\hat{x}_ i- x_i \|\|_ 2^{2}=\sum_ {i=1}^{N} \|\|WW^{T} x_i -x_i\|\|_ 2^{2}
$$

根据定义有：

$$
W W ^{T} x_i=(w_1,w_2,\cdots,w_d)\begin{bmatrix}w_1^{T}\\ w_2^{T}\\ \vdots \\ w_d^{T}\end{bmatrix} x_i=\sum_ {j=1}^{d}w_j(w_j^{T} x_i)
$$

由于 $w  _  j^{T} x  _  i$ 是标量，所以有： 

$$
W W^{T} x_i=\sum_ {j=1}^{d}(w_j^{T} x_i)w_j
$$

由于标量的转置等于它本身，所以有： 

$$
W W^{T} x_i=\sum_ {j=1}^{d}(x_i^Tw_j)w_j
$$

则有：

$$
\begin{aligned} \sum _ { i=1 }^{ N }{ { \| { \hat { x }  }_ { i }-{ x }_ { i } \|  }_ { 2 }^{ 2 } } &=\sum _ { i=1 }^{ N }{ { \| W{ W }^{ T }{ x }_ { i }-{ x }_ { i } \|  }_ { 2 }^{ 2 } }  \\ &=\sum _ { i=1 }^{ N }{ { \| \sum _ { j=1 }^{ d }{ \left( { x }_ { i }^{ T }{ w }_ { j } \right) { w }_ { j }-{ x }_ { i } }  \|  }_ { 2 }^{ 2 } }  \\ &=\sum _ { i=1 }^{ N }{ { \| { x }_ { i }-\sum _ { j=1 }^{ d }{ \left( { x }_ { i }^{ T }{ w }_ { j } \right) { w }_ { j } }  \|  }_ { 2 }^{ 2 } }  \end{aligned}
$$

根据$X$的定义，可以证明 $\|\|\cdot\|\|  _  F$ 为矩阵的 $Frobenius$ 范数：

$$
\|\|X- XWW^{T}\|\|_F^{2}=\sum_ {i=1}^{N}\|\|x_i- \sum_ {j=1}^{d}(x_i^{T} w_j)w_j\|\|_2^{2}
$$

证明：

$$
\begin{aligned} X-XW{ W }^{ T }&=X-\left[ \begin{matrix} { x }_ { 1 }^{ T } \\ \vdots  \\ { x }_ { N }^{ T } \end{matrix} \right] \left[ \begin{matrix} { w }_ { 1 } & \cdots  & { w }_ { d } \end{matrix} \right] \left[ \begin{matrix} { w }_ { 1 }^{ T } \\ \vdots  \\ { w }_ { d }^{ T } \end{matrix} \right]  \\ &=X-\begin{bmatrix} { x }_ { 1 }^{ T }{ w }_ { 1 } & \cdots  & { x }_ { 1 }^{ T }{ w }_ { d } \\ \vdots  & \ddots  & \vdots  \\ { x }_ { N }^{ T }{ w }_ { 1 } & \cdots  & { x }_ { N }^{ T }{ w }_ { d } \end{bmatrix}\left[ \begin{matrix} { w }_ { 1 }^{ T } \\ \vdots  \\ { w }_ { d }^{ T } \end{matrix} \right]  \\ &=X-\begin{bmatrix} \sum _ { k=1 }^{ d }{ { w }_ { k,1 }\times { x }_ { 1 }^{ T }{ w }_ { k } }  & \cdots  & \sum _ { k=1 }^{ d }{ { w }_ { k,n }\times { x }_ { 1 }^{ T }{ w }_ { k } }  \\ \vdots  & \ddots  & \vdots  \\ \sum _ { k=1 }^{ d }{ { w }_ { k,1 }\times { x }_ { N }^{ T }{ w }_ { k } }  & \cdots  & \sum _ { k=1 }^{ d }{ { w }_ { k,n }\times { x }_ { N }^{ T }{ w }_ { k } }  \end{bmatrix} \end{aligned}
$$

则有：

$$
\begin{aligned} { \| X-XW{ W }^{ T } \|  }_ { F }^{ 2 }&=\sum _ { i=1 }^{ N }{ \sum _ { j=1 }^{ n }{ { \left[ { x }_ { i,j }-\left( \sum _ { k=1 }^{ d }{ { w }_ { k,j }\times { x }_ { i }^{ T }{ w }_ { k } }  \right)  \right]  }^{ 2 } }  }  \\ &=\sum _ { i=1 }^{ N }{ { \| { x }_ { i }-\sum _ { k=1 }^{ d }{ \left( { x }_ { i }^{ T }{ w }_ { k } \right) { w }_ { k } }  \|  }_ { 2 }^{ 2 } }  \end{aligned}
$$

将最后的下标从 $k$ 替换为 $j$ 即可得证。


&emsp;&emsp;$PCA$ 降维要求重构误差最小。现在求解最优化问题：

$$
\begin{aligned} { W }^{ * }&=arg\max _ { W }{ \sum _ { i=1 }^{ N }{ { \| { \hat { x }  }_ { i }-{ x }_ { i } \|  }_ { 2 }^{ 2 } }  }  \\ &=arg\min _ { W }{ { \| X-XW{ W }^{ T } \|  }_ { F }^{ 2 } }  \\ &=arg\min _ { W }{ tr\left[ { \left( X-XW{ W }^{ T } \right)  }^{ T }\left( X-XW{ W }^{ T } \right)  \right]  }  \\ &=arg\min _ { W }{ tr\left[ \left( { X }^{ T }-W{ W }^{ T }{ X }^{ T } \right) \left( X-XW{ W }^{ T } \right)  \right]  }  \\ &=arg\min _ { w }{ tr\left[ { X }^{ T }X-{ X }^{ T }XW{ W }^{ T }-W{ W }^{ T }{ X }^{ T }X+W{ W }^{ T }{ X }^{ T }XW{ W }^{ T } \right]  }  \\ &=arg\min _ { w }{ tr\left[ tr\left( { X }^{ T }X \right) -tr\left( { X }^{ T }XW{ W }^{ T } \right) -tr\left( W{ W }^{ T }{ X }^{ T }X \right) +tr\left( W{ W }^{ T }{ X }^{ T }XW{ W }^{ T } \right)  \right]  }  \end{aligned}
$$

因为矩阵及其转置的迹相等，因此 

$$
tr(X^T X  W W^{T})=tr(W W^{T} X^TX ) 
$$

由于可以在 $tr(\cdot)$ 中调整矩阵的顺序，则

$$
tr(W W^{T} X^TX W W^{T})=tr(X^T X  W W^{T} W W^{T})
$$ 

考虑到：

$$
W^{T} W=\begin{bmatrix}w_1^{T}\\ \vdots \\ w_d^{T}\end{bmatrix}(w_1,\cdots,w_d) =\mathbf I_ {d\times d}
$$

代入上式有：

$$
tr(W W^{T}X^TX W W^{T})=tr(X^TX  W W^{T})
$$

于是有：

$$
W^{ \ * }=\arg\min_ {W}\left[tr(X^TX)-tr(X^TX WW^{T})\right]
$$

由于 $tr(X^TX)$ 与 $W$ 无关，因此：

$$
W^{ \ * } =\arg\min_ {W} -tr(X^{T} X WW^{T}) = \arg\max_ {W}tr(X^{T}X W W^{T})
$$

调整矩阵顺序，则有：

$$
W^{ \ * } =\arg\max_ {W}tr(W^T X ^{T} X W)
$$

其约束条件为：$W ^{T}W=\mathbf I  _  {d\times d}$。

&emsp;&emsp;$PCA$ 最优化问题需要求解就是 $X^T X$的特征值 。只需要对矩阵 $X^TX \in \mathbb R^{n\times n}$ 进行特征值分解，将求得的特征值排序：$\lambda  _  1\ge \lambda  _  2\ge \cdots \ge\lambda  _  n$。然后取前 $d$ 个特征值对应的单位特征向量构成坐标变换矩阵 $W=(w  _  1,w  _  2,\cdots,w  _  d)$。

&emsp;&emsp;当样本数据进行了中心化时 ，$\sum  _  {i=1}^{N}x  _  i x  _  i^{T}=X^{T}X$ 就是样本集的协方差矩阵。这也是为什么需要对样本进行中心化的一个原因。

### PCA 算法
#### 伪码
输入：
1. 样本集 $\mathbb D=x  _  1,x  _  2,\cdots,x  _  N $
2. 低维空间维数 d 

输出：投影矩阵$ W=(w  _  1,w  _  2,\cdots,w  _  d)$。

算法步骤：
1. 对所有样本进行中心化操作： 
   $$
   x_i \leftarrow x_i- \frac 1N \sum_ {j=1}^{N}x_j 
   $$
2. 计算样本的协方差矩阵 $X^TX$
3. 对协方差矩阵 $X^TX$ 做特征值分解。
4. 取最大的 $d$ 个特征值对应的单位特征向量 $w  _  1,w  _  2,\cdots,w  _  d$，构造投影矩阵 $W=(w  _  1,w  _  2,\cdots,w  _  d)$。
5. 投影矩阵 $W=(w  _  1,w  _  2,\cdots,w  _  d)$ 相当于是新的坐标空间。
6. 将样本点投影到投影矩阵上，将样本点投影到投影矩阵后才是降维后的样本。

##### $d$取值
通常低维空间维数 $d$ 的选取有两种方法：  
1. 通过交叉验证法选取较好的 $d$。“比较好”指的是在降维后的学习器的性能比较好。
2. 从算法原理的角度设置一个阈值，比如 $t=95\%$，然后选取使得下式成立的最小的 $d$ 的值：
   $$
   \frac{\sum_ {i=1}^{d }\lambda_i}{\sum_ {i=1}^{n}\lambda_i} \ge t
   $$
   其中 $\lambda  _  i$ 从大到小排列。

### 性质
&emsp;&emsp;从物理意义上看：给定协方差矩阵 $X^T X$，通过坐标变换将其对角化为矩阵：
$$
\Lambda=\begin{pmatrix} \lambda_1&0&0&\cdots&0\\ 0&\lambda_2&0&\cdots&0\\ \vdots&\vdots&\vdots&\ddots&\vdots\\ 0&0&0&\cdots&\lambda_n\\ \end{pmatrix}
$$

这相当于在新的坐标系中，任意一对特征之间的协方差为 $0$;单个特征的方差为 $\lambda  _  i,i=1,2,\cdots,n$。即：数据在每个维度上尽可能分散，且任意两个维度之间不相关。降维的过程就是寻找这样的一个坐标变换，也就是坐标变换矩阵 $W$。由于协方差矩阵 $X^T X$ 是对称矩阵，根据实对称矩阵的性质，这样的坐标变换矩阵一定存在。

&emsp;&emsp;$PCA$ 算法中，低维空间与高维空间必然不相同。因为末尾 $n-d$ 个最小的特征值对应的特征向量被抛弃了，这就是降维导致的结果。舍弃这部分信息之后能使得样本的采样密度增大（因为维数降低了），这是缓解维度灾难的重要手段。当数据受到噪声影响时，最小特征值对应的特征向量往往与噪声有关，将它们舍弃能在一定程度上起到降噪的效果。

&emsp;&emsp;$PCA$ 降低了输入数据的维度同时保留了主要信息/能量，但是这个主要信息只是针对训练集的，而且这个主要信息未必是重要信息。有可能舍弃了一些看似无用的信息，但是这些看似无用的信息恰好是重要信息，只是在训练集上没有很大的表现，所以 $PCA$ 也可能加剧了过拟合。

&emsp;&emsp;$PCA$ 中训练样本越多越好。如果训练样本太少，则训练集很有可能“偶然”近似落在同一个平面上。极端情况下，如果样本数量小于目标维度，比如样本数量为 $100$，目标维度为 $1000$ 维。则这 $100$ 个样本总可以构成一个 $1000$ 维的平面，且这样的平面有无穷多个。此时如果进行 $PCA$ 降维，则前几个特征值 $\lambda$ 占比非常大。

&emsp;&emsp;但是如果将样本数量扩充为 $10000$，则这些样本构成一个 $1000$ 维的平面的巧合就几乎很难成立。此时如果进行 $PCA$ 降维，则前几个特征值 $\lambda$ 占比就会降低。本质上是因为 $N$ 决定了协方差矩阵 $X^TX$ 的秩的上界。当 $N$ 较小时，$rank(X^TX)$ 也会很小，导致大量的特征值 $\lambda$ 为 $0$。

&emsp;&emsp;$PCA$ 不仅将数据压缩到低维，它也使得降维之后的数据各特征相互独立。注意：$PCA$ 推导过程中，并没有要求数据中心化；但是在推导协方差矩阵时，要求数据中心化。此时：

$$
Var[z]=\frac{1}{N-1}Z^{T}Z=\frac{1}{N-1}\Sigma^{2}
$$

其中：$\Sigma$ 为 $X^T X$ 的最大的 $d$ 个特征值组成的对角矩阵。$Z$ 为降维后的样本集组成的矩阵。

&emsp;&emsp;对于训练集、验证集、测试集，当对训练集进行 $PCA$ 降维时，也需要对验证集、测试集执行同样的降维。注意：对验证集、测试集执行中心化操作时，中心向量必须从训练集计算而来。不能使用验证集的中心向量，也不能用测试集的中心向量。


### 最大可分性
$PCA$ 降维的准则有两个：
1. 最近重构性：样本集中所有点，重构后的点距离原来的点的误差之和最小（就是前面介绍的内容）。
2. 最大可分性：样本点在低维空间的投影尽可能分开。

可以证明，最近重构性等价于最大可分性。证明如下：对于样本点 $x  _  i$， 它在降维后空间中的投影是 $z  _  i$。 则有： 

$$
z=W^Tx
$$

由于样本数据进行了中心化，则投影后样本点的方差是：

$$
\sum_ {i=1}^Nz_i z_i^T=\sum_ {i=1}^{N}W^T x_i x_i^{T}W
$$

根据 $X$ 的定义，有：

$$
tr(W^T X^T X W ) = \sum_ {i=1}^{N}W^T x_ix_i^{T}W 
$$

则样本点的方差最大的优化目标可写作：

$$
\max_ {W} tr(W^T X^{T} X W)\ s.t. W ^{T}W=I_ {d\times d}
$$

这就是前面最近重构性推导的结果。

> $I  _  {d\times d} $ 对角矩阵的性质特征列之间的是无相关性的，这样保证了特征之间的相互独立。

### PCA 与 LDA 的区别
&emsp;&emsp;$LDA$ 也可以用于降维。对于$2$维空间降低到$1$维直线的情况下，它设法将样例投影到某一条直线上，使得同类样例的投影点尽可能接近、异类样例的投影点尽可能远离。
1. $LDA$ 考虑的是：向类别区分最大的方向投影。如下图中的绿色投影直线。
2. $PCA$ 考虑的是：向方差最大的方向投影。如下图中的紫色投影直线。

因此$LDA$降维对于类别的区分效果要好的多。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/DimensionReduction/pca_lda.png?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">pca_lda</div>
</center>

### PCA 与 SVD
&emsp;&emsp;酉矩阵：若 $n$ 阶矩阵满足 $\mathbf U^H\mathbf U=\mathbf U\mathbf U^H=\mathbf I$，则它是酉矩阵。其中 $\mathbf U^H$ 为 $\mathbf U$ 的共轭转置。$\mathbf U$ 为酉矩阵的充要条件是：$\mathbf U^H=\mathbf U^{-1}$。

&emsp;&emsp;奇异值分解：假设 $M$ 是一个$m×n$的矩阵，如果存在一个分解：

$$
M=U\Sigma { V }^{ T }
$$

其中 $U$ 为 $m×m$ 的酉矩阵，$\Sigma$ 为 $m×n$ 的半正定对角矩阵，${ V }^{ T }$ 为 $V$ 的共轭转置矩阵，且为 $n \times n$ 的酉矩阵。这样的分解称为 $M$ 的奇异值分解 ，$\Sigma$ 对角线上的元素称为奇异值，$U$ 称为左奇异矩阵，${ V }^{ T }$ 称为右奇异矩阵。

&emsp;&emsp;对 $m×n$ 的矩阵 $M$,进行奇异值分解：

$$
{ M }_ { m\times n }={ U }_ { m\times m }{ \Sigma  }_ { m\times n }{ V }_ { n\times n }
$$

取其前 $r$ 个非 $0$ 奇异值（要保证前 $r$ 个奇异值能够保留矩阵中 $90\%$ 的能量信息），可以近似表示原来的矩阵 $M$

&emsp;&emsp;若是做数据降维可以取近似后的 $U$ 和 $V^T$，使用近似后的 $U$ 和 $V^T$ 可以做投影矩阵，如在 $PCA$ 中可以使用 $SVD$ 的右奇异矩阵 $V  _  (r×n)^T$ 来做投影矩阵，对原始矩阵 $X$ 做变换 $X  _  (m×n) V  _  (n×r)$；若是做数据压缩则可以使用近似后的 $M$。
    
&emsp;&emsp;$SVD$ 奇异值分解等价于 $PCA$ 主成分分析，核心都是求解 $M^T M$ 的特征值以及对应的单位特征向量。关系推导如下：

$$
\begin{aligned} { M }^{ T }M={ \left( U\Sigma { V }^{ T } \right)  }^{ T }U\Sigma { V }^{ T }=V\left( { \Sigma  }^{ T }\Sigma  \right) { V }^{ T } \\ M{ M }^{ T }=U\Sigma { V }^{ T }{ \left( U\Sigma { V }^{ T } \right)  }^{ T }=U\left( { \Sigma  }^{ T }\Sigma  \right) { U }^{ T } \end{aligned}
$$

其中，$M^T M$ 和 $MM^T$ 是方阵，$V^T$ 为 $M^T M$ 的特征向量，$U$ 为 $ MM^T$ 的特征向量。$M^T M$ 和 $MM^T$ 的特征值为 $M$ 的奇异值的平方。关系推导如下：

$$
M{ M }^{ T }=U\left( { \Sigma  }^{ T }\Sigma  \right) { U }^{ T }
$$

在等式的两边同时乘以矩阵$U$，得：

$$
\begin{aligned} M{ M }^{ T }U=U\left( { \Sigma  }^{ T }\Sigma  \right) { U }^{ T }U \\ \Rightarrow M{ M }^{ T }U=U\left( { \Sigma  }^{ T }\Sigma  \right)  \end{aligned}
$$

由于 ${ \Sigma  }^{ T }\Sigma $ 为单位矩阵，因此有 $U\left( { \Sigma  }^{ T }\Sigma  \right) =\left( { \Sigma  }^{ T }\Sigma  \right) U$
于是有：

$$
M{ M }^{ T }U=\left( { \Sigma  }^{ T }\Sigma  \right) U
$$

由此得出矩阵 $U$ 为 $M{ M }^{ T }$ 的特征向量；${ \Sigma  }^{ T }\Sigma $ 为 $M{ M }^{ T }$ 的特征值。

---

## 核化线性降维 KPCA

&emsp;&emsp;$PCA$ 方法假设从高维空间到低维空间的映射是线性的，但是在不少现实任务中可能需要非线性映射才能找到合适的低维空间来降维。非线性降维的一种常用方法是基于核技巧对线性降维方法进行核化 $kernelized$， 如核主成分分析 ($Kernelized PCA:KPCA$)，它是对 $PCA$ 的一种推广。

&emsp;&emsp;假定原始特征空间中的样本点 $x  _  i$ 通过映射 $\phi$ 映射到高维特征空间的坐标为 $x  _  {i,\phi}$，即 $x  _  {i,\phi}=\phi(x  _  i)$。且假设高维特征空间为 $n$ 维的，即：$x  _  {i,\phi} \in \mathbb R^{n}$。假定要将高维特征空间中的数据投影到低维空间中，投影矩阵为 $W$ 为 $n\times d$ 维矩阵。

根据 $PCA$ 推导的结果，求解方程：

$$
X_ {\phi}^{T} X_ {\phi} W=\lambda \mathbf W
$$

其中 $X  _  {\phi}=(x  _  {1,\phi}^T,x  _  {2,\phi}^T,\cdots,x  _  {N,\phi}^T)^T$ 为 $N \times n $ 维矩阵。于是有：

$$
\left(\sum_ {i=1}^{N}\phi(x_i)\phi(x_i)^{T}\right)\mathbf W=\lambda \mathbf W 
$$

&emsp;&emsp;通常并不清楚 $\phi$ 的解析表达式，因此并不会直接得到 $X  _  \phi$，所以无法直接求解方程：

$$
X_ {\phi}^{T}X_ {\phi} W=\lambda W 
$$

于是引入核函数：

$$
\kappa(x_i,x_j)=\phi(x_i)^{T}\phi(x_j)
$$

定义核矩阵：

$$
\mathbf K=\begin{bmatrix} \kappa(x_1,x_1)&\kappa(x_1,x_2)&\cdots&\kappa(x_1,x_N)\\ \kappa(x_2,x_1)&\kappa(x_2,x_2)&\cdots&\kappa(x_2,x_N)\\ \vdots&\vdots&\ddots&\vdots\\ \kappa(x_N,x_1)&\kappa(x_N,x_2)&\cdots&\kappa(x_N,x_N)\\ \end{bmatrix}
$$

则有： 

$$
X_ {\phi} X_ {\phi}^{T}=K
$$

定义 $\alpha  _  i=\frac{x  _  {i,\phi}^{T} W}{\lambda}$，则 $\alpha  _  i$ 为$1\times d$维行向量。定义：$A=(\alpha  _  1,\alpha  _  2,\cdots,\alpha  _  N)^{T}$ 为 $N \times d$ 维矩阵则有：

$$
\begin{aligned} W&=\frac { 1 }{ \lambda  } \left( \sum _ { i=1 }^{ N }{ { x }_ { i,\phi  }{ x }_ { i,\phi  }^{ T } }  \right) W \\ &=\sum _ { i=1 }^{ N }{ { x }_ { i,\phi  }\frac { { x }_ { i,\phi  }W }{ \lambda  }  }  \\ &=\sum _ { i=1 }^{ N }{ { x }_ { i,\phi  }{ \alpha  }_ { i } }  \\ &={ X }_ { \phi  }^{ T }A \end{aligned}
$$

将 $W=X  _  {\phi}^T A$代入$X  _  {\phi}^{T} X  _  {\phi} W=\lambda W$，有：

$$
X_ {\phi}^{T} X_ {\phi} X_ {\phi}^{T} A=\lambda X_ {\phi}^{T} A
$$

两边同时左乘以 $X  _  {\phi}$，再代入 $X  _  {\phi} X  _  {\phi}^{T}=K$ 有：

$$
KKA=\lambda K A
$$

通常会要求核矩阵可逆，上式两边同时左乘以 $K^{-1}$，则有：

$$
K A=\lambda A
$$

同样该问题也是一个特征值分解问题，取 $K$ 最大的 $d$ 个特征值对应的特征向量组成 $A$ 即可。

&emsp;&emsp;对于新样本 $x$， 其投影后第 $j$ 维的坐标为：

$$
\begin{aligned} { z }_ { j }&={ w }_ { j }^{ T }\phi \left( x \right)  \\ &=\sum _ { i=1 }^{ N }{ { \alpha  }_ { i,j }{ \phi \left( { x }_ { i } \right)  }^{ T }\phi \left( x \right)  }  \\ &=\sum _ { i=1 }^{ N }{ { \alpha  }_ { i,j }\kappa \left( { x }_ { i },x \right)  }  \end{aligned}
$$

其中 $\alpha  _  {i,j}$ 为行向量 $\alpha  _  i$ 的第 $ j$ 个分量。

可以看到：为了获取投影后的坐标，$KPCA$ 需要对所有样本求和，因此它的计算开销较大。
    

## 流形学习
&emsp;&emsp;流形学习 ($manifold learning$) 是一类借鉴了拓扑流形概念的降维方法。流形是在局部和欧氏空间同胚的空间，它在局部具有欧氏空间的性质，能用欧氏距离进行距离计算。如果低维流形嵌入到高维空间中，则数据样本在高维空间的分布虽然看起来非常复杂，但是在局部上仍然具有欧氏空间的性质。

&emsp;&emsp;当维数被降低至二维或者三维时，能对数据进行可视化展示，因此流形学习也可用于可视化。流形学习若想取得很好的效果，则必须对邻域保持样本密采样，但这恰恰是高维情形下面临的重大障碍。因此流形学习方法在实践中的降维性能往往没有预期的好。

&emsp;&emsp;流形学习对于噪音数据非常敏感。噪音数据可能出现在两个区域连接处：
1. 如果没有出现噪音，这两个区域是断路的。
2. 如果出现噪音，这两个区域是短路的。

### 多维缩放

#### 多维缩放原理
&emsp;&emsp;多维缩放 ($Multiple Dimensional Scaling:MDS$) 要求原始空间中样本之间的距离在低维空间中得到保持。假设 $N$ 个样本在原始空间中的距离矩阵为 $D=(d  _  {ij})  _  {N\times N}$:

$$
D=\begin{bmatrix} d_ {1,1}&d_ {1,2}&\cdots&d_ {1,N}\\ d_ {2,1}&d_ {2,2}&\cdots&d_ {2,N}\\ \vdots&\vdots&\ddots&\vdots\\ d_ {N,1}&d_ {N,2}&\cdots&d_ {N,N}\\ \end{bmatrix}
$$

其中 $d  _  {ij}=\|\|x  _  i-x  _  j\|\|$ 为样本 $x  _  i$ 到样本 $x  _  j$ 的距离。

&emsp;&emsp;假设原始样本是在 $n$ 维空间，目标是获取样本在 $n^{\prime}$ 维空间且欧氏距离保持不变，其中满足 $n^{\prime} \lt n$。假设样本集在原空间的表示为 $X \in \mathbb R^{N\times n}$，样本集在降维后空间的表示为 $Z \in \mathbb R^{N\times n^{\prime}}$。

$$
\begin{matrix} X=\left[ \begin{matrix} { x }_ { 1 }^{ T } \\ \vdots  \\ { x }_ { N }^{ T } \end{matrix} \right] =\begin{bmatrix} { x }_ { 1,1 } & { x }_ { 1,2 } & \cdots  & { x }_ { 1,n } \\ { x }_ { 2,1 } & { x }_ { 2,2 } & \cdots  & { x }_ { 2,n } \\ \vdots  & \vdots  & \ddots  & \vdots  \\ { x }_ { N,1 } & { x }_ { N,2 } & \cdots  & { x }_ { N,n } \end{bmatrix} & Z=\left[ \begin{matrix} { z }_ { 1 }^{ T } \\ \vdots  \\ { z }_ { N }^{ T } \end{matrix} \right] =\begin{bmatrix} { z }_ { 1,1 } & { z }_ { 1,2 } & \cdots  & { z }_ { 1,{ n }^{ \prime  } } \\ { z }_ { 2,1 } & { z }_ { 2,2 } & \cdots  & { z }_ { 2,{ { n }^{ \prime  } } } \\ \vdots  & \vdots  & \ddots  & \vdots  \\ { z }_ { N,1 } & { z }_ { N,2 } & \cdots  & { z }_ { N,{ n }^{ \prime  } } \end{bmatrix} \end{matrix}
$$

所求的正是 $Z$ 矩阵，同时也不知道 $n^{\prime}$。很明显：并不是所有的低维空间都满足线性映射之后，样本点之间的欧氏距离保持不变。


&emsp;&emsp;令 $B=ZZ^{T} \in \mathbb R^{N\times N}$，即 ：

$$
B=\begin{bmatrix} b_ {1,1}&b_ {1,2}&\cdots&b_ {1,N}\\ b_ {2,1}&b_ {2,2}&\cdots&b_ {2,N}\\ \vdots&\vdots&\ddots&\vdots\\ b_ {N,1}&b_ {N,2}&\cdots&b_ {N,N}\\ \end{bmatrix}
$$

其中 $b  _  {i,j}=z  _  i \cdot z  _  j$ 为降维后样本的内积。则根据降维前后样本的欧氏距离保持不变有：

$$
\begin{aligned} { d }_ { ij }^{ 2 }&={ \| { z }_ { i }-{ z }_ { j } \|  }^{ 2 } \\ &={ \| { z }_ { i } \|  }^{ 2 }+{ \| { z }_ { j } \|  }^{ 2 }-2{ z }_ { i }^{ T }{ z }_ { j } \\ &={ b }_ { i,i }+{ b }_ { j,j }-2{ b }_ { i,j } \end{aligned}
$$

假设降维后的样本集 $Z$ 被中心化，即 $\sum  _  {i=1}^{N}z  _  i=0$，则矩阵 $B$ 的每行之和均为零，每列之和均为零。即：

$$
\begin{aligned} \sum _ { i=1 }^{ N }{ { b }_ { i,j } } =0,j=1,2,\cdots ,N \\ \sum _ { j=1 }^{ N }{ { b }_ { i,j } } =0,i=1,2,\cdots ,N \end{aligned}
$$

于是有：

$$
\begin{aligned} \sum _ { i=1 }^{ N }{ { d }_ { i,j }^{ 2 } } &=\sum _ { i=1 }^{ N }{ { b }_ { i,i } } +N{ b }_ { j,j } \\ &=tr\left( B \right) +N{ b }_ { j,j } \\ \sum _ { j=1 }^{ N }{ { d }_ { i,j }^{ 2 } } &=\sum _ { j=1 }^{ N }{ { b }_ { j,j } } +N{ b }_ { i,i } \\ &=tr\left( B \right) +N{ b }_ { i,i } \\ \sum _ { i=1 }^{ N }{ \sum _ { j=1 }^{ N }{ { d }_ { i,j }^{ 2 } }  } &=\sum _ { i=1 }^{ N }{ \left( tr\left( B \right) +N{ b }_ { i,i } \right)  }  \\ &=2Ntr\left( B \right)  \end{aligned}
$$

令：

$$
\begin{aligned} { d }_ { i,\cdot  }^{ 2 }&=\frac { 1 }{ N } \sum _ { j=1 }^{ N }{ { d }_ { i,j }^{ 2 } }  \\ &=\frac { tr\left( B \right)  }{ N } +{ b }_ { i,i } \\ { d }_ { j,\cdot  }^{ 2 }&=\frac { 1 }{ N } \sum _ { i=1 }^{ N }{ { d }_ { i,j }^{ 2 } }  \\ &=\frac { tr\left( B \right)  }{ N } +{ b }_ { j,j } \\ { d }_ { \cdot ,\cdot  }^{ 2 }&=\frac { 1 }{ { N }^{ 2 } } \sum _ { i=1 }^{ N }{ \sum _ { j=1 }^{ N }{ { d }_ { i,j }^{ 2 } }  }  \\ &=2Ntr\left( B \right)  \end{aligned}
$$

代入 $d  _  {ij}^{2}=b  _  {i,i}+b  _  {j,j}-2b  _  {i,j}$，有：

$$
\begin{aligned} { b }_ { i,j }=\frac { { b }_ { i,i }+{ b }_ { j,j }-{ d }_ { i,j }^{ 2 } }{ 2 }  \\ =\frac { { d }_ { i,\cdot  }^{ 2 }+{ d }_ { j,\cdot  }^{ 2 }-{ d }_ { \cdot ,\cdot  }^{ 2 }-{ d }_ { i,j }^{ 2 } }{ 2 }  \end{aligned}
$$

右式根据 $d  _  {ij}$ 给出了 $b  _  {i,j}$，因此可以根据原始空间中的距离矩阵 $D$ 求出在降维后空间的内积矩阵 $B$。现在的问题是已知内积矩阵 $B=Z Z^{T}$，如何求得矩阵 $Z$。

&emsp;&emsp;对矩阵 $ B$ 做特征值分解，设 $B=V \Lambda V^T$，其中

$$
\Lambda=\begin{bmatrix}\lambda_1&0&0&\cdots&0\\ 0&\lambda_2&0&\cdots&0\\ \vdots&\vdots&\vdots&\ddots&\vdots\\ 0&0&0&\cdots&\lambda_N \end {bmatrix}
$$

为特征值构成的对角矩阵，$\lambda  _  1 \ge\lambda  _  2\ge\cdots\ge\lambda  _  N$，$\mathbf V$ 为特征向量矩阵。

&emsp;&emsp;假定特征值中有 $n^{ \ * }$ 个非零特征值，它们构成对角矩阵$\Lambda^{ \ * }=diag(\lambda  _  1,\lambda  _  2,\cdots,\lambda  _  {n^{ \ * }})$。令$V^{ \ * }$ 为对应的特征向量矩阵，则 $Z= V^{ \ * }\Lambda^{ \ * \;1/2}$。此时有 $n^{\prime}=n^{ \ * }$，$Z\in \mathbb R^{N\times n^{ \ * } }$。

&emsp;&emsp;在现实应用中为了有效降维，往往仅需要降维后的距离与原始空间中的距离尽可能相等，而不必严格相等。此时可以取 $n^{\prime} \ll n^{ \ * } \le n$ 个最大特征值构成对角矩阵 $\tilde \Lambda=diag(\lambda  _  1,\lambda  _  2,\cdots,\lambda  _  {n^{\prime}})$。令 $\tilde{V}$ 表示对应的特征向量矩阵，则 $Z= \tilde{V} \tilde\Lambda^{1/2} \in \mathbb R^{ N\times n^{\prime}}$。

#### 多维缩放算法
输入：
1. 距离矩阵 $D \in \mathbb R^{N\times N}$。
2. 低维空间维数 $n^{\prime}$。

输出：样本集在低维空间中的矩阵 $Z$。

算法步骤：
1. 根据下列式子计算 $d  _  {i,\cdot}^{2},d  _  {j,\cdot}^{2},d  _  {\cdot,\cdot}^{2}$：
   $$
   d_ {i,\cdot}^{2}=\frac 1N \sum_ {j=1}^{N}d_ {ij}^{2},\quad d_ {j,\cdot}^{2}=\frac 1N \sum_ {i=1}^{N}d_ {ij}^{2},\quad d_ {\cdot,\cdot}^{2}=\frac {1}{N^{2}}\sum_ {i=1}^{N}\sum_ {j=1}^{N}d_ {ij}^{2}
   $$
2. 根据下式计算矩阵 $B$：
   $$
   b_ {i,j}=\frac{d_ {i,\cdot}^{2}+d_ {j,\cdot}^{2}-d_ {\cdot,\cdot}^{2}-d_ {ij}^{2}}{2}
   $$   
3. 对矩阵 $B$ 进行特征值分解。
4. 取 $\tilde \Lambda$为$n^{\prime}$ 个最大特征值所构成的对角矩阵，$\tilde{V}$ 表示对应的特征向量矩阵，计算：$Z= \tilde{V} \tilde\Lambda^{1/2} \in \mathbb R^{ N\times n^{\prime}}$。

### 等度量映射

#### 算法
&emsp;&emsp;等度量映射 ($Isometric Mapping:Isomap$) 的基本观点是：低维流形嵌入到高维空间后，直接在高维空间中计算直线距离具有误导性。因为在高维空间中的直线距离在低维嵌入流形上是不可达的。

&emsp;&emsp;低维嵌入流形上，两点之间的距离是“测地线” (geodesic)距离。计算测地线距离的方法是：利用流形在局部上与欧氏空间同胚这个性质，对每个点基于欧氏距离找出它在低维流形上的近邻点， 然后就能建立一个近邻连接图。
1. 图中近邻点之间存在链接。
2. 图中非近邻点之间不存在链接。
于是计算两点之间测地线距离的问题转变为计算近邻连接图上两点之间的最短路径问题（可以通过著名的 $Dijkstra$ 算法或者 $Floyd$ 算法）。在得到任意两点的距离之后，就可以通过$MDS$算法来获得样本点在低维空间中的坐标。

##### $Isomap$算法：  
输入：
1. 样本集 $\mathbb D=\{x  _  1,x  _  2,\cdots,x  _  N\}$
2. 近邻参数 $k$。
3. 低维空间维数 $n^{\prime}$。

输出：样本集在低维空间中的矩阵 $Z$。

算法步骤：
1. 对每个样本点 $x  _  i$，计算它的 $k$ 近邻。同时将 $x  _  i$ 与它的 $ k $ 近邻的距离设置为欧氏距离，与其他点的距离设置为无穷大。
2. 调用最短路径算法计算任意两个样本点之间的距离，获得距离矩阵 $D \in \mathbb R^{N\times N}$。
3. 调用多维缩放 $MDS$ 算法，获得样本集在低维空间中的矩阵 $Z$。

#### 性质
$Isomap$ 算法有个很大的问题：对于新样本，如何将其映射到低维空间？常用的方法是：
1. 将训练样本的高维空间坐标作为输入，低维空间坐标作为输出，训练一个回归学习器。
2. 用这个回归学习器来对新样本的低维空间进行预测。

这仅仅是一个权宜之计，但是目前并没有更好的办法。如果将新样本添加到样本集中，重新调用 $Isomap$ 算法，则会得到一个新的低维空间。
1. 一方面计算量太大（每个新样本都要调用一次 $Isomap$ 算法）。
2. 另一方面每次降维后的空间都在变化，不利于降维后的训练过程。

对于近邻图的构建有两种常用方案：  
1. 一种方法是指定近邻点个数，比如指定距离最近的 $k$ 个点为近邻点 。这样得到的近邻图称作 $k$ 近邻图。
2. 另一种方法是指定距离阈值 $\epsilon$，距离小于 $\epsilon$ 的点被认为是近邻点。这样得到的近邻图称作 $\epsilon$ 近邻图。

这两种方案都有不足：  
1. 如果近邻范围过大，则距离很远的点也被误认作近邻，这就是“短路”问题。
2. 如果近邻范围过小，则图中有些区域可能与其他区域不存在连接，这就是“断路”问题 。 

短路问题和断路问题都会给后续的最短路径计算造成误导。

### 局部线性嵌入 LLE
&emsp;&emsp;与 $Isomap$ 试图保持邻域内样本之间的距离不同，局部线性嵌入 (Locally Linear Embedding:LLE)试图保持邻域内样本之间的线性关系。这种线性保持在二维空间中就是保持共线性，三维空间中就是保持共面性。

&emsp;&emsp;假定样本点 $x  _  i$ 的坐标能够通过它的邻域样本 $x  _  j,x  _  k,x  _  l$ 进行线性组合而重构出来，即：

$$
x_i=w_ {i,j}x_j +w_ {i,k}x_k +w_ {i,l}x_l
$$

$LLE$ 算法希望这种关系在低维空间中得到保持。

#### 重构系数求解
&emsp;&emsp;$LLE$ 首先为每个样本 $x  _  i$ 找到其近邻点下标集合 $\mathbb Q  _  i$， 然后计算基于 $\mathbb Q  _  i$ 中的样本点对 $x  _  i$ 进行线性重构的系数 $w  _  i$。定义样本集重构误差为（$w  _  {i,j}$ 为 $w  _  i$ 的分量）：

$$
err=\sum_ {i=1}^{N}\|\|x_i-\sum_ {j \in \mathbb Q_i}w_ {i,j}x_j \|\|_ 2^{2}
$$

目标是样本集重构误差最小，即：

$$
\min_ {w_1,w_2,\cdots,w_N}\sum_ {i=1}^{N}\|\|x_i-\sum_ {j \in \mathbb Q_i}w_ {i,j}x_j \|\|_ 2^{2}
$$

&emsp;&emsp;这样的解有无数个，对权重增加约束，进行归一化处理。即：

$$
\sum_ {j \in \mathbb Q_i}w_ {i,j}=1,i=1,2,\cdots,N
$$

现在就是求解最优化问题：

$$
\min_ {w_1,w_2,\cdots,w_N}\sum_ {i=1}^{N}\|\|x_i-\sum_ {j \in \mathbb Q_i}w_ {i,j}x_j \|\|_ 2^{2}\ s.t.\quad \sum_ {j \in \mathbb Q_i}w_ {i,j}=1,i=1,2,\cdots,N
$$

该最优化问题有解析解。令 $C  _  {j,k}=(x  _  i-x  _  j)^{T}(x  _  i-x  _  k)$，则可以解出：

$$
w_ {i,j}=\frac{\sum_ {k\in \mathbb Q_i}C_ {j,k}^{-1}}{\sum_ {l,s\in \mathbb Q_i}C_ {l,s}^{-1}} ,j \in \mathbb Q_i
$$

其中：$C  _  {j,k}$ 刻画了 $x  _  k$ 到 $x  _  i$ 的差向量，与 $x  _  j$ 到 $x  _  i$ 的差向量的内积；$ w  _  {i,j}$ 刻画了这些内积中，与 $ x  _  j$ 相关的内积的比例。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/DimensionReduction/lle.png?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">lle</div>
</center>

#### 低维空间保持
&emsp;&emsp;求出了线性重构的系数 $w  _  i$ 之后，$LLE$ 在低维空间中保持 $w  _  i$ 不变。设 $x  _  i$ 对应的低维坐标 $z  _  i$，已知线性重构的系数 $w  _  i$，定义样本集在低维空间中重构误差为：

$$
err^{\prime}=\sum_ {i=1}^{N}\|\|z_i-\sum_ {j \in \mathbb Q_i}w_ {i,j}z_j \|\|_ 2^{2}
$$

现在的问题是要求出 $z  _  i$，从而使得上式最小。即求解：

$$
\min_ {z_1,z_2,\cdots,z_N} \sum_ {i=1}^{N}\|\|z_i-\sum_ {j \in \mathbb Q_i}w_ {i,j}z_j \|\|_ 2^{2}
$$

&emsp;&emsp;令 $Z=(z  _  1^T,z  _  2^T,\cdots,z  _  N^T) ^T\in \mathbb R^{ N\times n^{\prime}}$，其中 $n^{\prime}$ 为低维空间的维数（$n$ 为原始样本所在的高维空间的维数）。令

$$
W=\begin{bmatrix} w_ {1,1}&w_ {1,2}&\cdots&w_ {1,N}\\ w_ {2,1}&w_ {2,2}&\cdots&w_ {2,N}\\ \vdots&\vdots&\ddots&\vdots\\ w_ {N,1}&w_ {N,2}&\cdots&w_ {N,N} \end{bmatrix}
$$

定义 $M=(I-W)^{T}(I-W)$，于是最优化问题可重写为：

$$
\min_ {Z}tr(Z^T M Z)
$$

该最优化问题有无数个解。添加约束 $Z^T Z=I  _  {n^\prime\times n^\prime}$，于是最优化问题为：

$$
\min_ {Z}tr(Z^T  M Z) \ s.t.\quad Z^{T}Z=I_ {n^\prime\times n^\prime}
$$

该最优化问题可以通过特征值分解求解：选取 $M$ 最小的 $n^{\prime}$ 个特征值对应的特征向量组成的矩阵即为 $Z$。


$LLE$ 中出现了两个重构误差。
1. 第一个重构误差：为了在原始空间中求解线性重构的系数 $w  _  i$。目标是：基于 $\mathbb Q  _  i$ 中的样本点对 $x  _  i$ 进行线性重构，使得重构误差最小。
2. 第二个重构误差：为了求解样本集在低维空间中的表示 $\mathbf Z$。目标是：基于线性重构的系数 $w  _  i$，将 $\mathbb Q  _  i$ 中的样本点对 $z  _  i$ 进行线性重构，使得重构误差最小。

#### LLE 算法
输入：
1. 样本集 $\mathbb D=\{x  _  1,x  _  2,\cdots,x  _  N\}$。
2. 近邻参数 $k$。
3. 低维空间维数 $n^{\prime}$。

输出：样本集在低维空间中的矩阵 $Z$。

算法步骤：
1. 对于样本集中的每个点 $x  _  i,i=1,2,\cdots,N$，执行下列操作：
   1. 确定 $x  _  i$ 的 $k$ 近邻，获得其近邻下标集合 $\mathbb Q  _  i$。
   2. 对于 $j \in \mathbb Q  _  i$，根据下式计算 $w  _  {i,j}$：
      $$
      \begin{aligned} { w }_ { i,j }&=\frac { \sum _ { k\in { Q }_ { i } }{ { C }_ { j,k }^{ -1 } }  }{ \sum _ { l,s\in { Q }_ { i } }{ { C }_ { l,s } }  }  \\ { C }_ { j,k }&={ \left( { x }_ { i }-{ x }_ { j } \right)  }^{ T }\left( { x }_ { i }-{ x }_ { k } \right)  \end{aligned}
      $$
   3. 对于 $j \notin \mathbb Q  _  i$，$w  _  {i,j}=0$
2. 根据 $w  _  {i,j}$ 构建矩阵 $W$。
3. 计算 $M=(I-W)^{T}(I-W)$。
4. 对 $M$ 进行特征值分解，取其最小的 $n^{\prime}$ 个特征值对应的特征向量，即得到样本集在低维空间中的矩阵 $Z$。

## 度量学习
&emsp;&emsp;在机器学习中对高维数据进行降维的主要目的是：希望找出一个合适的低维空间，在这个低维空间中进行学习能比原始空间性能更好。每个空间对应了在样本属性上定义的一个距离度量。寻找合适的空间，本质上就是在寻找一个合适的距离度量。

&emsp;&emsp;度量学习 ($metric learning$) 的思想就是：尝试直接学习出一个合适的距离度量。推广欧氏距离：对于两个 $n$ 维样本 $ x  _  i=(x  _  {i,1},x  _  {i,2},\cdots,x  _  {i,n})^{T},x  _  j=(x  _  {j,1},x  _  {j,2},\cdots,x  _  {j,n})^{T}$ ，假定不同的属性的重要性不同，因此引入了权重：

$$
dist_ {ed}^{2}(x_i,x_j)=\|\|x_i-x_j \|\|_2^{2}=w_1d_ {i,j,1}^2+w_2d_ {i,j,2}^{2}+\cdots+w_nd_ {i,j,n}^{2}
$$

其中 $d  _  {i,j,k}^{2}$ 表示 $x  _  i,x  _  j$ 在第 $k$ 维上的距离，$w  _  k \ge 0 $ 第 $k$ 维距离的权重。定义对角矩阵 $ W$ 为：

$$
W=\begin{bmatrix} w_1&0&0&\cdots&0\\ 0&w_2&0&\cdots&0\\ 0&0&w_3&\cdots&0\\ \vdots&\vdots&\vdots&\ddots&\vdots\\ 0&0&0&\cdots&w_n\\ \end{bmatrix}
$$

则 $dist  _  {ed}^{2}(x  _  i,x  _  j)=(x  _  i-x  _  j)^{T} W(x  _  i-x  _  j)$。上式中的权重矩阵 $W$ 可以通过学习确定。

&emsp;&emsp;前述假设权重矩阵 $W$  是对角矩阵，这意味着坐标轴是正交的，即属性之间无关。现实任务中可能会发生属性相关的情况，此时对应的坐标轴不再正交。于是可以将 $W$ 替换成一个普通的半正定对称矩阵 $M$，此时就得到了马氏距离 ($Mahalanobis distance$)：

$$
dist_ {mah}^{2}(x_i,x_j)=(x_i-x_j)^{T} M(x_i-x_j)
$$

其中的矩阵 $M$ 也称作度量矩阵，度量学习就是对 $M$ 进行学习。为了保持距离非负而且对称，则 $M$ 必须是半正定对称矩阵。即必有正交基 $P$，使得 $M=P P^{T}$。

&emsp;&emsp;对 $M$ 学习的目标是：将 $M$ 嵌入到学习器的评价指标中去，通过优化学习器的评价指标来求得 $M$。即：对 $M$ 的学习无法直接提出优化目标，而是将$M$的学习与学习器的学习作为一个整体，然后优化学习器的优化目标。

&emsp;&emsp;如果学习得的 $M$ 是一个低秩矩阵（假设秩为 $r  _  M \lt n$）， 可以找到一组正交基，其中正交基的数量为 $r  _  M$ ，该组正交基构成矩阵 $P$。于是度量学习的结果可以衍生出一个降维矩阵 $P$ ，能用于降维。降维后的低维空间就是该组正交基张成的空间。


## 概率PCA
&emsp;&emsp;定义隐变量 $z \in \mathbb R^d$，它属于低维空间（也称作隐空间，即隐变量所在的空间）。假设 $z$ 的先验分布为高斯分布：

$$
p(z)=\mathcal N(0,I)
$$

其均值为 $0$，协方差矩阵为 $ I$。

&emsp;&emsp;定义观测变量 $x \in \mathbb R^n$，它属于高维空间。假设条件概率分布 $p(x\mid z)$
也是高斯分布： 

$$
p(x \mid z)=\mathcal N( Wz+\mu,\sigma^2 I)
$$

其中：均值是的 $z$ 线性函数，$W\in \mathbb R^{n\times d}$ 为权重，$\mu$ 为偏置；协方差矩阵为 $\sigma^2 I$。则 $PPCA$ 模型生成观测样本的步骤为：

1. 首先以概率 $p(z)$ 生成隐变量 $z$。
2. 然后观测样本 $x$ 由如下规则生成：
   $$
   x=Wz+\mu +\epsilon 
   $$
   其中 $\epsilon$ 是一个 $n$ 维的均值为零、协方差矩阵为 $\sigma^2 I$ 的高斯分布的噪声：
   $$
    p(\epsilon)=\mathcal N(0,\sigma^2  I)
   $$

### 参数求解

#### 解析解
&emsp;&emsp;可以利用最大似然准则来确定参数 $W,\mu,\sigma^2$ 的解析解。根据边缘概率分布的定义有：

$$
p(x)=\int p(x\mid z) dz
$$

由于 $p(z)$ 和 $p(x\mid z)$ 均为高斯分布，因此 $p(x)$ 也是高斯分布。假设 $x$ 的其均值为 $\mu^\prime$，协方差为 $ C$ 。则：

$$
\begin{aligned} { \mu  }^{ \prime  }&=E\left[ x \right]  \\ &=E\left[ Wz+\mu +\epsilon  \right]  \\ &=\mu  \\ C&=cov\left[ x \right]  \\ &=E\left[ \left( Wz+\mu +\epsilon  \right) { \left( Wz+\mu +\epsilon  \right)  }^{ T } \right]  \\ &=E\left[ Wz{ z }^{ T }W \right] +E\left[ \epsilon { \epsilon  }^{ T } \right]  \\ &=W{ W }^{ T }+{ \sigma  }^{ 2 }I+\mu { \mu  }^{ T } \end{aligned}
$$

> 推导过程中假设 $z$ 和 $\epsilon$ 是相互独立的随机变量。

因此 $p(x) = \mathcal N(\mu,C)$。

&emsp;&emsp;给定数据集 $\mathbb D=\{x  _  1,\cdots,x  _  N\}$，则对数似然函数为：

$$
\begin{aligned} L&=\log { p\left( D;W,\mu ,{ \sigma  }^{ 2 } \right)  }  \\ &=\sum _ { i=1 }^{ N }{ \log { p\left( { x }_ { i };W,\mu ,{ \sigma  }^{ 2 } \right)  }  }  \\ &=-\frac { Nn }{ 2 } \log { \left( 2\pi  \right)  } -\frac { N }{ 2 } \log { \left| C \right|  } -\frac { 1 }{ 2 } \sum _ { i=1 }^{ N }{ { \left( { x }_ { i }-\mu  \right)  }^{ T }\left( { x }_ { i }-\mu  \right)  }  \end{aligned}
$$

其中 $\|\cdot\|$ 这里表示行列式的值。求解 $\frac{\partial L}{\partial {\mu}}=0$，解得 

$$
\mu=\bar{x}=\frac 1N\sum_ {i=1}^N x_i
$$

&emsp;&emsp;对数据集 $\mathbb D={x  _  1,\cdots,x  _  N}$ 进行零均值化，即：

$$
x_i \leftarrow x_i - \mu=x_i - \bar{x}
$$

则有：

$$
x=Wz+\epsilon
$$

因此 

$$
p(x) = \mathcal N(x; 0,C)
$$

其中 $C= \mathbb E[(Wz+ \epsilon)(Wz+ \epsilon)^T]=W W^T+\sigma^2 I$。

&emsp;&emsp;对数似然函数（忽略常数项 $-\frac{Nn}{2}\log(2\pi)$ ）：

$$
\begin{aligned} L&=\log { p\left( D;W,\mu ,{ \sigma  }^{ 2 } \right)  }  \\ &=-\frac { N }{ 2 } \log { \left| C \right|  } -\frac { 1 }{ 2 } \sum _ { i=1 }^{ N }{ { x }_ { i }^{ T }{ C }^{ -1 }{ x }_ { i } }  \\ &=-\frac { N }{ 2 } \left[ \log { \left| C \right|  } tr\left( { C }^{ -1 }S \right)  \right]  \end{aligned}
$$

其中 $S=\frac 1N \sum  _  {i=1}^Nx  _  i x  _  i^T$ 为协方差矩阵。 记：

$$
X=(x_1^T,\cdots,x_N^T)^T=\begin{bmatrix} x_1^T\\ \vdots\\ x_N^T \end{bmatrix}= \begin{bmatrix} x_ {1,1}&x_ {1,2}&\cdots&x_ {1,n}\\ x_ {2,1}&x_ {2,2}&\cdots&x_ {2,n}\\ \vdots&\vdots&\ddots&\vdots\\ x_ {N,1}&x_ {N,2}&\cdots&x_ {N,n}\\ \end{bmatrix}
$$

则 $S=X^T X$。

&emsp;&emsp;$Tipping and Bishop(1999b)$ 证明：$L$ 的所有驻点都可以写做：

$$
W=U_ {d}(\Lambda_d-\sigma^2 I)^{1/2} R
$$

其中：
1. $\mathbf U  _  d \in \mathbb R^{n\times d}$ 的列由协方差矩阵 $S$ 的任意 $d$ 个特征向量组成。
2. $\Lambda  _  d \in \mathbb R^{d\times d}$ 是对角矩阵，其元素是协方差矩阵 $S$ 对应的 $d$ 个特征值 $\lambda  _  i$。
3. $\mathbf R \in \mathbb R^{d\times d}$ 是任意一个正交矩阵。

当 $d$ 个特征向量被选择为前 $d$ 个最大的特征值对应的特征向量时，$L$ 取得最大值。其它的所有解都是鞍点。

&emsp;&emsp;假定协方差矩阵 $S$ 的特征值从大到小排列 $\lambda  _  1\ge\lambda  _  2\ge\cdots\ge\lambda  _  n$，对应的 $n$ 个特征向量为 $ u  _  1,\cdots,u  _  n$。则最大似然准则得到的解析解为：$U=(u  _  1,\cdots,u  _  d)$，它由前 $d$ 个特征向量组成。
$$
W=U_ {d}(\Lambda_d-\sigma^2 I)^{1/2}R
$$
其中，$\sigma^{2}=\frac{1}{n-d}\sum  _  {i=d+1}^n \lambda  _  i$，它就是与丢弃的维度相关连的平均方差。

&emsp;&emsp;$\mathbf R$ 是正交矩阵，因此它可以视作 $d$ 维隐空间的一个旋转矩阵。根据 

$$
C=W W^{T}+\sigma^{2} I= U_d(\Lambda_d-\sigma^{2} I) U_d^T+\sigma^{2} I
$$

则 $C$ 与 $R$ 无关。这表明：$p(x)$ 在隐空间中具有旋转不变性，因此 $R$ 可以选任意一个正交矩阵。这代表某种形式的统计不可区分性，因此有多个 $W$ 都会产生同样的密度函数 $p(x)$。

&emsp;&emsp;当 $R=I$时，$W =U  _  {d}(\Lambda  _  d-\sigma^2 I)^{1/2}$。此时 

$$
W=(\sqrt{\lambda_1-\sigma^2}u_1,\cdots,\sqrt{\lambda_d-\sigma^2}u_d) 
$$

它表示 $W$ 的列是对 $u  _  1,\cdots,u  _  d$ 进行缩放，缩放比例为 $\sqrt{\lambda  _  i-\sigma^2}$。由于 $u  _  i,u  _  j,i\ne j $ 是正交的，因此 $W$ 的列也是正交的。

&emsp;&emsp;当通过解析解直接求解时，可以直接令 $R=I$。但是当通过共轭梯度法或者 $EM$ 算法求解时，$R$ 可能是任意的，此时 $ W$ 的列不再是正交的。如果需要 $ W$ 是正交的，则需要恰当的后处理。

&emsp;&emsp;对于任意一个方向向量 $v$，其中 $v^Tv=1$，分布 $p(x)$ 在该方向上的方差为 $v^T Cv$。如果 $v$ 与 $\{u  _  1,\cdots,u  _  d\}$ 正交，即 $v$ 是被丢弃的特征向量的某个线性组合，则有 

$$
v^T U_d=0
$$

因此有 

$$
v^T C v=\sigma^2 
$$

如果 $v$ 就是 $\{u  _  1,\cdots,u  _  d\}$ 其中之一，即$v=u  _  i,1\le i\le d$，则有

$$
v^T C v=(\lambda_i-\sigma^2)+\sigma^2=\lambda_i
$$

可以看到：沿着特征向量 $u  _  i$ 方向上的方差 $\lambda  _  i$ 由两部分组成：
1. 单位方差的分布 $p(z)$ 通过线性映射$W$之后，在 $u  _  i$ 方向上的投影贡献了：$\lambda  _  i-\sigma^2$。
2. 噪声模型在所有方向上的各向同性的方差贡献了：$\sigma^2$。

因此：$PPCA$ 正确的描述了数据集 $\mathbb D$ 沿着主轴方向（即$u  _  1,\cdots,u  _  d$方向）的方差，并且用一个单一的均值 $\sigma^2$ 近似了所有剩余方向上的方差。

&emsp;&emsp;当 $d=n$ 时，不存在降维的过程。此时有 

$$
U_d=U ,\Lambda_d=\Lambda
$$

根据正交矩阵的性质：$U U^T=I$，以及 $R R^T$，则有：

$$
C =U(\Lambda-\sigma^{2 } I) U^T+\sigma^{2 }I=U \Lambda U^T=S
$$

&emsp;&emsp;由于计算时需要用到 $C^{-1}$，这涉及到一个 $n\times n$ 矩阵的求逆。可以考虑简化为：

$$
C^{-1}=\sigma^{-2} I-\sigma^{-2} W M^{-1} W^T 
$$

其中 $M=W^T W+\sigma^2 I \in R^{d\times d}$。计算复杂度从 $O(n^3)$ 降低到了 $O(d^3)$。

#### EM算法解
&emsp;&emsp;在 $PPCA$ 模型中，数据集 $\mathbb D$ 中的每个数据点 $x  _  i$ 都对应一个隐变量 $z  _  i$，于是可以使用 $EM$ 算法来求解模型参数。实际上 $PPCA$ 模型参数已经可以得到精确的解析解，看起来没必要使用 $EM$ 算法。但是 $EM$ 算法具有下列优势：
1. 在高维空间中，使用 $EM$ 算法而不是直接计算样本的协方差矩阵可能具有计算上的优势。
2. $EM$ 算法的求解步骤也可以推广到因子分析模型中，那里不存在解析解。
3. $EM$ 算法可以为缺失值处理提供理论支撑。

&emsp;&emsp;观测变量为 $x$，隐变量为 $z$，则完全数据为 $\{\mathbb D,\mathbb Z\}$，其中 $\mathbb D=\{x  _  1,\cdots,x  _  N\}$ 、 $\mathbb Z=\{z  _  1,\cdots,z  _  N\}$。其中 对数据集 $\mathbb D=\{x  _  1,\cdots,x  _  N\}$ 进行零均值化，即：

$$
x_i \leftarrow x_i - \mu=x_i - \bar{x}
$$

根据后验概率的定义以及高斯分布的性质，后验概率 

$$
p(z\mid x)=\mathcal N(M ^{-1}W^Tx,\sigma^2 M ^{-1})
$$

完全数据的对数似然函数为：

$$
\begin{aligned} \log { p\left( D,Z;W,{ \sigma  }^{ 2 } \right)  } &=\sum _ { i=1 }^{ N }{ \log { p\left( { x }_ { i },{ z }_ { i } \right)  }  }  \\ &=\sum _ { i=1 }^{ N }{ \left[ \log { p\left( { x }_ { i }|{ z }_ { i } \right)  } +\log { p\left( { z }_ { i } \right)  }  \right]  }  \end{aligned}
$$

其中：$p(x\mid z)=\mathcal N(Wz,\sigma^2 I)$; $p(z)=\mathcal N(0,I)$

$E$ 步，计算期望：

$$
\mathbb E_ {p(z\mid x)} \log p(\mathbb D,\mathbb Z;W,\sigma^2)=-\sum_ {i=1}^N\left[\frac n2\log(2\pi\sigma^2)+\frac 12 tr(\mathbb E_ {p(z\mid x)}[z_iz_i^T]) + \frac{1}{2\sigma^2}\|\|z_i\|\|_2^2-\frac{1}{\sigma^2}\mathbb E_ {p(z \mid x)}[z_i]^T W^T x_i  +\frac{1}{2\sigma^2}tr(\mathbb E_ {p(z\mid x)}[z_i z_i^T] W^T W)+\frac d2\log(2\pi) \right]
$$

其中：$\mathbb E  _  {p(z \mid x)}$ 表示计算期望的概率分布为后验概率分布 $p(z \mid x)$。 假设此时迭代的参数为 $ W  _  {old}$、$\sigma^2  _  {old}$，则有：

$$
\begin{aligned} { E }_ { p\left( z|x \right)  }\left[ { z }_ { i } \right] &={ M }_ { old }^{ -1 }{ W }_ { old }^{ T }{ x }_ { i } \\ { E }_ { p\left( z|x \right)  }\left[ { z }_ { i }{ z }_ { i }^{ T } \right] &={ cov }_ { p\left( z|x \right)  }\left[ { z }_ { i } \right] +{ E }_ { p\left( z|x \right)  }\left[ { z }_ { i } \right] { E }_ { p\left( z|x \right)  }{ \left[ { z }_ { i } \right]  }^{ T } \\ &={ \sigma  }_ { old }^{ 2 }{ M }_ { old }^{ -1 }+\left( { M }_ { old }^{ -1 }{ W }_ { old }^{ T }{ x }_ { i } \right) { \left( { M }_ { old }^{ -1 }{ W }_ { old }^{ T }{ x }_ { i } \right)  }^{ T } \end{aligned}
$$

$M$ 步，求最大化：

$$
W_ {new},\sigma^2_ {new}=\arg\max_ {W,\sigma^2} \mathbb E_ {p(z\mid x)} \log p(\mathbb D,\mathbb Z;W,\sigma^2)
$$      

解得：

$$
W_ {new}\leftarrow \left[\sum_ {i=1}^N x_i\mathbb E_ {p(z\mid x)}[z_i]^T\right]\left[\sum_ {i=1}^N E_ {p(z\mid x)}[z_i z_i^T]\right]^{-1}
$$

$$
\sigma^2_ {new}\leftarrow \frac{1}{Nn}\sum_ {i=1}^N\left[\|\|x_i\|\|^2-2\mathbb E_ {p(z \mid x)}[z_i]^T W_ {new}^T x_i+tr(\mathbb E_ {p(z \mid x)}[z_i z_i^T]) W_ {new}^T W_ {new}\right]
$$

$EM$ 算法的物理意义：
1. $E$ 步涉及到数据点 $x  _  i$ 在隐空间上的正交投影。
2. $M$ 步涉及到隐空间的重新估计，使得满足最大似然，其中投影固定。

&emsp;&emsp;一个简单的类比：考虑二维空间中的一组小球，令一维隐空间用一个固定的杆表示。现在使用很多个遵守胡克定律（存储的能量正比于弹簧长度的平方）的弹簧将每个小球与杆相连。
1. $E$ 步：保持杆固定，让附着的小球沿着杆上下滑动，使得能量最小。这使得每个小球独立地到达对应于它在杆上的正交投影位置。
2. $M$ 步：令附着点固定，然后松开杆，让杆到达能量最小的位置。

重复 $E$ 步和 $M$ 步，直到满足一个收敛准则。

&emsp;&emsp;$PPCA$ 的 $EM$ 算法的一个好处是大规模数据的计算效率。在高维空间中，$EM$ 算法每次迭代所需的计算量都比传统的$PCA$要小得多。
1. $PPCA$解析解算法中，对协方差矩阵进行特征分解的计算复杂度为$O(n^3)$。如果只需要计算前$d$个特征向量和它们的特征值，则可以使用 $O(dn^2)$复杂度的算法。然后计算协方差矩阵本身需要$O(Nn^2)$的计算量，因此不适合大规模数据。

2. $PPCA$的$EM$算法没有显式建立协方差矩阵，其计算量最大的步骤是涉及到对数据集求和的操作，计算代价为$O(Nnd)$。对于较大的$n$，有$d\ll n$，因此与$O(Nn^2)$相比，$EM$算法的计算量大大降低。这可以抵消$EM$算法需要多次迭代的缺陷。

&emsp;&emsp;$PPCA$的$EM$ 算法可以用一种在线的形式执行，其中$x  _  i$被读入、处理。然后在处理下一个数据$x  _  j$时丢弃$x  _  i$
1. $E$步中需要计算的量（一个$d$ 维向量和一个$d\times d$的矩阵 ) 可以分别对每个数据点单独计算。
2. $M$步中需要在数据点上累积求和，这个可以增量完成。

如果 $N$ 和 $n$ 都很大，则这种方式很有优势。

### 性质
&emsp;&emsp;概率 $PCA$ ($probabilistic \ PCA:PPCA$) 与传统的 $PCA$ 相比，有如下优势：
1. 概率 $PCA$ 能够描述数据集的主要特征，如期望、方差等。
2. 概率 $PCA$ 使用 $EM$ 算法求解。当只需要计算几个主要的特征向量的情况下，计算效率较高，它避免了计算协方差矩阵 $X^T X$ 。
3. 概率 $PCA$ 可以推广到混合概率分布，并且利用 $EM$ 算法训练。
4. 概率 $PCA$ 采用似然函数，这使得它可以与其它的概率密度模型进行比较。
5. 概率 $PCA$ 是一个生成式模型，可以用于生成新样本。

&emsp;&emsp;$PPCA$ 考虑的是低维空间到高维空间的映射，这与传统的 $PCA$ 观点不同。传统 $PCA$ 观点是高维空间到低维空间的映射。在 $PPCA$ 中，如果希望对数据进行降维，即从个高维空间映射到低维空间，则需要通过贝叶斯定理进行逆映射。根据后验概率分布 

$$
p(z \mid x)=\mathcal N( M ^{-1} W^T x,\sigma^2 M ^{-1})
$$

任意一点 $x$ 在低维空间中的投影均值为 

$$
\mathbb E[z \mid x]=M^{-1} W^T x
$$

如果取极限 $\sigma^2\rightarrow 0$，则 

$$
\mathbb E[z \mid x]\rightarrow (W^T W)^{-1} W^Tx
$$

这就是标准的 $PCA$ 模型。但此时后验协方差为 $0$，概率密度函数变得奇异。但是如果使用 $EM$ 算法求解，仍然可以得到一个合法的结果。对于 
$\sigma^2\gt0$ 的情况，低维空间中的投影会向着原点方向偏移。

&emsp;&emsp;目前在 $PCA$ 讨论中，假定低维空间的维度 $d$ 是给定的，实际上必须根据应用来选择一个合适的值。如果用于数据可视化，则一般选择为 $d=2$。如果特征值很自然的分成了两组：一组由很小的值组成，另一组由较大的值组成，两组之间有明显的区分。则$d$选取为较大的特征值的数量。
> 实际上这种明显的区分通常无法看到。

&emsp;&emsp;按照传统 $PCA$ 的方式，通过交叉验证法选取较好的 $d$，这种方式依赖于后续的模型。从算法原理的角度设置一个阈值，比如 $t=95\%$ ，然后选取使得下式成立的最小的 $d$ 的值：

$$
\frac{\sum_ {i=1}^{d }\lambda_i}{\sum_ {i=1}^{n}\lambda_i} \ge t
$$

这种方式需要指定阈值 $t$，从而将$d$的选择转移到 $t$ 的选择上。

&emsp;&emsp;基于 $PPCA$ 中的最大似然函数，使用交叉验证的方法，求解在验证集上对数似然函数最大的模型来确定维度的值。这种方法不依赖于后续的模型，但是缺点是计算量很大。利用贝叶斯 $PCA$ 自动寻找合适的 $d$。

&emsp;&emsp;贝叶斯 $PCA$：在$W$的每列上定义一个独立的高斯先验，每个这样的高斯分布都有一个独立的方差，由超参数 $\alpha  _  i$ 控制。因此：

$$
 p(W;\alpha)=\prod_ {i=1}^d\left(\frac{\alpha_i}{2\pi}\right)^{n/2}\exp\left(-\frac 12 \alpha_i w_i^T w_i\right)
$$

其中 $w  _  i$ 是 $W$ 的第 $i$ 列。
$\alpha$ 可以通过最大化 $\log p(\mathbb D)$ 来迭代的求解。最终某个 $\alpha  _  i$ 可能趋向于无穷大，对应的参数向量 $w  _  i$ 趋向于零，后验概率分布变成了原点处的 $\delta(\cdot)$ 函数。这就得到一个稀疏解。

&emsp;&emsp;这样低维空间的有效维度由有限的 $\alpha  _  i$ 的值确定。通过这种方式，贝叶斯方法自动的在提升数据拟合程度（使用较多的向量 $w  _  i$）和减小模型复杂度（压制某些向量 $w  _  i$）之间折中。

### 因子分析
&emsp;&emsp;因子分析：寻找变量之间的公共因子。如：随着年龄的增长，儿童的身高、体重会发生变化，具有一定的相关性。假设存在一个生长因子同时支配这两个变量，那么因子分析就是从大量的身高、体重数据中寻找该生长因子。

&emsp;&emsp;因子分析 ($Factor Analysis:FA$) 是一个线性高斯隐变量模型，它与 $PPCA$ 密切相关。因子分析的定义与 $PPCA$ 唯一差别是：给定隐变量 $z$ 的条件下，观测变量 $x$ 的条件概率分布的协方差矩阵是一个对角矩阵，而不是一个各向同性的协方差矩阵。即：

$$
p(x \mid z)=\mathcal N(Wz+\mu,\mathbf \Psi) 
$$

其中 $\mathbf\Psi$ 是一个 $n\times n$ 的对角矩阵。因此也可以认为 $PPCA$ 是一种特殊情形的因子分析。如果对 $x$ 进行了零均值化，则

$$
p(x \mid z)=\mathcal N(Wz ,\mathbf \Psi)
$$

与 $PPCA$ 模型相同，因子分析模型假设在给定隐变量 $z$ 的条件下，观测变量 $x$ 的各分量 $x  _  1,x  _  2,\cdots,x  _  n$ 是独立的。

&emsp;&emsp;在因子分析的文献中，$\mathbf W$ 的列描述了观测变量之间的相关性关系，被称作因子载入(factor loading)。
$\mathbf\Psi$ 的对角元素，表示每个变量的独立噪声方差，被称作唯一性 (uniqueness)。

观测变量的边缘概率分布为 

$$
p(x)=\mathcal N(0,C)
$$

其中 $C=W W^T+\mathbf\Psi$。与 $PPCA$ 相同，$FA$ 模型对于隐空间的选择具有旋转不变性。可以使用最大似然法来确定因子分析模型中的参数 $ W$、$\mathbf\Psi$ 的值。此时 $ W$、$\mathbf\Psi$ 的最大似然解不再具有解析解，因此必须用梯度下降法或者 $EM$ 算法迭代求解。  

$EM$ 算法的迭代步骤为：
1. $E$ 步：用旧的参数求期望：
   $$
   \begin{aligned} E\left[ { z }_ { i } \right] &=G{ W }^{ T }{ \Psi  }^{ -1 }{ x }_ { i } \\ E\left[ { z }_ { i }{ z }_ { i }^{ T } \right] &=cov\left[ { z }_ { i } \right] +E\left[ { z }_ { i } \right] { E\left[ { z }_ { i } \right]  }^{ T } \\ &=G+E\left[ { z }_ { i } \right] { E\left[ { z }_ { i } \right]  }^{ T } \end{aligned}
   $$
   其中 $G=(I+W^{T}\mathbf \Psi^{-1}W)^{-1}$。这里使用一个 $d\times d$ 的矩阵求逆表达式，而不是 $n\times n$ 的表达式。
2. $M$ 步：求最大化来获取新的参数。
   $$
   W_ {new}\leftarrow \left[\sum_ {i=1}^N x_i\mathbb E[x_i]^T\right]\left[\sum_ {i=1}^N\mathbb E[x_iz_i^T]\right]^{-1}\\ \mathbf \Psi_ {new}\leftarrow \text{diag}\left[S-W_ {new}\frac 1N\sum_ {i=1}^N\mathbb E[z_i]x_i^T\right]
   $$
   其中 $\text{diag}$ 将所有非对角线上的元素全部设置为零。

## 独立成分分析
&emsp;&emsp;独立成分分析 $ICA$ 用于从混合信号中分离出原始信号。本质上它并不是一个降维的算法，而是一个信号分离算法。

### 鸡尾酒会问题
&emsp;&emsp;假设酒会上有 $n$ 个人，他们可以同时说话。房间里散落了 $n$ 个声音接收器用于记录声音。酒会过后，从 $n$ 个声音接收器中采集到一组数据：

$$
\mathbb D=\{x_1,x_2,\cdots,x_N\}\ x_i=(x_ {i,1},x_ {i,2},\cdots,x_ {i,n})^T
$$   

&emsp;&emsp;任务的目标是：从这 $N$ 个时刻的采样数据中恢复出每个人说话的信号。这个过程也称作盲信号分离。随机变量 $x$ 表示观测随机变量，$x  _  i$ 是其第 $i $ 个采样值，其物理意义为：在时刻 $i$ 采集到的 $n$ 个声音信号。

定义
1. 第 $i$ 个人说话的信号为 $s  _  i$。它是一个随机变量，其分布为 $p  _  s( s  _  i)$。$s  _  {1,1},\cdots,s  _  {N,1}$ 为 $s  _  i$ 的 $N$ 个时刻的采样，记作 $u  _  i^{(s)}$。
2. $n$ 个人说话的信号 $s=(s  _  1,s  _  2,\cdots,s  _  n)^T$。它是一个 $n$ 维随机变量，分布为 $p  _  s(s)$。$s  _  {1},\cdots,s  _  {N}$ 为 $s$ 的 $N$ 个时刻的采样。
3. 第 $i$ 个声音接收器收到的信号为 $x  _  i$。它是一个随机变量，其分布为 $p  _  x(x  _  i)$。$x  _  {1,1},\cdots,x  _  {N,1}$ 为 $x  _  i$ 的 $N$ 个时刻的采样，记作 $ u  _  i^{(x)}$。
4. $n$ 个声音接收器收到的信号为 $x=(x  _  1,x  _  2,\cdots,x  _  n)^T$。它是一个 $n$ 维随机变量，分布为 $p  _  x(x)$。$x  _  {1},\cdots,x  _  {N}$ 为 $x$ 的 $N$ 个时刻的采样。

定义矩阵 $X$ 和矩阵 $S$ 为：

$$
X=\begin{bmatrix} x_1^T\\ \vdots\\ x_N^T \end{bmatrix}= \begin{bmatrix} x_ {1,1}&x_ {1,2}&\cdots&x_ {1,n}\\ x_ {2,1}&x_ {2,2}&\cdots&x_ {2,n}\\ \vdots&\vdots&\ddots&\vdots\\ x_ {N,1}&x_ {N,2}&\cdots&x_ {N,n}\\ \end{bmatrix}\quad \mathbf S=\begin{bmatrix} s_1^T\\ \vdots\\ s_N^T \end{bmatrix}= \begin{bmatrix} s_ {1,1}&s_ {1,2}&\cdots&s_ {1,n}\\ s_ {2,1}&s_ {2,2}&\cdots&s_ {2,n}\\ \vdots&\vdots&\ddots&\vdots\\ s_ {N,1}&s_ {N,2}&\cdots&s_ {N,n}\\ \end{bmatrix}
$$

其意义为：
1. $X$ 的每一行代表 $x$ 在时刻 $i$ 的采样 $x  _  i$；每一列代表信号 $x  _  j$ 在所有时刻的采样序列 $u  _  j^{(x)}$。
2. $S$ 的每一行代表 $s$ 在时刻 $i$ 的采样 $s  _  i$；每一列代表信号 $s  _  j$ 在所有时刻的采样序列 $u  _  j^{(s)}$。

$A=(a  _  {i,j})  _  {n\times n}$ 是一个未知的混合矩阵，它用于叠加 $n$ 个人说话的信号。则有： 

$$
X=S A^T 
$$

即：

$$
x =As
$$

其物理意义为：每个声音接收器采集的信号是所有人说话信号的线性叠加。

### 算法
&emsp;&emsp;现在 $X$ 是已知的，即信号 $x$ 是已知的。令 $W= A ^{-1}$，则有：

$$
s=W x
$$

$W$ 称作分离矩阵。如果没有任何先验知识，则无法同时确定信号 $s$ 和 $W$。

&emsp;&emsp;当 $W$ 的每个元素扩大 $2$ 倍，同时信号 $s$ 放大 $2$ 倍时，等式仍然成立。因此结果不是唯一的。当调整信号 $s$ 中各子信号的顺序，同时调整 $W$ 中各行的顺序，等式也仍然成立。因此结果不是唯一的。信号 $s$ 不能是多维高斯分布。假设 $s$ 是多维高斯分布： 

$$
p(s)=\mathcal N(0,I)
$$

则 $x$ 也是一个多维高斯分布，均值为 $0$，方差为 $\mathbb E[xx^T]=AA^T$。假设 $R$ 为任意一个正交矩阵，令 $A^{\prime}= AR$ ，则有： 

$$
A^{\prime}A^{\prime\;T}=ARR^TA^T=AA^T 
$$

这表示在给定信号 $s$ 的分布和 $x$ 的分布的情况下，参数 $A$ 的值并不是唯一的，因此无法分离出每个人说话的信号 $s  _  i$。

&emsp;&emsp;假设每个人发出的声音信号 $s  _  i$ 相互独立，则 $s$ 的概率分布为：

$$
p_s(s) = \prod_ {i=1}^n p_s(s_i)
$$

根据 $s=Wx$，有：

$$
p_x(x)=p_s(Wx)|W|
$$

其中 $\|\cdot\|$ 为行列式。记：

$$
W=\begin{bmatrix} w_ {1,1}&w_ {1,2}&\cdots&w_ {1,n}\\ w_ {2,1}&w_ {2,2}&\cdots&w_ {2,n}\\ \vdots&\vdots&\ddots&\vdots\\ w_ {n,1}&w_ {n,2}&\cdots&w_ {n,n}\\ \end{bmatrix}
$$

令 $w  _  i=(w  _  {i,1},w  _  {i,2},\cdots,w  _  {i,n})^T$，即它是由 $W$ 的第 $i$ 行组成。则有：

$$
p_s( Wx) = p_s(w_1^Tx,\cdots,w_n^Tx)=\prod_ {i=1}^n p_s(w_i^Tx)
$$

因此有：

$$
p_x(x)=\|W\|\prod_ {i=1}^np_s(w_i^Tx) 
$$

&emsp;&emsp;前面提到如果没有任何先验知识，则无法求解。这里需要假设 $p  _  s(s  _  i)$。首先，不能选取高斯分布。其次，考虑到概率密度函数由累计分布函数求导得到，一个方便的选择是：选择累计分布函数为 $sigmoid$ 函数 ：

$$
g(s) =\frac{1}{1+e^{-s}}
$$       

则概率密度函数为：

$$
 p_s(s) = g^\prime(s)=\frac{e^{s}}{(1+e^{s})^2}
$$

&emsp;&emsp;给定采样样本集 $\mathbb D=\{x  _  1,\cdots,x  _  N\}$，则对数似然函数为：

$$
L=\sum_ {i=1}^N\log p_x(x_i)=\sum_ {i=1}^N\left(\log |W|+\sum_ {j=1}^n\log p_s(w_j^Tx_i)\right)
$$

根据最大似然准则，可以采用梯度下降法求解 $L$ 的最大值。其中：根据矩阵微积分有：$\nabla  _  {W}\|W\|=\|W\|(W^{-1})^T$。则有：

$$
\nabla_ {W}L =\begin{bmatrix}1-2g(w_1^Tx_i)\\ 1-2g(w_2^Tx_i)\\ \vdots\\ 1-2g(w_n^Tx_i) \end{bmatrix} x_i^T +( W^{-1})^{T}
$$

当迭代求解出 $W$ 之后，通过 $s=Wx$。还原出原始信号。

&emsp;&emsp;最大似然估计时，假设 $x  _  i$ 和 $x  _  j$ 之间是相互独立的。事实上对于语音信号或者其他具有时间连续性依赖性的数据（如：温度），这个假设不能成立。但是当数据足够多，假设独立对于效果影响不大。如果事先打乱样本，则会加快梯度下降法的收敛速度。

### FastICA
&emsp;&emsp;$FastICA$ 的基本思想是：使得 $s  _  i=w  _  i^Tx$ 最不可能是高斯信号。度量随机变量 $s$ 的分布为高斯分布的程度：

1. 基于峰度 $kurtosis$ 的方法：
   $$
   \text{kurt}[s]=\mathbb E[s^4]-3(E[y^2])^2 
   $$
   对于高斯分布，其峰度为 $0$，因此如果 $\text{kurt}[s]$ 偏离 $0$ 值越远，则它越不可能是高斯分布。实际上该指标只能刻画一个分布是不是高斯分布，而无法描述一个分布偏离高斯分布的程度。因此该方法实际效果一般。
2. 基于负熵的方法：
   $$
   J[s] = H[s_ {gauss}]- H[s]
   $$   
   其中 $H$ 为随机变量的熵，$s  _  {gauss}$ 是一个高斯分布，其均值、方差与非高斯分布的 $s$ 的均值、方差相同。
   
   在信息论中可以证明：在相同方差的条件下，高斯分布的熵最大。因此可以认为 $J[s]$ 越大，$s$ 的分布偏离高斯分布越远。由于计算 $J[s]$ 必须需要知道 $s$ 的概率密度分布函数，实际任务中很难实现。因此通常采用近似公式 
   $$
   J[s] = \{\mathbb E[G(s_ {gauss})]-\mathbb E[G(s)]\}^2 
   $$
   来实现。其中 $G$ 为非线性函数，可以为：
   $$
   \begin{aligned} G(s)&=\tanh  (as) \\ G(s)&=s(-s^{ 2 }/2) \\ G(s)&=s^{ 3 } \end{aligned}
   $$
   其中 $1\le a\le 2$。其导数为 $G^\prime(s)$。


定义目标函数为 $\mathcal J =\sum  _  {i=1}^n J[s  _  i]$，采用梯度下降法求解。其迭代公式为：
$$
\begin{aligned} w&\leftarrow E[xG(w^{ T }x)]-E[G^{ \prime  }(w^{ T }x)]w \\ w&\leftarrow \frac { w }{ \|\|w\|\| }  \end{aligned}
$$
一次 $FastICA$ 算法能够估计出一个独立成分，为了估计出若干个独立成分，需要进行多次 $FastICA$ 算法来得到 $w  _  1,\cdots,w  _  n$。

&emsp;&emsp;为了防止这些向量收敛到同一个最大值（即：分解出同一个独立成分），当估计 $w  _  {i+1}$ 时，需要减去 $w  _  {i+1}$ 在之前得到的 $ w  _  1,\cdots,w  _  i$ 上的投影。即：
$$
\begin{aligned} w_ { i+1 }&\leftarrow E[xG(w_ { i+1 }^{ T }x)]-E[G^{ \prime  }(w_ { i+1 }^{ T }x)]w_ { i+1 } \\ w_ { i+1 }&\leftarrow w_ { i+1 }-\sum _ { k=1 }^{ i } (w_ { i+1 }^{ T }w_ { k })w_ { k }w_ { i+1 } \\ w_ { i+1 }&\leftarrow \frac { w_ { i+1 } }{ \|\|w_ { i+1 }\|\| }  \end{aligned}
$$
其中下标 $i+1$ 并不是迭代步数，而是第 $i+1$ 个 $\mathbf w$。


### 预处理
$ICA$ 中需要进行预处理，主要有数据中心化、白化两个步骤。

1. 数据中心化：

&emsp;&emsp;对数据集 $\mathbb D=\{x  _  1,\cdots,x  _  N\}$ 执行：

$$
x_i \leftarrow x_i- \frac 1N \sum_ {j=1}^{N}x_j
$$

$\bar{x} =\frac 1N \sum  _  {j=1}^{N}x  _  j$ 称作数据集 $\mathbb D$ 的中心向量，它的各元素就是各个特征的均值。该操作使得 $\mathbb E[x] =0$，这也意味着 $s$ 也是零均值的。

2. 白化

&emsp;&emsp;对 $\mathbb D$ 执行线性变化，使其协方差矩阵为单位矩阵 $I$。即： 

$$
E\left[ { x }^{ \prime  }{ x }^{ \prime T } \right] =I
$$

$x$ 的协方差矩阵为 $X^TX$（经过数据中心化之后）， 设其特征值为 $\lambda  _  1,\cdots,\lambda  _  n$，对应的特征向量组成的矩阵为 $ \mathbf E$，则有：

$$
X^TX=E \mathbf \Lambda E^T 
$$

其中 $\mathbf\Lambda=\text{diag}(\lambda  _  1,\cdots,\lambda  _  n)$。

&emsp;&emsp;令：$x^\prime =E\mathbf\Lambda^{-1/2}E^Tx$，则有： 
    
$$
E\left[ { x }^{ \prime  }{ x }^{ \prime T } \right] =I
$$

若 $x^\prime$ 的协方差矩阵为单位矩阵，则根据 $s=W x^\prime$ 有： 

$$
\mathbb E[ss^T] = WW^T
$$

根据假设，$s$ 中各信号 $s  _  1,s  _  2,\cdots,s  _  n$ 是相互独立的，因此 $s$ 的协方差矩阵必须是对角矩阵。能够对 $s  _  i$ 进行缩放时，相应的 $w  _  i$ 进行同样缩放，等式仍然成立。即：最终的解与 $w  _  i$ 幅度无关。因此可以选择 $w  _  i$ 的长度为 $1$。因此有： 

$$
\mathbb E[ss^T] = WW^T=I
$$

$WW^T=I$，即 $w  _  i,w  _  j,i\ne j$ 相互正交且长度为 $1$。这也是 $FastICA$ 算法中需要对 $w  _  {i+1}$ 进行归一化和正交化的原因。这使得矩阵 $W$ 的参数从 $n^2$ 个降低到 $n\times (n-1)/2$ 个，减小了算法的计算复杂度。
        

## t-SNE
&emsp;&emsp;t-SNE ($t-distributed stochastic neighbor embedding$) 是一种非线性降维算法，它是由 $SNE$ 发展而来。


<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/DimensionReduction/t-sne_optimise.gif?raw=true"
    width="540" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">tsne</div>
</center>


### SNE
&emsp;&emsp;$SNE$ 的基本思想：如果两个样本在高维相似，则它们在低维也相似。$SNE$ 主要包含两步：
1. 构建样本在高维的概率分布。
2. 在低维空间里重构这些样本的概率分布，使得这两个概率分布之间尽可能相似。

&emsp;&emsp;在数据集 $\mathbb D=\{x  _  1,\cdots,x  _  N\}$中，给定一个样本$x  _  i$，然后计算$\{x  _  1,\cdots, x  _  {i-1}, x  _  {i+1},\cdots,x  _  N\}$ 是 $ x  _  i$ 的邻居的概率。$SNE$ 假设：如果 $x  _  j$ 与 $x  _  i$ 越相似，则 $x  _  j$ 是 $x  _  i$ 的邻居的概率越大。相似度通常采用欧几里得距离来衡量，两个样本距离越近则它们越相似。概率 $p(x  _  j\mid x  _  i)$ 通常采用指数的形式：

$$
p(x_j\mid x_i) \propto \exp\left(-\|\|x_j-x_i\|\|^2/(2\sigma_i^2)\right)
$$

对 $j=1,2,\cdots,N,j\ne i$ 进行归一化有：

$$
p(x_j\mid x_i) = \frac{\exp\left(-\|\|x_j-x_i\|\|^2/(2\sigma_i^2)\right)}{\sum_ {k\ne i}\exp\left(-\|\|x_k-x_i\|\|^2/(2\sigma_i^2)\right)}
$$

其中 $\sigma  _  i$ 是与 $  _  i$ 相关的、待求得参数，它用于对距离进行归一化。定义 $p  _  {j\mid i}=p(x  _  j\mid x  _  i)$。由于挑选 $x  _  j$ 时排除了 $ x  _  i$，因此有 $p  _  {i\mid i} = 0$。定义概率分布 $P  _  i=(p  _  {1\mid i},\cdots,p  _  {N\mid i})$，它刻画了所有其它样本是 $x  _  i$ 的邻居的概率分布。

&emsp;&emsp;假设经过降维，样本 $x  _  i\in\mathbb R^n$ 在低维空间的表示为 $z  _  i\in \mathbb R^d$，其中 $d\le n$。定义：

$$
q_ {j\mid i} = q(z_j\mid z_i) = \frac{\exp\left(-\|\|z_j-z_i\|\|^2 \right)}{\sum_ {k\ne i}\exp\left(-\|\|z_k-z_i\|\|^2\right)}
$$

其中 $q  _  {j\mid i}$ 表示给定一个样本 $z  _  i$，然后计算 $z  _  1,\cdots, z  _  {i-1}, z  _  {i+1},\cdots,z  _  N$ 是 $z  _  j$ 的邻居的概率。这里选择 $\sigma^2 = \frac 12$ 为固定值。同样地，有 $q  _  {i\mid i} = 0$。定义概率分布 $Q  _  i=(q  _  {1\mid i},\cdots,q  _  {N\mid i})$ ，它刻画了所有其它样本是 $z  _  i$ 的邻居的概率分布。

&emsp;&emsp;对于样本 $x  _  i$，如果降维的效果比较好，则有 $p  _  {j\mid i} = q  _  {j\mid i},i=1,2,\cdots,N$。即：降维前后不改变 $ x  _  i$ 周围的样本分布。
对于 $x  _  i$，定义其损失函数为分布 $P  _  i$ 和 $ Q  _  i $ 的距离，通过 $KL$ 散度来度量。对于全体数据集 $\mathbb D$，整体损失函数为：

$$
L = \sum_ {i=1}^N KL(P_i\|\|Q_i) =\sum_ {i=1}^N\sum_ {j=1}^N p_ {j\mid i} \log \frac{p_ {j\mid i}}{q_ {j\mid i}}
$$    

$KL$ 散度具有不对称性，因此不同的错误对应的代价是不同的。给定样本 $x  _  i $：
1. 对于高维距离较远的点 $x  _  j$，假设 $p  _  {j\mid i}=0.2$， 如果在低维映射成距离比较近的点，假设 $q  _  {j\mid i} = 0.8$。则该点的代价为 $0.2\log \frac{0.2}{0.8} = -0.277$。
2. 对于高维距离较近的点 $x  _  j$，假设 $p  _  {j\mid i}=0.8$， 如果在低维映射成距离比较远的点，假设 $q  _  {j\mid i} = 0.2$。则该点的代价为 $0.8\log \frac{0.8}{0.2} = 1.11$。

&emsp;&emsp;因此 $SNE$ 倾向于将高维空间较远的点映射成低位空间中距离较近的点。这意味着 $SNE$ 倾向于保留高维数据中的局部特征（因为远处的特征会被扭曲）。因此 $SNE$ 更关注局部结构而忽视了全局结构。

&emsp;&emsp;从 $p  _  {j\mid i} \propto \exp\left(-\|\|x  _  j-x  _  i\|\|^2/(2\sigma  _  i^2)\right)$ 可以看到：$\sigma  _  i$ 是与 $ x  _  i $ 相关的、用于对距离进行归一化的参数。
1. 若 $\sigma  _  i$ 较大，则概率 $p  _  {j\mid i} \gg 0$ 的样本 $x  _  j$ 更多，它们覆盖的范围更广，概率分布 $P  _  i$ 的熵越大。
2. 若 $\sigma  _  i$ 较小，则概率 $p  _  {j\mid i} \gg 0$ 的样本 $x  _  j$ 更少，它们覆盖的范围更窄，概率分布 $P  _  i$ 的熵越小。

定义困惑度为：

$$
\text{Perp}(P_i) = 2^{H(P_i)}
$$

其中 $H(P  _  i)$ 表示概率分布 $P  _  i$ 的熵。困惑度刻画了 $x  _  i$ 附近的有效近邻点个数。通常选择困惑度为 $5-50$ 之间。它表示：对于给定的 $x  _  i$，只考虑它周围最近的 $5-50$ 个样本的分布。给定困惑度之后，可以用二分搜索来寻找合适的 $\sigma  _  i$。

&emsp;&emsp;当 $\sigma  _  i$ 已经求得，可以根据数据集 $\mathbb D$ 以及公式 $p  _  {j\mid i} = \frac{\exp\left(-\|\|x  _  j-x  _  i\|\|^2/(2\sigma  _  i^2)\right)}{\sum  _  {k\ne i}\exp\left(-\|\|x  _  k-x  _  i\|\|^2/(2\sigma  _  i^2)\right)}$ 来求出 $p  _  {j\mid i}$。剔除 $L$ 中的已知量（$\sum  _  i\sum  _  j p  _  {j\mid i}\log p  _  {j\mid i}$），则有：

$$
L = -\sum_ {i=1}^N\sum_ {j=1}^N p_ {j\mid i} \log q_ {j\mid i} 
$$

可以通过梯度下降法求解损失函数的极小值。记 $y  _  {i,j}=-\|\|z  _  j-z  _  i\|\|^2$ ，则有 

$$
q_ {j\mid i} = \frac{\exp\left( y_ {i,j} \right)}{\sum_ {k\ne i}\exp\left( y_ {i,k} \right)} 
$$

考虑到 $softmax$ 交叉熵的损失函数 $\sum  _  {}y  _  {true}\log q $ 的梯度为 $y  _  {true} - q$。令分布$p  _  {j\mid i}$ 为样本的真实标记 $y  _  {true}$ ，则有：

$$
\begin{aligned} { \nabla  }_ { { y }_ { i,j } }\left( \sum _ { j=1 }^{ N }{ { p }_ { j|i }\log { { q }_ { j|i } }  }  \right) &={ p }_ { j|i }-{ q }_ { j|i } \\ { \nabla  }_ { { z }_ { i } }\left( \sum _ { j=1 }^{ N }{ { p }_ { j|i }\log { { q }_ { j|i } }  }  \right)& ={ \nabla  }_ { { y }_ { i,j } }\left( \sum _ { j=1 }^{ N }{ { -p }_ { j|i }\log { { q }_ { j|i } }  }  \right) \times { \nabla  }_ { { z }_ { i } }{ y }_ { i,j } \\ &=-2\left( { p }_ { j|i }-{ q }_ { j|i } \right) \times \left( { z }_ { i }-{ z }_ { j } \right)  \\ { \nabla  }_ { { z }_ { j } }\left( \sum _ { i=1 }^{ N }{ { p }_ { j|i }\log { { q }_ { j|i } }  }  \right) &={ \nabla  }_ { { y }_ { i,j } }\left( \sum _ { i=1 }^{ N }{ { -p }_ { j|i }\log { { q }_ { j|i } }  }  \right) \times { \nabla  }_ { { z }_ { j } }{ y }_ { i,j } \\ &=-2\left( { p }_ { j|i }-{ q }_ { j|i } \right) \times \left( { z }_ { j }-{ z }_ { i } \right)  \end{aligned}
$$

考虑梯度 $\nabla  _  {z  _  i}  L$ ，有两部分对它产生贡献：
1. 给定 $x  _  i$ 时，梯度的贡献为：$-\sum  _  j\nabla  _  {z  _  i}\left(\sum  _  {j=1}^N p  _  {j\mid i} \log q  _  {j\mid i}\right)$。
2. 给定 $x  _  j$ 时，梯度的贡献为：$-\sum  _  j\nabla  _  {z  _  i}\left(\sum  _  {j=1}^N p  _  {i\mid j} \log q  _  {i\mid j}\right)$。

因此有：

$$
\begin{aligned} { \nabla  }_ { { z }_ { i } }L&=-\sum _ { j }{ \left( -2\left( { p }_ { j|i }-{ q }_ { j|i } \right) \times \left( { z }_ { i }-{ z }_ { j } \right)  \right) +\left( -2\left( { p }_ { i|j }-{ q }_ { i|j } \right) \times \left( { z }_ { i }-{ z }_ { j } \right)  \right)  }  \\ &=\sum _ { j }{ 2\left( { p }_ { j|i }-{ q }_ { j|i }+{ p }_ { i|j }-{ q }_ { i|j } \right) \left( { z }_ { i }-{ z }_ { j } \right)  }  \end{aligned}
$$

该梯度可以用分子之间的引力和斥力进行解释：低维空间中的点 $z  _  i$ 的位置是由其它所有点对其作用力的合力决定。
1. 某个点 $z  _  j$ 对 $z  _  i$ 的作用力的方向：沿着 $z  _  i - z  _  j$ 的方向。
2. 某个点 $z  _  j$ 对 $z  _  i$ 的作用力的大小：取决于 $z  _  j$ 和 $z  _  i$ 之间的距离。

为了避免陷入局部最优解，可以采用采用基于动量的随机梯度下降法：

$$
\begin{aligned} v&\leftarrow \alpha v-\epsilon { \nabla  }_ { { z }_ { i } }L \\ { z }_ { i }&\leftarrow { z }_ { i }+v \end{aligned}
$$

其中 $\epsilon$ 为学习率，$\alpha \in [0,1)$ 为权重衰减系数。每次迭代过程中引入一些高斯噪声，然后逐渐减小该噪声。

### 对称 SNE
&emsp;&emsp;在 $SNE$ 中使用的是条件概率分布 $p  _  {j\mid i }$ 和 $q  _  {j\mid i}$ ，它们分别表示在高维和低维下，给定第 $i$ 个样本的情况下，第 $j$ 个样本的分布。而对称 $SNE$ 中使用联合概率分布 $p  _  {i,j}$ 和 $q  _  {i,j}$ ，它们分别表示在高维和低维下，第 $i$ 个样本和第 $j$ 个样本的联合分布。其中：

$$
\begin{aligned} { p }_ { i,j }&=\frac { exp\left( \frac { -{ \| { x }_ { i }-{ x }_ { j } \|  }^{ 2 } }{ 2{ \sigma  }^{ 2 } }  \right)  }{ \sum _ { k }{ \sum _ { l,k\neq l }{ exp\left( \frac { -{ \| { x }_ { k }-{ x }_ { l } \|  }^{ 2 } }{ 2{ \sigma  }^{ 2 } }  \right)  }  }  }  \\ { q }_ { i,j }&=\frac { exp\left( -{ \| { z }_ { i }-{ z }_ { j } \|  }^{ 2 } \right)  }{ \sum _ { k }{ \sum _ { l,k\neq l }{ exp\left( -{ \| { z }_ { k }-{ z }_ { l } \|  }^{ 2 } \right)  }  }  }  \\ { p }_ { i,j }&=0 \\ { p }_ { i,j }&=0 \end{aligned}
$$

根据定义可知 $p  _  {i,j}$ 和 $q  _  {i,j}$ 都满足对称性：$p  _  {i,j}=p  _  {j,i}$ 、$q  _  {i,j}=q  _  {j,i}$。

&emsp;&emsp;上述定义的 $p  _  {i,j}$ 存在异常值问题。当 $x  _  i$ 是异常值时，对所有的 $ x  _  j,j\ne i$， 有 $\|\|x  _  i-x  _  j\|\|^2$ 都很大。这使得 $ p  _  {i,j}, j=1,2,\cdots,N $ 都几乎为 $0$。这就使得 $x  _  i$ 的代价：

$$
\sum_ {j}p_ {i,j} \log q_ {i,j} \rightarrow 0
$$

即：无论 $z  _  i$ 周围的点的分布如何，它们对于代价函数的影响忽略不计。而在原始 $SNE$ 中，可以保证 $\sum  _  {j=1}^Np  _  {j\mid i} =1$，$z  _  i$ 周围的点的分布会影响代价函数。

&emsp;&emsp;为解决异常值问题，定义： 
$$
p_ {i, j} = \frac{p_ {j\mid i} +p_ {i\mid j}}{2N} 
$$

这就使得 $\sum  _  {j=1}^N p  _  {i,j} \gt \frac 1{2N}$，从而使得 $ z  _  i$ 周围的点的分布对代价函数有一定的贡献。注意：这里并没有调整 $q  _  {i,j}$ 的定义。

&emsp;&emsp;对称 $SNE$ 的目标函数为：

$$
L = \sum_ {i=1}^N KL(P_i\|\|Q_i) =- \sum_ {i=1}^N\sum_ {j=1}^N p_ {i,j} \log q_ {i,j}
$$

根据前面的推导有：

$$
\nabla_ {z_i} L= \sum_j 4(p_ {i,j}-q_ {i,j})(z_i-z_j) 
$$

其中：$p  _  {i, j} = \frac{p  _  {j\mid i} +p  _  {i\mid j}}{2N},q  _  {i, j} = \frac{\exp\left(-\|\|z  _  i-z  _  j\|\|^2 \right)}{\sum  _  {k}\sum  _  {l,k\ne l}\exp\left(-\|\|z  _  k-z  _  l\|\|^2\right)}$。
实际上对称 $SNE$ 的效果只是略微优于原始 $SNE$ 的效果。

### 拥挤问题
&emsp;&emsp;拥挤问题 ($Crowding Problem$)：指的是 $SNE$ 的可视化效果中，不同类别的簇挤在一起，无法区分开来。拥挤问题本质上是由于高维空间距离分布和低维空间距离分布的差异造成的。

&emsp;&emsp;考虑 $n$ 维空间中一个以原点为中心、半径为 $1$ 的超球体。在球体内部随机选取一个点，则该点距离原点的距离为 $r$ 的概率密度分布为：

$$
p(r) = \lim_ {\Delta r\rightarrow 0 }\frac{(r+\Delta r)^n-r^n}{\Delta r} = nr^{n-1}
$$

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/DimensionReduction/sne_crowding.png?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">sne_crowding</div>
</center>

累计概率分布为：

$$
F(r)=\int_ {0}^r p(r)dr = r^n
$$

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/DimensionReduction/sne_crowding2.png?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">sne_crowding2</div>
</center>

可以看到：随着空间维度的增长，采样点在原点附近的概率越低、在球体表面附近的概率越大。如果直接将这种距离分布关系保留到低维，则就会出现拥挤问题。


### t-SNE
$t-SNE$ 通过采用不同的分布来解决拥挤问题：
1. 在高维空间下使用高斯分布将距离转换为概率分布。
2. 在低维空间下使用 $t$ 分布将距离转换为概率分布。这也是 $t-SNE$ 的名字的由来。

$t-SNE$ 使用自由度为 $1$ 的 $t$ 分布。此时有：

$$
q_ {i,j}=\frac{(1+\|\|z_i-z_j\|\|^2)^{-1}}{\sum_ {k}\sum_ {l,l\ne k}(1+\|\|z_k-z_l\|\|^2)^{-1}}
$$

则梯度为：

$$
 \nabla_ {z_i}  L= \sum_j 4(p_ {i,j}-q_ {i,j})(z_i-z_j)(1+\|\|z_i-z_j\|\|^2)^{-1}
    
$$

也可以选择自由度超过 $1$ 的 $t$ 分布。自由度越高，越接近高斯分布。

$t$ 分布相对于高斯分布更加偏重长尾。可以看到：
1. 对于高维空间相似度较大的点（如下图中的 $q1$），相比较于高斯分布， $t$ 分布在低维空间中的距离要更近一点。
2. 对于高维空间相似度较小的点（如下图中的 $q2$ ），相比较于高斯分布，$t$ 分布在低维空间中的距离要更远一点。
    
即：同一个簇内的点（距离较近）聚合的更紧密，不同簇之间的点（距离较远）更加疏远。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/DimensionReduction/sne_norm_t_dist_cost.png?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">t_dist</div>
</center>
   
优化过程中的技巧：
1. 提前压缩 ($early compression$)：开始初始化的时候，各个点要离得近一点。这样小的距离，方便各个聚类中心的移动。可以通过引入 $L2$ 正则项(距离的平方和)来实现。
2. 提前放大 ($early exaggeration$)：在开始优化阶段，$p  _  {i,j}$ 乘以一个大于 $1$ 的数进行扩大，来避免 $q  _  {i,j}$ 太小导致优化太慢的问题。比如前$50$ 轮迭代，$p  _  {i,j}$ 放大四倍。

$t-SNE$ 的主要缺点：
1. $t-SNE$ 主要用于可视化，很难用于降维。有两个原因：
    1. $t-SNE$没有显式的预测部分，所以它无法对测试样本进行直接降维。一个解决方案是：构建一个回归模型来建立高维到低维的映射关系，然后通过该模型来对测试样本预测其低维坐标。
    2. $t-SNE$ 通常用于 $2$ 维或者 $3$ 维的可视化。如果数据集相互独立的特征数量如果较大，则映射到 $2-3$ 维之后信息损失严重。
2. $t-SNE$ 中的距离、概率本身没有意义，它们主要用于描述样本之间的概率分布。
3. $t-SNE$ 代价函数是非凸的，可能得到局部最优解。
4. $t-SNE$ 计算开销较大，训练速度慢。其计算复杂度为 $O(N^2)$ 。经过优化之后可以达到 $O(N\log N)$。
        

### t-SNE 改进
$2014$ 年 $Mattern$ 在论文 $Accelerating t-SNE using Tree-Based Algorithms$ 中对 $t-SNE$ 进行了改进，主要包括两部分：
1. 使用 $kNN$ 图来表示高维空间中点的相似度。
2. 优化了梯度的求解过程。

#### kNN 图的相似度表示
&emsp;&emsp;注意到 $p(x  _  j\mid x  _  i)$ 的表达式：

$$
p(x_j\mid x_i) = \frac{\exp\left(-\|\|x_j-x_i\|\|^2/(2\sigma_i^2)\right)}{\sum_ {k\ne i}\exp\left(-\|\|x_k-x_i\|\|^2/(2\sigma_i^2)\right)}
$$

每个数据点 $x  _  i$ 都需要计算 $\sum  _  {k\ne i}\exp\left(-\|\|x  _  k-x  _  i\|\|^2/(2\sigma  _  i^2)\right)$ ，这一项需要计算所有其他样本点到 $x  _  i$ 的距离。当数据集较大时，这一项的计算量非常庞大。事实上，如果两个点相距较远，则它们互为邻居的概率非常小，因为 

$$
\lim_ {\|\|x_j-x_i\|\|\rightarrow \infty}{\exp\left(-\|\|x_j-x_i\|\|^2/(2\sigma_i^2)\right)} = 0 
$$

因此在构建高维空间的点的相似度关系时，只需要考虑 $x  _  i$ 最近的若干个邻居点即可。

&emsp;&emsp;考虑与点 $x  _  i$ 最近的 $\lfloor 3\text{Perp}\rfloor$ 个点，其中 $\text{Perp}$ 为点 $x  _  i$ 的周围点的概率分布的困惑度。记这些邻居结点的集合为 $\mathbb N  _  i$，则有：

$$
\begin{aligned} { p }_ { j|i }&=p\left( { x }_ { j }|{ x }_ { i } \right)  \\ &=\frac { exp\left( \frac { -{ \| { x }_ { i }-{ x }_ { j } \|  }^{ 2 } }{ 2{ \sigma  }_ { i }^{ 2 } }  \right)  }{ \sum _ { k\in { N }_ { i } }{ exp\left( \frac { -{ \| { x }_ { k }-{ x }_ { j=i } \|  }^{ 2 } }{ 2{ \sigma  }^{ 2 } }  \right)  }  }  \\ { p }_ { j|i }&=0 \\ j&\in { N }_ { i } \\ j&\neq { N }_ { i } \end{aligned}
$$

这种方法会大大降低计算量。但是需要首先构建高维空间的 $kNN$ 图，从而快速的获取 $x  _  i$ 最近的 $\lfloor 3\text{Perp}\rfloor$ 个点。

$Maaten$ 使用 $VP$ 树 ($vantage-point tree$) 来构建 $ kNN$ 图，可以在 $O(\text{Perp}N\log N)$ 的计算复杂度内得到一个精确的 $kNN$ 图。

#### 梯度求解优化
&emsp;&emsp;对 $\nabla  _  {z  _  i} L$ 进行变换。定义 

$$
Z=\sum_ {k}\sum_ {l,l\ne k}(1+\|\|z_k-z_l\|\|^2)^{-1} 
$$

则根据：

$$
q_ {i,j}=\frac{(1+\|\|z_i-z_j\|\|^2)^{-1}}{\sum_ {k}\sum_ {l,l\ne k}(1+\|\|z_k-z_l\|\|^2)^{-1}}
$$   

有：

$$
(1+\|\|z_i-z_j\|\|^2)^{-1}=q_ {i,j}\times Z 
$$

则有：

$$
\begin{aligned} { \nabla  }_ { { z }_ { i } }L&=\sum _ { j }{ 4\left( { p }_ { i|j }-{ q }_ { i|j } \right) \left( { z }_ { i }-{ z }_ { j } \right) { \left( 1+{ \| { z }_ { i }-{ z }_ { j } \|  }^{ 2 } \right)  }^{ -1 } }  \\ &=\sum _ { j }{ 4\left( { p }_ { i|j }-{ q }_ { i|j } \right) \left( { z }_ { i }-{ z }_ { j } \right) { q }_ { i,j }Z }  \\ &=4\left[ \sum _ { j }{ { p }_ { i,j }{ q }_ { i,j }Z\left( { z }_ { i }-{ z }_ { j } \right)  } -\sum _ { j }{ { q }_ { i,j }^{ 2 }Z\left( { z }_ { i }-{ z }_ { j } \right)  }  \right]  \end{aligned}
$$

定义引力为：

$$
F_ {attr} = \sum_ {j}p_ {i,j}q_ {i,j}Z (z_i-z_j)
$$

斥力为：

$$
F_ {rep}=\sum_ {j}q_ {i,j}^2Z(z_i-z_j)
$$

则有：

$$
 \nabla_ {z_i} L=4(F_ {attr}-F_ {rep}) 
$$

&emsp;&emsp;引力部分的计算比较简单。考虑到 $q  _  {i,j}\times Z = (1+\|\|z  _  i-z  _  j\|\|^2)^{-1}$，则有：
$$
F_ {attr} = \sum_ {j}p_ {i,j}q_ {i,j}Z (z_i-z_j)=\sum_jp_ {i,j}\frac{z_i-z_j}{1+\|\|z_i-z_j\|\|^2}
$$

根据 $4\lim  _  {\|\|z  _  i-z  _  j\|\|\rightarrow \infty}\frac{z  _  i-z  _  j}{1+\|\|z  _  i-z  _  j\|\|^2}= 0$，则只可以忽略较远的结点。 仅考虑与点 $z  _  i$ 最近的 $\lfloor 3\text{Perp}\rfloor$ 个点，则引力部分的计算复杂度为 $O(\text{Perp}N)$。

&emsp;&emsp;斥力部分的计算比较复杂，但是仍然有办法进行简化。考虑下图中的三个点，其中 $ \|\|z  _  i-z  _  j\|\| \simeq \|\|z  _  i-z  _  k\|\| \gg \|\|z  _  j-z  _  k\|\|$。此时认为点 $z  _  j$ 和 $z  _  k$ 对点 $z  _  i$ 的斥力是近似相等的。


<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/DimensionReduction/tsne2.png?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">tsne2</div>
</center>
    
&emsp;&emsp;事实上这种情况在低维空间中很常见，甚至某片区域中每个点对 $z  _  i$ 的斥力都可以用同一个值来近似，如下图所示。假设区域 $\mathbb A$ 中 $4$ 个点对 $z  _  i$ 产生的斥力都是近似相等的，则可以计算这 $4$ 个点的中心（虚拟的点）产生的斥力 $ F  _  {\mathbb A  _  c}$，则区域  $\mathbb A$ 产生的总的斥力为$4 F  _  {\mathbb A  _  c}$。
        
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/DimensionReduction/tsne3.png?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">tsne3</div>
</center>

&emsp;&emsp;$Matten$ 使用四叉树来完成区域搜索任务，并用该区域中心点产生的斥力作为整个区域的斥力代表值。并非所有区域都满足该近似条件，这里使用$Barnes-Hut$算法搜索并验证符合近似条件的点-区域组合 。

&emsp;&emsp;事实上可以进一步优化，近似区域到区域之间的斥力。如下所示为区域 $\mathbb A$ 和区域 $\mathbb B$ 中，任意两个结点之间的斥力都可以用 $F  _  {\mathbb A\mathbb B  _  c}$ 来近似。其中 $F  _  {\mathbb A\mathbb B  _  c}$ 代表区域 $\mathbb A$ 的中心（虚拟的点）和区域 $\mathbb B$ 的中心（虚拟的点）产生的斥力。同样也需要判断两个区域之间的斥力是否满足近似条件。这里采用了 $Dual-tree$ 算法搜索并验证符合近似条件的区域-区域组合 。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/DimensionReduction/tsne4.png?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">tsne4</div>
</center>

## LargeVis
数据可视化的本质任务是：在低维空间中保存高维数据的内在结构。即：
1. 如果高维空间中两个点相似，则低维空间中它们距离较近。
2. 如果高维空间中两个点不相似，则低维空间中它们距离较远。

$t-SNE$ 及其变体的核心思想：
1. 从高维数据中提取数据之间的内在结构。这是通过相似度、条件概率分布来实现的。
2. 将该结构投影到低维空间。这是通过$KL$散度来实现的。

$t-SNE$的重要缺陷：
1. 计算 $kNN$ 图存在计算瓶颈。$t-SNE$ 构建 $kNN$  图时采用Vantage-Point树，其性能在数据维度增加时明显恶化。
2. 当数据量变大时，可视化效率明显恶化。
3. $t-SNE$ 的参数对于不同数据集非常敏感，需要花费较多的时间在调参上。

$LargeVis$ 是一种不同的可视化技术，可以支持百万级别的数据点可视化。与 $t-SNE$ 相比，它主要在以下方面进行了改进：
1. 通过随机投影树来构建近似$kNN$ 图。
2. 提出一个概率模型，该模型的目标函数可以通过异步随机梯度得到优化。

### 近似 $kNN$ 图

####  随机投影树
&emsp;&emsp;随机投影树和 $kd$ 树一样，都是一种分割$n$维数据空间的数据结构。$kd$ 树严格按照坐标轴来划分空间，树的深度正比于空间的维度 $n$。当 $n$ 的数值很大（即数据空间维度很高）时，$kd$ 树的深度会非常深。
随机投影树划分空间的方式比较灵活，其深度为 $O\log(N)$，其中 $N$ 为样本的数量。
        
随机投影树建立过程：
1. 将数据集 $\mathbb D=\{x  _  1,\cdots,x  _  N\}$中的所有点放入根节点。
2. 随机选取一个从原点出发的向量 $x  _  1$，与这个向量垂直的空间$S  _  1$ 将根节点划分为两部分。
   1. 将 $S1$ 左侧的点划分到左子树。$S  _  1$左侧的点是满足$x  _  i\cdot v  _  1\le 0$ 的点。
   2. 将 $S1$ 右侧的点划分到右子树。$S  _  1$右侧的点是满足$ x  _  i\cdot v  _  1\gt 0$ 的点。
3. 重复划分左子树和右子树，使得每个子树包含的点的数量符合要求。

#### 构建 kNN 图
&emsp;&emsp;$LargeVis$ 通过随机投影树来构建近似$kNN$图。首先建立随机投影树。对于数据点 $x  _  i$，找到树中对应的叶节点 $\text{node}  _  i$。然后将叶节点 $\text{node}  _  i$ 对应的子空间中的所有数据点都作为数据点 $x  _  i$ 的候选邻居。

&emsp;&emsp;单棵随机投影树构建的$kNN$图精度不高。为了提高 $kNN$ 图的精度，通常需要建立多个随机投影树 $\text{Tree}  _  1,\text{Tree}  _  2,\cdots,\text{Tree}  _  T$。
对于数据点  $x  _  i$，对每颗树 $\text{Tree}  _  t$，找到树 $\text{Tree}  _  t$ 中对应的叶节点 $\text{node}  _  i^{(t)}$。然后将叶节点 $\text{node}  _  i^{(t)}$ 对应的子空间中的所有数据点都作为数据点 $x  _  i$ 的候选邻居。
> 这种做法的缺点是：为了得到足够高的精度，需要大量的随机投影树，这导致计算量较大。
    
$LargeVis$使用邻居探索技术来构建高精度的 $kNN$ 图。其基本思路是：邻居的邻居也可能是我的邻居。
1. 首先建立多个随机投影树 $\text{Tree}  _  1,\text{Tree}  _  2,\cdots,\text{Tree}  _  T$，这里 $T$ 比较小。
2. 对于数据点 $x  _  i$，在所有这些随机投影树中寻找其候选邻居。
3. 将这些候选邻居的候选邻居都作为 $x  _  i$ 的候选邻居。

这种方法只需要构建少量的随机投影树，就可以得到足够高精度的 $kNN$ 树。
    

### LargeVis 概率模型
$LargeVis$ 根据 $kNN$ 图来定义图 $G=(V,E)$：
1. 顶点集 $V$：它就是所有高维空间中的数据点的集合。
2. 边集 $E$：它就是 $kNN$ 中所有边的集合。其中边的权重 $w  _  {i,j} = p  _  {i,j} = \frac{p  _  {i\mid j}+p  _  {j\mid i}}{2N}$。
        
将该图$G$的结构投影到低维空间保持不变。定义低维数据点 $z  _  i$和$z  _  j$ 存在边的概率为：

$$
P(e_ {i,j}=1)=f(\|\|z_i-z_j\|\|) 
$$

其中：$e  _  {i,j}=1$ 表示点 $z  _  i$ 和 $z  _  j$ 存在边。$f(\cdot)$ 为函数，可以为：$f(x)=\frac{1}{1+ax^2}$ 或者 $f(x)=\frac{1}{1+\exp(x^2)}$。

定义数据点 $z  _  i$ 和 $z  _  j$ 存在边、且其权重为 $w  _  {i,j}$ 的概率为：$P(e  _  {i,j}=w  _  {i,j})=P(e  _  {i,j}=1)^{w  _  {i,j}}$。
考虑数据集 $\mathbb D =\{x  _  1,\cdots,x  _  N\}$，则对数似然函数为：

$$
L = \sum_ {(i,j) \in E} w_ {i,j}\log P(e_ {i,j}=1) + \sum_ {(i,j) \in \bar E} \gamma\log (1-P(e_ {i,j}=1))
$$

其中：
1. $(i,j) \in E$ 表示$E$中的边代表的顶点对。这些顶点对也称作正边。
2. $\bar E$ 表示不存在边的那些顶点对的集合。这些顶点对也称作负边 。
3. $\gamma$ 是所有负边的权重。负边的权重是统一的。

&emsp;&emsp;事实上在$kNN$图中，正边的数量较少，负边的数量非常庞大，因此计算 $L$ 的代价较高。$LargeVis$ 利用负采样技术进行优化。对图 $G$ 中的每个顶点 $i $，$LargeVis$ 仅仅从以 $i$ 为一个端点的负边中随机采样 $ M$ 个顶点 $ j  _  1,j  _  2,\cdots,j  _  M$ 来计算 $L$。其中采样概率 $P  _  n(j)\propto d  _  j^{0.75}$ ，$d  _  j$ 为顶点 $j$ 的度（与它相连的边的数量）。则对数似然函数为：

$$
L = \sum_ {(i,j) \in E} w_ {i,j}\left[\log P(e_ {i,j}=1) + \sum_ {m=1}^M\mathbb E_ {j_m\sim P_n(j)} [\gamma\log (1-P(e_ {i,j_m}=1))] \right]
$$

其中：$j  _  m$ 表示随机采样的 $M$ 个顶点。

&emsp;&emsp;由于 $\nabla  _  {z  _  i} L $ 中 $w  _  {i,j}$ 作为乘积项出现的，而网络中边的权重 $w  _  {i,j}$ 变化范围很大，因此梯度变化会较大。这对训练不利，也就是所谓的梯度剧增和消失问题 ($gradient explosion and vanishing problem$)。

&emsp;&emsp;为了解决这个问题，$LargeVis$ 对正边也采用随机采样：若正边的权重为 $w  _  {i,j}$，则将其转换为 $w  _  {i,j}$ 个权重为 $1$ 的二元边，再随机从这些二元边中进行采样。
1. 每条边的权重都是 $1$，这解决了梯度变化范围大的问题。
2. 因为权重较大的边转换得到的二元边更多，被采样的概率越大，所以保证了正确性和合理性。

&emsp;&emsp;$LargeVis$ 使用异步随机梯度下降来进行训练。这在稀疏图上非常有效，因为不同线程采样的边所连接的两个节点很少有重复的，不同线程之间几乎不会产生冲突。
    
&emsp;&emsp;$LargeVis$ 每一轮随机梯度下降的时间复杂度为 $O(d\times M)$，其中 $M$ 为负样本个数，$d$ 为低维空间的维度。随机梯度下降的步数和样本数量 $N$ 成正比，因此总的时间复杂度为 $ O(d\times M\times N)$ ，与样本数量呈线性关系。