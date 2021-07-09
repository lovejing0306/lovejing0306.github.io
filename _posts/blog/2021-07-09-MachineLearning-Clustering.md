---
layout: post
title: 聚类
categories: [MachineLearning]
description: 聚类
keywords: MachineLearning
---


聚类
---


&emsp;&emsp;在无监督学习 ($unsupervised learning$) 中，训练样本的标记信息是未知的。其目标：通过对无标记训练样本的学习来揭露数据的内在性质以及规律。

一个经典的无监督学习任务：寻找数据的最佳表达 ($representation$)。常见的有：
1. 低维表达：试图将数据（位于高维空间）中的信息尽可能压缩在一个较低维空间中。
2. 稀疏表达：将数据嵌入到大多数项为零的一个表达中。该策略通常需要进行维度扩张。
3. 独立表达：使数据的各个特征相互独立。

&emsp;&emsp;无监督学习应用最广的是聚类 ($clustering$)。给定数据集 $\mathbb D=\{x  _  1,x  _  2,\cdots,x  _  N\}$，聚类试图将数据集中的样本划分为 $K$ 个不相交的子集 $\{\mathbb C  _  1,\mathbb C  _  2,\cdots,\mathbb C  _  K\}$，每个子集称为一个簇 ($cluster$)。其中：$\mathbb C  _  k \bigcap  _  {k\ne l} \mathbb C  _  l=\phi$,$\mathbb D=\bigcup   _  {k=1}^{K}\mathbb C  _  k$。通过这样的划分，每个簇可能对应于一个潜在的概念。这些概念对于聚类算法而言，事先可能是未知的。聚类过程仅仅能自动形成簇结构，簇所对应的概念语义需要由使用者来提供。

&emsp;&emsp;聚类可以作为一个单独的过程，用于寻找数据内在的分布结构；也可以作为其他学习任务的前驱过程。如对数据先进行聚类，然后对每个簇单独训练模型。

&emsp;&emsp;聚类问题本身是病态的，即：没有某个标准来衡量聚类的效果。可以简单的度量聚类的性质，如每个聚类的元素到该类中心点的平均距离。但是实际上不知道这个平均距离对应于真实世界的物理意义。可能很多不同的聚类都很好地对应了现实世界的某些属性，它们都是合理的。如：在图片识别中包含的图片有：红色卡车、红色汽车、灰色卡车、灰色汽车。可以聚类成：红色一类、灰色一类；也可以聚类成：卡车一类、汽车一类。
> 解决该问题的一个做法是：利用深度学习来进行分布式表达，可以对每个车辆赋予两个属性：一个表示颜色、一个表示型号。


## 性能度量
&emsp;&emsp;聚类的性能度量也称作聚类的有效性指标 ($validity index$)。直观上看，希望同一簇的样本尽可能彼此相似，不同簇的样本之间尽可能不同。即：簇内相似度 ($intra-cluster similarity$) 高，且簇间相似度 ($inter-cluster similarity$) 低。

&emsp;&emsp;聚类的性能度量分两类：  
1. 外部指标 ($external index$)：聚类结果与某个参考模型 ($reference model$) 进行比较。
2. 内部指标 ($internal index$)：直接考察聚类结果而不利用任何参考模型。

### 外部指标
&emsp;&emsp;对于数据集 $\mathbb D=\{x  _  1,x  _  2,\cdots,x  _  N\}$，假定通过聚类给出的簇划分为 $C=\{\mathbb C  _  1,\mathbb C  _  2,\cdots,\mathbb C  _  K\}$。参考模型给出的簇划分为 $C^{ \ * }=\{\mathbb C  _  1^{ \ * },\mathbb C  _  2^{ \ * },\cdots,\mathbb C  _  {K^\prime}^{ \ * }\}$，其中 $K$ 和 $ K^\prime$ 不一定相等 。

&emsp;&emsp;令 $\lambda,\lambda^{ \ * }$ 分别表示 $C,C^{ \ * }$ 的簇标记向量。定义：

$$
\begin{aligned}
   a&=\left| SS \right|,SS=\left\{ \left( \mathop{x}_{i},\mathop{x}_{j} \right)|\mathop{\lambda }_{i}=\mathop{\lambda }_{j},\mathop{\lambda }_{i}^{ \ * }=\mathop{\lambda }_{j}^{ \ * },i < j \right\}  \\
   b&=\left| SD \right|,SD=\left\{ \left( \mathop{x}_{i},\mathop{x}_{j} \right)|\mathop{\lambda }_{i}=\mathop{\lambda }_{j},\mathop{\lambda }_{i}^{ \ * }\ne \mathop{\lambda }_{j}^{ \ * },i < j \right\}  \\
   c&=\left| DS \right|,DS=\left\{ \left( \mathop{x}_{i},\mathop{x}_{j} \right)|\mathop{\lambda }_{i}\ne \mathop{\lambda }_{j},\mathop{\lambda }_{i}^{ \ * }=\mathop{\lambda }_{j}^{ \ * },i < j \right\}  \\
   d&=\left| DD \right|,DD=\left\{ \left( \mathop{x}_{i},\mathop{x}_{j} \right)|\mathop{\lambda }_{i}\ne \mathop{\lambda }_{j},\mathop{\lambda }_{i}^{ \ * }\ne \mathop{\lambda }_{j}^{ \ * },i < j \right\}  \\
\end{aligned}
$$

其中 $ \|\cdot \|$ 表示集合的元素的个数。各集合的意义为：
1. $SS$：包含了同时隶属于 $C,C^{ \ * }$ 的样本对。
2. $SD$：包含了隶属于 $C$，但是不隶属于 $C^{ \ * }$ 的样本对。
3. $DS$：包含了不隶属于 $C$，但是隶属于 $C^{ \ * }$ 的样本对。
4. $DD$：包含了既不隶属于 $C$，又不隶属于 $C^{ \ * }$ 的样本对。

由于每个样本对 $(x  _  i,x  _  j), i\lt j$ 仅能出现在一个集合中，因此有 $a+b+c+d=\frac {N(N-1)}{2}$。
    
下述性能度量的结果都在 $[0,1]$ 之间。这些值越大，说明聚类的性能越好。
    

#### Jaccard系数
$Jaccard$ 系数 $Jaccard Coefficient:JC$：

$$
JC=\frac {a}{a+b+c}
$$   

它刻画了：所有的同类的样本对（要么在 $C$ 中属于同类，要么在 $C^{ \ * }$ 中属于同类）中，同时隶属于 $C,C^{ \ * }$ 的样本对的比例。

#### FM指数
$FM$ 指数 ($Fowlkes and Mallows Index:FMI$)：

$$
FMI=\sqrt{\frac {a}{a+b}\cdot \frac{a}{a+c}}
$$

它刻画的是：
1. 在 $C$ 中同类的样本对中，同时隶属于 $C^{ \ * }$ 的样本对的比例为 $p1=\frac{a}{a+b}$。
2. 在 $C^{ \ * }$ 中同类的样本对中，同时隶属于 $C$ 的样本对的比例为 $p2=\frac{a}{a+c}$。
3. $FM$ 就是 $p1$ 和 $p2$ 的几何平均。

#### Rand指数
$Rand$ 指数 $Rand Index:RI$：

$$
RI=\frac{a+d}{N(N-1)/2}
$$

它刻画的是：同时隶属于 $C,C^{ \ * }$ 的同类样本对（这种样本对属于同一个簇的概率最大）与既不隶属于 $C$、 又不隶属于 $C^{ \ * }$ 的非同类样本对（这种样本对不是同一个簇的概率最大）之和，占所有样本对的比例。这个比例其实就是聚类的可靠程度的度量。

#### ARI指数
使用 $RI$ 有个问题：对于随机聚类，$RI$ 指数不保证接近 $0$（可能还很大）。$ARI$ 指数就通过利用随机聚类来解决这个问题。

定义一致性矩阵为：

$$
\begin{matrix}
	&		\mathbb{C}_{1}^{ \ * }&		\mathbb{C}_{2}^{ \ * }&		\cdots&		\mathbb{C}_{K^{'}}^{ \ * }&		sums\\
	\mathbb{C}_1&		n_{1,1}&		n_{1,2}&		\cdots&		n_{1,K^{'}}&		s_1\\
	\mathbb{C}_2&		n_{2,1}&		n_{2,2}&		\cdots&		n_{2,K^{'}}&		s_2\\
	\vdots&		\vdots&		\vdots&		\ddots&		\vdots&		\vdots\\
	\mathbb{C}_K&		n_{K,1}&		n_{K,2}&		\cdots&		n_{K,K^{'}}&		s_K\\
	sums&		t_1&		t_2&		\cdots&		t_K&		N\\
\end{matrix}
$$

其中：
1. $s  _  i$ 为属于簇 $\mathbb C  _  i$ 的样本的数量，$t  _  i$ 为属于簇 $\mathbb C  _  i^{ \ * }$ 的样本的数量。
2. $n  _  {i,j}$ 为同时属于簇 $\mathbb C  _  i$ 和簇 $\mathbb C  _  i^{ \ * }$ 的样本的数量。

则根据定义有：

$$
a=\sum_i\sum_jC_{n_{i,j}}^2 
$$

其中 $C  _  {n}^2 = \frac{n(n-1)}{2}$ 表示组合数。数字 $2$ 是因为需要提取两个样本组成样本对。

定义 $ARI$ 指数 ($Adjusted Rand Index$):

$$
ARI=\frac{\sum_i{\sum_j{C_{n_{i,j}}^{2}}}-\frac{\left[ \sum_i{C_{s_i}^{2}}\times \sum_j{C_{t_j}^{2}} \right]}{C_{N}^{2}}}{\frac{1}{2}\left[ \sum_i{C_{s_i}^{2}}\times \sum_j{C_{t_j}^{2}} \right] -\frac{\left[ \sum_i{C_{s_i}^{2}}\times \sum_j{C_{t_j}^{2}} \right]}{C_{N}^{2}}}
$$

其中：
1. $\sum  _  i\sum  _  jC^2  _  {n  _  {i,j}}$：表示同时隶属于 $C$,$C^{ \ * }$ 的样本对。
2. $\frac 12\left[\sum  _  iC^2  _  {s  _  i}+\sum  _  jC^2  _  {t  _  j}\right]$：表示最大的样本对。即：无论如何聚类，同时隶属于 $C$,$C^{ \ * }$ 的样本对不会超过该数值。
 3. $\left[\sum  _  iC^2  _  {s  _  i}\times \sum  _  jC^2  _  {t  _  j}\right]/C  _  N^2$：表示在随机划分的情况下，同时隶属于 $C$,$C^{ \ * }$ 的样本对的期望。
    1. 随机挑选一对样本，一共有 $C  _  N^2$ 种情形。
    2. 这对样本隶属于 $C$ 中的同一个簇，一共有 $\sum  _  iC^2  _  {s  _  i}$ 种可能。
    3. 这对样本隶属 $C^{ \ * }$ 中的同一个簇，一共有 $\sum  _  jC^2  _  {t  _  i}$ 种可能。
    4. 这对样本隶属于 $C$ 中的同一个簇、且属于 $C^{ \ * }$ 中的同一个簇，一共有 $\sum  _  iC^2  _  {s  _  i}\times \sum  _  jC^2  _  {t  _  j}$ 种可能。
    5. 则在随机划分的情况下，同时隶属于 $C$,$C^{ \ * }$ 的样本对的期望（平均样本对）为：$\left[\sum  _  iC^2  _  {s  _  i}\times\sum  _  jC^2  _  {t  _  j}\right]/C  _  N^2$。

### 内部指标
&emsp;&emsp;对于数据集 $\mathbb D=\{x  _  1,x  _  2,\cdots,x  _  N\}$，假定通过聚类给出的簇划分为 $C=\{\mathbb C  _  1,\mathbb C  _  2,\cdots,\mathbb C  _  K\}$。定义：
$$
\begin{aligned}
	avg\left( \mathbb{C}_ k \right) &=\frac{2}{\left| \mathbb{C}_ k \right|\left( \left| \mathbb{C}_ k \right|-1 \right)}\sum_{x_i,x_j\in \mathbb{C}_ k,i\ne j}{distance\left( x_i,y_j \right) ,k=1,2,\cdots ,K}\\
	diam\left( \mathbb{C}_ k \right) &=\underset{x_i,x_j\in \mathbb{C}_ k,i\ne j}{\max}distance\left( x_i,x_j \right) ,k=1,2,\cdots ,K\\
	d_{\min}\left( \mathbb{C}_ k,\mathbb{C}_ l \right) &=\underset{x_i\in \mathbb{C}_ k,x_j\in \mathbb{C}_ l}{\min}distance\left( x_i,x_j \right) ,k=1,2,\cdots ,K;k\ne l\\
	d_{cen}\left( \mathbb{C}_ k,\mathbb{C}_ l \right) &=distance\left( \mu _ k,\mu _ l \right) ,k=1,2,\cdots ,K;k\ne l\\
\end{aligned}
$$

其中：
1. $\text{distance}(x  _  i,x  _  j)$ 表示两点 $x  _  i,x  _  j$ 之间的距离；
2. $\mu   _  k$ 表示簇 $\mathbb C  _  k$ 的中心点；
3. $\mu   _  l$ 表示簇 $\mathbb C  _  l$ 的中心点；
4. $\text{distance}( \mu   _  k, \mu   _  l)$ 表示簇 $\mathbb C  _  k,\mathbb C  _  l$ 的中心点之间的距离。

上述定义的意义为：
1. $\text{avg}(\mathbb C  _  k)$：簇 $\mathbb C  _  k$ 中每对样本之间的平均距离。
2. $\text{diam}(\mathbb C  _  k)$：簇 $\mathbb C  _  k$ 中距离最远的两个点的距离。
3. $d  _  {min}(\mathbb C  _  k,\mathbb C  _  l)$：簇 $\mathbb C  _  k,\mathbb C  _  l $ 之间最近的距离。
4. $d  _  {cen}(\mathbb C  _  k,\mathbb C  _  l)$：簇 $\mathbb C  _  k,\mathbb C  _  l $ 中心点之间的距离。 

#### DB指数
$DB$ 指数 ($Davies-Bouldin Index:DBI$)：

$$
DBI=\frac 1K \sum_{k=1}^{K}\max_{k\ne l}\left(\frac{\text{avg}(\mathbb C_k)+\text{avg}(\mathbb C_l)}{d_{cen}(\mathbb C_k,\mathbb C_l)}\right)
$$

其物理意义为：
1. 给定两个簇，每个簇样本距离均值之和比上两个簇的中心点之间的距离作为度量。该度量越小越好。
2. 给定一个簇 $k$，遍历其它的簇，寻找该度量的最大值。
3. 对所有的簇，取其最大度量的均值。

显然 $DBI$ 越小越好。
1. 如果每个簇样本距离均值越小（即簇内样本距离都很近），则 $DBI$ 越小。
2. 如果簇间中心点的距离越大（即簇间样本距离相互都很远），则 $DBI$ 越小。

#### Dunn指数
$Dunn$ 指数 ($Dunn Index:DI$)：

$$
DI= \frac{\min_{k\ne l} d_{min}(\mathbb C_k,\mathbb C_l)}{\max_{i}\text{diam}(\mathbb C_i)}
$$

其物理意义为：任意两个簇之间最近的距离的最小值，除以任意一个簇内距离最远的两个点的距离的最大值。

显然 $DI$ 越大越好。
1. 如果任意两个簇之间最近的距离的最小值越大（即簇间样本距离相互都很远），则 $DI$ 越大。
2. 如果任意一个簇内距离最远的两个点的距离的最大值越小（即簇内样本距离都很近），则 $DI$ 越大。

### 距离度量

#### 闵可夫斯基距离($Minkowski distance$)：
&emsp;&emsp;给定样本 $x  _  i=(x  _  {i,1},x  _  {i,2},\cdots,x  _  {i,n})^{T},x  _  j=(x  _  {j,1},x  _  {j,2},\cdots,x  _  {j,n})^{T}$，则闵可夫斯基距离定义为：
   $$
    \text{distance}(x_i, x_j )=\left(\sum_{d=1}^{n}|x_{i,d}-x_{j,d}|^{p}\right)^{1/p}
   $$
   1. 当 $p=2$ 时，闵可夫斯基距离就是欧式距离 ($Euclidean distance$)：
       $$
       \text{distance}(x_i,x_j )=||x_i-x_j||_ {2}=\sqrt {\sum_ {d=1}^{n}|x_{i,d}-x_{j,d}|^{2}}
       $$
   2. 当 $p=1$ 时，闵可夫斯基距离就是曼哈顿距离($Manhattan distance$)：
      $$
      \text{distance}(x_i, x_j )=||x_i-x_j||_ {1}= \sum_{d=1}^{n}|x_{i,d}-x_{j,d}|
      $$

#### $VDM$距离($Value Difference Metric$)：

&emsp;&emsp;考虑非数值类属性（如属性取值为：中国，印度，美国，英国），令 $m  _  {d,a}$ 表示 $x  _  d=a$ 的样本数；$m  _  {d,a,k}$表示$x  _  d=a$ 且位于簇 $\mathbb C  _  k$ 中的样本的数量。则在属性 $ d $ 上的两个取值 $a,b$ 之间的 $VDM$ 距离为：
$$
VDM_p(a,b)=\left(\sum_{k=1}^{K}\left|\frac {m_{d,a,k}}{m_{d,a}}-\frac {m_{d,b,k}}{m_{d,b}}\right|^{p}\right)^{1/p}
$$    

该距离刻画的是：属性取值在各簇上的频率分布之间的差异。


&emsp;&emsp;当样本的属性为数值属性与非数值属性混合时，可以将闵可夫斯基距离与 $VDM$ 距离混合使用。假设属性 $x  _  1,x  _  2,\cdots,x  _  {n  _  c}$ 为数值属性， 属性 $x  _  {n  _  c+1},x  _  {n  _  c+2},\cdots,x  _  {n}$ 为非数值属性。则：

$$
\text{distance}(x_i, x_j )=\left (\sum_{d=1}^{n_c}|x_{i,d}-x_{j,d}|^{p}+\sum_{d=n_c+1}^{n}VDM_p(x_{i,d},x_{j,d})^p\right)^{1/p}
$$

当样本空间中不同属性的重要性不同时，可以采用加权距离。以加权闵可夫斯基距离为例：
$$
\text{distance}(x_i, x_j )=\left(\sum_{d=1}^{n}w_d\times|x_{i,d}-x_{j,d}|^{p}\right)^{1/p}\ w_d \ge 0,d=1,2,\cdots,n;\quad \sum_{d=1}^{n}w_d=1
$$    

&emsp;&emsp;这里的距离函数都是事先定义好的。在有些实际任务中，有必要基于数据样本来确定合适的距离函数。这可以通过距离度量学习 ($distance metric learning$) 来实现。

这里的距离度量满足三角不等式： 

$$
\text{distance}(x_i, x_j ) \le \text{distance}(x_i, x_k )+\text{distance}(x_k, x_j)
$$

在某些任务中，根据数据的特点可能需要放松这一性质。如：美人鱼和人距离很近，美人鱼和鱼距离很近，但是人和鱼的距离很远。这样的距离称作非度量距离 ($non-metric distance$)。
