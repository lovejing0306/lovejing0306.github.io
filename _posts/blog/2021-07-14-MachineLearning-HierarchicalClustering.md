---
layout: post
title: 层次聚类
categories: [MachineLearning]
description: 层次聚类
keywords: MachineLearning
---


层次聚类
---


层次聚类 ($hierarchical \ clustering$) 试图在不同层次上对数据集进行划分，从而形成树形的聚类结构。

## AGNES 算法
层次凝聚 ($AGglomerative NESting,AGNES$) 是一种采用自底向上聚合策略的层次聚类算法。
 
### 思路  
&emsp;&emsp;首先将数据集中的每个样本看作一个初始的聚类簇；然后在算法运行的每一步中，找出距离最近的两个聚类簇进行合并。合并过程不断重复，直到达到预设的聚类簇的个数。

### 距离度量
由于每个簇就是一个集合，因此只需要采用关于集合的某个距离即可。给定聚类簇 $\mathbb C  _  i$和$\mathbb C  _  j$，有三种距离：  

#### 最小距离

$$
d_{min}(\mathbb C_i,\mathbb C_j)=\min_{x_i \in \mathbb C_i,x_j \in \mathbb C_j}distance(x_i,x_j)
$$

最小距离由两个簇的最近样本决定。

#### 最大距离
$$
d_{max}(\mathbb C_i,\mathbb C_j)=\max_{x_i \in \mathbb C_i,x_j \in \mathbb C_j}distance(x_i,x_j)
$$

最大距离由两个簇的最远样本决定。
        
#### 平均距离

$$
d_{avg}(\mathbb C_i,\mathbb C_j)=\frac{1}{|\mathbb C_i||\mathbb C_j|}\sum_{x_i \in \mathbb C_i}\sum_{x_j \in \mathbb C_j}distance(x_i,x_j)
$$

平均距离由两个簇的所有样本决定。

$AGNES$ 算法可以采取上述任意一种距离：
1. 当 $AGNES$ 算法的聚类簇距离采用 $d  _  {min}$ 时，称作单链接 ($single-linkage$) 算法。
2. 当 $AGNES$ 算法的聚类簇距离采用 $d  _  {max}$ 时，称作全链接 ($complete-linkage$) 算法。
3. 当 $AGNES$ 算法的聚类簇距离采用 $d  _  {avg}$ 时，称作均链接 ($average-linkage$) 算法 。

### 伪码
输入：
1. 数据集 $\mathbb D=\{x  _  1,x  _  2,\cdots,x  _  N\}$
2. 聚类簇距离度量函数 $d(\cdot,\cdot)$
3. 聚类簇数量 $K$

输出：簇划分 $C=\{\mathbb C  _  1,\mathbb C  _  2,\cdots,\mathbb C  _  K\}$

算法步骤：
1. 初始化：将每个样本都作为一个簇
    $$
    \mathbb C_i=\{x_i\} ,i=1,2,\cdots,N
    $$
2. 迭代，终止条件为聚类簇的数量为 $K$。迭代过程为：计算聚类簇之间的距离，找出距离最近的两个簇，将这两个簇合并。
   > 每进行一次迭代，聚类簇的数量就减少一些。

### 优缺点
$AGNES$ 算法的优点：
1. 距离容易定义，使用限制较少。
2. 可以发现聚类的层次关系。

$AGNES$ 算法的缺点：
1. 计算复杂度较高。
2. 算法容易聚成链状。


## DIANA算法
&emsp;&emsp;层次分裂是一种采用自顶向下聚类策略的层次聚类算法，它首先把所有待聚类的样本点都放到同一个簇中，并计算这个簇中最不相似的两个样本点；将这两个样本点分别放到两个不同的簇中，其它样本点所属的簇由它们和这两个样本点之间的相似度决定；重复上述过程，直到每个样本自成一簇，或者达到了某个终止条件。

&emsp;&emsp;$DIANA$ 是最具代表性的自顶向下层次聚类算法之一。
