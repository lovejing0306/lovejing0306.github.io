---
layout: post
title: 原型聚类
categories: [MachineLearning]
description: 原型聚类
keywords: MachineLearning
---


原型聚类
---


原型聚类 ($prototype-based clustering$) 假设聚类结构能通过一组原型刻画。常用的原型聚类有：
1. $k$ 均值算法 ($k-means$)。
2. 学习向量量化算法 ($Learning Vector Quantization:LVQ$)。
3. 高斯混合聚类 ($Mixture-of-Gaussian$)。

## k-means 算法

### k-means
#### 描述
&emsp;&emsp;$k-means$ 是一种典型的基于距离的聚类算法，其通过距离函数来度量样本间的相似性。

#### 原理
&emsp;&emsp;尝试找出使平方误差函数值最小的 $k$ 个划分。

&emsp;&emsp;给定样本集 $D=\{x  _  1,x  _  2,\cdots,x  _  N\}$，假设一个划分为 $C=\{\mathbb C  _  1,\mathbb C  _  2,\cdots,\mathbb C  _  K\}$。定义该划分的平方误差为：

$$
err=\sum_{k=1}^{K}\sum_{x_i \in \mathbb C_k}||x_i-\mu_k||_ 2^{2}
$$

其中 $\mu  _  k=\frac {1}{\|\mathbb C  _  k\|}\sum  _  {x  _  i \in \mathbb C  _  k}x  _  i$ 是簇 $\mathbb C  _  k$ 的均值向量。

&emsp;&emsp;$err$ 刻画了簇类样本围绕簇均值向量的紧密程度，其值越小，则簇内样本相似度越高。$k-means$ 算法的优化目标为：最小化 $err$ ，即
：

$$
\min_{C} \sum_{k=1}^{K}\sum_{x_i \in C_k}||x_i-\mu_k||_ 2^{2} 
$$

#### 伪码
输入：
1. 样本集 $\mathbb D=\{x  _  1,x  _  2,\cdots,x  _  N\}$。
2. 聚类簇数 $K$。

输出：簇划分 $\mathcal C=\{\mathbb C  _  1,\mathbb C  _  2,\cdots,\mathbb C  _  K\}$。
        
算法步骤：
1. 从 $\mathbb D$ 中随机选择 $K$ 个样本作为初始均值向量 ${\mu  _  1,\mu  _  2,\cdots,\mu  _  K}$。
2. 重复迭代直到算法收敛，迭代过程：
    1. 初始化阶段：取 $\mathbb C  _  k=\phi,k=1,2,\cdots,K$
    2. 划分阶段：令 $i=1,2,\cdots,N$：
        1. 计算 $x  _  i$ 的簇标记：$\lambda  _  i=\arg\min  _  {k}\|\|x  _  i-\mu  _  k\|\|  _  2 ,k \in \{1,2,\cdots,K\}$。即：将 $x  _  i$ 离哪个簇的均值向量最近，则该样本就标记为那个簇。
        2. 然后将样本 $x  _  i$ 划入相应的簇：$\mathbb C  _  {\lambda  _  i}= \mathbb C  _  {\lambda  _  i} \bigcup\{x  _  i\}$。
                    
    3. 重计算阶段：计算 $\hat{\mu}  _  k$：$\hat{\mu}  _  k =\frac {1}{\|\mathbb C  _  k\|}\sum  _  {x  _  i \in \mathbb C  _  k}x  _  i$。
    4. 终止条件判断：
        1. 如果对所有的 $k \in \{1,2,\cdots,K\}$，都有$\hat{\mu}  _  k=\mu  _  k$，则算法收敛，终止迭代。
        2. 否则重赋值 $\mu  _  k=\hat{\mu}  _  k$。

#### 度量准则
一种用于度量聚类效果的指标是误差平方和 ($Sum of Squared Error, SSE$)，$SSE$ 值越小表示数据点越接近于它们的质心，聚类效果也越好。

#### 优点
1. 计算复杂度低，为 $O(N\times K\times q)$，其中 $q$ 为迭代次数。通常 $K$ 和 $q$ 要远远小于$N$，此时复杂度相当于 $O(N)$。
2. 思想简单，容易实现。
3. 当簇是密集的、球状或团状，且簇与簇之间区别明显时，聚类效果较好。

#### 缺点
1. $k-means$ 中的 $k$ 值需要指定，而该值往往难以估计
2. 分类结果严重依赖于分类中心的初始化。通常进行多次 $k-means$，然后选择最优的那次作为最终聚类结果。
3. 结果不一定是全局最优的，只能保证局部最优。
4. 对噪声敏感。因为簇的中心是取平均，因此聚类簇很远地方的噪音会导致簇的中心点偏移。
5. 无法解决不规则形状的聚类。
6. 无法处理离散特征，如：国籍、性别等。
7. 当数据量很大时，算法的开销很大

#### 性质
1. $k-means$ 实际上假设数据是呈现球形分布，实际任务中很少有这种情况。与之相比，$GMM$ 使用更加一般的数据表示，即高斯分布。
2. $k-means$ 假设各个簇的先验概率相同，但是各个簇的数据量可能不均匀。
3. $k-means$ 使用欧式距离来衡量样本与各个簇的相似度。这种距离实际上假设数据的各个维度对于相似度的作用是相同的。
4. $k-means$ 中，各个样本点只属于与其相似度最高的那个簇，这实际上是硬分簇。  
5. $k-means$ 的迭代步骤可以看成是 $E$ 步和 $M$ 步。$E$步：固定类别中心向量重新标记样本；$M$ 步：固定标记样本调整类别中心向量。

#### 高斯混合模型和 $k-means$ 的异同
相同点：
1. 需要指定 $k$ 值；
2. 需要指定初始值，如 $k-means$ 的中心点，$GMM$ 的各个参数；
3. 都含有 $EM$ 算法的思想；

不同点：
1. 优化的目标函数不同，$k-means$ 最短距离，$GMM$ 是最大化似然函数
2. $E$ 步的指标不同，$k-means$ 是点到中心的距离（找出距离最短的中心），$GMM$ 是求解样本来自各个模型的概率（找出概率最大的模型）。

        

### k-means++
#### 描述
&emsp;&emsp;$k-means$ 算法中随机选择的初始聚类中心可能会造成聚类的结果与数据实际分布的结果相差较大。$k-means++$ 算法是对 $k-means$ 算法初始聚类中心选取的一种改进。

#### 原理
&emsp;&emsp;$k-means++$ 算法认为，选取的初始聚类中心之间的相互距离要尽可能的远。
    
#### 伪码
输入：
1. 样本集 $\mathbb D=\{x  _  1,x  _  2,\cdots,x  _  N\}$。
2. 聚类簇数 $K$。

输出：簇划分 $ C=\{\mathbb C  _  1,\mathbb C  _  2,\cdots,\mathbb C  _  K\}$。

算法步骤：
1. 从 $\mathbb D$ 中随机选择 $1$ 个样本作为初始均值向量组 ${\mu  _  1,}$。
2. 迭代，直到初始均值向量组有 $K$ 个向量。假设初始均值向量组为 ${\mu  _  1,\cdots,\mu  _  m}$。迭代过程如下：
    1. 对每个样本 $x  _  i$，分别计算其与 $\mu  _  1,\cdots,\mu  _  m$ 之间的距离，取这些距离中的最小值记做 $d  _  i=\min  _  {\mu  _  j} \|\|x  _  i-\mu  _  j\|\|$。
    2. 对样本 $x  _  i$，其设置为初始均值向量的概率正比于 $d  _  i$，即：离所有的初始均值向量越远，则越可能被选中为下一个初始均值向量。
    3. 以概率分布 $P=\{d  _  1,d  _  2,\cdots,d  _  N\}$ （未归一化的）随机挑选一个样本作为下一个初始均值向量 $\mu  _  {m+1}$。
3. 一旦挑选出初始均值向量组 ${\mu  _  1,\cdots,\mu  _  K}$，剩下的迭代步骤与 $k-means$ 相同。
            

### ~~k-modes~~
&emsp;&emsp;$k-modes$ 属于 $k-means$ 的变种，它主要解决 $k-means$ 无法处理离散特征的问题。$k-modes$ 与 $k-means$ 有两个不同点（假设所有特征都是离散特征）：
1. 距离函数不同。
   在 $k-modes$ 算法中，距离函数为：
   $$
   \text{distance}(x_i, x_j)=\sum_{d=1}^n I(x_{i,d}=x_{j,d})
   $$
   其中 $I(\cdot)$ 为示性函数。上式的意义为：样本之间的距离等于它们之间属性值相同的个数。

2. 簇中心的更新规则不同。
   在 $k-modes$ 算法中，簇中心每个属性的取值为：簇内该属性出现频率最大的那个值。
   $$
   \hat\mu_{k,d} = \arg\max_{v} \sum_{x_i\in \mathbb C_k} I(x_{i,d}=v)
   $$
   其中 $v$ 的取值空间为所有样本在第 $d$ 个属性上的取值。

### mini-batch k-means
#### 描述
&emsp;&emsp;$mini-batch \ k-means$ 属于 $k-means$ 的变种，它主要用于减少 $k-means$ 的计算时间。$mini-batch  \ k-means$ 算法每次训练时随机抽取小批量的数据，然后用这个小批量数据训练。这种做法减少了 $k-means$ 的收敛时间，其效果略差于标准算法。

#### 伪码
输入：
1. 样本集 $\mathbb D=\{x  _  1,x  _  2,\cdots,x  _  N\}$。
2. 聚类簇数 $K$

输出：簇划分 $C=\{\mathbb C  _  1,\mathbb C  _  2,\cdots,\mathbb C  _  K\}$。

算法步骤：
1. 从 $\mathbb D$ 中随机选择 $ K$ 个样本作为初始均值向量 $\{\mu  _  1,\mu  _  2,\cdots,\mu  _  K\}$。
2. 重复迭代直到算法收敛，迭代过程：
    1. 初始化阶段：取 $\mathbb C  _  k=\phi,k=1,2,\cdots,K$
    2. 划分阶段：随机挑选一个 $Batch$ 的样本集合 $\mathbb B={x  _  {b  _  1},\cdots,x  _  {b  _  M}}$，其中 $M$ 为批大小。
        1. 计算 $x  _  i,i=b  _  1,\cdots,b  _  M$ 的簇标记：
           $$
           \lambda_i=\arg\min_{k}\|\|x_i-\mu_k\|\|_ 2 ,k \in \{1,2,\cdots,K\} 
           $$
           即：将 $x  _  i$ 离哪个簇的均值向量最近，则该样本就标记为那个簇。
        2. 然后将样本 $x  _  i,i=b  _  1,\cdots,b  _  M$ 划入相应的簇：$\mathbb C  _  {\lambda  _  i}= \mathbb C  _  {\lambda  _  i} \bigcup\{x  _  i\}$。
    3. 重计算阶段：计算 $\hat{\mu}  _  k$：$\hat{\mu}  _  k =\frac {1}{\|\mathbb C  _  k\|}\sum  _  {x  _  i \in \mathbb C  _  k}x  _  i$。
    4. 终止条件判断：
        1. 如果对所有的 $k \in \{1,2,\cdots,K\}$，都有 $\hat{\mu}  _  k=\mu  _  k$，则算法收敛，终止迭代。
        2. 否则重赋值 $\mu  _  k=\hat{\mu}  _  k$。

## 高斯混合聚类
&emsp;&emsp;高斯混合聚类采用概率模型来表达聚类原型。对 $n$ 维样本空间 $\mathcal X$ 中的随机向量 $x$，若 $x$ 服从高斯分布，则其概率密度函数为：

$$
p(x\mid \mu,\Sigma)=\frac {1}{(2\pi)^{n/2}|\Sigma|^{1 /2}}\exp\left(-\frac 12(x- \mu)^{T}\Sigma^{-1}(x- \mu)\right)
$$

其中 $\mu=(\mu  _  1,\mu  _  2,\cdots,\mu  _  n)^{T}$ 为 $n$ 维均值向量，$\Sigma$ 是 $n\times n$ 的协方差矩阵。$x$ 的概率密度函数由参数 $\mu,\Sigma$ 决定。

&emsp;&emsp;定义高斯混合分布： 

$$
p_{\mathcal M}=\sum_{k=1}^{K}\alpha_k p(x\mid \mu_k,\Sigma_k)
$$

该分布由 $K$ 个混合成分组成，每个混合成分对应一个高斯分布。其中: $\mu  _  k,\Sigma  _  k$ 是第 $k$ 个高斯混合成分的参数。$\alpha  _  k \gt 0$ 是相应的混合系数，满足 $\sum  _  {k=1}^{K}\alpha  _  k=1$。之所有要保证权重的和为 $1$，是因为概率密度函数必须满足在 $\left( +\infty ,-\infty \right) $ 内的积分值为 $1$ 。

&emsp;&emsp;假设训练集 $\mathbb D=\{x  _  1,x  _  2,\cdots,x  _  N\}$ 的生成过程是由高斯混合分布给出。令随机变量 $Z \in \{1,2,\cdots,K\}$ 表示生成样本 $x$ 的高斯混合成分序号，$Z$ 的先验概率$P(Z =k)=\alpha  _  k$。生成样本的过程分为两步：
1. 首先根据概率分布 $\alpha  _  1,\alpha  _  2,\cdots,\alpha  _  K$ 生成随机变量 $Z$。
2. 再根据 $Z$ 的结果，比如 $Z =k$， 根据概率 $p(x \mid \mu  _  k,\Sigma  _  k)$ 生成样本。

&emsp;&emsp;根据贝叶斯定理， 若已知输出为 $x  _  i$，则 $Z$ 的后验分布为：

$$
p_{\mathcal M}(Z =k\mid x_i)=\frac{P(Z =k)p_{\mathcal M}(x_i \mid Z =k)}{p_{\mathcal M}(x_i)} = \frac{\alpha_k p(x_i\mid \mu_k,\Sigma_k)}{\sum_{l=1}^{K}\alpha_l p(x_i\mid \mu_l,\Sigma_l)}
$$    

其物理意义为：所有导致输出为 $x  _  i$ 的情况中，$Z =k$ 发生的概率。

&emsp;&emsp;当高斯混合分布已知时，高斯混合聚类将样本集 $\mathbb D$ 划分成 $K$ 个簇 $C=\{\mathbb C  _  1,\mathbb C  _  2,\cdots,\mathbb C  _  K\}$。 对于每个样本 $x  _  i$ ，给出 它的簇标记 $\lambda  _  i$ 为：

$$
\lambda_i=\arg\max_k p_{\mathcal M}(Z =k\mid x_i)
$$

即 如果 $x  _  i$ 最有可能是 $Z =k$ 产生的，则将该样本划归到簇 $\mathbb C  _  k$。这就是通过最大后验概率确定样本所属的聚类。
    
现在的问题是，如何学习高斯混合分布的参数。由于涉及到隐变量 $Z$，可以采用 $EM$ 算法求解。具体求解参考 $EM$ 算法的章节部分。

