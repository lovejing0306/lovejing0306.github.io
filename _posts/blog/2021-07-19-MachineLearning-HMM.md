---
layout: post
title: 隐马尔可夫模型
categories: [MachineLearning]
description: 隐马尔可夫模型
keywords: MachineLearning
---


隐马尔可夫模型
---


## 隐马尔可夫模型HMM
    
&emsp;&emsp;隐马尔可夫模型（ $Hidden \ Markov \  model,HMM$ ）是关于时序的概率模型，描述由一个隐藏的马尔可夫链随机生成不可观测的状态随机序列，再由各个状态生成一个观测而产生观测随机序列的过程。属于生成模型。
1. 隐藏的马尔可夫链随机生成的状态序列称作状态序列。
2. 每个状态生成一个观测，而由此产生的观测的随机序列称作观测序列。
3. 序列的每一个位置又可以看作是一个时刻。

### 基本概念
&emsp;&emsp;设 $\mathbb Q=\{{\mathbf q}  _  1,{\mathbf q}  _  2,\cdots,{\mathbf q}  _  Q\}$ 是所有可能的状态的集合， $\mathbb V=\{{\mathbf v}  _  1,{\mathbf v}  _  2,\cdots,{\mathbf v}  _  V\}$ 是所有可能的观测的集合，其中 $Q$ 是可能的状态数量， $V$  是可能的观测数量； $\mathbb Q$ 是状态的取值空间， $\mathbb V$ 是观测的取值空间；每个观测值 ${\mathbf v}  _  i$ 可能是标量，也可能是一组标量构成的集合，因此这里用加粗的黑体表示。状态值的表示也类似。

&emsp;&emsp;设 $\mathbf I=( i  _  1,i  _  2,\cdots,i  _  T)$ 是长度为 $T$ 的状态序列， $\mathbf O=(o  _  1,o  _  2,\cdots,o  _  T)$ 是对应的观测序列。其中， $i  _  t\in\{1,\cdots,Q\}$ 是一个随机变量，代表状态 $\mathbf q  _  {i  _  t}$ ； $o  _  t\in\{1,\cdots,V\}$  是一个随机变量，代表观测 $\mathbf v  _  {o  _  t}$ 。

&emsp;&emsp;设 $\mathbf A$ 为状态转移概率矩阵：

$$
\mathbf A=\begin{bmatrix} a_{1,1} & a_{1,2} & \cdots & a_{1,Q} \\ a_{2,1} & a_{2,2} & \cdots & a_{2,Q} \\ \vdots &\vdots &\vdots&\vdots \\ a_{Q,1} & a_{Q,2} & \cdots & a_{Q,Q} \end{bmatrix}
$$

其中 $a  _  {i,j}=P({i}  _  {t+1}=j\mid {i}  _  t=i)$ ，表示在时刻 $t$ 处于状态 ${\mathbf q}  _  i$ 的条件下，在时刻 $t+1$ 时刻转移到状态 ${\mathbf q}  _  j$ 的概率。

&emsp;&emsp;设 $\mathbf B$ 为观测概率矩阵：

$$
\mathbf B=\begin{bmatrix} b_1(1) & b_1(2) & \cdots & b_1(V) \\ b_2(1) & b_2(2) & \cdots & b_2(V) \\ \vdots &\vdots &\vdots&\vdots \\ b_Q(1) & b_Q(2) & \cdots & b_Q(V) \end{bmatrix}
$$

其中 $b  _  j(k)=P({o}  _  t=k\mid {i}  _  t=j)$ ，表示在时刻 $t$ 处于状态 ${\mathbf q}  _  j$ 的条件下生成观测 ${\mathbf v}  _  k$ 的概率。
    
&emsp;&emsp;设 $\pi$ 是初始状态概率矩阵： $\pi=(\pi  _  1,\pi  _  2,\cdots,\pi  _  Q)^{T},\quad \pi  _  i=P({i}  _  1=i)$ 是时刻 $t=1$ 时处于状态 ${\mathbf q}  _  i$ 的概率。根据定义有： 

$$
\sum_{i=1}^{Q}\pi_i=1
$$

&emsp;&emsp;隐马尔可夫模型由初始状态概率向量 $\pi$ 、状态转移概率矩阵 $\mathbf A$ 以及观测概率矩阵 $\mathbf B$ 决定。因此隐马尔可夫模型 $\lambda$ 可以用三元符号表示，即：

$$
\lambda=(\mathbf A,\mathbf B,\pi) 
$$

其中， $\mathbf A,\mathbf B,\pi$ 称为隐马尔可夫模型的三要素。
> 状态转移概率矩阵 $\mathbf A$ 和初始状态概率向量 $\pi$ 确定了隐藏的马尔可夫链，生成不可观测的状态序列。
> 观测概率矩阵 $\mathbf B$ 确定了如何从状态生成观测，与状态序列一起确定了如何产生观测序列。

&emsp;&emsp;从定义可知，隐马尔可夫模型做了两个基本假设：
1. 齐次性假设：即假设隐藏的马尔可夫链在任意时刻 $t$ 的状态只依赖于它前一时刻的状态，与其他时刻的状态和观测无关，也与时刻 $t$ 无关，即：
    $$
    P({i}_ t\mid {i}_ {t-1},{o}_ {t-1},\cdots,{i}_ 1,{o}_ 1)=P({i}_ t\mid {i}_ {t-1}),\quad t=1,2,\cdots,T
    $$
2. 观测独立性假设，即假设任意时刻的观测值只依赖于该时刻的马尔可夫链的状态，与其他观测及状态无关，即：
    $$
    P({o}_ t\mid {i}_ T,{o}_ T,\cdots,{i}_ {t+1},{o}_ {t+1},{i}_ {t},{i}_ {t-1},{o}_ {t-1},\cdots,{i}_ 1,{o}_ 1)=P({o}_ t\mid {i}_ t),\quad t=1,2,\cdots,T
    $$

## HMM 基本问题

隐马尔可夫模型的 $3$ 个基本问题：
1. 概率计算问题（也称为评估问题）：给定模型 $\lambda=(\mathbf A,\mathbf B,\vec \pi)$ 和观测序列 $\mathbf O=({o}  _  1,{o}  _  2,\cdots,{o}  _  T)$ ，计算观测序列 $\mathbf O$ 出现的概率 $P(\mathbf O;\lambda)$ ，即评估模型 $\lambda$ 与观察序列  $\mathbf O$ 之间的匹配程度。
2. 学习问题：已知观测序列 $\mathbf O=({o}  _  1,{o}  _  2,\cdots,{o}  _  T)$ ，估计模型 $\lambda=(\mathbf A,\mathbf B,\vec \pi)$ 的参数，使得在该模型下观测序列概率 $P(\mathbf O;\lambda)$ 最大，即用极大似然估计的方法估计参数。
    > 根据训练数据是包括观测序列和对应的状态序列还是只有观测序列，可以分别由监督学习和非监督学习实现。
3. 预测问题（也称为解码问题）：已知模型 $\lambda=(\mathbf A,\mathbf B,\vec \pi)$ 和观测序列 $\mathbf O=({o}  _  1,{o}  _  2,\cdots,{o}  _  T)$ ，求对给定观测序列的条件概率 $P(\mathbf I\mid \mathbf O)$ 最大的状态序列 $\mathbf I=({i}  _  1,{i}  _  2,\cdots,{i}  _  T)$ ，即给定观测序列，求最可能对应的状态序列。
    > 如：在语音识别任务中，观测值为语音信号，隐藏状态为文字。解码问题的目标就是：根据观测的语音信号来推断最有可能的文字序列。

### 概率计算问题
&emsp;&emsp;给定隐马尔可夫模型 $\lambda=(\mathbf A,\mathbf B,\vec \pi)$ 和观测序列 $\mathbf O=({o}  _  1,{o}  _  2,\cdots,{o}  _  T)$ ，概率计算问题需要计算在模型 $\lambda$ 下观测序列 $\mathbf O$ 出现的概率 $P(\mathbf O;\lambda)$ 。
    
最直接的方法是按照概率公式直接计算：通过列举所有可能的、长度为 $T$ 的状态序列 $\mathbf I=({i}  _  1,{i}  _  2,\cdots,{i}  _  T)$ ，求各个状态序列 $\mathbf I$ 与观测序列 $\mathbf O=({o}  _  1,{o}  _  2,\cdots,{o}  _  T)$ 的联合概率 $P(\mathbf O,\mathbf I;\lambda)$ ，然后对所有可能的状态序列求和，得到 $P(\mathbf O;\lambda)$ 。
    
状态序列 $\mathbf I=({i}  _  1,{i}  _  2,\cdots,{i}  _  T)$ 的概率为：

$$
P(\mathbf I;\lambda)=\pi_{{i}_ 1}a_{{i}_ 1,{i}_ 2}a_{{i}_ 2,{i}_ 3}\cdots a_ {{i}_ {T-1},{i}_ T}
$$

给定状态序列 $\mathbf I=({i}  _  1,{i}  _  2,\cdots,{i}  _  T)$ ，观测序列 $\mathbf O=({o}  _  1,{o}  _  2,\cdots,{o}  _  T)$ 的条件概率为：

$$
P(\mathbf O\mid \mathbf I;\lambda)=b_{{i}_ 1}({o}_ 1)b_{{i}_ 2}({o}_ 2)\cdots b_{{i}_ T}({o}_ T)
$$

$\mathbf O$ 和 $\mathbf I$ 同时出现的联合概率为：

$$
P(\mathbf O,\mathbf I;\lambda)=P(\mathbf O\mid \mathbf I;\lambda)P(\mathbf I;\lambda)=\pi_{{i}_ 1}a_{{i}_ 1,{i}_ 2}a_{{i}_ 2,{i}_ 3}\cdots a_{{i}_ {T-1},{i}_ T}b_{{i}_ 1}({o}_ 1)b_{{i}_ 2}({o}_ 2)\cdots b_{{i}_ T}({o}_ T)
$$

对所有可能的状态序列 $\mathbf I$ 求和，得到观测序列 $\mathbf O$ 的概率：

$$
P(\mathbf O;\lambda)=\sum_{\mathbf I} P(\mathbf O,\mathbf I;\lambda)=\sum_{{i}_ 1,{i}_ 2,\cdots,{i}_ T} \pi_{{i}_ 1}a_{{i}_ 1,{i}_ 2}a_{{i}_ 2,{i}_ 3}\cdots a_{{i}_ {T-1},{i}_ T}b_{{i}_ 1}({o}_ 1)b_{{i}_ 2}({o}_ 2)\cdots b_{{i}_ T}({o}_ T)
$$

上式的算法复杂度为 $O(T\times Q^{T})$ ，太复杂，实际应用中不太可行。
        

#### 前向算法
&emsp;&emsp;给定隐马尔可夫模型 $\lambda=(\mathbf A,\mathbf B,\vec \pi)$ ，定义前向概率：在时刻 $t$ 时的观测序列为 ${o}  _  1,{o}  _  2,\cdots,{o}  _  t$ ， 且时刻 $t$ 时状态为 ${\mathbf q}  _  {i}$  的概率为前向概率，记作：

$$
\alpha_t(i)=P({o}_ 1,{o}_ 2,\cdots,{o}_ t,{i}_ t=i;\lambda)
$$

根据定义， $\alpha  _  t(j)$ 是在时刻 $t$ 时观测到 ${o}  _  1,{o}  _  2,\cdots,{o}  _  t$ ，且在时刻 $t$ 处于状态 ${\mathbf q}  _  j$ 的前向概率。则有：

1. $\alpha  _  t(j)\times a  _  {j,i}$ ：为在时刻 $t$ 时观测到 ${o}  _  1,{o}  _  2,\cdots,{o}  _  t$ ，且在时刻 $t$ 处于状态 ${\mathbf q}  _  j$ ，且在 $t+1$ 时刻处在状态 ${\mathbf q}  _  i$ 的概率。

2. $\sum  _  {j=1}^{Q}\alpha  _  t(j)\times a  _  {j,i}$ ：为在时刻 $t$ 观测序列为 ${o}  _  1,{o}  _  2,\cdots,{o}  _  t$ ，并且在时刻 $t+1$ 时刻处于状态 ${\mathbf q}  _  i$ 的概率。
        
3. 考虑 $b  _  i(o  _  {t+1})$ ，则得到前向概率的地推公式：

$$
\alpha_{t+1}(i)=\left[\sum_{j=1}^{Q}\alpha_t(j)a_{j,i}\right]b_i(o_{t+1})
$$

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/HMM/forward_HMM_prob.png?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Forward HMM Prob</div>
</center>
        
##### 伪码
输入：
1. 隐马尔可夫模型 $\lambda=(\mathbf A,\mathbf B,\pi)$ 
2. 观测序列 $\mathbf O=({o}  _  1,{o}  _  2,\cdots,{o}  _  T)$ 

输出：观测序列概率 $P(\mathbf O;\lambda)$ 
        
算法步骤：
1. 计算初值： $\alpha  _  1(i)=\pi  _  ib  _  i(o  _  1),\quad i=1,2,\cdots,Q$ 。该初值是初始时刻的状态 $i  _  1=i$ 和观测 $o  _  1$ 的联合概率。
2. 递推：对于 $t=1,2,\cdots,T-1$ ：
    $$
     \alpha_{t+1}(i)=\left[\sum_{j=1}^{Q}\alpha_t(j)a_{j,i}\right]b_i(o_{t+1}),\quad i=1,2,\cdots,Q
    $$
3. 终止： $P(\mathbf O;\lambda)=\sum  _  {i=1}^{Q}\alpha  _  T(i)$ 。
    因为 $\alpha  _  T(i)$ 表示在时刻 $T$ ，观测序列为 ${o}  _  1,{o}  _  2,\cdots,{o}  _  T$ ，且状态为 ${\mathbf q}  _  i$ 的概率。对所有可能的 $Q$ 个状态 ${\mathbf q}  _  i$ 求和则得到 $P(\mathbf O;\lambda)$ 。
            
前向算法是基于状态序列的路径结构递推计算 $P(\mathbf O;\lambda)$ 。其高效的关键是局部计算前向概率，然后利用路径结构将前向概率“递推”到全局。算法复杂度为 $O(TQ^{2})$ 。

#### 后向算法
&emsp;&emsp;给定隐马尔可夫模型 $\lambda=(\mathbf A,\mathbf B,\vec \pi)$ ，定义后向概率：在时刻 $t$ 的状态为 ${\mathbf q}  _  i$ 的条件下，从时刻 $t+1$ 到 $T$ 的观测序列为 $o  _  {t+1},o  _  {t+2},\cdots,o  _  T$  的概率为后向概率，记作：

$$
\beta_t(i)=P(o_{t+1},o_{t+2},\cdots,o_T\mid i_t=i;\lambda) 
$$

&emsp;&emsp;在时刻 $t$ 状态为 ${\mathbf q}  _  i$ 的条件下，从时刻 $t+1$ 到 $T$ 的观测序列为 $o  _  {t+1},o  _  {t+2},\cdots,o  _  T$ 的概率可以这样计算：
1. 考虑 $t$ 时刻状态 ${\mathbf q}  _  i$ 经过 $a  _  {i,j}$ 转移到 $t+1$ 时刻的状态 ${\mathbf q}  _  j$ 。
    1. $t+1$ 时刻状态为 ${\mathbf q}  _  j$ 的条件下，从时刻 $t+2$ 到 $T$ 的观测序列为观测序列为 $o  _  {t+2},o  _  {t+3},\cdots,o  _  T$ 的概率为 $\beta  _  {t+1}(j)$ 。
    2. $t+1$ 时刻状态为 ${\mathbf q}  _  j$ 的条件下，从时刻 $t+1$ 到 $T$ 的观测序列为观测序列为 $o  _  {t+1}, o  _  {t+2},\cdots,o  _  T$ 的概率为 $b  _  j(o  _  {t+1})\times \beta  _  {t+1}(j)$ 。
2. 考虑所有可能的 ${\mathbf q}  _  j$ ，则得到 $\beta  _  t(j)$ 的递推公式：
    $$
    \beta_t(i)=\sum_{j=1}^{Q} a_{i,j}b_j(o_{t+1})\beta_{t+1}(j)
    $$

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/HMM/backward_HMM_prob.png?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Backward HMM Prob</div>
</center>
        
##### 伪码
1. 输入：
    1. 隐马尔可夫模型 $\lambda=(\mathbf A,\mathbf B,\pi)$ 
    2. 观测序列 $\mathbf O=({o}  _  1,{o}  _  2,\cdots,{o}  _  T)$ 
2. 输出： 观测序列概率 $P(\mathbf O;\lambda)$ 
3. 算法步骤：
    1. 计算初值： $\beta  _  T(i)=1,\quad i=1,2,\cdots,Q$ 对最终时刻的所有状态 ${\mathbf q}  _  i$ ，规定 $\beta  _  T(i)=1$ 。
    2. 递推：对 $t=T-1,T-2,\cdots,1$ :
        $$
        \beta_t(i)=\sum_{j=1}^{Q} a_{i,j}b_j(o_{t+1})\beta_{t+1}(j),\quad i=1,2,\cdots,Q
        $$
    3. 终止： $P(\mathbf O;\lambda)=\sum  _  {i=1}^{Q}\pi  _  ib  _  i(o  _  1)\beta  _  1(i)$ ， $\beta  _  1(i)$ 为在时刻 $1$ ， 状态为 ${\mathbf q}  _  i$ 的条件下，从时刻 $2$ 到 $T$ 的观测序列为 $o  _  {2},o  _  {3},\cdots,o  _  T$  的概率。对所有的可能初始状态 ${\mathbf q}  _  i$ （由 $\pi  _  i$ 提供其概率）求和并考虑 $o  _  1$ 即可得到观测序列为 $o  _  {1},o  _  {2},\cdots,o  _  T$ 的概率。
            

#### 统一形式
&emsp;&emsp;利用前向概率和后向概率的定义，可以将观测序列概率统一为：

$$
P(\mathbf O;\lambda)=\sum_{i=1}^{Q}\sum_{j=1}^{Q}\alpha_t(i)a_{i,j}b_j(o_{t+1})\beta_{t+1}(j),\quad t=1,2,\cdots,T-1
$$

当 $t=1$ 时，就是后向概率算法；当 $t=T-1$ 时，就是前向概率算法。

其意义为，在时刻 t：
1. $\alpha  _  t(i)$ 表示：已知时刻 $t$ 时的观测序列为 ${o}  _  1,{o}  _  2,\cdots,{o}  _  t$ 、且时刻 $t$ 时状态为 ${\mathbf q}  _  i$ 的概率。
2. $\alpha  _  t(i)a  _  {i,j}$ 表示：已知时刻 $t$ 时的观测序列为 ${o}  _  1,{o}  _  2,\cdots,{o}  _  t$ 、且时刻 $t$ 时状态为 ${\mathbf q}  _  i$ 、且 $t+1$ 时刻状态为 ${\mathbf q}  _  j$ 的概率。
3. $\alpha  _  t(i)a  _  {i,j}b  _  j(o  _  {t+1})$ 表示：已知时刻 $t+1$ 时的观测序列为 ${o}  _  1,{o}  _  2,\cdots,{o}  _  {t+1}$ 、且时刻 $t$ 时状态为 ${\mathbf q}  _  i$ 、且 $t+1$ 时刻状态为 ${\mathbf q}  _  j$ 的概率。
4. $\alpha  _  t(i)a  _  {i,j}b  _  j(o  _  {t+1})\beta  _  {t+1}(j)$ 表示：已知观测序列为 ${o}  _  1,{o}  _  2,\cdots,{o}  _  {T}$ 、且时刻 $t$ 时状态为 ${\mathbf q}  _  i$ 、且 $t+1$ 时刻状态为 ${\mathbf q}  _  j$ 的概率。
5. 对所有可能的状态 $\mathbf q  _  i,\mathbf q  _  j$ 取值，即得到上式。

&emsp;&emsp;根据前向算法有：

$$
\alpha_{t+1}(j)=\sum_{i=1}^Q\alpha_t(i)a_{i,j}b_j(o_{t+1})
$$

则得到：
$$
\begin{aligned}
P(\mathbf O;\lambda)&=\sum_{i=1}^{Q}\sum_{j=1}^{Q}\alpha_t(i)a_{i,j}b_j(o_{t+1})\beta_{t+1}(j)\\ 
&=\sum_{j=1}^{Q}\left[\sum_{i=1}^{Q}\alpha_t(i)a_{i,j}b_j(o_{t+1})\right]\beta_{t+1}(j) \\
&=\sum_{j=1}^Q\alpha_{t+1}(j)\beta_{t+1}(j)\quad\\ 
t&=1,2,\cdots,T-1
\end{aligned}
$$

由于 $t$ 的形式不重要，因此有：

$$
P(\mathbf O;\lambda)=\sum_{j=1}^Q\alpha_{t}(j)\beta_{t}(j),\quad t=1,2,\cdots,T
$$

&emsp;&emsp;给定模型 $\lambda=(\mathbf A,\mathbf B,\pi)$ 和观测序列 $\mathbf O$ 的条件下，在时刻 $t$ 处于状态 ${\mathbf q}  _  i$ 的概率记作：

$$
\gamma_t(i)=P(i_t=i\mid \mathbf O;\lambda)
$$

根据定义：

$$
\gamma_t(i)=P(i_t=i\mid \mathbf O;\lambda)=\frac{P(i_t=i,\mathbf O;\lambda)}{P(\mathbf O;\lambda)}
$$

根据前向概率和后向概率的定义，有： 

$$
\alpha_t(i)\beta_t(i)=P(i_t=i,\mathbf O;\lambda)
$$

则有：

$$
\begin{aligned}
\gamma_t(i)&=\frac{P(i_t=i,\mathbf O;\lambda)}{P(\mathbf O;\lambda)} \\
&=\frac{\alpha_t(i)\beta_t(i)}{P(\mathbf O;\lambda)} \\
&=\frac{\alpha_t(i)\beta_t(i)}{\sum_{j=1}^{Q}\alpha_t(j)\beta_t(j)}
\end{aligned}
$$

&emsp;&emsp;给定模型 $\lambda=(\mathbf A,\mathbf B,\pi)$ 和观测序列 $\mathbf O$ ，在时刻 $t$ 处于状态 ${\mathbf q}  _  i$ 且在 $t+1$ 时刻处于状态 ${\mathbf q}  _  j$ 的概率记作：

$$
\xi_t(i,j)=P(i_t=i,i_{t+1}=j\mid \mathbf O;\lambda)
$$

根据

$$
\begin{aligned}
\xi_t(i,j)&=P(i_t=i,i_{t+1}=j\mid \mathbf O;\lambda) \\
&=\frac{P(i_t=i,i_{t+1}=j,\mathbf O;\lambda)}{P(\mathbf O;\lambda)}\\
&=\frac{P(i_t=i,i_{t+1}=j,\mathbf O;\lambda)}{\sum_{u=1}^{Q}\sum_{v=1}^{Q}P(i_t=u,i_{t+1}=v,\mathbf O;\lambda)}
\end{aligned}
$$

考虑到前向概率和后向概率的定义有： 

$$
P(i_t=i,i_{t+1}=j,\mathbf O;\lambda)=\alpha_t(i)a_{i,j}b_j(o_{t+1})\beta_{t+1}(j)
$$

因此有：

$$
 \xi_t(i,j)=\frac{\alpha_t(i)a_{i,j}b_j(o_{t+1})\beta_{t+1}(j)}{\sum_{u=1}^{Q}\sum_{v=1}^{Q}\alpha_t(u)a_{u,v}b_v(o_{t+1})\beta_{t+1}(v)}
$$

&emsp;&emsp;一些期望值：
1. 在给定观测 $\mathbf O$ 的条件下，状态 $i$ 出现的期望值为： $\sum  _  {t=1}^{T}\gamma  _  t(i)$ 。
2. 在给定观测 $\mathbf O$ 的条件下，从状态 $i$ 转移的期望值： $\sum  _  {t=1}^{T-1}\gamma  _  t(i)$ 。
    1. 这里的转移，表示状态 $i$ 可能转移到任何可能的状态。
    2. 假若在时刻 $T$ 的状态为 ${\mathbf q}  _  i$ ，则此时不可能再转移，因为时间最大为 $T$ 。
3. 在观测 $\mathbf O$ 的条件下，由状态 $i$ 转移到状态 $j$ 的期望值： $\sum  _  {t=1}^{T-1}\xi  _  t(i,j)$ 。
        

### 学习问题
根据训练数据的不同，隐马尔可夫模型的学习方法也不同：
1. 训练数据包括观测序列和对应的状态序列：通过监督学习来学习隐马尔可夫模型。
2. 训练数据仅包括观测序列：通过非监督学习来学习隐马尔可夫模型。

#### 监督学习
假设数据集为 $\mathbb D=\{(\mathbf O  _  1,\mathbf I  _  1),(\mathbf O  _  2,\mathbf I  _  2),\cdots,(\mathbf O  _  N,\mathbf I  _  N)\}$ 。其中：
1. $\mathbf O  _  1,\cdots,\mathbf O  _  N$ 为 $N$ 个观测序列； $\mathbf I  _  1,\cdots,\mathbf I  _  N$ 为对应的 $N$ 个状态序列。
2. 序列 $\mathbf O  _  k$ ， $\mathbf I  _  k$ 的长度为 $T  _  k$ ，其中数据集 $\mathbf O  _  1,\cdots,\mathbf O  _  N$ 之间的序列长度可以不同。

可以利用极大似然估计来估计隐马尔可夫模型的参数。
1. 转移概率 $a  _  {i,j}$ 的估计：设样本中前一时刻处于状态 $i$ 、且后一时刻处于状态 $j$ 的频数为 $A  _  {i,j}$ ，则状态转移概率 $a  _  {i,j}$ 的估计是：
    $$
    \hat a_{i,j}=\frac{A_{i,j}}{\sum_{u=1}^{Q}A_{i,u}} ,\quad i=1,2,\cdots,Q;j=1,2,\cdots,Q
    $$
2. 观测概率 $b  _  j(k)$ 的估计：设样本中状态为 $j$ 并且观测为 $k$ 的频数为 $B  _  {j,k}$ ，则状态为 $j$ 并且观测为 $k$ 的概率 $b  _  j(k)$ 的估计为：
    $$
     \hat b_j(k)=\frac{B_{j,k}}{\sum_{v=1}^{V}B_{j,v}},\quad j=1,2,\cdots,Q;k=1,2,\cdots,V
    $$
3. 初始状态概率的估计：设样本中初始时刻（即： $t=1$  ）处于状态 $i$ 的频数为 $C  _  i$ ，则初始状态概率 $\pi  _  i$ 的估计为：
    $$
    \hat \pi_i=\frac{C_i}{\sum_{j=1}^Q C_j},\quad i=1,2,\cdots,Q
    $$

#### 无监督学习
&emsp;&emsp;监督学习需要使用人工标注的训练数据。由于人工标注往往代价很高，所以经常会利用无监督学习的方法。隐马尔可夫模型的无监督学习通常使用 $Baum-Welch$ 算法求解。

&emsp;&emsp;在隐马尔可夫模型的无监督学习中，数据集为 $\mathbb D=\{\mathbf O  _  1,\mathbf O  _  2,\cdots,\mathbf O  _  N\}$ 。其中： $\mathbf O  _  1,\cdots,\mathbf O  _  N$ 为 $N$ 个观测序列；序列 $\mathbf O  _  k$ 的长度为 $T  _  k$ ，其中数据集中 $\mathbf O  _  1,\cdots,\mathbf O  _  N$ 之间的序列长度可以不同。

&emsp;&emsp;将观测序列数据看作观测变量 $\mathbf O$ ， 状态序列数据看作不可观测的隐变量 $\mathbf I$ ，则隐马尔可夫模型事实上是一个含有隐变量的概率模型：

$$
P(\mathbf O;\lambda)=\sum_{\mathbf I} P(\mathbf O\mid \mathbf I;\lambda)P(\mathbf I;\lambda)
$$

其参数学习可以由 $EM$ 算法实现。
    
&emsp;&emsp; $E$ 步：求 $Q$ 函数（其中 $\bar\lambda$ 是参数的当前估计值）

$$
Q(\lambda,\bar \lambda)=\sum_{j=1}^N\left(\sum_{\mathbf I} P(\mathbf I\mid \mathbf O=\mathbf O_j;\bar\lambda)\log P(\mathbf O=\mathbf O_j,\mathbf I;\lambda)\right)
$$

将 $P(\mathbf I\mid \mathbf O=\mathbf O  _  j;\bar\lambda)=\frac{P(\mathbf I,\mathbf O=\mathbf O  _  j;\bar\lambda)}{P(\mathbf O  _  j;\bar\lambda)}$ 代入上式，有：

$$
Q(\lambda,\bar \lambda)=\sum_{j=1}^N\frac{1}{P(\mathbf O_j;\bar\lambda)}\left(\sum_{\mathbf I} P(\mathbf I,\mathbf O=\mathbf O_j;\bar\lambda)\log P(\mathbf I,\mathbf O=\mathbf O_j;\lambda)\right)
$$

在给定参数 $\bar\lambda$ 时， $P(\mathbf O  _  j;\bar\lambda)$ 是已知的常数，记做 $\tilde P  _  j$ 。

在给定参数 $\bar\lambda$ 时， $P(\mathbf I,\mathbf O=\mathbf O  _  j;\bar\lambda)$ 是 $\mathbf I$ 的函数，记做 $\tilde P  _  j(\mathbf I)$ 。
        

根据 $P(\mathbf O,\mathbf I;\lambda)=\pi  _  {{i}  _  1}b  _  {{i}  _  1}({o}  _  1)a  _  {{i}  _  1,{i}  _  2}b  _  {{i}  _  2}({o}  _  2)\cdots a  _  {{i}  _  {T-1},{i}  _  T}b  _  {{i}  _  T}({o}  _  T)$ 得到：

$$
 Q(\lambda,\bar \lambda)=\sum_{j=1}^N\frac{1}{\tilde P_j}\left(\sum_{\mathbf I }(\log \pi_{{i}_ 1})\tilde P_j(\mathbf I)+\sum_{\mathbf I}\left(\sum_{t=1}^{T_j-1}\log a_{i_t,i_{t+1}}\right)\tilde P_j(\mathbf I)\ +\sum_{\mathbf I}\left(\sum_{t=1}^{T_j}\log b_{{i}_ t}({o}_ t^{(j)})\right)\tilde P_j(\mathbf I)\right)
$$

其中：$T  _  j$ 表示第 $j$ 个序列的长度， $o  _  t^{(j)}$ 表示第 $j$ 个观测序列的第 $t$ 个位置。
        
&emsp;&emsp; $M$ 步：求 $Q$ 函数的极大值：

$$
\bar\lambda^{ \< new \> }\leftarrow \arg\max_{\lambda} Q(\lambda,\bar\lambda)
$$

极大化参数在 $Q$ 函数中单独的出现在3个项中，所以只需要对各项分别极大化。

&emsp;&emsp; $\frac{\partial Q(\lambda,\bar\lambda)}{\partial \pi  _  i}=0$ ：

$$
\begin{aligned}
 \frac{\partial Q(\lambda,\bar\lambda)}{\partial \pi_i}=\frac{\partial (\sum_{j=1}^N\frac{1}{\tilde P_j}\sum_{\mathbf I }(\log \pi_{{i}_1})\tilde P_j(\mathbf I))}{\partial \pi_i}\\ 
 =\sum_{j=1}^N\frac{1}{\tilde P_j} \sum_{i_1=1}^Q P(i_1,\mathbf O=\mathbf O_j;\bar\lambda)\frac{\partial\log \pi_{i_1}}{\partial \pi_i}
 \end{aligned}
$$

将 $\pi  _  Q=1-\pi  _  1-\cdots-\pi  _  {Q-1}$ 代入，有：

$$
\frac{\partial Q(\lambda,\bar\lambda)}{\partial \pi_i}=\sum_{j=1}^N\frac{1}{\tilde P_j}\left(\frac{P(i_1=i,\mathbf O=\mathbf O_j;\bar\lambda)}{\pi_i}-\frac{P(i_1=Q,\mathbf O=\mathbf O_j;\bar\lambda)}{\pi_Q}\right)=0
$$

将 $\tilde P  _  j=P(\mathbf O=\mathbf O  _  j;\bar\lambda)$ 代入，即有：

$$
\pi_i \propto \sum_{j=1}^N P(i_1=i\mid \mathbf O=\mathbf O_j;\bar\lambda)
$$

考虑到 $\sum  _  {i=1}^Q\pi  _  i=1$ ，以及 $\sum  _  {i=1}^Q\sum  _  {j=1}^N P(i  _  1=i\mid \mathbf O=\mathbf O  _  j;\bar\lambda)=N$ ， 则有：

$$
 \pi_i=\frac{\sum_{j=1}^N P(i_1=i\mid \mathbf O=\mathbf O_j;\bar\lambda)}{N}
$$

其物理意义为：统计在给定参数 $\bar\lambda$ ，已知 $\mathbf O=\mathbf O  _  j$ 的条件下 $i  _  1=i$ 的出现的频率。它就是 $i  _  1=i$ 的后验概率的估计值。
            
&emsp;&emsp; $\frac{\partial Q(\lambda,\bar\lambda)}{\partial a  _  {i,j}}=0$ ：同样的处理有：

$$
\frac{\partial Q(\lambda,\bar\lambda)}{\partial a_{i,j}}=\sum_{k=1}^N\frac{1}{\tilde P_k}\sum_{t=1}^{T_k-1}\left(\frac{P(i_t=i,i_{t+1}=j,\mathbf O=\mathbf O_k;\bar\lambda)}{a_{i,j}}-\frac{P(i_t=i,i_{t+1}=Q,\mathbf O=\mathbf O_k;\bar\lambda)}{a_{i,Q}}\right)
$$

得到：

$$
a_{i,j}\propto \sum_{k=1}^N\sum_{t=1}^{T_k-1}P(i_t=i,i_{t+1}=j\mid \mathbf O=\mathbf O_k;\bar\lambda)
$$

考虑到 $\sum  _  {j=1}^Q a  _  {i,j}=1$ ，则有：

$$
\begin{aligned}
a_{i,j}&=\frac {\sum_{k=1}^N\sum_{t=1}^{T_k-1}P(i_t=i,i_{t+1}=j\mid \mathbf O=\mathbf O_k;\bar\lambda)}{\sum_{j^\prime=1}^Q \sum_{k=1}^N\sum_{t=1}^{T_k-1}P(i_t=i,i_{t+1}=j^\prime\mid \mathbf O=\mathbf O_k;\bar\lambda)}\\ 
&=\frac {\sum_{k=1}^N\sum_{t=1}^{T_k-1}P(i_t=i,i_{t+1}=j\mid \mathbf O=\mathbf O_k;\bar\lambda)}{ \sum_{k=1}^N\sum_{t=1}^{T_k-1}P(i_t=i\mid \mathbf O=\mathbf O_k;\bar\lambda)}
\end{aligned}
$$

其物理意义为：统计在给定参数 $\bar\lambda$ ，已知 $\mathbf O=\mathbf O  _  j$ 的条件下，统计当 $i  _  t=i$ 的情况下 $i  _  {t+1}=j$ 的出现的频率。它就是 $i  _  {t+1}=j\mid i  _  t=i$ 的后验概率的估计值。
            
&emsp;&emsp; $\frac{\partial Q(\lambda,\bar\lambda)}{\partial b  _  j(k)}=0$ ：同样的处理有：

$$
\frac{\partial Q\left( \lambda ,\overline{\lambda } \right)}{\partial b_j\left( k \right)}=\sum_{i=1}^N{\frac{1}{\widetilde{p}_i}}\sum_{t=1}^{T_i}{\left( \frac{P\left( i_t=j,o_t=k,O=O_i;\overline{\lambda } \right)}{b_j\left( k \right)}-\frac{P\left( i_t=j,o_t=k,O=O_i;\overline{\lambda } \right)}{b_j\left( V \right)} \right)}
$$

得到：

$$
b_j(k) \propto \sum_{i=1}^N\sum_{t=1}^{T_i}P(i_t=j,o_t=k\mid \mathbf O=\mathbf O_i;\bar\lambda)
$$

其中如果第 $i$ 个序列 $\mathbf O  _  i$ 的第 $t$ 个位置 $o  _  t^{(i)}\ne k$ ，则 $P(i  _  t=j,o  _  t=k\mid \mathbf O=\mathbf O  _  i;\bar\lambda)=0$ 。考虑到 $\sum  _  {k=1}^V b  _  j(k)=1$ ，则有：

$$
\begin{aligned}
b_j(k)&= \frac{ \sum_{i=1}^N\sum_{t=1}^{T_i}P(i_t=j,o_t=k\mid \mathbf O=\mathbf O_i;\bar\lambda)}{\sum_{k^\prime=1}^V \sum_{i=1}^N\sum_{t=1}^{T_i}P(i_t=j,o_t=k^\prime\mid \mathbf O=\mathbf O_i;\bar\lambda)}\\ 
&=\frac{\sum_{i=1}^N\sum_{t=1}^{T_i}P(i_t=j,o_t=k\mid \mathbf O=\mathbf O_i;\bar\lambda)}{\sum_{i=1}^N\sum_{t=1}^{T_i}P(i_t=j\mid \mathbf O=\mathbf O_i;\bar\lambda)}
\end{aligned}
$$

其物理意义为：统计在给定参数 $\bar\lambda$ ，已知 $\mathbf O=\mathbf O  _  j$ 的条件下，统计当 $i  _  t=j$ 的情况下 $o  _  {t}=k$ 的出现的频率。它就是 $o  _  {t}=k\mid i  _  t=j$ 的后验概率的估计值。
            
&emsp;&emsp;令 $\gamma  _  t^{(s)}(i)=P(i  _  t=i\mid \mathbf O=\mathbf O  _  s;\bar\lambda)$ ，其物理意义为：在序列 $\mathbf O  _  s$ 中，第 $ t $ 时刻的隐状态为 $i$ 的后验概率。

令 $\xi  _  t^{(s)}(i,j)=P(i  _  t=i,i  _  {t+1}=j\mid \mathbf O=\mathbf O  _  s;\bar\lambda)$ ，其物理意义为：在序列 $\mathbf O  _  s$ 中，第 $t$ 时刻的隐状态为 $i$ 、且第 $t+1$ 时刻的隐状态为 $j$ 的后验概率。则 $M$ 步的估计值改写为：

$$
\begin{aligned}
\pi_i&=\frac{\sum_{s=1}^N P(i_1=i\mid \mathbf O=\mathbf O_s;\bar\lambda)}{N}=\frac{\sum_{s=1}^N\gamma_1^{(s)}(i)}{N}\\ 
a_{i,j}&=\frac {\sum_{s=1}^N\sum_{t=1}^{T_s-1}P(i_t=i,i_{t+1}=j\mid \mathbf O=\mathbf O_s;\bar\lambda)}{\sum_{s=1}^N\sum_{t=1}^{T_s-1}P(i_t=i\mid \mathbf O=\mathbf O_s;\bar\lambda)}=\frac {\sum_{s=1}^N\sum_{t=1}^{T_s-1}\xi_t^{(s)}(i,j)}{\sum_{s=1}^N\sum_{t=1}^{T_s-1}\gamma_t^{(s)}(i)}\\ 
b_j(k)&=\frac{\sum_{s=1}^N\sum_{t=1}^{T_s}P(i_t=j,o_t=k\mid \mathbf O=\mathbf O_s;\bar\lambda)}{\sum_{s=1}^N\sum_{t=1}^{T_i}P(i_t=j\mid \mathbf O=\mathbf O_s;\bar\lambda)}=\frac{\sum_{s=1}^N\sum_{t=1}^{T_s}\gamma_t^{(s)}(j)\mathbb I(o_t^{(s)}=k)}{\sum_{s=1}^N\sum_{t=1}^{T_i}\gamma_t^{(s)}(j)}
\end{aligned}
$$

其中 $\mathbb I(o  _  t^{(s)}=k)$ 为示性函数，其意义为：当 $\mathbf O  _  s$ 的第 $t$ 时刻为 $k$ 时，取值为 $1$ ；否则取值为 $0$ 。
    
####  $Baum-Welch$ 算法伪码
输入：观测数据 $\mathbb D=\ {\mathbf O  _  1,\mathbf O  _  2,\cdots,\mathbf O  _  N \ }$ 

输出：隐马尔可夫模型参数

算法步骤：
1. 初始化： $n=0$ ，选取 $a  _  {i,j}^{<0>},b  _  j(k)^{<0>},\pi  _  i^{<0>}$ ，得到模型 $\lambda^{<0>}=(\mathbf A^{<0>},\mathbf B^{<0>},\pi^{<0>})$ 
2. 迭代，迭代停止条件为：模型参数收敛。迭代过程为：
    1. 求使得 $Q$ 函数取极大值的参数：
    2. 判断模型是否收敛。如果不收敛，则 $n\leftarrow n+1$ ，继续迭代。
                
3. 最终得到模型 $\lambda^{\ < n \ >}=(\mathbf A^{\ < n \ >},\mathbf B^{\ < n \ >},\pi^{\ < n \ >})$ 。
            

### 预测问题

#### 近似算法
&emsp;&emsp;近似算法思想：在每个时刻 $t$ 选择在该时刻最有可能出现的状态 $i  _  t^{ \ * }$ ，从而得到一个状态序列 $\mathbf I^{ \ * }=(i  _  1^{ \ * },i  _  2^{ \ * },\cdots,i  _  T^{ \ * })$ ，然后将它作为预测的结果。
    
&emsp;&emsp;近似算法：给定隐马尔可夫模型 $\lambda=(\mathbf A,\mathbf B,\pi)$ ，观测序列 $\mathbf O=({o}  _  1,{o}  _  2,\cdots,{o}  _  T)$ ，在时刻 $t$ 它处于状态 ${\mathbf q}  _  i$ 的概率为：

$$
\gamma_t(i)=\frac{\alpha_t(i)\beta_t(i)}{P(\mathbf O;\lambda)}=\frac{\alpha_t(i)\beta_t(i)}{\sum_{j=1}^{Q}\alpha_t(j)\beta_t(j)}
$$

在时刻 $t$ 最可能的状态： 

$$
i_t^{ \ * }=\arg\max_{1 \le i \le Q} \gamma_t(i)
$$

##### 优缺点
优点
1. 计算简单

缺点
1. 不能保证预测的状态序列整体是最有可能的状态序列，因为预测的状态序列可能有实际上不发生的部分
2. 近似算法是局部最优（每个点最优），但是不是整体最优的。
3. 近似算法无法处理这种情况：转移概率为 $0$ 。因为近似算法没有考虑到状态之间的迁移。

#### 维特比算法
&emsp;&emsp;维特比算法用动态规划来求解隐马尔可夫模型预测问题。它用动态规划求解概率最大路径（最优路径），这时一条路径对应着一个状态序列。
    
&emsp;&emsp;维特比算法根据动态规划原理，最优路径具有这样的特性：如果最优路径在时刻 $t$ 通过结点 $i  _  t^{ \ * }$ ， 则这一路径从结点 $i  _  t^{ \ * }$ 到终点 $i  _  T^{ \ * }$ 的部分路径，对于从 $i  _  t^{ \ * }$ 到 $i  _  T^{ \ * }$ 的所有可能路径来说，也必须是最优的。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/HMM/wtb1.png?raw=true"
    width="420" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">wtb1</div>
</center>
        
只需要从时刻 $t=1$ 开始，递推地计算从时刻 $1$ 到时刻 $t$ 且时刻 $t$ 状态为 $i,i=1,2,\cdots,N$ 的各条部分路径的最大概率（以及取最大概率的状态）。于是在时刻 $t=T$ 的最大概率即为最优路径的概率 $P^{ \ * }$ ，最优路径的终结点  $i  _  T^{ \ * }$ 也同时得到。之后为了找出最优路径的各个结点，从终结点 $i  _  T^{ \ * }$ 开始，由后向前逐步求得结点 $i  _  {T-1}^{ \ * },\cdots,i  _  1^{ \ * }$ ，得到最优路径 $\mathbf I^{ \ * }=(i  _  1^{ \ * },i  _  2^{ \ * },\cdots,i  _  T^{ \ * })$ 。
        
&emsp;&emsp;定义在时刻 $t$ 状态为 $i$ 的所有单个路径 $({i}  _  1,{i}  _  2,\cdots,{i}  _  t)$ 中概率最大值为：

$$
\delta_t(i)=\max_{i_1,i_2,\cdots,i_{t-1}} P(i_t=i,i_{t-1},\cdots,i_1,o_t,\cdots,o_1;\lambda),\quad i=1,2,\cdots,Q
$$

> 它就是算法导论中《动态规划》一章提到的“最优子结构”

则根据定义，得到变量 $\delta$ 的递推公式：

$$
\delta_{t+1}(i)=\max_{i_1,i_2,\cdots,i_{t}} P(i_{t+1}=i,i_{t},\cdots,i_1,o_{t+1},\cdots,o_1;\lambda)=\max_{1 \le j \le Q} \delta_t(j)\times a_{j,i}\times b_i(o_{t+1})\ i=1,2,\cdots,Q;t=1,2,\cdots,T-1
$$

&emsp;&emsp;定义在时刻 $t$ 状态为 $i$ 的所有单个路径中概率最大的路径的第 $t-1$ 个结点为：

$$
\Psi_t(i)=\arg\max_{1 \le j \le Q} \delta_{t-1}(j)a_{j,i} ,\quad i=1,2,\cdots,Q
$$

它就是最优路径中，最后一个结点（其实就是时刻 $t$ 的 ${\mathbf q}  _  i$ 结点）的前一个结点。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/HMM/dynamic_wtb.png?raw=true"
    width="720" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">dynamic wtb</div>
</center>
    
##### 伪码
输入：
1. 隐马尔可夫模型 $\lambda=(\mathbf A,\mathbf B,\pi)$ 
2. 观测序列 $\mathbf O=({o}  _  1,{o}  _  2,\cdots,{o}  _  T)$ 

输出：最优路径 $\mathbf I^{ \ * }=(i  _  1^{ \ * },i  _  2^{ \ * },\cdots,i  _  T^{ \ * })$ 

算法步骤：
1. 初始化：因为第一个结点的之前没有结点 ，所以有：
    $$
    \delta_1(i)=\pi_ib_i(o_1),\Psi_1(i)=0,\quad i=1,2,\cdots,Q
    $$
2. 递推：对 $t=2,3,\cdots,T$ 
    $$
    \begin{aligned}
    \delta_{t}(i)&=\max_{1 \le j \le Q} \delta_{t-1}(j)a_{j,i}b_i(o_{t}),\quad i=1,2,\cdots,Q;t=1,2,\cdots,T\\ 
    \Psi_t(i)&=\arg\max_{1 \le j \le Q} \delta_{t-1}(j)a_{j,i},\quad i=1,2,\cdots,Q
    \end{aligned}
    $$
3. 终止：$P^{ \ * }=\max  _  {1 \le i \le Q}\delta  _  T(i),\quad i^{ \ * }  _  T=\arg\max  _  {1 \le i \le Q} \delta  _  T(i)$ 。
            
4. 最优路径回溯：对 $t=T-1,T-2,\cdots,1 :i^{ \ * }\  _  t=\Psi\  _  {t+1}(i^{ \ * }  _  {t+1})$ 。

5. 最优路径 $\mathbf I^{ \ * }=(i  _  1^{ \ * },i  _  2^{ \ * },\cdots,i  _  T^{ \ * })$ 。
            

## 最大熵马尔科夫模型MEMM
$HMM$ 存在两个基本假设：
1. 观察值之间严格独立。
2. 状态转移过程中，当前状态仅依赖于前一个状态（一阶马尔科夫模型）。
    
如果放松第一个基本假设，则得到最大熵马尔科夫模型 $MEMM$ 。

&emsp;&emsp;最大熵马尔科夫模型并不通过联合概率建模，而是学习条件概率 $P(i  _  t\mid i  _  {t-1},o  _  t)$ 。它刻画的是：在当前观察值 $o  _  t$ 和前一个状态 $i  _  {t-1}$ 的条件下，当前状态 $i  _  t$ 的概率。
    
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/HMM/MEMM.png?raw=true"
    width="820" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">MEMM</div>
</center>
    
&emsp;&emsp; $MEMM$ 通过最大熵算法来学习。根据最大熵推导的结论：

$$
\begin{aligned}
P_w(y\mid x)&=\frac{1}{Z_w(x)} \exp\left(\sum_{i=1}^{n}w_i f_i(x,y)\right)\\
Z_w(x)&=\sum_y \exp\left(\sum_{i=1}^{n}w_i f_i(x,y)\right)
\end{aligned}
$$

这里 $x$ 就是当前观测 $o  _  t$ 和前一个状态 $i  _  {t-1}$ ，因此： $x=(i  _  {t-1},o  _  t)$ 。这里 $y$ 就是当前状态 $i  _  t$ ，因此： $y=i  _  t$ 。因此得到：

$$
 \begin{aligned}
 P_w(i_t\mid i_{t-1},o_t)&=\frac{1}{Z_w(i_{t-1},o_t)} \exp\left(\sum_{i=1}^{n}w_i f_i(i_t, i_{t-1},o_t)\right)\\ 
 Z_w(i_{t-1},o_t)&=\sum_{i_t} \exp\left(\sum_{i=1}^{n}w_i f_i(i_t, i_{t-1},o_t)\right)
 \end{aligned}
$$
    
> $MEMM$ 的参数学习使用最大熵中介绍的 $IIS$ 算法或者拟牛顿法，解码任务使用维特比算法。
    
&emsp;&emsp;标注偏置问题，如下图所示，通过维特比算法解码得到：

$$
 \begin{aligned}
P(1\rightarrow 1\rightarrow 1\rightarrow 1)&= 0.4\times 0.45\times 0.5 = 0.09 \\ 
P(2\rightarrow 2\rightarrow 2\rightarrow 2)&= 0.2\times 0.3 \times 0.3 = 0.018\\ 
P(1\rightarrow 2\rightarrow 1\rightarrow 2)&= 0.6\times 0.2\times 0.5 = 0.06\\ 
P(1\rightarrow 1\rightarrow 2\rightarrow 2)&= 0.4\times 0.55\times 0.3 = 0.066
 \end{aligned}
$$

可以看到：维特比算法得到的最优路径为 $1\rightarrow 1 \rightarrow 1 \rightarrow 1$ 。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/HMM/MEMM_bias.png?raw=true"
    width="520" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">MEMM bias</div>
</center>

    
实际上，状态 $1$ 倾向于转换到状态 $2$ ；同时状态 $2$ 也倾向于留在状态 $2$ 。但是由于状态 $2$ 可以转化出去的状态较多，从而使得转移概率均比较小。而维特比算法得到的最优路径全部停留在状态 $1$ ，这样与实际不符。
        
$MEMM$ 倾向于选择拥有更少转移的状态，这就是标记偏置问题。
        
&emsp;&emsp;标记偏置问题的原因是：计算 $P  _  w(i  _  t\mid i  _  {t-1},o  _  t)$ 仅考虑局部归一化，它仅仅考虑指定位置的所有特征函数。如上图中， $P  _  w(i  _  t\mid i  _  {t-1}=2,o  _  t)$ 只考虑在 $(i  _  {t-1}=2,o  _  t)$ 这个结点的归一化。
1. 对于 $(i  _  {t-1}=2,o  _  t)$ ，其转出状态较多，因此每个转出概率都较小。
2. 对于 $(i  _  {t-1}=1,o  _  t)$ ，其转出状态较少，因此每个转出概率都较大。

$CRF$ 解决了标记偏置问题，因为 $CRF$ 是全局归一化的：

$$
P(\mathbf Y\mid \mathbf X)=\frac 1Z \exp\left(\sum_{j=1}^{K_1}\sum_{i=1}^{n-1}\lambda_jt_j(Y_i,Y_{i+1},\mathbf X,i)+\sum_{k=1}^{K_2}\sum_{i=1}^{n}\mu_ks_k(Y_i,\mathbf X,i)\right)
$$

它考虑了所有位置、所有特征函数。