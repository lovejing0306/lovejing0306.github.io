---
layout: post
title: XGBoost
categories: [MachineLearning]
description: XGBoost
keywords: MachineLearning
---


XGBoost
---


&emsp;&emsp;$xgboost$ 也是使用与梯度提升相同的前向分步算法。其区别在于：$xgboost$ 通过结构风险极小化来确定下一个决策树的参数 $\Theta  _  m$：

$$
\hat\Theta_m=\arg\min_{\Theta_m}\sum_{i=1}^{N}L(\tilde y_i,f_m(x_i))+\Omega(h_m(x))
$$

其中 $\Omega(h  _  m)$ 为第 $m$ 个决策树的正则化项，这是 $xgboost$ 和 $GBT$ 的一个重要区别。

&emsp;&emsp;定义：

$$
\hat y_{i}^{<m-1>}=f_{m-1}(x_i),\quad g_i=\frac{\partial L(\tilde y_i,\hat y_{i}^{<m-1>})}{\partial \,\hat y_{i}^{<m-1>}},\quad h_i=\frac{\partial^2 L(\tilde y_i,\hat y_{i}^{<m-1>})}{\partial ^2\,\hat y_{i}^{<m-1>}}
$$

其中，$g  _  i$ 为 $L(\tilde y  _  i,\hat y  _  {i}^{\<m-1\>})$ 在 $\hat y  _  {i}^{\<m-1\>}$ 的一阶导数；$h  _  i$ 为 $L(\tilde y  _  i,\hat y  _  {i}^{\<m-1\>})$ 在 $\hat y  _  {i}^{\<m-1\>}$ 的二阶导数。

&emsp;&emsp;对目标函数 $L$ 执行二阶泰勒展开：

$$
\begin{aligned} L&=\sum _ { i=1 }^{ N }{ L\left( { \tilde { y }  }_ { i },{ f }_ { m }\left( { x }_ { i } \right)  \right)  } +\Omega \left( { h }_ { m }\left( x \right)  \right)  \\ &=\sum _ { i=1 }^{ N }{ L\left( { \tilde { y }  }_ { i },{ \hat { y }  }_ { i }^{ <m-1> }+{ h }_ { m }\left( { x }_ { i } \right)  \right)  } +\Omega \left( { h }_ { m }\left( x \right)  \right)  \\ &\simeq \sum _ { i=1 }^{ N }{ \left[ L\left( { \tilde { y }  }_ { i },{ \hat { y }  }_ { i }^{ <m-1> } \right) +{ g }_ { i }{ h }_ { m }\left( { x }_ { i } \right) +\frac { 1 }{ 2 } { h }_ { i }{ h }_ { m }^{ 2 }\left( { x }_ { i } \right)  \right] +\Omega \left( { h }_ { m }\left( x \right)  \right) +constant }  \end{aligned}
$$

提升树模型只采用一阶泰勒展开，这是 $xgboost$ 和 $GBT$ 的另一个重要区别。
    
&emsp;&emsp;对一个决策树 $h  _  m(x)$ 不考虑复杂的推导过程，仅考虑决策树的效果：给定输入 $\mathbf{x}$，该决策树将该输入经过不断的划分，最终划分到某个叶结点上去；给定一个叶结点，该叶结点有一个输出值。

&emsp;&emsp;因此将决策树拆分成结构部分 $q(\cdot)$ 和叶结点权重部分 $\mathbf{\vec w}=(w  _  1,w  _  2,\cdots,w  _  T)$，其中 $T$ 为叶结点的数量。结构部分 $q(x)$ 的输出是叶结点编号 $d$ 。它的作用是将输入 $x$ 映射到编号为$d$的叶结点；叶结点权重部分就是每个叶结点的值。它的作用是输出编号为 $d$ 的叶结点的值 $w  _  d$。因此决策树改写为：

$$
h_m(x)=w_{q(x)} 
$$

## 结构分
&emsp;&emsp;定义一个决策树的复杂度为：

$$
\Omega(h\_m(x))=\gamma T+\frac 12\lambda \sum\_{j=1}^T w_j^2 
$$

其中，$T$ 为叶结点的个数；$w  _  j$ 为每个叶结点的输出值；$\gamma,\lambda\ge 0$ 为系数，控制这两个部分的比重。叶结点越多，则决策树越复杂；每个叶结点输出值的绝对值越大，则决策树越复杂。

&emsp;&emsp;将树的拆分、树的复杂度代入 $L$ 的二阶泰勒展开，有：

$$
L\simeq \sum _{ i=1 }^{ N }{ \left[ { g }_{ i }{ w }_{ q\left( { x }_{ i } \right)  }+\frac { 1 }{ 2 } { h }_{ i }{ w }_{ q\left( { x }_{ i } \right)  }^{ 2 } \right]  } +\gamma T+\frac { 1 }{ 2 } \lambda \sum _{ j=1 }^{ T }{ { w }_{ j }^{ 2 } } +constant
$$

对于每个样本 $x  _  i$，它必然被划分到树 $h  _  m$ 的某个叶结点。定义划分到叶结点 $j$ 的样本的集合为：$\mathbb I  _  j=\{i \mid q(x  _  i)=j\}$。则有：

$$
L \simeq \sum_{j=1}^T\left[ \left( \sum_{i\in \mathbb I_j}g_i\right)w_j+\frac 12 \left(\sum_{i\in\mathbb I_j}h_i +\lambda \right)w_j^2\right]+\gamma T+\text{constant}
$$

定义：

$$
\mathbf G_j=\sum_{i\in \mathbb I_j}g_i,\; \mathbf H_j=\sum_{i\in \mathbb I_j}h_i 
$$

其中，$\mathbf G  _  j$ 刻画了隶属于叶结点 $j$ 的那些样本的一阶偏导数之和；$\mathbf H  _  j$ 刻画了隶属于叶结点 $j$ 的那些样本的二阶偏导数之和。
    
则上式化简为：

$$
L \simeq \sum_{j=1}^T\left[ \mathbf G_jw_j+\frac 12 \left(\mathbf H_j+\lambda \right)w_j^2\right]+\gamma T+\text{constant}
$$

假设 $w  _  j$ 与 $T,\mathbf G  _  j,\mathbf H  _  j$ 无关，对 $w  _  j$ 求导等于，则得到： 

$$
w_j^{\*}=-\frac{\mathbf G_j}{\mathbf H_j+\lambda} 
$$

忽略常数项，于是定义目标函数为：

$$
L^{\*}=-\frac12 \sum_{j=1}^T\frac{\mathbf G_j^2}{\mathbf H_j+\lambda}+\gamma T
$$

&emsp;&emsp;在推导过程中假设 $w  _  j$ 与 $T,\mathbf G  _  j,\mathbf H  _  j$ 无关，这其实假设树的结构是已知的。事实上 $\mathcal L^{\*}$ 是与 $T$ 相关的，甚至与树的结构相关，因此定义 $L^{\*}$ 为结构分。结构分刻画了：当已知树的结构时目标函数的最小值。
    

## 分解结点
&emsp;&emsp;如何得到最佳的树的结构，从而使得目标函数全局最小。

### 贪心算法
&emsp;&emsp;对现有的叶结点加入一个分裂，考虑分裂之后目标函数降低多少。如果目标函数下降，则说明可以分裂；如果目标函数不下降，则说明该叶结点不宜分裂。

&emsp;&emsp;对于一个叶结点，假如给定其分裂点，定义划分到左子结点的样本的集合为；$\mathbb I  _  {L}=\{i\mid q(x  _  i)=L\}$；定义划分到右子结点的样本的集合为：$\mathbb I  _  {R}=\{i\mid q(x  _  i)=R\}$ 。则有：

$$
\begin{aligned} { G }_ { L }&=\sum _ { i\in { I }_ { L } }{ { g }_ { i } }  \\ { G }_ { R }&=\sum _ { i\in { I }_ { R } }{ { g }_ { i } }  \\ { H }_ { L }&=\sum _ { i\in { I }_ { L } }{ { h }_ { i } }  \\ { H }_ { R }&=\sum _ { i\in { I }_ { R } }{ { h }_ { i } }  \\ G&=\sum _ { i\in { I }_ { L } }{ { g }_ { i } } +\sum _ { i\in { I }_ { R } }{ { g }_ { i } }  \\ &={ G }_ { L }+{ G }_ { R } \\ H&=\sum _ { i\in { I }_ { L } }{ { h }_ { i } } +\sum _ { i\in { I }_ { R } }{ { h }_ { i } }  \\ &={ H }_ { L }+{ H }_ { R } \end{aligned}
$$

&emsp;&emsp;定义叶结点的分裂增益为：

$$
Gain=\frac { 1 }{ 2 } \left[ \frac { { G }_{ L }^{ 2 } }{ { H }_{ L }+\lambda  } +\frac { { G }_{ R }^{ 2 } }{ { H }_{ R }+\lambda  } +\frac { { G }^{ 2 } }{ H+\lambda  }  \right] -\lambda 
$$

其中：$\frac { { G }  _  { L }^{ 2 } }{ { H }  _  { L }+\lambda  } $ 表示：该叶结点的左子树的结构分；$\frac { { G }  _  { L }^{ 2 } }{ { H }  _  { L }+\lambda  } $ 表示：该叶结点的右子树的结构分；$\frac { { G }^{ 2 } }{ H+\lambda  } $ 表示：如果不分裂，则该叶结点本身的结构分；$-\lambda $ 表示：因为分裂导致叶结点数量增大 $1$，从而导致增益的下降。

&emsp;&emsp;问题是：不知道分裂点。对于每个叶结点，存在很多个分裂点，且可能很多分裂点都能带来增益。解决的办法是：对于叶结点中的所有可能的分裂点进行一次扫描。然后计算每个分裂点的增益，选取增益最大的分裂点作为本叶结点的最优分裂点。

最优分裂点贪心算法：
1. 输入：
   1. 数据集 $\mathbb D = \{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\}$ ，其中样本 $ x  _  i = (x  _  {i,1},x  _  {i,2},\cdots,x  _  {i,n})^T$。
   2. 属于当前叶结点的样本集的下标集合 $\mathbb I$。
2. 输出：当前叶结点最佳分裂点。
3. 算法：
   1. 初始化：$score \leftarrow 0,\;\mathbf G\leftarrow \sum  _  {i\in \mathbb I} g  _  i,\;\mathbf H\leftarrow \sum  _  {i\in \mathbb I}h  _  i$。
   2. 遍历各维度：$k=1,\cdots,n$
      1. 初始化：$\mathbf G  _  L\leftarrow0,\;\mathbf H  _  L\leftarrow 0$
      2. 遍历各拆分点：沿着第 $k$ 维 ：
         1. 如果第 $k$ 维特征为连续值，则将当前叶结点中的样本从小到大排序。然后用 $j$ 顺序遍历排序后的样本下标：
         $$
         \begin{aligned} { G }_{ L }&\leftarrow { G }_{ L }+{ g }_{ j } \\ { G }_{ R }&\leftarrow G-{ G }_{ L } \\ { H }_{ L }&\leftarrow { H }_{ L }+{ h }_{ j } \\ { H }_{ R }&\leftarrow H-{ H }_{ L } \end{aligned}
         $$
         $$
         score\leftarrow \max { \left( score,\frac { { G }_{ L }^{ 2 } }{ { H }_{ L }+\lambda  } +\frac { { G }_{ R }^{ 2 } }{ { H }_{ R }+\lambda  } +\frac { { G }^{ 2 } }{ H+\lambda  }  \right)  } 
         $$
         2. 如果第 $k$ 维特征为离散值 $\{a  _  1,a  _  2,\cdots,a  _  {m  _  k}\}$，设当前叶结点中第 $k$ 维取值 $a  _  {j}$ 样本的下标集合为 $\mathbb I  _  {j }$，则遍历 $j=1,2,\cdots,m  _  k$：
         $$
         \begin{aligned} { G }_{ L }&\leftarrow \sum _{ i\in { I }_{ j } }{ { g }_{ i } }  \\ { G }_{ R }&\leftarrow G-{ G }_{ L } \\ { H }_{ L }&\leftarrow \sum _{ i\in { I }_{ j } }{ { h }_{ i } }  \\ { H }_{ R }&\leftarrow H-{ H }_{ L } \end{aligned}
         $$
         $$
         score\leftarrow \max { \left( score,\frac { { G }_{ L }^{ 2 } }{ { H }_{ L }+\lambda  } +\frac { { G }_{ R }^{ 2 } }{ { H }_{ R }+\lambda  } +\frac { { G }^{ 2 } }{ H+\lambda  }  \right)  } 
         $$
   3. 选取最大的 $score$ 对应的维度和拆分点作为最优拆分点。

&emsp;&emsp;分裂点贪心算法尝试所有特征和所有分裂位置，从而求得最优分裂点。当样本太大且特征为连续值时，这种暴力做法的计算量太大。

### 近似算法
&emsp;&emsp;近似算法寻找最优分裂点时不会枚举所有的特征值，而是对特征值进行聚合统计，然后形成若干个桶。然后仅仅将桶边界上的特征的值作为分裂点的候选，从而获取计算性能的提升。

假设数据集 $\mathbb D = \{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\}$，样本 $ x  _  i = (x  _  {i,1},x  _  {i,2},\cdots,x  _  {i,n})^T$。对第 $k$ 个特征进行分桶：
1. 如果第 $k$ 个特征为连续特征，则执行百分位分桶，得到分桶的区间为：$\mathbb S  _  k=\{s  _  {k,1},s  _  {k,2},\cdots,s  _  {k,l}\}$，其中 $s  _  {k,1}\lt s  _  {k,2}\lt\cdots\lt s  _  {k,l}$。分桶的数量、分桶的区间都是超参数，需要仔细挑选。
2. 如果第 $k$ 个特征为离散特征，则执行按离散值分桶，得到的分桶为：$\mathbb S  _  k=\{s  _  {k,1},s  _  {k,2},\cdots,s  _  {k,l}\}$，其中 $s  _  {k,1},s  _  {k,2},\cdots,s  _  {k,l}$ 为第 $k$ 个特征的所有可能的离散值。分桶的数量 $l$ 就是所有样本在第 $k$ 个特征上的取值的数量。
        
最优分裂点近似算法：
1. 输入：
   1. 数据集 $\mathbb D = \{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\}$ ，其中样本$ x  _  i = (x  _  {i,1},x  _  {i,2},\cdots,x  _  {i,n})^T$。
   2. 属于当前叶结点的样本集的下标集合 $\mathbb I$。
2. 输出：当前叶结点最佳分裂点。
3. 算法：
   1. 对每个特征进行分桶。 假设对第 $k$ 个特征上的值进行分桶为：$\mathbb S  _  k=\{s  _  {k,1},s  _  {k,2},\cdots,s  _  {k,l}\}$。如果第 $k$ 个特征为连续特征，则要求满足 $s  _  {k,1}\lt s  _  {k,2}\lt\cdots\lt s  _  {k,l}$。
   2. 初始化：$score \leftarrow 0,\;\mathbf G\leftarrow \sum  _  {i\in \mathbb I} g  _  i,\;\mathbf H\leftarrow \sum  _  {i\in \mathbb I}h  _  i$。
   3. 遍历各维度：$k=1,\cdots,n$
      1. 初始化：$\mathbf G  _  L\leftarrow0,\;\mathbf H  _  L\leftarrow 0$
      2. 遍历各拆分点，即遍历 $j =1,2,\cdots,l$：
         1. 如果是连续特征，则设叶结点的样本中，第 $k$ 个特征取值在区间 $(s  _  {k,j},s  _  {k,j+1}]$ 的样本的下标集合为 $\mathbb I  _  j$，则：
         $$
         \begin{aligned} { G }_ { L }&\leftarrow { G }_ { L }+\sum _ { i\in { I }_ {  j } }{ { g }_ { i } }  \\ { G }_ { R }&\leftarrow G-{ G }_ { L } \\ { H }_ { L }&\leftarrow { H }_ { L }+\sum _ { i\in { I }_ { j } }{ { h }_ { i } }  \\ { H }_ { R }&\leftarrow H-{ H }_ { L } \end{aligned}
         $$
         $$
         score\leftarrow \max { \left( score,\frac { { G }_{ L }^{ 2 } }{ { H }_{ L }+\lambda  } +\frac { { G }_{ R }^{ 2 } }{ { H }_{ R }+\lambda  } +\frac { { G }^{ 2 } }{ H+\lambda  }  \right)  } 
         $$
         2. 如果是离散特征，则设叶结点的样本中，第 $k$ 个特征取值等于 $s  _  {k,j}$ 的样本的下标集合为$ \mathbb I  _  j$，则：
         $$
         \begin{aligned} { G }_ { L }&\leftarrow \sum _ { i\in { I }_ { j } }{ { g }_ { i } }  \\ { G }_ { R }&\leftarrow G-{ G }_ { L } \\ { H }_ { L }&\leftarrow \sum _ { i\in { I }_ { j } }{ { h }_ { i } }  \\ { H }_ { R }&\leftarrow H-{ H }_ { L } \end{aligned}
         $$
         $$
         score\leftarrow \max { \left( score,\frac { { G }_{ L }^{ 2 } }{ { H }_{ L }+\lambda  } +\frac { { G }_{ R }^{ 2 } }{ { H }_{ R }+\lambda  } +\frac { { G }^{ 2 } }{ H+\lambda  }  \right)  } 
         $$
      3. 选取最大的 $score$ 对应的维度和拆分点作为最优拆分点。

分桶有两种模式：
1. 全局模式：在算法开始时，对每个维度分桶一次，后续的分裂都依赖于该分桶并不再更新。
   1. 优点是：只需要计算一次，不需要重复计算。
   2. 缺点是：在经过多次分裂之后，叶结点的样本有可能在很多全局桶中是空的。
2. 局部模式：除了在算法开始时进行分桶，每次拆分之后再重新分桶。
   1. 优点是：每次分桶都能保证各桶中的样本数量都是均匀的。
   2. 缺点是：计算量较大。

全局模式会构造更多的候选拆分点。而局部模式会更适合构建更深的树。

&emsp;&emsp;分桶时的桶区间间隔大小是个重要的参数。区间间隔越小，则桶越多，则划分的越精细，候选的拆分点就越多。
    

## 加权分桶
&emsp;&emsp;假设候选样本的第$k$维特征，及候选样本的损失函数的二阶偏导数为：

$$
\mathcal D_k=\{(x_{1,k},h_1),(x_{2,k},h_2),\cdots,(x_{N,k},h_N)\}
$$

定义排序函数：

$$
r_k(z)=\frac{\sum\_{\{i\mid (x_{i,k},h_i)\in \mathcal D_k,x_{i,k}\lt z\}} h_i}{\sum_{\{i\mid (x_{i,k},h_i)\in \mathcal D_k\}} h_i}
$$

它刻画的是：第 $k$ 维小于 $z$ 的样本的 $h$ 之和，占总的 $h$ 之和的比例。

&emsp;&emsp;$xgboost$ 的作者提出了一种带权重的桶划分算法。定义候选样本的下标集合为 $\mathbb I$，拆分点 $\mathbb S  _  k=\{s  _  {k,1},s  _  {k,2},\cdots,s  _  {k,l}\}$ 定义为：

$$
s_{k,1}=\min_{i\in \mathbb I} x_{i,k},\; s_{k,l}=\max_ {i\in \mathbb I}x_{i,k},\quad |r_k(s_{k,j})-r_k(s_{k,j+1})|\lt \epsilon
$$

其中 $x  _  {i,k}$ 表示样本 $ x  _  i$ 的第 $k$ 个特征。即：
1. 最小的拆分点是所有样本第 $k$ 维的最小值。
2. 最大的拆分点是所有样本第 $k$ 维的最大值。
3. 中间的拆分点：选取拆分点，使得相邻拆分点的排序函数值小于  $\epsilon$（分桶的桶宽）。
   1. 其意义为：第 $k$ 维大于等于$s  _  {k,j}$，小于 $s  _  {k,j+1}$ 的样本的 $h$ 之和，占总的 $h$ 之和的比例小于 $\epsilon$。
   2. 这种拆分点使得每个桶内的以 $h$ 为权重的样本数量比较均匀，而不是样本个数比较均匀。

&emsp;&emsp;上述拆分的一个理由是：根据损失函数的二阶泰勒展开有：

$$
\begin{aligned} L&\simeq \sum _ { i=1 }^{ N }{ \left[ L\left( { \tilde { y }  }_ { i },{ \hat { y }  }_ { i }^{ <m-1> } \right) +{ g }_ { i }{ h }_ { m }\left( { x }_ { i } \right) +\frac { 1 }{ 2 } { h }_ { i }{ h }_ { m }^{ 2 }\left( { x }_ { i } \right)  \right] +\Omega \left( { h }_ { m }\left( x \right)  \right) +constant }  \\ &=\sum _ { i=1 }^{ N }{ \frac { 1 }{ 2 } { h }_ { i }\left[ \frac { 2{ g }_ { i } }{ { h }_ { i } } { h }_ { m }\left( { x }_ { i } \right)  \right] +\Omega \left( { h }_ { m }\left( x \right)  \right) +constant }  \\ &=\sum _ { i=1 }^{ N }{ \frac { 1 }{ 2 } { h }_ { i }{ \left( { h }_ { m }\left( { x }_ { i } \right) -\frac { { g }_ { i } }{ { h }_ { i } }  \right)  }^{ 2 }+{ \Omega  }^{ \prime  }\left( { h }_ { m }\left( x \right)  \right) +constant }  \end{aligned}
$$

对于第 $m$ 个决策树，它等价于样本 $x  _  i$ 的真实标记为 $\frac{g  _  i}{h  _  i}$、权重为 $h  _  i$、损失函数为平方损失函数。因此分桶时每个桶的权重为 $h$。
    

## 缺失值
&emsp;&emsp;真实场景中，有很多可能导致产生稀疏。如：数据缺失、某个特征上出现很多 $0$ 项、人工进行 $one-hot$ 编码导致的大量的 $0$。理论上，数据缺失和数值 $0$ 的含义是不同的，数值 $0$ 是有效的。实际上，数值$0$的处理方式类似缺失值的处理方式，都视为稀疏特征。在 $xgboost$ 中，数$0$的处理方式和缺失值的处理方式是统一的。这只是一个计算上的优化，用于加速对稀疏特征的处理速度。对于稀疏特征，只需要对有效值进行处理，无效值则采用默认的分裂方向。
> 注意：每个结点的默认分裂方向可能不同。
        
&emsp;&emsp;在 $xgboost$ 算法的实现中，允许对数值 $0$ 进行不同的处理。可以将数值 $0$ 视作缺失值，也可以将其视作有效值。如果数值 $0$ 是有真实意义的，则建议将其视作有效值。

缺失值处理算法：
1. 输入：
   1. 数据集 $\mathbb D = \{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\}$ ，其中样本 $x  _  i = (x  _  {i,1},x  _  {i,2},\cdots,x  _  {i,n})^T$。
   2. 属于当前叶结点的样本的下标集合 $\mathbb I$。
   3. 属于当前叶结点，且第 $k$ 维特征有效的样本的下标集合 $\mathbb I  _  k=\{i\in\mathbb I\mid x  _  {k,i}\ne missing\}$。
2. 输出：当前叶结点最佳分裂点。
3. 算法：
   1. 初始化：$score \leftarrow 0,\;\mathbf G\leftarrow \sum  _  {i\in \mathbb I} g  _  i,\;\mathbf H\leftarrow \sum  _  {i\in \mathbb I}h  _  i$。
   2. 遍历各维度：$k=1,\cdots,n$  
      1. 先从左边开始遍历：
         1. 初始化：$\mathbf G  _  L\leftarrow0,\;\mathbf H  _  L\leftarrow 0$
         2. 遍历各拆分点：沿着第 $k$ 维，将当前有效的叶结点的样本从小到大排序。这相当于所有无效特征值的样本放在最右侧，因此可以保证无效的特征值都在右子树。然后用 $j$ 顺序遍历排序后的样本下标：
         $$
         \begin{aligned} { G }_{ L }&\leftarrow { G }_{ L }+{ g }_{ j } \\ { G }_{ R }&\leftarrow G-{ G }_{ L } \\ { H }_{ L }&\leftarrow { H }_{ L }+{ h }_{ j } \\ { H }_{ R }&\leftarrow H-{ H }_{ L } \end{aligned}
         $$
         $$
         score\leftarrow \max { \left( score,\frac { { G }_{ L }^{ 2 } }{ { H }_{ L }+\lambda  } +\frac { { G }_{ R }^{ 2 } }{ { H }_{ R }+\lambda  } +\frac { { G }^{ 2 } }{ H+\lambda  }  \right)  } 
         $$
      2. 再从右边开始遍历：
         1. 初始化：$\mathbf G  _  R\leftarrow0,\;\mathbf H  _  R\leftarrow 0$
         2. 遍历各拆分点：沿着$k$维，将当前叶结点的样本从大到小排序。这相当于所有无效特征值的样本放在最左侧，因此可以保证无效的特征值都在左子树。然后用 $j$ 逆序遍历排序后的样本下标：
         $$
         \begin{aligned} { G }_{ R }&\leftarrow { G }_{ R }+{ g }_{ j } \\ { G }_{ L }&\leftarrow G-{ G }_{ R } \\ { H }_{ R }&\leftarrow { H }_{ R }+{ h }_{ j } \\ { H }_{ L }&\leftarrow H-{ H }_{ R } \end{aligned}
         $$
         $$
         score\leftarrow \max { \left( score,\frac { { G }_{ L }^{ 2 } }{ { H }_{ L }+\lambda  } +\frac { { G }_{ R }^{ 2 } }{ { H }_{ R }+\lambda  } +\frac { { G }^{ 2 } }{ H+\lambda  }  \right)  } 
         $$
   3. 选取最大的 $score$ 对应的维度和拆分点作为最优拆分点。

&emsp;&emsp;缺失值处理算法中，通过两轮遍历可以确保稀疏值位于左子树和右子树的情形。
    

## 其他优化

### 正则化
$xgboost$ 在学习过程中使用了如下的正则化策略来缓解过拟合：
1. 通过学习率 $\nu$ 来更新模型：
$$
f_m(x)=f_{m-1}(x)+\nu h_m(x;\Theta_m),\;0\lt \nu \le1
$$ 
2. 类似于随机森林，采取随机属性选择。

### 计算速度提升
$xgboost$ 在以下方面提出改进来提升计算速度：
1. 预排序 $pre-sorted$。
2. $cache-aware$ 预取。
3. $Out-of-Core$ 大数据集。

#### 预排序
$xgboost$ 提出 $column block$ 数据结构来降低排序时间。
1. 每一个 $block$ 代表一个属性，样本在该 $block$ 中按照它在该属性的值排好序。
2. 这些 $block$ 只需要在程序开始的时候计算一次，后续排序只需要线性扫描这些 $block$ 即可。
3. 由于属性之间是独立的，因此在每个维度寻找划分点可以并行计算。

&emsp;&emsp;$block$ 可以仅存放样本的索引，而不是样本本身，这样节省了大量的存储空间。如：$block  _  1$ 代表所有样本在 $feature  _  1$ 上的从小到大排序：$sample  _  no1,sample  _  no2,....$。其中样本编号出现的位置代表了该样本的排序。
    

#### 预取
&emsp;&emsp;由于在 $column block$ 中，样本的顺序会被打乱，这会使得从导数数组中获取 $g  _  i$ 时的缓存命中率较低。因此 $xgboost$ 提出了 $cache-aware$ 预取算法，用于提升缓存命中率。

&emsp;&emsp;$xgboost$ 会以 $minibatch$ 的方式累加数据，然后在后台开启一个线程来加载需要用到的导数 $g  _  i$。这里有个折中：$minibatch$ 太大，则会引起 $cache miss$；太小，则并行程度较低。


#### Out-of-Core
$xgboost$ 利用硬盘来处理超过内存容量的大数据集。其中使用了下列技术：
1. 使用 $block$ 压缩技术来缓解内存和硬盘的数据交换 $IO$ ： 数据按列压缩，并且在硬盘到内存的传输过程中被自动解压缩。
2. 数据随机分片到多个硬盘，每个硬盘对应一个预取线程，从而加大"内存-硬盘"交换数据的吞吐量。