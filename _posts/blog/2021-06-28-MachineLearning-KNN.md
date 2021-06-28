---
layout: post
title: KNN
categories: [MachineLearning]
description: KNN
keywords: MachineLearning
---


KNN
---


## K 近邻算法
$k$ 近邻 ($k-Nearest \ Neighbor:kNN$) 是一种基本的分类与回归方法。
1. 分类问题：对新的样本，根据其 $k$ 个最近邻的训练样本的类别，通过多数表决等方式进行预测。
2. 回归问题：对新的样本，根据其 $k$ 个最近邻的训练样本标签值的均值作为预测值。

$k$ 近邻法不具有显式的学习过程，它是直接预测。它是“惰性学习” ($lazy \ learning$) 的著名代表。
1. 它实际上利用训练数据集对特征向量空间进行划分，并且作为其分类的"模型"。
2. 这类学习技术在训练阶段仅仅将样本保存起来，训练时间开销为零，等到收到测试样本后再进行处理。那些在训练阶段就对样本进行学习处理的方法称作“急切学习” ($eager learning$)。

### 优点
1. $k$ 近邻法是个非参数学习算法，它没有任何参数（$k$ 是超参数，而不是需要学习的参数）。
2. $k$ 近邻模型具有非常高的容量，这使得它在训练样本数量较大时能获得较高的精度。
3. 既可以用来做分类也可以用来做回归。

### 缺点
1. 计算成本很高。因为需要构建一个 $N\times N$ 的距离矩阵，其计算量为 $O\left( { N }^{ 2 } \right) $，其中 $N$ 为训练样本的数量。当数据集是几十亿个样本时，计算量是不可接受的。
2. 在训练集较小时，泛化能力很差，非常容易陷入过拟合。
3. 无法判断特征的重要性。
4. 存在样本不均衡问题。
5. $k$ 值不易选取，$k$ 值的选择会对$k$近邻的结果产生重大的影响。

### 三要素
1. $k$ 值选择。
2. 距离度量。
3. 决策规则。

#### k 值选择
&emsp;&emsp;当 $k=1$ 时 $k$ 近邻算法称为最近邻算法，此时将训练集中与 $x$ 最近的点的类别作为 $x$ 的分类。$k$ 值的选择会对 $k$ 近邻法的结果产生重大影响。
1. 若 $k$ 值较小，则相当于用较小的邻域中的训练样本进行预测，"学习"的偏差减小。此时，只有与输入样本较近的训练样本才会对预测起作用，预测结果会对近邻的样本点非常敏感。若近邻的训练样本点刚好是噪声，则预测会出错。即 $k$ 值减小意味着模型整体变复杂，易发生过拟合 。
   1. 优点：减少"学习"的偏差。
   2. 缺点：增大"学习"的方差（即波动较大）。
2. 若 $k$ 值较大，则相当于用较大的邻域中的训练样本进行预测。这时输入样本较远的训练样本也会对预测起作用，使预测偏离预期的结果。即 $k$ 值增大意味着模型整体变简单。
   1. 优点：减少"学习"的方差（即波动较小）。
   2. 缺点：增大"学习"的偏差。

应用中，$k$ 值一般取一个较小的数值。通常采用交叉验证法来选取最优的 $k$ 值。
    
#### 距离度量
&emsp;&emsp;特征空间中两个样本点的距离是两个样本点的相似程度的反映。$k$ 近邻模型的特征空间一般是$n$维实数向量空间 $\mathbb R^{n}$，其距离一般为欧氏距离，也可以是一般的 $L  _  p$ 距离：

$$
{ L }_ { p }\left( { x }_ { i },{ x }_ { j } \right) ={ \left( \sum _ { l=1 }^{ n }{ { \left| { x }_ { i,l }-{ x }_ { j,l } \right|  }^{ p } }  \right)  }^{ \frac { 1 }{ p }  },p\ge 1
$$

其中，${ x }  _  { i },{ x }  _  { j }\in \mathcal X={ \mathbb R }^{ n }$，${ x }  _  { i }={ \left( { x }  _  { i,1 },{ x }  _  { i,2 },\cdots ,{ x }  _  { i,n } \right)  }^{ T }$。

当 $p=2$ 时，为欧氏距离：

$$
{ L }_ { 2 }\left( { x }_ { i },{ x }_ { j } \right) ={ \left( \sum _ { l=1 }^{ n }{ { \left| { x }_ { i,l }-{ x }_ { j,l } \right|  }^{ 2 } }  \right)  }^{ \frac { 1 }{ 2 }  }
$$

当 $p=1$ 时，为曼哈顿距离：

$$
{ L }_ { 1 }\left( { x }_ { i },{ x }_ { j } \right) =\sum _ { l=1 }^{ n }{ { \left| { x }_ { i,l }-{ x }_ { j,l } \right|  } } 
$$

当 $p=\infty$ 时，为各维度距离中的最大值：

$$
{ L }_ { \infty  }\left( { x }_ { i },{ x }_ { j } \right) =\max _ { l }{ \left| { x }_ { i,l }-{ x }_ { j,l } \right|  } 
$$

不同的距离度量所确定的最近邻点是不同的。
    
#### ~~决策规则~~

##### 分类决策规则
&emsp;&emsp;分类决策通常采用多数表决，也可以基于距离的远近进行加权投票：距离越近的样本权重越大。

&emsp;&emsp;多数表决等价于经验风险最小化。设分类的损失函数为 $0-1$ 损失函数，分类函数为：

$$
f:\mathbb R^{n} \rightarrow \{c_1,c_2,\cdots,c_K\}
$$

&emsp;&emsp;给定样本 $x \in \mathcal X $，其最邻近的 $k$ 个训练点构成集合 $\mathcal N  _  k(x)$。设涵盖 $\mathcal N  _  k(x)$ 区域的类别为 $c  _  m$（这是待求的未知量，但它肯定是 $c  _  1,c  _  2,\cdots,c  _  K$ 之一），则损失函数为：

$$
L=\frac { 1 }{ k } \sum _{ { x }_{ i }\in \mathcal N_k(x) }{ I\left( { \tilde { y }  }_{ i }\neq { c }_{ m } \right)  } =1-\frac { 1 }{ k } \sum _{ { x }_{ i }\in \mathcal N_k(x) }{ I\left( { \tilde { y }  }_{ i }={ c }_{ m } \right)  } 
$$

$L$ 就是训练数据的经验风险。要使经验风险最小，则使得 $\sum   _  { { x }  _  { i }\in \mathcal N  _  k(x) }{ I\left( { \tilde { y }  }  _  { i }={ c }  _  { m } \right)  } $ 最大。即多数表决：

$$
{ c }_ { m }=arg\max _ { { c }_ { m } }{ \sum _ { { x }_ { i }\in \mathcal N_k(x) }{ I\left( { \tilde { y }  }_ { i }={ c }_ { m } \right)  }  } 
$$

##### 回归决策规则
&emsp;&emsp;回归决策通常采用均值回归，也可以基于距离的远近进行加权投票：距离越近的样本权重越大。

&emsp;&emsp;均值回归等价于经验风险最小化。设回归的损失函数为均方误差。给定样本 $x \in \mathcal X $，其最邻近的 $k$ 个训练点构成集合 $\mathcal N  _  k(x)$。设涵盖 $\mathcal N  _  k(x)$ 区域的输出为 $\hat y$，则损失函数为：

$$
L=\frac { 1 }{ k } \sum _ { { x }_ { i }\in \mathcal N_k(x) }{ { \left( { \tilde { y }  }_ { i }-\hat { y }  \right)  }^{ 2 } } 
$$

$L$ 就是训练数据的经验风险。要使经验风险最小，则有：

$$
\hat { y } =\frac { 1 }{ k } \sum _ { { x }_ { i }\in \mathcal N_k(x) }{ { \tilde { y }  }_ { i } } 
$$

即：均值回归。
    
### ~~k 近邻算法~~
#### 分类算法
1. 输入：
   1. 训练数据集 $\mathbb D=\{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\},x  _  i \in \mathcal X \subseteq \mathbb R^{n},\tilde y  _  i \in \mathcal Y=\{c  _  1,c  _  2,\cdots,c  _  K\}$
   2. 给定样本 $x$
2. 输出：样本 $x$ 所属的类别 $y$
3. 步骤：
   1. 根据给定的距离度量，在 $\mathbb D$ 中寻找与 $x$ 最近邻的 $k$ 个点。定义涵盖这 $k$ 个点的 $x$ 的邻域记作 $\mathcal N  _  k(x)$。
   2. 从 $\mathcal N  _  k(x)$ 中，根据分类决策规则（如多数表决）决定 $x$ 的类别 $y$：
      $$
        y=arg\max _ { { c }_ { m } }{ \sum _ { { x }_ { i }\in \mathcal N_k(x) }{ I\left( { \tilde { y }  }_ { i }={ c }_ { m } \right)  }  } 
      $$

#### 回归算法
1. 输入：
   1. 训练数据集$\mathbb D=\{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\},x  _  i \in \mathcal X \subseteq \mathbb R^{n},\tilde y  _  i \in \mathcal Y \subseteq \mathbb R$
   2. 给定样本 $x$
2. 输出：样本 $x$ 的输出 $y$
3. 步骤：
   1. 根据给定的距离度量，在 $\mathbb D$ 中寻找与 $x$ 最近邻的 $k$ 个点。定义涵盖这 $k$ 个点的 $x$ 的邻域记作 $\mathcal N  _  k(x)$。
   2. 从 $\mathcal N  _  k(x)$ 中，根据回归决策规则（如均值回归）决定 $x$ 的输出 $y$：
      $$
      y=\frac { 1 }{ k } \sum _ { { x }_ { i }\in \mathcal N_k(x) }{ { \tilde { y }  }_ { i } } 
      $$

## kd树
&emsp;&emsp;实现 $k$ 近邻法时，主要考虑的问题是：如何对训练数据进行快速 $k$ 近邻搜索。最简单的实现方法：线性扫描。此时要计算输入样本与每个训练样本的距离。当训练集很大时，计算非常耗时。解决办法是：使用 $kd$ 树 来提高 $k$ 近邻搜索的效率。

&emsp;&emsp;$kd$ 树是一种对$k$维空间中的样本点进行存储以便对其进行快速检索的树型数据结构。它是二叉树，表示对 $k$ 维空间的一个划分。构造 $kd$ 树的过程相当于不断的用垂直于坐标轴的超平面将 $k$ 维空间切分的过程。$kd$ 树的每个结点对应于一个 $k$ 维超矩形区域。

&emsp;&emsp;通常，依次选择坐标轴对空间切分，使用被选中的坐标轴上的实例点的中位数作切分点，这样得到的 $kd$ 树是平衡的。

### kd树构建算法
输入：$k$ 维空间样本集 $\mathbb D=\{x  _  1,x  _  2,\cdots,x  _  N\},x  _  i \in \mathcal X \subseteq \mathbb R^{k}$

输出：$kd$ 树

算法步骤：
1. 构造根结点。

    根结点对应于包含$\mathbb D$的$k$维超矩形。选择 $x_ 1$ 为轴，以 $\mathbb D$ 中所有样本的 $x  _  1$ 坐标的中位数 $ x  _  1^{ \* }$ 为切分点，将根结点的超矩形切分为两个子区域，切分产生深度为 $1$ 的左、右子结点。切分超平面为：$x  _  1=x  _  1^{\*}$。
    1. 左子结点对应于坐标 $x_ 1< x  _  1^{\*}$ 的子区域。
    2. 右子结点对应于坐标 $x_ 1> x  _  1^{\*}$ 的子区域。
    3. 落在切分超平面上的点 $x_  1=x_  1^{\*}$ 保存在根结点。
2. 对深度为 $j$ 的结点，选择 $x  _  l$ 为切分的坐标轴继续切分，$l=j\pmod k+1$。本次切分之后，树的深度为 $j+1$。这里取模而不是 $ l=j+1$，因为树的深度可以超过维度 $k$。此时切分轴又重复回到 $x_  l$，轮转坐标轴进行切分。
3. 直到所有结点的两个子域中没有样本存在时，切分停止。此时形成 $kd$ 树的区域划分。

### kd 树搜索算法
#### 伪码 
输入：
   1. 已构造的 $kd$ 树
   2. 测试点 $x$

输出：$x$ 的最近邻测试点

步骤：
   1. 初始化：当前最近点为 $x_  {nst}=null$，当前最近距离为 $\text{distance}_  {nst}=\infty$。
   2. 在 $ kd $ 树中找到包含测试点$x$的叶结点： 从根结点出发，递归向下访问 $ kd $ 树（即：执行二叉搜索），在访问过程中记录下访问的各结点的顺序，存放在先进后出队列 $ Queue $中，以便于后面的回退：
      1. 若测试点 $x$ 当前维度的坐标小于切分点的坐标，则查找当前结点的左子结点。
      2. 若测试点 $x$ 当前维度的坐标大于切分点的坐标，则查找当前结点的右子结点。
   3. 循环，结束条件为 $ Queue $ 为空。循环步骤为：
      1. 从 $ Queue$ 中弹出一个结点，设该结点为 $x  _  q$。计算 $x$ 到 $x  _  q$ 的距离，假设为 $\text{distance}  _  q$。若 $ \text{distance}  _  q \lt \text{distance}  _  {nst}$，则更新最近点与最近距离：
         $$
         \text{distance}_{nst}=\text{distance}_q,\quad x_{nst}= x_q
         $$
      2. 如果 $x  _  q$ 为中间节点：考察以 $x$ 为球心、以 $\text{distance}  _  {nst}$ 为半径的超球体是否与 $x  _  q$ 所在的超平面相交。如果相交：
         1. 若 $ Queue$ 中已经访问过了 $x  _  q$ 的左子树，则继续二叉搜索 $x  _  q$ 的右子树。
         2. 若 $ Queue$ 中已经访问过了 $x  _  q$ 的右子树，则继续二叉搜索 $x  _  q$ 的左子树。
         二叉搜索的过程中，仍然在 $ Queue$ 中记录搜索的各结点。
                
   4. 循环结束时，$x  _  {nst}$ 就是 $x$ 的最近邻点。

#### 时间复杂度           
$ kd $ 树搜索的平均计算复杂度为 $ O(\log N) $， $N$为训练集大小。$ kd $  树适合 $N >> k$ 的情形，当 $N$ 与维度 $k$ 接近时效率会迅速下降。

#### 搜索过程    
通常最近邻搜索只需要检测几个叶结点即可：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/KNN/kdshow1.png?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Depthwise Convolutional</div>
</center>

但是如果样本点的分布比较糟糕时，需要几乎遍历所有的结点：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/KNN/kdshow2.png?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Depthwise Convolutional</div>
</center>    

#### 示例
假设有 $6$ 个二维数据点：$\mathbb D=\{(2,3),(5,4),(9,6),(4,7),(8,1),(7,2)\}$。构建 $ kd $ 树的过程：
1. 首先从 $x$ 轴开始划分，根据 $x$ 轴的取值 $2,5,9,4,8,7$ 得到中位数为 $7$ ，因此切分线为：$x=7$。左子空间（记做 $\mathbb D  _  1$）包含点 $(2,3),(5,4),(4,7)$，切分轴轮转，从 $y$ 轴开始划分，切分线为：$y=4$。右子空间（记做 $\mathbb D  _  2$）包含点 $(9,6),(8,1)$，切分轴轮转，从 $y$ 轴开始划分，切分线为：$y=6$。
2. $\mathbb D  _  1$ 的左子空间（记做 $\mathbb D  _  3$）包含点 $(2,3)$，切分轴轮转，从 $x$ 轴开始划分，切分线为 $x=2$。其左子空间记做 $\mathbb D  _  7$，右子空间记做 $\mathbb D  _  8$。由于 $\mathbb D  _  7,\mathbb D  _  8 $ 都不包含任何点，因此对它们不再继续拆分。
3. $\mathbb D  _  1$ 的右子空间（记做 $\mathbb D  _  4$）包含点 $(4,7)$，切分轴轮转，从 $x$ 轴开始划分，切分线为：$x=4$。其左子空间记做 $\mathbb D  _  9$，右子空间记做 $\mathbb D  _  {10}$。由于 $\mathbb D  _  9,\mathbb D  _  {10}$ 都不包含任何点，因此对它们不再继续拆分。
4. $\mathbb D  _  2$ 的左子空间（记做 $\mathbb D  _  5$）包含点 $(8,1)$，切分轴轮转，从 $x$ 轴开始划分，切分线为：$x=8$。其左子空间记做 $\mathbb D  _  {11}$，右子空间记做 $\mathbb D  _  {12}$。由于 $\mathbb D  _  {11},\mathbb D  _  {12}$ 都不包含任何点，因此对它们不再继续拆分。
5. $\mathbb D  _  2$ 的右子空间（记做 $\mathbb D  _  6$）不包含任何点，停止继续拆分。

最终得到样本空间拆分图如下：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/KNN/kd_tree1.png?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Depthwise Convolutional</div>
</center>

样本空间结构图如下：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/KNN/kd_tree2.png?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Depthwise Convolutional</div>
</center>


$ kd $ 树如下：

&emsp;&emsp;$ kd $ 树以树的形式，根据样本空间的拆分，重新组织了数据集的样本点。每个结点都存放着位于划分平面上数据点。由于样本空间结构图中的叶区域不包含任何数据点，因此叶区域不会被划分。因此 $ kd $ 树的高度要比样本空间结构图的高度少一层。从 $ kd $ 树中可以清晰的看到坐标轮转拆分。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/KNN/kd_tree3.png?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Depthwise Convolutional</div>
</center>


假设需要查询的点是 $P=(2.1,3.1)$ 。
1. 首先从 $kd$ 树进行二叉查找，最终找到叶子节点 $(2,3)$，查找路径为：$Queue=<(7,2),(5,4),(2,3)>$ 。
2. $Queue$ 弹出结点$(2,3)$：$P$ 到 $(2,3)$的距离为 $0.1414$ ，该距离作为当前最近距离，$(2,3)$ 作为候选最近邻点。
3. $Queue$ 弹出结点$(5,4)$：$P$ 到 $(5,4)$的距离为 $3.03$ 。候选最近邻点仍然为 $(2,3)$，当前最近距离仍然为 $0.1414$ 。
4. 因为结点 $(5,4)$ 为中间结点，考察以 $P$ 为圆心，以 $0.1414$ 为半径的圆是否与 $y=4$ 相交。结果不相交，因此不用搜索 $(5,4)$ 的另一半子树。
5. $Queue$ 弹出结点 $(7,2)$：$P$ 到 $(7,2)$ 的距离为 $5.02$ 。候选最近邻点仍然为 $(2,3)$，当前最近距离仍然为 $0.1414$ 。
6. 因为结点 $(7,2)$ 为中间结点，考察以 $P$ 为圆心，以 $0.1414$ 为半径的圆是否与 $x=7$ 相交。结果不相交，因此不用搜索 $(7,2)$ 的另一半子树。
7. 现在 $Queue$ 为空，迭代结束。因此最近邻点为 $(2,3)$ ，最近距离为 $0.1414$。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/KNN/kd_tree4.png?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Depthwise Convolutional</div>
</center>

假设需要查询的点是 $P=(2,4.5)$ 。
1. 首先从 $ kd $ 树进行二叉查找，最终找到叶子节点 $(4,7)$，查找路径为：$Queue=<(7,2),(5,4),(4,7)>$ 。
2. $Queue$ 弹出结点 $(4,7)$ ：$P$ 到 $(4,7)$ 的距离为 $3.202$ ，该距离作为当前最近距离，$(4,7)$ 作为候选最近邻点。
3. $Queue$ 弹出结点 $(5,4)$ ：$P$ 到 $(5,4)$ 的距离为 $3.041$ ，该距离作为当前最近距离，$(5,4)$ 作为候选最近邻点。因为 $(5,4)$ 为中间结点，考察以 $P$ 为圆心，以 $3.041$ 为半径的圆是否与 $y=4$ 相交。结果相交，因此二叉搜索 $(5,4)$ 的另一半子树，得到新的查找路径为：$Queue=<(7,2),(2,3)>$ 。
4. $Queue$ 弹出结点 $(2,3)$ ：$P$ 到 $(2,3)$ 的距离为 $1.5$ ，该距离作为当前最近距离， $(2,3)$ 作为候选最近邻点。
5. $Queue$ 弹出结点 $(7,2)$ ：$P$ 到 $(7,2)$ 的距离为 $5.59$ 。候选最近邻点仍然为 $(2,3)$，当前最近距离仍然为 $1.5$。因为结点 $(7,2)$ 为中间结点，考察以 $P$ 为圆心，以 $1.5$ 为半径的圆是否与 $x=7$ 相交。结果不相交，因此不用搜索 $(7,2)$ 的另一半子树。
6. 现在 $Queue$ 为空，迭代结束。因此最近邻点为 $(2,3)$，最近距离为 $1.5$。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/KNN/kd_tree5.png?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Depthwise Convolutional</div>
</center>
