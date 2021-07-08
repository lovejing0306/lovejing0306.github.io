---
layout: post
title: SVM
categories: [MachineLearning]
description: SVM
keywords: MachineLearning
---


SVM
---


&emsp;&emsp;支持向量机 ($Support \ Vector\ Machines:SVM$) 是一种二分类模型。它是定义在特征空间上的几何间隔最大的线性分类器。
1. 间隔最大使得支持向量机有别于感知机（如果数据集是线性可分的，那么感知机获得的模型可能有很多个，而支持向量机选择的是间隔最大的那一个）。
2. 支持向量机还支持核技巧，从而使它成为实质上的非线性分类器。

## 问题分类
1. 当训练数据线性可分时，通过硬间隔最大化，学习一个线性分类器，即线性可分支持向量机（也称作硬间隔支持向量机）。
2. 当训练数据近似线性可分时，通过软间隔最大化，学习一个线性分类器，即线性支持向量机（也称为软间隔支持向量机）。
3. 当训练数据不可分时，通过使用核技巧以及软间隔最大化，学习一个非线性分类器，即非线性支持向量机。

&emsp;&emsp;当输入空间为欧氏空间或离散集合，特征空间为希尔伯特空间时，将输入向量从输入空间映射到特征空间，得到特征向量。支持向量机的学习是在特征空间进行的。
1. 线性可分支持向量机、线性支持向量机假设这两个空间的元素一一对应，并将输入空间中的输入向量映射为特征空间中的特征向量。
2. 非线性支持向量机利用一个从输入空间到特征空间的非线性转换将输入向量映射为特征向量。
   1. 特征向量之间的内积就是核函数，使用核函数可以学习非线性支持向量机。
   2. 非线性支持向量机等价于隐式的在高维的特征空间中学习线性支持向量机，这种方法称作核技巧。

&emsp;&emsp;欧氏空间是有限维度的，希尔伯特空间为无穷维度的。
1. 欧式空间 $\subseteq$ 希尔伯特空间 $\subseteq$ 内积空间 $\subseteq$ 赋范空间。
   1. 欧式空间，具有很多美好的性质。
   2. 若不局限于有限维度，就来到了希尔伯特空间（从有限到无限是一个质变，很多美好的性质消失了，一些非常有悖常识的现象会出现）。
   3. 如果再进一步去掉完备性，就来到了内积空间。
   4. 如果再进一步去掉"角度"的概念，就来到了赋范空间。此时还有“长度”和“距离”的概念。
2. 越抽象的空间具有的性质越少，在这样的空间中能得到的结论就越少
3. 如果发现了赋范空间中的某些性质，那么前面那些空间也都具有这个性质。

## 支持向量机的优点
1. 有严格的数学理论支持，可解释性强；
2. 能找出对任务至关重要的样本（即：支持向量）；
3. 采用核技巧之后，可以处理非线性问题；
4. 泛化能力较强（其目标函数是结构风险极小化）；
5. $SVM$ 的凸优化特性保证了具有全局最优解；
6. 适用于规模较小的数据集；

## 支持向量机的缺点
1. 训练时间长。当采用 $SMO$ 算法时，由于每次都需要挑选一对参数，因此时间复杂度为 $O(N^2)$，其中 $N$ 为 $\mathbf{\alpha}$ 的长度，也就是训练样本的数量；
2. 当采用核技巧时，如果需要存储核矩阵，则空间复杂度为 $O(N^2)$；
3. 模型预测时，预测时间与支持向量的个数成正比。当支持向量的数量较大时，预测计算复杂度较高；
4. 当训练数据中存在噪声时，模型的性能不是很好；
5. 对非线性问题没有通用的解决方案，有时很难找到一个合适的核函数；
6. 当训练数据集很多时，效率不是很高；

因此支持向量机目前只适合小批量样本的任务，无法适应百万甚至上亿样本的任务。

## SVM 中核函数的选择技巧
1. 如果样本数量小于特征数，使用线性核；
2. 如果样本数量大于特征数，可使用非线性核将样本映射到更高维度；
3. 如果样本数量等于特征数，可使用非线性核将样本映射到高维空间；
4. 对于 (1) 中的情况，可以先对数据进行降维，然后使用非线性核。
    

## 线性可分支持向量机
&emsp;&emsp;给定一个特征空间上的训练数据集 $\mathbb D=\{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\}$，其中 $ x  _  i \in \mathcal X = \mathbb R^{n},\tilde y  _  i \in \mathcal Y=\{+1,-1\},i=1,2,\cdots,N$。
$x  _  i$ 为第 $i$ 个特征向量，也称作实例；$\tilde y  _  i$ 为 $x  _  i$ 的类标记；$(x  _  i,\tilde y  _  i)$ 称作样本点。
1. 当 $\tilde y  _  i=+1$ 时，称 $x  _  i$ 为正例。
2. 当 $\tilde y  _  i=-1$ 时，称 $x  _  i$ 为负例。

&emsp;&emsp;假设训练数据集是线性可分的，则学习的目标是在特征空间中找到一个分离超平面，能将实例分到不同的类。分离超平面对应于方程 $ w \cdot x+b=0$， 它由法向量 $w$ 和截距 $b$ 决定，可以用 $(w,b)$ 来表示。

&emsp;&emsp;给定线性可分训练数据集，通过间隔最大化学习得到的分离超平面为：

$$
w^{ \ * }\cdot x+b^{ \ * }=0 
$$

相应的分类决策函数：

$$
f(x)=\text{sign}(w^{ \ * }\cdot x+b^{ \ * })
$$

称之为线性可分支持向量机。

&emsp;&emsp;当训练数据集线性可分时，存在无穷个分离超平面可以将两类数据正确分开。
1. 感知机利用误分类最小的策略，求出分离超平面。但是此时的解有无穷多个。
2. 线性可分支持向量机利用间隔最大化求得最优分离超平面，这样的解只有唯一的一个。

### 函数间隔
可以使用一个点距离分离超平面的远近来表示分类预测的确信度：
1. 一个点距离分离超平面越远，则该点的分类越可靠；
2. 一个点距离分离超平面越近，则该点的分类则不那么确信。

在超平面 $w \cdot x+b=0$ 确定的情况下:
1. $ \|w \cdot x  _  i+b \|$ 能够相对地表示点 $x  _  i$ 距离超平面的远近。
2. $w \cdot x  _  i+b$ 的符号与类标记 $\tilde y  _  i$ 的符号是否一致能表示分类是否正确
   1. $w \cdot x  _  i+b \gt 0 $ 时，即 $x  _  i$ 位于超平面上方，将 $x  _  i$ 预测为正类。此时若 $\tilde y  _  i=+1$ 则分类正确；否则分类错误。
   2. $w \cdot x  _  i+b \lt 0 $ 时，即 $x  _  i$ 位于超平面下方，将 $x  _  i$ 预测为负类。此时若 $\tilde y  _  i=-1$ 则分类正确；否则分类错误。
            
可以用 $y(w \cdot x+b)$ 来表示分类的正确性以及确信度，就是函数间隔的概念。
1. 符号决定了正确性。
2. 范数决定了确信度。

对于给定的训练数据集 $\mathbb D$ 和超平面 $(w,b)$，定义超平面 $(w,b)$ 关于样本点 $(x  _  i,\tilde y  _  i)$ 的函数间隔为： 

$$
\hat\gamma_i=\tilde y_i(w \cdot x_i+b)
$$

定义超平面 $(w,b)$ 关于训练集 $\mathbb D$ 的函数间隔为超平面 $(w,b)$ 关于 $\mathbb D$ 中所有样本点 $(x  _  i,\tilde y  _  i)$ 的函数间隔之最小值：

$$
\hat\gamma=\min_ {\mathbb D} \hat\gamma_i
$$

### 几何间隔
&emsp;&emsp;如果成比例的改变 $w$ 和 $b$，比如将它们改变为 $100w$ 和 $100b$，超平面 $100w \cdot x+100b=0$ 还是原来的超平面，但是函数间隔却成为原来的 $100$ 倍。这时就需要几何间隔。

&emsp;&emsp;对于给定的训练数据集 $\mathbb D$ 和超平面 $(w,b)$，定义超平面 $(w,b)$ 关于样本点 $(x  _  i,\tilde y  _  i)$ 的几何间隔为：

$$
 \gamma_i=\tilde y_ i(\frac{w}{\|\|w \|\|_ 2}\cdot x_i+\frac{b}{\|\|w\|\|_ 2})
$$

定义超平面 $(w,b)$ 关于训练集 $\mathbb D$ 的几何间隔为超平面 $(w,b)$ 关于 $\mathbb D$ 中所有样本点 $(x  _  i,\tilde y  _  i)$ 的几何间隔之最小值：

$$
\gamma=\min_ {\mathbb D} \gamma_i 
$$

&emsp;&emsp;由定义可知函数间隔和几何间隔有下列的关系：

$$
\quad \gamma=\frac{\hat\gamma}{\|\|w\|\|_ 2}
$$

如果 $\|\|w\|\|  _  2=1$，那么函数间隔和几何间隔相等。如果超平面参数 $w$ 和 $b$ 成比例的改变（超平面未改变），函数间隔也按此比例改变，而几何间隔不变。  

### 硬间隔最大化
&emsp;&emsp;支持向量机学习基本思想：线性可分支持向量机利用几何间隔最大化求最优分离超平面。几何间隔最大化又称作硬间隔最大化。

&emsp;&emsp;几何间隔最大化的物理意义：不仅将正负实例点分开，而且对于最难分辨的实例点（距离超平面最近的那些点），也有足够大的确信度来将它们分开。

&emsp;&emsp;求解几何间隔最大的分离超平面可以表示为约束的最优化问题：

$$
\max_ {w,b} \gamma\ s.t. \quad \tilde y_i(\frac{w}{\|\|w\|\|_ 2}\cdot x_i+\frac{b}{\|\|w\|\|_ 2}) \ge \gamma, i=1,2,\cdots,N
$$

考虑几何间隔和函数间隔的关系，改写问题为：

$$
\max_ {w,b} \frac{\hat\gamma}{\|\|w\|\|_ 2}\ s.t. \quad \tilde y_i(w \cdot x_i+b) \ge \hat\gamma, i=1,2,\cdots,N
$$

&emsp;&emsp;函数间隔 $\hat \gamma$ 的大小并不影响最优化问题的解。假设将 $w,b$ 按比例的改变为 $\lambda w,\lambda b$，此时函数间隔变成 $ \lambda \hat\gamma$ （这是由于函数间隔的定义）：
1. 这一变化对求解最优化问题的不等式约束没有任何影响。
2. 这一变化对最优化目标函数也没有影响。

因此取 $\hat\gamma =1$，则最优化问题改写为：

$$
 \max_ {w,b} \frac{1}{\|\|w\|\|_ 2}\ s.t. \quad \tilde y_i(w \cdot x_i+b) \ge 1, i=1,2,\cdots,N
$$

注意到 $\max \frac{1}{\|\|w\|\|  _  2}$ 和 $\min \frac 12 \|\|w\|\|  _  2^{2}$ 是等价的，于是最优化问题改写为：

$$
 \min_ {w,b} \frac 12 \|\|w\|\|_ 2^{2}\ s.t. \quad \tilde y_i(w \cdot x_i+b) -1 \ge 0, i=1,2,\cdots,N
$$

这是一个凸二次规划问题。

&emsp;&emsp;凸优化问题 ，指约束最优化问题：
$$
 \min_ {w}f(w)\ s.t. \quad g_j(w) \le0,j=1,2,\cdots,J\ h_k(w)=0,k=1,2,\cdots,K
$$

其中：
1. 目标函数 $f(w)$ 和约束函数 $g  _  j(w)$ 都是 $\mathbb R^{n}$ 上的连续可微的凸函数。
2. 约束函数 $h  _  k(w)$ 是 $\mathbb R^{n}$ 上的仿射函数。$h(x)$ 称为仿射函数，如果它满足 $ h(x)=a \cdot x+b,\quad a\in \mathbb R^{n},b \in \mathbb R, x \in \mathbb R^{n}$
当目标函数 $f(w)$ 是二次函数且约束函数 $g  _  j(w)$ 是仿射函数时，上述凸最优化问题成为凸二次规划问题。

线性可分支持向量机原始算法：
1. 输入：线性可分训练数据集 $\mathbb D=\{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\}$，其中 $x  _  i \in \mathcal X = \mathbb R^{n},\tilde y  _  i \in \mathcal Y=\{+1,-1\},i=1,2,\cdots,N$
2. 输出：
   1. 最大几何间隔的分离超平面
   2. 分类决策函数
3. 算法步骤：
   1. 构造并且求解约束最优化问题：
      $$
      \min_ {w,b} \frac 12 \|\|w\|\|_ 2^{2}\ s.t. \quad \tilde y_i(w \cdot x_i+b) -1 \ge 0, i=1,2,\cdots,N
      $$
      求得最优解 $w^{ \ * },b^{ \ * }$
   2. 由此得到分离超平面：$w^{ \ * }\cdot x+b^{ \ * }=0$，以及分类决策函数：$f(x)=\text{sign}(w^{ \ * }\cdot x+b^{ \ * })$。

可以证明：若训练数据集 $\mathbb D$ 线性可分，则可将训练数据集中的样本点完全正确分开的最大间隔分离超平面存在且唯一。
    

### 支持向量
&emsp;&emsp;在训练数据集线性可分的情况下，训练数据集中与分离超平面距离最近的样本点的实例称为支持向量。支持向量是使得约束条件等号成立的点，即 $\tilde y  _  i(w \cdot x  _  i+b) -1=0$：
1. 对于正实例点，支持向量位于超平面 $H  _  1:w \cdot x+b=1$
2. 对于负实例点，支持向量位于超平面 $H  _  2:w \cdot x+b=-1$

超平面 $H  _  1$、$H  _  2$ 称为间隔边界， 它们和分离超平面 $w \cdot x+b=0$ 平行，且没有任何实例点落在 $H  _  1$、$H  _  2$ 之间。
在 $H  _  1$、$H  _  2$ 之间形成一条长带，分离超平面位于长带的中央。长带的宽度称为 $H  _  1$、$H  _  2$ 之间的距离，也即间隔，间隔大小为 $\frac{2}{\|\|w\|\|}  _  2$。


<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/SVM/linear_svm.png?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Linear SVM </div>
</center>

&emsp;&emsp;在决定分离超平面时，只有支持向量起作用，其他的实例点并不起作用。如果移动支持向量，将改变所求的解；如果在间隔边界以外移动其他实例点，甚至去掉这些点，则解是不变的。

&emsp;&emsp;由于支持向量在确定分离超平面中起着决定性作用，所以将这种分离模型称为支持向量机。支持向量的个数一般很少，所以支持向量机由很少的、重要的训练样本确定。

### 对偶算法
&emsp;&emsp;将线性可分支持向量机的最优化问题作为原始最优化问题，应用拉格朗日对偶性，通过求解对偶问题得到原始问题的最优解。这就是线性可分支持向量机的对偶算法。

对偶算法的优点：
1. 对偶问题往往更容易求解。
2. 引入了核函数，进而推广到非线性分类问题。

&emsp;&emsp;原始问题：
$$
 \min_ {w,b} \frac 12 \|\|w\|\|_ 2^{2}\ s.t. \quad \tilde y_i(w \cdot x_i+b) -1 \ge 0, i=1,2,\cdots,N
$$

定义拉格朗日函数：

$$
L(w,b,\vec\alpha)=\frac 12 \|\|w\|\|_ 2^{2}-\sum_ {i=1}^{N}\alpha_i\tilde y_i(w \cdot x_i +b)+\sum_ {i=1}^{N} \alpha_i
$$

其中 $\alpha=(\alpha  _  1,\alpha  _  2,\cdots,\alpha  _  N)^{T}$ 为拉格朗日乘子向量。

&emsp;&emsp;根据拉格朗日对偶性，原始问题的对偶问题是极大极小问题：

$$
 \max_\alpha\min_ {w,b}L(w,b,\alpha)
$$

先求 $\min  _  {w,b}L(w,b,\alpha)$。拉格朗日函数分别为 $w,b$ 求偏导数，并令其等于 $0$

$$
\begin{aligned} { \nabla  }_ { w }L\left( w,b,\alpha  \right) &=w-\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }{ \tilde { y }  }_ { i }{ x }_ { i } } =0 \\ { \nabla  }_ { b }L\left( w,b,\alpha  \right) &=\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }{ \tilde { y }  }_ { i } } =0 \end{aligned}
$$   

得：

$$
w=\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }{ \tilde { y }  }_ { i }{ x }_ { i } } ,\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }{ \tilde { y }  }_ { i } } =0
$$

代入拉格朗日函数：

$$
\begin{aligned} L\left( w,b,\alpha  \right) &=\frac { 1 }{ 2 } \sum _ { i=1 }^{ N }{ \sum _ { j=1 }^{ N }{ { \alpha  }_ { i }{ \alpha  }_ { j }{ \tilde { y }  }_ { i }{ \tilde { y }  }_ { j }\left( { x }_ { i }\cdot { x }_ { j } \right)  }  } -\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }{ \tilde { y }  }_ { i }\left[ \left( \sum _ { j=1 }^{ N }{ { \alpha  }_ { j }{ \tilde { y }  }_ { j }{ x }_ { j } }  \right) { x }_ { i }+b \right]  } +\sum _ { i=1 }^{ N }{ { \alpha  }_ { i } }  \\ &=-\frac { 1 }{ 2 } \sum _ { i=1 }^{ N }{ \sum _ { j=1 }^{ N }{ { \alpha  }_ { i }{ \alpha  }_ { j }{ \tilde { y }  }_ { i }{ \tilde { y }  }_ { j }\left( { x }_ { i }\cdot { x }_ { j } \right)  }  } +\sum _ { i=1 }^{ N }{ { \alpha  }_ { i } }  \end{aligned}
$$

即

$$
\min _ { w,b }{ L\left( w,b,\alpha  \right) =-\frac { 1 }{ 2 } \sum _ { i=1 }^{ N }{ \sum _ { j=1 }^{ N }{ { \alpha  }_ { i }{ \alpha  }_ { j }{ \tilde { y }  }_ { i }{ \tilde { y }  }_ { j }\left( { x }_ { i }\cdot { x }_ { j } \right)  }  } +\sum _ { i=1 }^{ N }{ { \alpha  }_ { i } }  } 
$$

对偶问题极大值为：

$$
\begin{aligned} \max _ { \alpha  }{ -\frac { 1 }{ 2 } \sum _ { i=1 }^{ N }{ \sum _ { j=1 }^{ N }{ { \alpha  }_ { i }{ \alpha  }_ { j }{ \tilde { y }  }_ { i }{ \tilde { y }  }_ { j }\left( { x }_ { i }\cdot { x }_ { j } \right)  }  } +\sum _ { i=1 }^{ N }{ { \alpha  }_ { i } }  }  \\ s.t.&\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }{ \tilde { y }  }_ { i } } =0 \\ &{ \alpha  }_ { i }\ge 0,i=1,2,\cdots ,N \end{aligned}
$$

设对偶最优化问题的 $\alpha$ 的解为 $\alpha^{ \ * }=(\alpha  _  1^{ \ * },\alpha  _  2^{ \ * },\cdots,\alpha  _  N^{ \ * })$，则根据 $KKT$ 条件有：

$$
\begin{aligned} { \nabla  }_ { w }L\left( { w }^{ * },{ b }^{ * },{ \alpha  }^{ * } \right) &={ w }^{ * }-\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }^{ * }{ \tilde { y }  }_ { i }{ x }_ { i } }  \\ { \nabla  }_ { b }L\left( { w }^{ * },{ b }^{ * },{ \alpha  }^{ * } \right) &=\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }^{ * }{ \tilde { y }  }_ { i } } =0 \\ { \alpha  }_ { i }^{ * }\left[ { \tilde { y }  }_ { i }\left( { w }^{ * }\cdot { x }_ { i }+{ b }^{ * } \right) -1 \right] &=0,i=1,2,\cdots ,N \\ { \tilde { y }  }_ { i }\left( { w }^{ * }\cdot { x }_ { i }+{ b }^{ * } \right) -1&\ge 0,i=1,2,\cdots ,N \\ { \alpha  }_ { i }^{ * }&\ge 0,i=1,2,\cdots ,N \end{aligned}
$$

根据第一个式子，有：

$$
w^{ \ * }=\sum_ {i=1}^{N}\alpha_i^{ \ * }\tilde y_ix_i 
$$

由于 $\alpha^{ \ * }$ 不是零向量（若它为零向量，则 $w^{ \ * }$ 也为零向量，矛盾），则必然存在某个 $j$ 使得 $\alpha  _  j^{ \ * } \gt 0$。根据第三个式子，此时必有$\tilde y  _  j(w^{ \ * } \cdot x  _  j+b^{ \ * })-1=0$。同时考虑到$\tilde y  _  j^{2}=1$，得到：

$$
 b^{ \ * }=\tilde y_j-\sum_ {i=1}^{N}\alpha_i^{ \ * }\tilde y_i(x_i \cdot x_j)
$$

于是分离超平面写作：

$$
\sum_ {i=1}^{N}\alpha_i^{ \ * }\tilde y_i(x \cdot x_i)+b^{ \ * }=0 
$$

分类决策函数写作：

$$
f(x)=\text{sign}\left(\sum_ {i=1}^{N}\alpha_i^{ \ * }\tilde y_i(x \cdot x_i)+b^{ \ * }\right) 
$$

上式称作线性可分支持向量机的对偶形式。可以看到：分类决策函数只依赖于输入 $x$ 和训练样本的内积。

线性可分支持向量机对偶算法：
1. 输入：线性可分训练数据集 $\mathbb D=\{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\}$，其中 $x  _  i \in \mathcal X = \mathbb R^{n},\tilde y  _  i \in \mathcal Y=\{+1,-1\},i=1,2,\cdots,N$
2. 输出：
   1. 最大几何间隔的分离超平面
   2. 分类决策函数
3. 算法步骤：
   1. 构造并且求解约束最优化问题：
      $$
      \begin{aligned} \min _ { \alpha  }{ \frac { 1 }{ 2 } \sum _ { i=1 }^{ N }{ \sum _ { j=1 }^{ N }{ { \alpha  }_ { i }{ \alpha  }_ { j }{ \tilde { y }  }_ { i }{ \tilde { y }  }_ { j }\left( { x }_ { i }\cdot { x }_ { j } \right)  }  } -\sum _ { i=1 }^{ N }{ { \alpha  }_ { i } }  }  \\ s.t.&\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }{ \tilde { y }  }_ { i } } =0 \\ &{ \alpha  }_ { i }\ge 0,i=1,2,\cdots ,N \end{aligned}
      $$
      求得最优解 $\alpha^{ \ * }=(\alpha  _  1^{ \ * },\alpha  _  2^{ \ * },\cdots,\alpha  _  N^{ \ * })^{T}$。  
   2. 计算 $w^{ \ * }=\sum  _  {i=1}^{N}\alpha\  _  i^{ \ * }\tilde y  _  ix  _  i$。
   3. 选择 $\alpha^{ \ * }$ 的一个正的分量 $\alpha  _  j^{ \ * } \gt 0$，计算
      $$
      b^{ \ * }=\tilde y_j-\sum_ {i=1}^{N}\alpha_i^{ \ * }\tilde y_i(x_i \cdot x_j)
      $$
   4. 由此得到分离超平面：
      $$
      w^{ \ * }\cdot x+b^{ \ * }=0 
      $$
      以及分类决策函数：
      $$
      f(x)=\text{sign}(w^{ \ * }\cdot x+b^{ \ * })
      $$

&emsp;&emsp;$w^{ \ * },b^{ \ * }$ 只依赖于 $\alpha  _  i^{ \ * } \gt 0$ 对应的样本点 $x  _  i,\tilde y  _  i$，而其他的样本点对于 $w^{ \ * },b^{ \ * }$ 没有影响。将训练数据集里面对应于 $\alpha  _  i^{ \ * } \gt 0$ 的样本点对应的实例 $x  _  i$ 称为支持向量。
对于 $\alpha  _  i^{ \ * } \gt 0$ 的样本点，根据 $\alpha  _  i^{ \ * }[\tilde y  _  i(w^{ \ * }\cdot x  _  i+b^{ \ * })-1]=0$，有：$w^{ \ * }\cdot x  _  i+b^{ \ * }=\pm 1$。
即 $x  _  i$ 一定在间隔边界上。这与原始问题给出的支持向量的定义一致。


## 线性支持向量机

### 描述
&emsp;&emsp;当训练数据近似线性可分时，通过软间隔最大化，也可学得一个线性分类器，即线性支持向量机，又称为软间隔支持向量机。

### 原理
&emsp;&emsp;线性支持向量机利用几何间隔最大化和误分类点个数最小化求最优分离超平面。这样做保证了分类的正确性，并且使分类确信度最大化。

### 原始问题
&emsp;&emsp;设训练集为$\mathbb D=\{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\}$，其中 $x  _  i \in \mathcal X = \mathbb R^{n},\tilde y  _  i \in \mathcal Y=\{+1,-1\},i=1,2,\cdots,N$。假设训练数据集不是线性可分的，这意味着某些样本点 $(x  _  i,\tilde y  _  i)$ 不满足函数间隔大于等于 $1$ 的约束条件。

&emsp;&emsp;对每个样本点 $(x  _  i,\tilde y  _  i)$ 引进一个松弛变量 $\xi  _  i \ge 0$，使得函数间隔加上松弛变量大于等于 $1$。即约束条件变成了： 

$$
\tilde y_i(w \cdot x_i + b) \ge 1-\xi_i 
$$

对每个松弛变量 $\xi  _  i$，支付一个代价 $\xi  _  i$。目标函数由原来的 $\frac 12 \|\|w\|\|^{2}  _  2$ 变成：

$$
\min \frac 12 \|\|w\|\|^{2}_2+C\sum_ {i=1}^{N}\xi_i
$$

这里$C \gt 0$ 称作惩罚参数，一般由应用问题决定。$C$ 值越大，惩罚越大，要求误分类点的个数越少且几何间隔越小，但 $C$ 值过大时模型会表现出过拟合现象；$C$ 值越小，与上述情况相反。

&emsp;&emsp;相对于硬间隔最大化，$\frac 12 \|\|w\|\|^{2}  _  2+C\sum  _  {i=1}^{N}\xi  _  i$ 称为软间隔最大化。于是线性近似的线性支持向量机的学习问题变成了凸二次规划问题：

$$
\begin{aligned} \min _ { w,b,\xi  }{ \frac { 1 }{ 2 } { \left\| w \right\|  }_ { 2 }^{ 2 }+C\sum _ { i=1 }^{ N }{ { \xi  }_ { i } }  }  \\ s.t.&{ \tilde { y }  }_ { i }\left( w\cdot { x }_ { i }+b \right) \ge 1-{ \xi  }_ { i } \\ &{ \xi  }_ { i }\ge 0,i=1,2,\cdots ,N \end{aligned}
$$

这称为线性支持向量机的原始问题。因为这是个凸二次规划问题，因此解存在。可以证明 $w$ 的解是唯一的；$b$ 的解不是唯一的，$b$ 的解存在于一个区间。

&emsp;&emsp;对于给定的线性不可分的训练集数据，通过求解软间隔最大化问题得到的分离超平面为：

$$
w^{ \ * }\cdot x+b^{ \ * }=0 
$$

以及相应的分类决策函数：

$$
f(x)=w^{ \ * }\cdot x+b^{ \ * } 
$$

称之为线性支持向量机。

### 对偶问题
&emsp;&emsp;定义拉格朗日函数为：

$$
\begin{aligned} L\left( w,b,\xi ,\alpha ,\mu  \right) =\frac { 1 }{ 2 } { \left\| w \right\|  }_ { 2 }^{ 2 }+C\sum _ { i=1 }^{ N }{ { \xi  }_ { i } } -\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }\left[ { \tilde { y }  }_ { i }\left( w\cdot { x }_ { i }+b \right) -1+{ \xi  }_ { i } \right]  } -\sum _ { i=1 }^{ N }{ { \mu  }_ { i }{ \xi  }_ { i } }  \\ { \alpha  }_ { i }\ge 0,{ \mu  }_ { i }\ge 0 \end{aligned}
$$

原始问题是拉格朗日函数的极小极大问题；对偶问题是拉格朗日函数的极大极小问题。

&emsp;&emsp;先求 $L\left( w,b,\xi ,\alpha ,\mu  \right) $ 对 $w,b,\xi$ 的极小。根据偏导数为 $0$：

$$
\begin{aligned} { \nabla  }_ { w }L\left( w,b,\xi ,\alpha ,\mu  \right) &=w-\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }{ \tilde { y }  }_ { i }{ x }_ { i } } =0 \\ { \nabla  }_ { b }L\left( w,b,\xi ,\alpha ,\mu  \right) &=-\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }{ \tilde { y }  }_ { i } } =0 \\ { \nabla  }_ { { \xi  }_ { i } }L\left( w,b,\xi ,\alpha ,\mu  \right) &=C-{ \alpha  }_ { i }-{ \mu  }_ { i }=0 \end{aligned}
$$

得到：

$$
\begin{aligned} w=\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }{ \tilde { y }  }_ { i }{ x }_ { i } }  \\ \sum _ { i=1 }^{ N }{ { \alpha  }_ { i }{ \tilde { y }  }_ { i } } =0 \\ C-{ \alpha  }_ { i }-{ \mu  }_ { i }=0 \end{aligned}
$$

再求极大问题：将上面三个等式代入拉格朗日函数：

$$
\max _ { \alpha ,\mu  }{ \min _ { w,b,\xi  }{ L\left( w,b,\xi ,\alpha ,\mu  \right) =\max _ { \alpha ,\mu  }{ \left[ -\frac { 1 }{ 2 } \sum _ { i=1 }^{ N }{ \sum _ { j=1 }^{ N }{ { \alpha  }_ { i }{ \alpha  }_ { j }{ \tilde { y }  }_ { i }{ \tilde { y }  }_ { j }\left( { x }_ { i }\cdot { x }_ { j } \right)  }  } +\sum _ { i=1 }^{ N }{ { \alpha  }_ { i } }  \right]  }  }  } 
$$

于是得到对偶问题：

$$
\begin{aligned} \min _ { \alpha  }{ \frac { 1 }{ 2 } \sum _ { i=1 }^{ N }{ \sum _ { j=1 }^{ N }{ { \alpha  }_ { i }{ \alpha  }_ { j }{ \tilde { y }  }_ { i }{ \tilde { y }  }_ { j }\left( { x }_ { i }\cdot { x }_ { j } \right)  }  } -\sum _ { i=1 }^{ N }{ { \alpha  }_ { i } }  }  \\ s.t.\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }{ \tilde { y }  }_ { i } } =0 \\ { 0\le \alpha  }_ { i }\le C,i=1,2,\cdots ,N \end{aligned}
$$

根据 $KKT$ 条件：

$$
\begin{aligned} { \nabla  }_ { w }L\left( { w }^{ * },{ b }^{ * },{ \xi  }^{ * },{ \alpha  }^{ * },{ \mu  }^{ * } \right) &=w-\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }^{ * }{ \tilde { y }  }_ { i }{ x }_ { i } } =0 \\ { \nabla  }_ { b }L\left( { w }^{ * },{ b }^{ * },{ \xi  }^{ * },{ \alpha  }^{ * },{ \mu  }^{ * } \right) &=-\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }^{ * }{ \tilde { y }  }_ { i } } =0 \\ { \nabla  }_ { { \xi  }_ { i } }L\left( { w }^{ * },{ b }^{ * },{ \xi  }^{ * },{ \alpha  }^{ * },{ \mu  }^{ * } \right) &=C-{ \alpha  }_ { i }^{ * }-{ \mu  }_ { i }^{ * }=0 \\ { \alpha  }_ { i }^{ * }\left[ { \tilde { y }  }_ { i }\left( { w }^{ * }\cdot { x }_ { i }+{ b }^{ * } \right) -1+{ \xi  }_ { i }^{ * } \right] &=0 \\ { \mu  }_ { i }^{ * }{ \xi  }_ { i }^{ * }&=0 \\ { \tilde { y }  }_ { i }\left( { w }^{ * }\cdot { x }_ { i }+{ b }^{ * } \right) -1+{ \xi  }_ { i }^{ * }&\ge 0 \\ { \xi  }_ { i }^{ * }&\ge 0 \\ C&\ge { \alpha  }_ { i }^{ * }\ge 0 \\ { \mu  }_ { i }^{ * }&\ge 0 \\ i=1,2,\cdots ,N \end{aligned}
$$

则有：

$$
w^{ \ * }=\sum_ {i=1}^{N}\alpha^{ \ * }\tilde y_ix_i 
$$
1. 若 ${ \alpha  }  _  { i }^{ * }\le C$，${ \xi  }  _  { i }^{ * }=0$，则支持向量 $x  _  i$ 签好落在间隔边界上；
2. 若 ${ \alpha  }  _  { i }^{ * }=C$，$0<{ \xi  }  _  { i }^{ * }<1$，则分类正确，$x  _  i$ 在间隔边界与分离超平面之间；
3. 若 ${ \alpha  }  _  { i }^{ * }=C$，${ \xi  }  _  { i }=1$，则 $x  _  i$在分离超平面上；
4. 若 ${ \alpha  }  _  { i }^{ * }=C$，${ \xi  }  _  { i }>1$，则 $x  _  i$位于分离超平面一侧。

分离超平面为： 

$$
\sum_ {i=1}^{N}\alpha^{ \ * }\tilde y_i(x_i \cdot x)+b^{ \ * }=0 
$$

分类决策函数为：

$$
f(x)=\text{sign}\left[\sum_ {i=1}^{N}\alpha^{ \ * }\tilde y_i(x_i \cdot x)+b^{ \ * }\right]
$$

线性支持向量机对偶算法：
1. 输入：训练数据集 $\mathbb D=\{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\}$，其中 $x  _  i \in \mathcal X = \mathbb R^{n},\tilde y  _  i \in \mathcal Y=\{+1,-1\},i=1,2,\cdots,N$
2. 输出：
   1. 分离超平面
   2. 分类决策函数
3. 算法步骤：
   1. 选择惩罚参数 $C\gt 0$，构造并且求解约束最优化问题：
      $$
      \begin{aligned} \min _ { \alpha  }{ \frac { 1 }{ 2 } \sum _ { i=1 }^{ N }{ \sum _ { j=1 }^{ N }{ { \alpha  }_ { i }{ \alpha  }_ { j }{ \tilde { y }  }_ { i }{ \tilde { y }  }_ { j }\left( { x }_ { i }\cdot { x }_ { j } \right)  }  } -\sum _ { i=1 }^{ N }{ { \alpha  }_ { i } }  }  \\ s.t.\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }{ \tilde { y }  }_ { i } } =0 \\ { 0\le \alpha  }_ { i }\le C,i=1,2,\cdots ,N \end{aligned}
      $$
      
      求得最优解 $\alpha^{ \ * }=(\alpha  _  1^{ \ * },\alpha  _  2^{ \ * },\cdots,\alpha  _  N^{ \ * })^{T}$。
   2. 计算：$w^{ \ * }=\sum  _  {i=1}^{N}\alpha  _  i^{ \ * }\tilde y  _  ix  _  i$。
   3. 选择 $\alpha^{ \ * }$ 的一个合适的分量 $C \gt \alpha  _  j^{ \ * } \gt 0$，计算： 
      $$
      b^{ \ * }=\tilde y_j-\sum_ {i=1}^{N}\alpha_i^{ \ * }\tilde y_i(x_i \cdot x_j) 
      $$
      可能存在多个符合条件的 $\alpha  _  j^{ \ * }$。这是由于原始问题中，对$b$的解不唯一。所以实际计算时可以取在所有符合条件的样本点上的平均值。
   4. 由此得到分离超平面：
      $$
      w^{ \ * }\cdot x+b^{ \ * }=0
      $$
      以及分类决策函数： 
      $$
      f(x)=\text{sign}(w^{ \ * }\cdot x+b^{ \ * }) 
      $$

### 支持向量
&emsp;&emsp;在线性不可分的情况下，对偶问题的解 $\alpha^{ \ * }=(\alpha  _  1^{ \ * },\alpha  _  2^{ \ * },\cdots,\alpha  _  N^{ \ * })^{T}$ 中，对应于 $\alpha  _  i^{ \ * } \gt 0$ 的样本点 $(x  _  i,\tilde y  _  i)$ 的实例点 $x  _  i$ 称作支持向量，它是软间隔的支持向量。

&emsp;&emsp;线性不可分的支持向量比线性可分时的情况复杂一些，根据 $\nabla  _  {\xi  _  i} L(w,b,\xi,\alpha,\mu)=C-\alpha  _  i-\mu  _  i=0$，以及$ \mu  _  j^{ \ * }\xi  _  j^{ \ * }=0$，则：
1. 若 $\alpha  _  i^{ \ * } \lt C$，则 $\mu  _  i \gt 0$， 则松弛量 $\xi  _  i =0$。此时：支持向量恰好落在了间隔边界上。
2. 若 $\alpha  _  i^{ \ * } = C$， 则 $\mu  _  i =0$，于是 $\xi  _  i$ 可能为任何正数：
   1. 若 $0 \lt \xi  _  i \lt 1$，则支持向量落在间隔边界与分离超平面之间，分类正确。
   2. 若 $\xi  _  i= 1$，则支持向量落在分离超平面上。
   3. 若 $\xi  _  i \gt 1$，则支持向量落在分离超平面误分类一侧，分类错误。

### 合页损失函数
定义取正函数为：

$$
\text{plus}(z)= \begin{cases} z, & z \gt 0 \\ 0, & z \le 0 \end{cases}
$$

定义合页损失函数为： 

$$
L(\tilde y,\hat y)=\text{plus}(1-\tilde y \hat y)
$$

其中 $\tilde y$ 为样本的标签值，$\hat y$ 为样本的模型预测值。则线性支持向量机就是最小化目标函数：

$$
 \sum_ {i=1}^{N}\text{plus}(1-\tilde y_i(w \cdot x_i+b))+\lambda\|\|w\|\|^{2}_ 2,\quad \lambda \gt 0
$$

合页损失函数的物理意义：
1. 当样本点 $(x  _  i,\tilde y  _  i)$ 被正确分类且函数间隔（确信度）$\tilde y  _  i(w \cdot x  _  i+b)$ 大于 $1$ 时，损失为 $0$
2. 当样本点 $(x  _  i,\tilde y  _  i)$ 被正确分类且函数间隔（确信度）$\tilde y  _  i(w \cdot x  _  i+b)$ 小于等于 $1$ 时损失为 $1-\tilde y  _  i(w \cdot x  _  i+b)$
3. 当样本点 $(x  _  i,\tilde y  _  i)$ 未被正确分类时损失为 $1-\tilde y  _  i(w \cdot x  _  i+b)$

可以证明：线性支持向量机原始最优化问题等价于最优化问题：
$$
\min_ {w,b}\sum_ {i=1}^{N}\text{plus}(1-\tilde y_i(w \cdot x_i+b))+\lambda\|\|w\|\|^{2}_ 2,\quad \lambda \gt 0
$$

合页损失函数图形如下：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/SVM/hinge_loss.png?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Hinge Loss </div>
</center>

感知机的损失函数为 $plus\left( -\tilde { y } \left( w\cdot x+b \right)  \right) $，相比之下合页损失函数不仅要分类正确，而且要确信度足够高（确信度为 $1$）时，损失才是 $0$。即合页损失函数对学习有更高的要求。

&emsp;&emsp;$0-1$ 损失函数通常是二分类问题的真正的损失函数，合页损失函数是 $0-1$ 损失函数的上界。因为 $0-1$ 损失函数不是连续可导的，因此直接应用于优化问题中比较困难。通常都是用 $0-1$ 损失函数的上界函数构成目标函数，这时的上界损失函数又称为代理损失函数。

&emsp;&emsp;理论上 $SVM$ 的目标函数可以使用梯度下降法来训练。但存在三个问题：
1. 合页损失函数部分不可导。这可以通过 $sub-gradient descent$ 来解决。
2. 收敛速度非常慢。
3. 无法得出支持向量和非支持向量的区别。

## 非线性支持向量机
&emsp;&emsp;对于给定的训练集 $\mathbb D=\{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\}$，其中 $x  _  i \in \mathcal X = \mathbb R^{n},\tilde y  _  i \in \mathcal Y=\{+1,-1\},i=1,2,\cdots,N$，如果能用 $\mathbb R^{n}$ 中的一个超曲面将正负实例正确分开，则称这个问题为非线性可分问题。

&emsp;&emsp;设原空间为 $\mathcal X \subset \mathbb R^{2}, x=(x  _  1,x  _  2)^{T} \in \mathcal X$，新的空间为 $\mathcal Z \subset \mathbb R^{2}, z=(z  _  1,z  _  2)^{T}\in \mathcal Z$。定义从原空间到新空间的变换（映射）为：

$$
z=\phi(x)=(x_1^{2},x^{2}_ 2)^{T}
$$

则经过变换 $z=\phi(x)$：
1. 原空间 $\mathcal X \subset \mathbb R^{2}$ 变换为新空间 $\mathcal Z \subset \mathbb R^{2}$，原空间中的点相应地变换为新空间中的点。
2. 原空间中的椭圆 $w  _  1x  _  1^{2}+w  _  2x  _  2^{2}+b=0$ 变换为新空间中的直线 $w  _  1z  _  1+w  _  2z  _  2+b=0$。
3. 若在变换后的新空间，直线 $w  _  1z  _  1+w  _  2z  _  2+b=0$ 可以将变换后的正负实例点正确分开，则原空间的非线性可分问题就变成了新空间的线性可分问题。

用线性分类方法求解非线性分类问题分两步：
1. 首先用一个变换将原空间的数据映射到新空间。
2. 再在新空间里用线性分类学习方法从训练数据中学习分类模型。
这一策略称作核技巧。
    

### 核函数

#### 核函数定义
&emsp;&emsp;设 $\mathcal X$ 是输入空间（欧氏空间 $\mathbb R^{n}$ 的子集或者离散集合），$\mathcal H$ 为特征空间（希尔伯特空间）。若果存在一个从 $\mathcal X$ 到 $\mathcal H$ 的映射 $\phi(x):\mathcal X \rightarrow \mathcal H$，使得所有的 $x,z \in\mathcal X$， 函数$K(x,z)=\phi(x) \cdot \phi(z)$，则称 $K(x,z)$为核函数。
> 核函数将原空间中的任意两个向量 $x,z$，映射为特征空间中对应的向量之间的内积。

&emsp;&emsp;实际任务中，通常直接给定核函数 $K(x,z)$，然后用解线性分类问题的方法求解非线性分类问题的支持向量机。学习是隐式地在特征空间进行的，不需要显式的定义特征空间和映射函数。通常直接计算 $K(x,z)$ 比较容易，反而是通过 $\phi(x)$ 和 $\phi(z)$ 来计算 $K(x,z)$ 比较困难。
1. 首先特征空间 $\mathcal H$ 一般是高维的，甚至是无穷维的，映射 $\phi(x)$ 不容易定义。
2. 其次核函数关心的是希尔伯特空间两个向量的内积，而不关心这两个向量的具体形式。因此对于给定的核函数，特征空间 $\mathcal H$ 和映射函数 $\phi(x)$ 取法并不唯一。
   1. 可以取不同的特征空间 $\mathcal H$。
   2. 即使是在同一个特征空间 $\mathcal H$ 里，映射函数 $\phi(x)$ 也可以不同。

&emsp;&emsp;在线性支持向量机的对偶形式中，无论是目标函数还是决策函数都只涉及输入实例之间的内积。在对偶问题的目标函数中的内积 $x  _  i \cdot x  _  j$ 可以用核函数 $K(x  _  i,x  _  j)=\phi(x  _  i) \cdot \phi(x  _  j)$ 来代替。此时对偶问题的目标函数成为：

$$
 L(\alpha)=\frac 12 \sum_ {i=1}^{N}\sum_ {j=1}^{N}\alpha_i\alpha_j\tilde y_i\tilde y_jK(x_i,x_j)-\sum_ {i=1}^{N}\alpha_i
$$       

分类决策函数中的内积也可以用核函数代替：

$$
f(x)=\text{sign}\left(\sum_ {i=1}^{N}\alpha_i^{ \ * }\tilde y_iK(x_i,x)+b^{ \ * }\right) 
$$

核函数替代法，等价于：
1. 首先经过映射函数 $\phi$ 将原来的输入空间变换到一个新的特征空间。
2. 然后将输入空间中的内积 $x  _  i \cdot x  _  j$ 变换为特征空间中的内积 $\phi(x  _  i) \cdot \phi(x  _  j)$。
3. 最后在新的特征空间里从训练样本中学习线性支持向量机。

&emsp;&emsp;若映射函数$\phi$为非线性函数，则学习到的含有核函数的支持向量机是非线性分类模型。若映射函数$\phi$为线性函数，则学习到的含有核函数的支持向量机依旧是线性分类模型。

#### 核函数选择
&emsp;&emsp;在实际应用中，核函数的选取往往依赖领域知识，最后通过实验验证来验证核函数的有效性。

&emsp;&emsp;若已知映射函数 $\phi$，那么可以通过 $\phi(x)$ 和 $\phi(z)$ 的内积求得核函数 $K(x,z)$。现在问题是：不用构造映射 $\phi$， 那么给定一个函数 $K(x,z)$ 判断它是否是一个核函数？也就是说，$K(x,z)$ 满足什么条件才能成为一个核函数？

&emsp;&emsp;可以证明：设 $K:\mathcal X \times \mathcal X \rightarrow \mathbb R$ 是对称函数，则 $K(x,z)$ 为正定核函数的充要条件是：对任意 $x  _  i \in \mathcal X,i=1,2,\cdots,N$，$K(x,z)$ 对应的 $Gram$ 矩阵：$K=[K(x  _  i,x  _  j)]  _  {N\times N}$ 是半正定矩阵。

&emsp;&emsp;对于一个具体函数 $K(x,z)$ 来说，检验它为正定核函数并不容易。因为要求对任意有限输入集 $\{x  _  1,x  _  2,\cdots,x  _  N\}$ 来验证 $K(\cdot,\cdot)$ 对应的 $Gram$ 矩阵是否为半正定的。因此，实际问题中往往应用已有的核函数。

常用核函数：
1. 多项式核函数：
   $$
   K(x,z)=(x \cdot z+1)^{p}
   $$
   对应的支持向量机是一个 $p$ 次多项式分类器。
2. 高斯核函数：
   $$
   K(x,z)=\exp(-\frac{||x-z||^{2}}{2\sigma^{2}})
   $$
   它是最常用的核函数，对应于无穷维空间中的点积。它也被称作径向基函数 ($radial \ basis \ function:RBF$)，因为其值从 $x$ 沿着 $z$ 向外辐射的方向减小。对应的支持向量机是高斯径向基函数分类器 ($radial \ basis \ function$) 。
   
   $\gamma $ 与 $\sigma $ 关系：
   $$
   \gamma =\frac { 1 }{ 2{ \sigma  }^{ 2 } } 
   $$
   $\sigma $ 值越小，$\gamma $ 值越大，曲线越瘦高，数据分布越集中，会造成只作用于支持向量附近，会产生过拟合。
   * $\gamma $ 越大，支持向量越少。
   * $\gamma $ 越小，支持向量越多。

3. $Sigmod$ 核函数：
   $$
   K(x,z)=\tanh(\gamma(x \cdot z)+r)
   $$ 
   对应的支持向量机实现的就是一种神经网络。
        

### 学习算法
非线性支持向量机学习算法：
1. 输入：训练数据集 $\mathbb D=\{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\}$，其中$ x  _  i \in \mathcal X = \mathbb R^{n},\tilde y  _  i \in \mathcal Y=\{+1,-1\},i=1,2,\cdots,N$。
2. 输出：分类决策函数
3. 算法步骤：
   
   选择适当的核函数 $K(x,z)$ 和惩罚参数 $C\gt 0$，构造并且求解约束最优化问题：

   $$
   \begin{aligned} \min _ { \alpha  }{ \frac { 1 }{ 2 } \sum _ { i=1 }^{ N }{ \sum _ { j=1 }^{ N }{ { \alpha  }_ { i }{ \alpha  }_ { j }{ \tilde { y }  }_ { i }{ \tilde { y }  }_ { j }K\left( { x }_ { i }\cdot { x }_ { j } \right)  }  } -\sum _ { i=1 }^{ N }{ { \alpha  }_ { i } }  }  \\ s.t.\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }{ \tilde { y }  }_ { i } } =0 \\ { 0\le \alpha  }_ { i }\le C,i=1,2,\cdots ,N \end{aligned}
   $$

   求得最优解 $\alpha^{ \ * }=(\alpha  _  1^{ \ * },\alpha  _  2^{ \ * },\cdots,\alpha  _  N^{ \ * })^{T}$
   > 当$K(x,z)$ 是正定核函数时，该问题为凸二次规划问题，解是存在的。
   
   计算： 
   $$
   w^{ \ * }=\sum_ {i=1}^{N}\alpha_i^{ \ * }\tilde y_ix_i 
   $$

   选择 $\alpha^{ \ * }$的一个合适的分量$C \gt \alpha  _  j^{ \ * } \gt 0$，计算：
   $$
   b^{ \ * }=\tilde y_j-\sum_ {i=1}^{N}\alpha_i^{ \ * }\tilde y_iK(x_i,x_j) 
   $$

   构造分类决策函数：

   $$
   f(x)=\text{sign}\left(\sum_ {i=1}^{N}\alpha_i^{ \ * } \tilde y_iK(x_i,x)+b^{ \ * }\right) 
   $$

## 支持向量回归
&emsp;&emsp;支持向量机不仅可以用于分类问题，也可以用于回归问题。给定训练数据集 $\mathbb D=\{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\}$，其中 $x  _  i \in \mathcal X = \mathbb R^{n},\tilde y  _  i \in \mathcal Y=\mathbb R$。
1. 对于样本 $(x  _  i,\tilde y  _  i)$，传统的回归模型通常基于模型输出 $f(x  _  i)$ 与真实输出 $\tilde y  _  i$ 之间的差别来计算损失。当且仅当 $f(x  _  i)$ 与 $\tilde y  _  i$ 完全相同时，损失才为零。
2. 支持向量回归 (Support Vector Regression:SVR) 不同：它假设能容忍 $f(x  _  i)$ 与 $\tilde y  _  i$ 之间最多有 $\epsilon$ 的偏差。仅当 $\|f(x  _  i)-\tilde y  _  i \| \gt \epsilon$ 时，才计算损失。支持向量回归相当于以 $f(x  _  i)$ 为中心，构建了一个宽度为 $2\epsilon$ 的间隔带。若训练样本落在此间隔带内则被认为是预测正确的。

### 原始问题
$SVR$ 问题形式化为：

$$
f(x)=w \cdot x+b \ \min_ {w,b}\frac 12 \|\|w\|\|_ 2^{2}+C\sum_ {i=1}^{N}L_\epsilon\left(f(x_i)-\tilde y_i\right)
$$

其中：$C$ 为罚项常数。若 $C$ 较大，则倾向于 $f(x  _  i)$ 与 $\tilde y  _  i$ 之间较小的偏差；若 $C$ 较小，则能容忍 $f(x  _  i)$ 与 $\tilde y  _  i$ 之间较大的偏差。$L  _  \epsilon$ 为损失函数。其定义为：

$$
L_\epsilon(z)=\begin{cases} 0&, \text{if} |z| \le \epsilon\  |z|-\epsilon&,\text{else} \end{cases}
$$

> 线性回归中，损失函数为 L(z)=z^{2}


<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/SVM/L_epsilon.png?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> L Epsilon </div>
</center>

引入松弛变量 $\xi  _  i,\hat\xi  _  i$，将上式写做：

$$
\begin{aligned} \min _ { w,b,{ \xi  }_ { i },{ \hat { \xi  }  }_ { i } }{ \frac { 1 }{ 2 } { \left\| w \right\|  }_ { 2 }^{ 2 }+C\sum _ { i=1 }^{ N }{ \left( { \xi  }_ { i }+{ \hat { \xi  }  }_ { i } \right)  }  }  \\ s.t.f\left( { x }_ { i } \right) -{ \tilde { y }  }_ { i }\le \epsilon +{ \xi  }_ { i } \\ { \tilde { y }  }_ { i }-f\left( { x }_ { i } \right) \le \epsilon +{ \hat { \xi  }  }_ { i } \\ { \xi  }_ { i }\ge 0,{ \hat { \xi  }  }_ { i }\ge 0,i=1,2,\cdots ,N \end{aligned}
$$

这就是 $SVR$ 原始问题。

### 对偶问题
&emsp;&emsp;引入拉格朗日乘子，$\mu  _  i \ge 0,\hat \mu  _  i \ge 0,\alpha  _  i \ge 0,\hat\alpha  _  i \ge 0$，定义拉格朗日函数：

$$
L(w,b,\alpha,\hat{\alpha},\xi,\hat{\xi},\mu,\hat{\mu}) \\ =\frac 12 \|\|w\|\|_2^{2}+C\sum_ {i=1}^{N}( \xi_i+\hat\xi_i)-\sum_ {i=1}^{N}\mu_i\xi_i-\sum_ {i-1}^{N}\hat\mu_i\hat\xi_i +\sum_ {i=1}^{N}\alpha_i\left( f(x_i)-\tilde y_i-\epsilon-\xi_i \right)+\sum_ {i-1}^{N}\hat\alpha_i\left(\tilde y_i-f(x_i)-\epsilon-\hat\xi_i\right)
$$

根据拉格朗日对偶性，原始问题的对偶问题是极大极小问题：

$$
\max_ {\alpha,\hat{\alpha}}\min_ {w,b,\xi,\hat{\xi}}L(w,b,\alpha,\hat{\alpha},\xi,\hat{\xi},\vec\mu,\hat{\mu})
$$

先求极小问题，根据 $L(w,b,\alpha,\hat{\alpha},\xi,\hat{\xi},\mu,\hat{\mu}) $ 对 $w,b,\xi,\hat{\xi}$ 偏导数为零可得：

$$
\begin{aligned} w&=\sum _ { i=1 }^{ N }{ \left( { \hat { \alpha  }  }_ { i }-{ \alpha  }_ { i } \right) { x }_ { i } }  \\ 0&=\sum _ { i=1 }^{ N }{ \left( { \hat { \alpha  }  }_ { i }-{ \alpha  }_ { i } \right)  }  \\ C&={ \alpha  }_ { i }+{ \mu  }_ { i } \\ C&={ \hat { \alpha  }  }_ { i }+{ \hat { \mu  }  }_ { i } \end{aligned}
$$

再求极大问题（取负号变极小问题）：

$$
\begin{aligned} \min _ { \alpha ,\hat { \alpha  }  }{ \sum _ { i=1 }^{ N }{ \left[ { \tilde { y }  }_ { i }\left( { \hat { \alpha  }  }_ { i }-{ \alpha  }_ { i } \right) -\epsilon \left( { \hat { \alpha  }  }_ { i }+{ \alpha  }_ { i } \right)  \right]  } -\frac { 1 }{ 2 } \sum _ { i=1 }^{ N }{ \sum _ { j=1 }^{ N }{ \left( { \hat { \alpha  }  }_ { i }-{ \alpha  }_ { i } \right) \left( { \hat { \alpha  }  }_ { j }-{ \alpha  }_ { j } \right) { x }_ { i }^{ T }{ x }_ { j } }  }  }  \\ s.t.\sum _ { i=1 }^{ N }{ \left( { \hat { \alpha  }  }_ { i }-{ \alpha  }_ { i } \right)  } =0 \\ 0\le { \alpha  }_ { i },{ \hat { \alpha  }  }_ { i }\le C \end{aligned}
$$

上述过程需要满足 $KKT$ 条件，即：

$$
\begin{cases} \alpha_i\left( f(x_i)-\tilde y_i-\epsilon-\xi_i \right)=0\\ \hat\alpha_i\left(\tilde y_i-f(x_i)-\epsilon-\hat\xi_i\right)=0\\ \alpha_i\hat\alpha_i=0\\ \xi_i\hat\xi_i=0\ (C-\alpha_i)\xi_i=0\\ (C-\hat\alpha_i)\hat\xi_i=0 \end{cases}
$$

可以看出，当样本 $(x  _  i,\tilde y  _  i)$ 不落入 $\epsilon$ 间隔带中时，对应的 $\alpha  _  i,\hat\alpha  _  i$ 才能取非零值：
1. 当且仅当 $f(x  _  i)-\tilde y  _  i-\epsilon-\xi  _  i=0$ 时，$\alpha  _  i$ 能取非零值
2. 当且仅当 $\tilde y  _  i-f(x  _  i)-\epsilon-\hat\xi  _  i=0$ 时，$\hat\alpha  _  i$ 能取非零值

此外约束 $f(x  _  i)-\tilde y  _  i-\epsilon-\xi  _  i=0$ 与 $\tilde y  _  i-f(x  _  i)-\epsilon-\hat\xi  _  i=0$不能同时成立，因此$\alpha  _  i,\hat\alpha  _  i $中至少一个为零。

设最终解$\alpha$中，存在$C \gt \alpha  _  j \gt 0$，则有：
$$
\begin{aligned} b&={ \tilde { y }  }_ { j }+\epsilon -\sum _ { i=1 }^{ N }{ \left( { \hat { \alpha  }  }_ { i }-{ \alpha  }_ { j } \right) { x }_ { i }^{ T }{ x }_ { j } }  \\ f\left( x \right) &=\sum _ { i=1 }^{ N }{ \left( { \hat { \alpha  }  }_ { i }-{ \alpha  }_ { i } \right) { x }_ { i }^{ T }x+b }  \end{aligned}
$$

最后若考虑使用核技巧，则 $SVR$ 可以表示为：

$$
f(x)=\sum_ {i=1}^{N}(\hat\alpha_i-\alpha_i)K(x_i,x)+b
$$

## SVDD

### one class 分类
&emsp;&emsp;通常分类问题是两类或者多类，但有一种分类为一类 ($one class$) 的分类问题：它只有一个类，预测结果为是否属于这个类。

&emsp;&emsp;一类分类的策略是：训练出一个最小的超球面把正类数据包起来。识别一个新的数据点时，如果这个数据点落在超球面内，则属于正类；否则不是。

&emsp;&emsp;示例：给定一些用户的购物行为日志，其中包括两类用户：
1. 购买了某个商品的用户。可以肯定该类用户对于该商品是感兴趣的（标记为正例）。
2. 未购买某个商品的用户。此时无法断定该用户是对该商品感兴趣，还是不感兴趣（无法标记为反例）。

&emsp;&emsp;现在给定一群新的用户，预测这些用户中，哪些可能对该商品有兴趣。如果简单的使用二类分类问题，则有两个问题：
1. 未购买商品的用户，不一定是对该商品不感兴趣，可能是由于某些原因未能购买。
2. 通常未购买商品的用户数量远大于购买用户的数量。如果使用二类分类，则容易造成正负样本不均匀。

### SVDD 算法
&emsp;&emsp;$Support Vector Domain Description:SVDD$ 可以用于一类分类问题。给定训练集 $\mathbb D=\{x  _  1,x  _  2,\cdots,x  _  N\}$，这些样本都是属于同一类。$SVDD$ 的优化目标是：求一个中心为 $\mathbf{o}$，半径为 $R$ 的最小球面，使得 $\mathbb D$ 中的样本都在该球面中。

&emsp;&emsp;类似 $SVR$，$SVDD$ 允许一定程度上的放松，引入松弛变量。对松弛变量 $\xi  _  i$，其代价为 $C\xi  _  i$。

$$
\begin{aligned} L\left( R,o,\xi  \right) ={ R }^{ 2 }+\sum _ { i=1 }^{ N }{ { \xi  }_ { i } }  \\ s.t.{ \left\| { x }_ { i }-o \right\|  }_ { 2 }^{ 2 }\le { R }^{ 2 }+{ \xi  }_ { i } \\ { \xi  }_ { i }\ge 0 \\ i=1,2,\cdots ,N \end{aligned}
$$

其中 $C\gt 0$ 为惩罚系数：若 $C$ 较大，则不能容忍那些球面之外的点，因此球面会较大；若 $C$ 较小，则给予球面之外的点较大的弹性，因此球面会较小。

&emsp;&emsp;$SVDD$ 的求解也是采用拉格朗日乘子法：

$$
L(R,\mathbf{ o},\alpha,\xi,\gamma)=R^2+C\sum_ {i=1}^{N}\xi_i-\sum_ {i=1}^{N}\alpha_i\left(R^2+\xi_i-||x_i-\mathbf{ o}||_2^2\right)-\sum_ {i=1}^{N}\gamma_i\xi_i\  s.t. \alpha_i\ge 0,\gamma_i\ge 0,\xi_i\ge 0
$$

根据拉格朗日对偶性，原始问题的对偶问题是极大极小问题：

$$
\max_ {\alpha,\gamma}\min_ {R,\mathbf{ a},\xi}L(R,\mathbf{ o},\vec\alpha,\vec\xi,\vec\gamma) 
$$

先求极小问题：根据 $L(R,\mathbf{ o},\alpha,\xi,\gamma)$对$R$,$\mathbf{o},\xi$ 偏导数为零可得：

$$
\begin{aligned} \sum _ { i=1 }^{ N }{ { \alpha  }_ { i } } =1 \\ o=\frac { \sum _ { i=1 }^{ N }{ { \alpha  }_ { i }{ x }_ { i } }  }{ \sum _ { i=1 }^{ N }{ { \alpha  }_ { i } }  } =\frac { \sum _ { i=1 }^{ N }{ { \alpha  }_ { i }{ x }_ { i } }  }{ \sum _ { i=1 }^{ N }{ { \alpha  }_ { i } }  }  \\ C-{ \alpha  }_ { i }-{ \gamma  }_ { i }=0,i=1,2,\cdots ,N \end{aligned}
$$

代入拉格朗日函数有：

$$
 \begin{aligned} L=\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }\left( { x }_ { i }\cdot { x }_ { i } \right)  } -\sum _ { i=1 }^{ N }{ \sum _ { j=1 }^{ N }{ { \alpha  }_ { i }{ \alpha  }_ { j }\left( { x }_ { i }\cdot { x }_ { j } \right)  }  }  \\ s.t.0\le { \alpha  }_ { i }\le C \\ \sum _ { i=1 }^{ N }{ { \alpha  }_ { i } } =1 \end{aligned}
$$

引入核函数：

$$
 \begin{aligned} L=\sum _ { i=1 }^{ N }{ { \alpha  }_ { i }K\left( { x }_ { i }\cdot { x }_ { i } \right)  } -\sum _ { i=1 }^{ N }{ \sum _ { j=1 }^{ N }{ { \alpha  }_ { i }{ \alpha  }_ { j }K\left( { x }_ { i }\cdot { x }_ { j } \right)  }  }  \\ s.t.0\le { \alpha  }_ { i }\le C \\ \sum _ { i=1 }^{ N }{ { \alpha  }_ { i } } =1 \end{aligned}
$$

其解法类似支持向量机的解法。

&emsp;&emsp;判断一个新的数据点 $\mathbf{z}$ 是否属于这个类，主要看它是否在训练出来的超球面内：若 

$$
||\mathbf{ z}-\mathbf{o}||_ 2^2\le R^2
$$

则判定为属于该类。如果使用支持向量，则判定准则为：

$$
(\mathbf{z}\cdot \mathbf{z}) -2\sum_ {i=1}^N\alpha_i(\mathbf{z}\cdot x_i)+\sum_ {i=1}^N\sum_ {j=1}^N\alpha_i\alpha_j(x_i\cdot x_j) \le R^2 
$$

如果是用核函数，则判定准则为：

$$
K(\mathbf{ z}\cdot \mathbf{z}) -2\sum_ {i=1}^N\alpha_iK(\mathbf{z}\cdot x_i)+\sum_ {i=1}^N\sum_ {j=1}^N\alpha_i\alpha_jK(x_i\cdot x_j) \le R^2
$$

## 序列最小最优化方法
&emsp;&emsp;支持向量机的学习问题可以形式化为求解凸二次规划问题。这样的凸二次规划问题具有全局最优解，并且有多种算法可以用于这一问题的求解。当训练样本容量非常大时，这些算法往往非常低效。而序列最小最优化 ($Sequential Minimal Optimization:SMO$）算法可以高效求解。

&emsp;&emsp;$SMO$ 算法的思路：若所有变量都满足条件，则最优化问题的解就得到了；否则，选择两个变量的同时固定其他所有变量，针对这两个变量构建一个二次规划子问题。这个二次规划子问题关于这两个变量的解应该更接近原始二次规划问题的解，因为这会使得原始二次规划问题的目标函数值变得更小。更重要的是，这个二次规划子问题可以通过解析的方法求解。此时子问题有两个变量，至少存在一个变量不满足约束条件（否则就是所有变量满足条件了）。假设其中一个是违反约束最严重的那个，另一个由约束等式自动确定：$\sum  _  {i=1}^N\alpha  _  i\tilde y  _  i=0$。

&emsp;&emsp;$SMO$ 算法将原始问题不断地分解为子问题并且对子问题求解，进而达到求解原问题的目的。整个 $SMO$ 算法包括两部分：
1. 求解两个变量二次规划的解析方法。
2. 选择变量的启发式方法。

### 子问题的求解
&emsp;&emsp;假设选择的两个变量是 $\alpha  _  1,\alpha  _  2$， 其他变量 $\alpha  _  i,i=3,4,\cdots,N$ 是固定的。于是 $SMO$ 的最优化问题的子问题为：

$$
\begin{aligned} \min _ { { \alpha  }_ { 1 },{ \alpha  }_ { 2 } }{ L\left( { \alpha  }_ { 1 },{ \alpha  }_ { 2 } \right) \\ =\frac { 1 }{ 2 } { K }_ { 11 }{ \alpha  }_ { 1 }^{ 2 }+\frac { 1 }{ 2 } { K }_ { 22 }{ \alpha  }_ { 2 }^{ 2 }+\tilde { y } _ { 1 }{ \tilde { y }  }_ { 2 }{ K }_ { 12 }{ \alpha  }_ { 1 }{ \alpha  }_ { 2 }-\left( { \alpha  }_ { 1 }+{ \alpha  }_ { 2 } \right) +\tilde { y } _ { 1 }{ \alpha  }_ { 1 }\sum _ { i=1 }^{ 3 }{ \tilde { y } _ { i }{ \alpha  }_ { i }{ K }_ { i1 } } +\tilde { y } _ { 2 }{ \alpha  }_ { 2 }\sum _ { i=1 }^{ 3 }{ \tilde { y } _ { i }{ \alpha  }_ { i }{ K }_ { i2 } }  }  \\ s.t.{ \alpha  }_ { 1 }\tilde { y } _ { 1 }+{ \alpha  }_ { 2 }\tilde { y } _ { 2 }=-\sum _ { i=1 }^{ 3 }{ \tilde { y } _ { i }{ \alpha  }_ { i } } =\gamma  \\ 0\le { \alpha  }_ { i }\le C,i=1,2 \end{aligned}
$$

其中 $K  _  {ij}=K(x  _  i,x  _  j),i,j=1,2,\cdots,N,\quad \gamma$ 为常数，且目标函数式中省略了不含 $\alpha  _  1,\alpha  _  2$ 的常数项。

#### 取值范围约束
$\alpha  _  1,\alpha  _  2$ 的约束条件为：

当 $\tilde y  _  1,\tilde y  _  2$ 异号时，$\alpha  _  1,\alpha  _  2$ 位于直线 $ \|\alpha  _  1-\alpha  _  2 \|=\gamma$，且在矩形范围内。矩形的长为 $C$，宽为 $C$， 起始点坐标为 $(0,0)$： 


<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/SVM/smo_1.png?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> SMO 1 </div>
</center>


此时 $\alpha  _  2$ 和 $\alpha  _  1$ 的取值范围为：
1. 当 $\alpha  _  2\ge \alpha  _  1$ 时（上面那条线）：
   $$
   \begin{aligned} &{ \gamma \le \alpha  }_ { 2 }\le C \\ &0\le { \alpha  }_ { 1 }\le C-\gamma  \end{aligned}
   $$
2. 当 $\alpha  _  2\gt \alpha  _  1$ 时（下面那条线）：
   $$
   \begin{aligned} &{ \gamma \le \alpha  }_ { 1 }\le C \\ &0\le { \alpha  }_ { 2 }\le C-\gamma  \end{aligned}
   $$

当 $\tilde y  _  1,\tilde y  _  2$ 同号时，$\alpha  _  1,\alpha  _  2$ 位于直线 $\alpha  _  1+\alpha  _  2=\gamma$，且在矩形范围内。矩形的长为 $C$，宽为 $C$， 起始点坐标为 $(0,0)$： 


<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/SVM/smo_2.png?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> SMO 2 </div>
</center>

此时 $\alpha  _  1,\alpha  _  2$ 取值范围为：
1. 当 $\gamma \ge C$ 时：（上面那条线）
   $$
   \begin{aligned} \gamma -C\le { \alpha  }_ { 1 }\le C \\ \gamma -C\le { \alpha  }_ { 2 }\le C \end{aligned}
   $$
2. 当 $\gamma \lt C$ 时：（下面那条线）
   $$
   \begin{aligned} 0\le { \alpha  }_ { 1 }\le \gamma  \\ 0\le { \alpha  }_ { 2 }\le \gamma  \end{aligned}
   $$

假设 $\alpha  _  2$ 的最优解为 $\alpha  _  2^{new}$，其初始可行解为 $\alpha  _  2^{old}$；$\alpha  _  1$ 的初始可行解为 $\alpha  _  1^{old}$。既然是初始可行解，则需要满足约束条件。因此有

$$
\begin{aligned} \left| { \alpha  }_ { 1 }^{ old }-{ \alpha  }_ { 2 }^{ old } \right| &=\gamma ,if{ \tilde { y }  }_ { 1 }{ \neq \tilde { y }  }_ { 2 } \\ { \alpha  }_ { 1 }^{ old }+{ \alpha  }_ { 2 }^{ old }&=\gamma ,if{ \tilde { y }  }_ { 1 }={ \tilde { y }  }_ { 2 } \end{aligned}
$$

假设 $\alpha  _  2^{new}$ 的取值范围为 $[L,H]$，则有：
1. 当 $\tilde y  _  1\ne \tilde y  _  2$ 时，若$\alpha  _  2^{new} \ge \alpha  _  1^{new}$，则$\gamma \le \alpha  _  2^{new} \le C$；若$ \alpha  _  1^{new} \gt \alpha  _  2^{new}$，则$0 \le \alpha  _  2 ^{new}\le C-\gamma$。根据：
   $$
   \gamma =\left| { \alpha  }_ { 1 }^{ old }-{ \alpha  }_ { 2 }^{ old } \right| =\begin{cases} { \alpha  }_ { 2 }^{ old }-{ \alpha  }_ { 1 }^{ old },if{ \alpha  }_ { 2 }^{ old }-{ \ge \alpha  }_ { 1 }^{ old } \\ { \alpha  }_ { 1 }^{ old }-{ \alpha  }_ { 2 }^{ old },else \end{cases}
   $$
   则有：
   $$
   \begin{aligned} L&=\max { \left( 0,{ \alpha  }_ { 2 }^{ old }-{ \alpha  }_ { 1 }^{ old } \right)  }  \\ H&=\min { \left( C,C+{ \alpha  }_ { 2 }^{ old }-{ \alpha  }_ { 1 }^{ old } \right)  }  \end{aligned}
   $$
2. 当 $\tilde y  _  1=\tilde y  _  2$ 时，若 $\gamma \ge C$，则$\gamma-C \le \alpha  _  2^{new} \le C$；若 $\gamma \lt C$， 则$0 \le \alpha  _  2^{new} \le \gamma$。根据：
   $$
   \gamma ={ \alpha  }_ { 1 }^{ old }+{ \alpha  }_ { 2 }^{ old }
   $$
   则有：
   $$
   \begin{aligned} L&=\max { \left( 0,{ \alpha  }_ { 2 }^{ old }+{ \alpha  }_ { 1 }^{ old }-C \right)  }  \\ H&=\min { \left( C,{ \alpha  }_ { 2 }^{ old }{ +\alpha  }_ { 1 }^{ old } \right)  }  \end{aligned}
   $$
  
#### 解析解
&emsp;&emsp;令 $g\left( x \right) =\sum   _  { i=1 }^{ N }{ { \alpha  }  _  { i }{ \tilde { y }  }  _  { i }K\left( { x }  _  { i },x \right) +b } $。它表示解得 $ \alpha,b$ 参数之后，对 $x$ 的预测值。预测值的正负代表了分类的结果。令

$$
\begin{aligned} { E }_ { i }&=g\left( { x }_ { i } \right) -{ \tilde { y }  }_ { i } \\ &=\left( \sum _ { j=1 }^{ N }{ { \alpha  }_ { j }{ \tilde { y }  }_ { j }K\left( { x }_ { j },{ x }_ { i } \right) +b }  \right) -{ \tilde { y }  }_ { i } \end{aligned}
$$

其中，$i=1,2$。$E  _  1$ 表示 $g(x  _  1)$ 的预测值与真实输出 $\tilde y  _  1 $ 之差。$E  _  2$ 表示 $g(x  _  2)$ 的预测值与真实输出 $\tilde y  _  2$ 之差。

根据 $\alpha  _  1\tilde y  _  1+\alpha  _  2\tilde y  _  2 =\gamma$，将 $\alpha  _  1=\frac{\gamma-\tilde y  _  2\alpha  _  2}{\tilde y  _  1}$ 代入 $ L(\alpha  _  1,\alpha  _  2)$ 中，求解 $d\frac{L(\alpha  _  2)}{d\alpha  _  2}=0$ ，即可得到 $\alpha  _  2$ 的最优解（不考虑约束条件）：

$$
\alpha_2^{new,unc}=\alpha_2^{old}+\frac{\tilde y_2(E_1-E_2)}{\eta}
$$

其中：$\eta=K  _  {11}+K  _  {22}-2K  _  {12}$；$E  _  1,E  _  2$ 中的 $\alpha  _  1,\alpha  _  2$ 分别为 $\alpha  _  1^{old},\alpha  _  2^{old}$ （它们表示初始的可行解，用于消掉 $\gamma$）。

将 $\alpha  _  2^{new,unc}$ 截断，则得到 $\alpha  _  2^{new}$ 的解析解为：

$$
{ \alpha  }_ { 2 }^{ new }=\begin{cases} H,&{ \alpha  }_ { 2 }^{ new,unc }>H \\ { \alpha  }_ { 2 }^{ new,unc },&L\le { \alpha  }_ { 2 }^{ new,unc }\le H \\ L,&{ \alpha  }_ { 2 }^{ new,unc } < L \end{cases}
$$

其中 $\alpha  _  1^{old},\alpha  _  2^{old}$ 为初始可行解，$\alpha  _  2^{new}$ 为最终解。

根据 $\tilde y  _  1\alpha  _  1^{new}+\tilde y  _  2\alpha  _  2^{new}=\gamma=\tilde y  _  1\alpha  _  1^{old}+\tilde y  _  2\alpha  _  2^{old}$，以及 $\tilde y  _  1^2=\tilde y  _  2^2=1$，得到 $\alpha  _  1^{new}$：

$$
 \alpha_1^{new}=\alpha_1^{old}+\tilde y_1\tilde y_2(\alpha_2^{old}-\alpha_2^{new}) 
$$

其中 $\alpha  _  1^{old},\alpha  _  2^{old}$ 为初始可行解，$\alpha  _  1^{new}$ 为最终解。

### 变量选
&emsp;&emsp;$SMO$ 算法在每个子问题中选择两个变量进行优化，其中至少一个变量是违反约束条件的。如果都不违反约束条件，则说明已经求解了。

#### 外层循环
&emsp;&emsp;第一个变量的选择过程称为外层循环。外层循环在训练样本中选择违反约束条件最严重的样本点，并将对应的变量作为第一个变量。具体来讲，就是检验训练样本点 $(x  _  i,\tilde y  _  i)$ 是否满足约束条件 ($KKT$ 条件)：

$$
\begin{aligned} { \alpha  }_ { i }=0\Leftrightarrow { \tilde { y }  }_ { i }g\left( { x }_ { i } \right) \ge 1 \\ 0<{ \alpha  }_ { i } < C\Leftrightarrow { \tilde { y }  }_ { i }g\left( { x }_ { i } \right) =1 \\ { \alpha  }_ { i }=C\Leftrightarrow { \tilde { y }  }_ { i }g\left( { x }_ { i } \right) \le 1 \end{aligned}
$$

其中，$g\left( { x }  _  { i } \right) =\sum   _  { j=1 }^{ N }{ { \alpha  }  _  { j }{ \tilde { y }  }  _  { j }K\left( { x }  _  { j },{ x }  _  { i } \right) +b } $

&emsp;&emsp;检验时，外层循环首先遍历所有满足条件 $0 \lt \alpha  _  i \lt C$ 的样本点，即间隔边界上的支持向量点。如果这些样本点都满足条件，再遍历整个训练集，检验是否满足条件。

#### 6.2.2 内存循环
&emsp;&emsp;第二个变量的选择过程称为内层循环。假设已经在外层循环中找到第一个变量 $ \alpha  _  1$，现在要在内层循环中找到第二个变量 $ \alpha  _  2$。第二个变量选择标准是希望能够使得 $\alpha  _  2$ 有足够大的变化。

&emsp;&emsp;由前面式子可知，$\alpha  _  2^{new}$ 依赖于 $E  _  1-E  _  2$。一种简单的做法是选择 $\alpha  _  2$，使得对应的 $|E\  _  1-E\  _  2|$ 最大。因为 $\alpha  _  1$ 已经确定， $E  _  1$ 也已经确定。
1. 如果 $E  _  1$ 为正数，则选择最小的 $E  _  i$ 作为 $E  _  2$。
2. 如果 $E  _  1$ 为负数，则选择最大的 $E  _  i$ 作为 $E  _  2$。
> 为了节省计算时间，可以将所有 $E _ i$ 值保存在一个列表中。

&emsp;&emsp;特殊情况下，若内层循环找到的 $\alpha  _  2$ 不能使得目标函数有足够的下降，则采用以下的启发式规则继续选择 $\alpha  _  2$：
1. 遍历在间隔边界上的支持向量点，依次将其对应的变量作为 $\alpha  _  2$ 试用，直到目标函数有足够的下降。
2. 若还是找不到合适的 $\alpha  _  2$ ，则遍历训练数据集寻找。
3. 若还是找不到合适的 $\alpha  _  2$，则放弃找到的 $\alpha  _  1$，再通过外层循环寻求另外的 $\alpha  _  1$。

#### 参数更新
&emsp;&emsp;每次完成两个变量的优化后，都要重新计算 $b$。根据约束条件有，当 $0 \lt \alpha  _  1^{new} \lt C$ 时：

$$
\sum_ {i=1}^{N}\alpha_i\tilde y_iK_ {i1}+b=\tilde y_1
$$

代入 $E  _  1$ 的定义式有：
$$
b_1^{new}=-E_1-\tilde y_1K_ {11}(\alpha_1^{new}-\alpha_1^{old})-\tilde y_2K_ {21}(\alpha_2^{new}-\alpha_2^{old})+b^{old}
$$

同样，当 $0 \lt \alpha  _  2^{new} \lt C$ 时有：

$$
b_2^{new}=-E_2-\tilde y_1K_ {12}(\alpha_1^{new}-\alpha_1^{old})-\tilde y_2K_ {22}(\alpha_2^{new}-\alpha_2^{old})+b^{old} 
$$

如果 $\alpha  _  1^{new},\alpha  _  2^{new}$ 同时满足 $0\lt\alpha  _  i^{new}\lt C$ ，则有：

$$
b_1^{new}=b_2^{new}
$$

如果 $\alpha  _  1^{new},\alpha  _  2^{new}$ 或者为 $0$，或者为 $C$，则 $ [b  _  1^{new},b  _  2^{new}]$ 区间内的数都可以作为 b^{new}。此时一个选择是：

$$
b^{new}=\frac{b_1^{new}+b_2^{new}}{2}
$$

每次完成两个变量的优化后，需要更新对应的 $E  _  i$ 值。$E  _  i$ 的更新要用到 $ b^{new}$ 以及所有支持向量对应的 $\alpha  _  j$。

### SMO算法
输入：
1. 训练数据集 $\mathbb D=\{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\}$，其中 $x  _  i \in \mathcal X = \mathbb R^{n},\tilde y  _  i \in \mathcal Y=\{+1,-1\}$
2. 精度 $\varepsilon$

输出：近似解 $\hat {\alpha}$

算法步骤：
1. 取初值 $\alpha^{(0)}=0$，$k=0$
2. 选取优化变量 $\alpha  _  1^{(k)},\alpha  _  2^{(k)}$，解析求解两个变量的最优化问题，求得最优解 $\alpha  _  1^{(k+1)},\alpha  _  2^{(k+1)}$，更新 $\alpha$ 为 $\alpha^{(k+1)}$。
3. 若在精度 $\varepsilon$ 范围内满足停机条件：
   $$
   \begin{aligned} \sum _ { i=1 }^{ N }{ { \alpha  }_ { i }{ \tilde { y }  }_ { i } } =0 \\ 0\le { \alpha  }_ { i }\le C,i=1,2,\cdots ,N \\ { \tilde { y }  }_ { i }g\left( { x }_ { i } \right) =\begin{cases} \ge 1,{ \alpha  }_ { i }=0 \\ =1,0<{ \alpha  }_ { i } < C \\ \le 1,{ \alpha  }_ { i }=C \end{cases} \end{aligned}
   $$
   则退出迭代并令 $\hat {\alpha}=\alpha^{(k+1)}$；否则令 $k=k+1$ ，继续迭代。其中 $g\left( { x }  _  { i } \right) =\sum   _  { j=1 }^{ N }{ { \alpha  }  _  { j }{ \tilde { y }  }  _  { j }K\left( { x }  _  { j },{ x }  _  { i } \right) +b } $。
