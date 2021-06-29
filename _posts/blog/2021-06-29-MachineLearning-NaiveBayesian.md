---
layout: post
title: 贝叶斯分类器
categories: [MachineLearning]
description: 贝叶斯分类器
keywords: MachineLearning
---


贝叶斯分类器
---


## 贝叶斯定理
&emsp;&emsp;设 $ \mathbb S $ 为试验$E$的样本空间；${ B }  _  { 1 },{ B }  _  { 2 },\cdots ,{ B }  _  { n }$ 为 $E$ 的一组事件。若 ${ B }  _  { i }\cap { B }  _  { j }=\emptyset ,i\neq j,i,j=1,2,\cdots ,n$ 且 ${ B }  _  { 1 }\cup { B }  _  { 2 }\cup \cdots \cup { B }  _  { n }=\mathbb S$ 则称 ${ B }  _  { 1 },{ B }  _  { 2 },\cdots ,{ B }  _  { n }$ 为样本空间 $ \mathbb S $ 的一个划分。

&emsp;&emsp;如果 ${ B }  _  { 1 },{ B }  _  { 2 },\cdots ,{ B }  _  { n }$ 为样本空间 $ \mathbb S $ 的一个划分，则对于每次试验，事件 ${ B }  _  { 1 },{ B }  _  { 2 },\cdots ,{ B }  _  { n }$ 中有且仅有一个事件发生。

&emsp;&emsp;全概率公式：设试验 $E$ 的样本空间为 $ \mathbb S $，$A$ 为 $E$ 的事件，${ B }  _  { 1 },{ B }  _  { 2 },\cdots ,{ B }  _  { n }$ 为样本空间 $ \mathbb S $ 的一个划分，且 $p\left( { B }  _  { i } \right) \ge 0\left( i=1,2,\cdots ,n \right) $。则有：

$$
p\left( A \right) =p\left( A|{ B }_ { 1 } \right) p\left( { B }_ { 1 } \right) +p\left( A|{ B }_ { 2 } \right) p\left( { B }_ { 2 } \right) +\cdots +p\left( A|{ B }_ { n } \right) p\left( { B }_ { n } \right) =\sum _ { i=1 }^{ n }{ p\left( A|{ B }_ { i } \right) p\left( { B }_ { i } \right)  } 
$$

&emsp;&emsp;贝叶斯定理：设试验 $E$ 的样本空间为 $ \mathbb S $，$A$ 为 $E$ 的事件，${ B }  _  { 1 },{ B }  _  { 2 },\cdots ,{ B }  _  { n }$ 为样本空间 $ \mathbb S $ 的一个划分，且 $p\left( A \right) >0,p\left( { B }  _  { i } \right) \ge 0\left( i=1,2,\cdots ,n \right) $，则有：

$$
p\left( { B }_ { i }|A \right) =\frac { p\left( A|{ B }_ { i } \right) p\left( { B }_ { i } \right)  }{ \sum _ { j=1 }^{ n }{ p\left( A|{ B }_ { j } \right) p\left( { B }_ { j } \right)  }  } 
$$
    
## 先验概率、后验概率

$$
p\left( A|B \right) =\frac { p\left( B|A \right) p\left( A \right)  }{ p\left( B \right)  } 
$$

其中，

$p\left( A \right)$ 是 $A$ 的先验概率或边缘概率。之所以称为"先验"是因为它不考虑任何$B$方面的因素；

$p\left( B \right) $ 是 $B$ 的先验概率或边缘概率；

$p\left( A|B \right) $ 是已知 $B$ 发生后$A$的条件概率，也由于得自 $B$ 的取值而被称作 $A$ 的后验概率；

$p(B|A)$ 是已知 $A$ 发生后 $B$ 的条件概率，也由于得自 $A$ 的取值而被称作 $B$ 的后验概率。


## 朴素贝叶斯
### 描述
&emsp;&emsp;朴素贝叶斯是基于贝叶斯定理与特征独立假设的分类方法，之所以称之为“朴素”是由于做了特征间相互独立的假设。

### 原理
&emsp;&emsp;对于给定的待分类项，求解在此项出现的条件下各个类别出现的概率，哪个类别概率最大，就把待分类项标记为那个类别。

&emsp;&emsp;设输入空间 $\mathcal X \subseteq \mathbb R^{n} $ 为 $n$ 维向量的集合 ，输出空间为类标记集合 $\mathcal Y=\left\{ { c }  _  { 1 },{ c }  _  { 2 },\cdots ,{ c }  _  { k } \right\} $。令 $x={ \left( { x }  _  { 1 },{ x }  _  { 2 },\cdots ,{ x }  _  { n } \right)  }^{ T }$ 为定义在 $ \mathcal X $ 上的随机向量，$y$ 为定义在 $\mathcal Y$ 上的随机变量。令 $p\left( x,y \right) $ 为 $x$ 和 $y$ 的联合概率分布，假设训练数据集 $\mathbb D=\left\{ \left( { x }  _  { 1 },{ y }  _  { 1 } \right) ,\left( { x }  _  { 2 },{ y }  _  { 2 } \right) ,\cdots ,\left( { x }  _  { m },{ y }  _  { m } \right)  \right\}$ 由 $p\left( x,y \right) $ 独立同分布产生。

&emsp;&emsp;朴素贝叶斯通过训练数据集 学习联合概率分布 $p\left( x,y \right) $。具体学习下列概率分布：

1. 先验概率分布：$p\left( y \right) $。
2. 条件概率分布：$p\left( x|y \right) =p\left( { x }  _  { 1 },{ x }  _  { 2 },\cdots ,{ x }  _  { n }|y \right) $。

&emsp;&emsp;朴素贝叶斯法对条件概率做了特征独立性假设：$p\left( x|y \right) =p\left( { x }  _  { 1 },{ x }  _  { 2 },\cdots ,{ x }  _  { n }|y \right) =\prod   _  { j=1 }^{ n }{ p\left( { x }  _  { j }|y \right)  } $。这意味着在分类确定的条件下，用于分类的特征是条件独立的；该假设使得朴素贝叶斯法变得简单，但是可能牺牲一定的分类准确率。

&emsp;&emsp;根据贝叶斯定理：

$$
p\left( y|x \right) =\frac { p\left( x|y \right) p\left( y \right)  }{ \sum _ { { y }^{ \prime  } }{ p\left( x|{ y }^{ \prime  } \right) p\left( { y }^{ \prime  } \right)  }  } 
$$

考虑分类特征的条件独立假设有：

$$
p\left( y|x \right) =\frac { p\left( y \right) \prod _ { j=1 }^{ n }{ p\left( { x }_ { j }|y \right)  }  }{ \sum _ { { y }^{ \prime  } }{ p\left( x|{ y }^{ \prime  } \right) p\left( { y }^{ \prime  } \right)  }  } 
$$

则朴素贝叶斯分类器表示为：

$$
f\left( x \right) =arg\max _ { y\in \mathcal Y }{ \frac { p\left( y \right) \prod _ { j=1 }^{ n }{ p\left( { x }_ { j }|y \right)  }  }{ \sum _ { { y }^{ \prime  } }{ p\left( x|{ y }^{ \prime  } \right) p\left( { y }^{ \prime  } \right)  }  }  } 
$$

由于上式的分母 $p\left( x \right) $ 与 $y$ 的取值无关，则分类器重写为：

$$
f\left( x \right) =arg\max _ { y\in \mathcal Y }{ p\left( y \right) \prod _ { j=1 }^{ n }{ p\left( { x }_ { j }|y \right)  }  } 
$$


### 算法
&emsp;&emsp;在朴素贝叶斯中，学习意味着估计概率：$p\left( y \right) $，$p\left( { x }  _  { i }|y \right) $。可以用极大似然，估计相应概率。先验概率 $p\left( y \right) $ 的极大似然估计为：

$$
p\left( y={ c }_ { k } \right) =\frac { \sum _ { i=1 }^{ m }{ I\left( { \tilde { y }  }_ { i }={ c }_ { k } \right)  }  }{ m } 
$$

&emsp;&emsp;设第 $j$ 个特征 ${ x }  _  { j }$ 可能的取值为 $\{ { a }  _  { j,1 },{ a }  _  { j,2 },\cdots ,{ a }  _  { j,{ s }  _  { j } } \}$，则条件概率 $p\left( { x }  _  { j }={ a }  _  { j,l }|y={ c }  _  { k } \right) $ 的极大似然估计为：

$$
p\left( { x }_ { j }={ a }_ { j,l }|y={ c }_ { k } \right) =\frac { \sum _ { i=1 }^{ m }{ I\left( { x }_ { i,j }={ a }_ { j,l }|{ \tilde { y }  }_ { i }={ c }_ { k } \right)  }  }{ \sum _ { i=1 }^{ m }{ I\left( { \tilde { y }  }_ { i }={ c }_ { k } \right)  }  } 
$$

其中：$j=1,2,\cdots ,n$；$l=1,2,\cdots ,{ s }  _  { j }$；$k=1,2,\cdots ,K$；$I$ 为示性函数；${ x }  _  { i,j }$ 表示第 $i$ 个样本的第 $j$ 个特征。

### 伪码        
输入
1. 训练集$\mathbb D=\{ \left( { x }  _  { 1 },{ y }  _  { 1 } \right) ,\left( { x }  _  { 2 },{ y }  _  { 2 } \right) ,\cdots ,\left( { x }  _  { m },{ y }  _  { m } \right)  \}$。${ x }  _  { i }={ \left( { x }  _  { i,1 }{ x }  _  { i,2 },\cdots ,{ x }  _  { i,n } \right)  }^{ T }$，${ x }  _  { i,j }$ 表示第 $i$ 个样本的第 $j$ 个特征。其中 ${ x }  _  { i,j }\in \{ { a }  _  { j,1 },{ a }  _  { j,2 },\cdots ,{ a }  _  { j,{ s }  _  { j } } \} $，${ a }  _  { j,l }$ 为第 $j$ 个特征可能取到的第 $l$ 个值。
2. 实例 $x$。
            
输出：实例 $x$ 的分类<br>
        
算法步骤
1. 计算先验概率以及条件概率：

$$
p\left( y={ c }_ { k } \right) =\frac { \sum _ { i=1 }^{ m }{ I\left( { \tilde { y }  }_ { i }={ c }_ { k } \right)  }  }{ m } 
$$
   其中，$k=1,2,\cdots ,K$；
$$
p\left( { x }_ { j }={ a }_ { j,l }|y={ c }_ { k } \right) =\frac { \sum _ { i=1 }^{ m }{ I\left( { x }_ { i,j }={ a }_ { j,l }|{ \tilde { y }  }_ { i }={ c }_ { k } \right)  }  }{ \sum _ { i=1 }^{ m }{ I\left( { \tilde { y }  }_ { i }={ c }_ { k } \right)  }  } 
$$
   其中：$j=1,2,\cdots ,n$；$l=1,2,\cdots ,{ s }  _  { j }$；$k=1,2,\cdots ,K$。
2. 对于给定的实例 $x={ \left( { x }  _  { 1 },{ x }  _  { 2 },\cdots ,{ x }  _  { n } \right)  }^{ T }$，计算：$p\left( y={ c }  _  { k } \right) \prod   _  { j=1 }^{ n }{ p\left( { x }  _  { j }|y={ c }  _  { k } \right)  } $。
3. 确定实例 $x$ 的分类：$\hat { y } =arg\max   _  { { c }  _  { k } }{ p\left( y={ c }  _  { k } \right) \prod   _  { j=1 }^{ n }{ p\left( { x }  _  { j }|y={ c }  _  { k } \right)  }  } $。

### 优点
1. 性能好，速度快，可以避免维度灾难
2. 支持大规模数据的并行学习，且天然的支持增量学习
3. 既可用于二分类又可用于多分类
4. 算法简单、所估计参数少、对缺失数据不太敏感
5. 对小规模数据表现很好

### 缺点
~~1. 无法给出分类概率，因此难以应用于需要分类概率的场景。~~
2. 朴素贝叶斯假设样本各个特征之间相互独立，这个假设在实际应用中往往不成立，从而影响分类正确性
3. 对数据输入的表达形式很敏感

### 贝叶斯估计
&emsp;&emsp;在估计概率 $p\left( { x }  _  { i }|y \right) $ 的过程中，分母 $\sum   _  { i=1 }^{ m }{ I\left( { \tilde { y }  }  _  { i }={ c }  _  { k } \right)  } $ 可能为 $0$。这是由于训练样本太少才导致 ${ c }  _  { k }$ 的样本数为 $0$。而真实的分布中，${ c }  _  { k }$ 的样本并不为 $0$。

&emsp;&emsp;解决的方案是采用贝叶斯估计（最大后验估计）。假设第 $ j $ 个特征 ${ x }  _  { j }$ 可能的取值为 $\{ { a }  _  { j,1 },{ a }  _  { j,2 },\cdots ,{ a }  _  { j,{ s }  _  { j } } \} $，贝叶斯估计假设在每个取值上都有一个先验的计数 $\lambda$。即：

$$
{ p }_{ \lambda  }\left( { x }_{ j }={ a }_{ j,l }|y={ c }_{ k } \right) =\frac { \sum _{ i=1 }^{ m }{ I\left( { x }_{ i,j }={ a }_{ j,l }|{ \tilde { y }  }_{ i }={ c }_{ k } \right)  } +\lambda  }{ \sum _{ i=1 }^{ m }{ I\left( { \tilde { y }  }_{ i }={ c }_{ k } \right)  } +{ s }_{ j }\lambda  } 
$$

其中，$j=1,2,\cdots ,n$；$l=1,2,\cdots ,{ s }  _  { j }$；$k=1,2,\cdots ,K$。它等价于在 ${ x }  _  { j }$ 的各个取值的频数上赋予了一个正数 $\lambda$。若 ${ c }  _  { k }$ 的样本数为 $0$，则它假设特征 ${ x }  _  { j }$ 每个取值的概率为 $\frac { 1 }{ { s }  _  { j } } $，即等可能的。

&emsp;&emsp;采用贝叶斯估计后，$p\left( y \right) $ 的贝叶斯估计调整为:

$$
{ p }_ { \lambda  }\left( y={ c }_ { k } \right) =\frac { \sum _ { i=1 }^{ m }{ I\left( { \tilde { y }  }_ { i }={ c }_ { k } \right)  } +\lambda  }{ m+K\lambda  } 
$$

其中，$K$为类别个数。当 $\lambda=0$ 时，为极大似然估计；当 $\lambda=1$ 时，为拉普拉斯平滑。若 ${ c }  _  { k }$ 的样本数为 $0$，则假设赋予它一个非零的概率 $\frac { \lambda  }{ m+K\lambda  } $。

## 半朴素贝叶斯分类器
&emsp;&emsp;朴素贝叶斯法对条件概率做了特征的独立性假设：$p\left( x|y \right) =p\left( { x }  _  { 1 },{ x }  _  { 2 },\cdots ,{ x }  _  { n }|y \right) =\prod   _  { j=1 }^{ n }{ p\left( { x }  _  { j }|y \right)  } $。但是现实任务中这个假设有时候很难成立。若对特征独立性假设进行一定程度上的放松，这就是半朴素贝叶斯分类器 semi-naive Bayes classifiers。

&emsp;&emsp;半朴素贝叶斯分类器原理：适当考虑一部分特征之间的相互依赖信息，从而既不需要进行完全联合概率计算，又不至于彻底忽略了比较强的特征依赖关系。
    

### 独依赖估计 OED
&emsp;&emsp;独依赖估计 ($One-Dependent Estimator:OED$) 是半朴素贝叶斯分类器最常用的一种策略。它假设每个特征在类别之外最多依赖于一个其他特征，即：

$$
p\left( x|y \right) =p\left( { x }_ { 1 },{ x }_ { 2 },\cdots ,{ x }_ { n }|y \right) =\prod _ { j=1 }^{ n }{ p\left( { x }_ { j }|y,{ x }_ { j }^{ P } \right)  } 
$$

其中，${ x }  _  { j }^{ P }$ 为特征 ${ x }  _  { j }$ 所依赖的特征，称作的 ${ x }  _  { j }$ 父特征。

&emsp;&emsp;如果父属性已知，那么可以用贝叶斯估计来估计概率值 $p\left( { x }  _  { j }|y,{ x }  _  { j }^{ P } \right) $。现在的问题是：如何确定每个特征的父特征？不同的做法产生不同的独依赖分类器。
    
#### SPODE
&emsp;&emsp;最简单的做法是：假设所有的特征都依赖于同一个特征，该特征称作超父。然后通过交叉验证等模型选择方法来确定超父特征。这就是 ($SPODE:Super-Parent ODE$) 方法。假设节点 $Y$ 代表输出变量 $y$ ，节点 $Xj$ 代表属性 ${ x }  _  { j }$。下图给出了超父特征为 ${ x }  _  { 1 }$ 时的 $SPODE$。


<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/NaiveBayesian/spode.png?raw=true"
    width="720" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> SPODE </div>
</center>

    
#### TAN
&emsp;&emsp;$TAN$ ($Tree Augmented naive Bayes$) 是在最大带权生成树算法基础上，通过下列步骤将特征之间依赖关系简化为如下图所示的树型结构：
1. 计算任意两个特征之间的条件互信息。记第 $i$ 个特征 ${ x }  _  { i }$ 代表的结点为 $\mathbf X  _  i$，标记代表的节点为 $\mathbf{Y}$ 则有:
   $$
   I\left( { X }_ { i },{ X }_ { j }|Y \right) =\sum _ { y }{ \sum _ { { x }_ { i } }{ \sum _ { { x }_ { j } }{ p\left( { x }_ { i },{ x }_ { j }|y \right) \log { \frac { p\left( { x }_ { i },{ x }_ { j }|y \right)  }{ p\left( { x }_ { i }|y \right) p\left( { x }_ { j }|y \right)  }  }  }  }  } 
   $$
   如果两个特征 ${ x }  _  { i }$ 和 ${ x }  _  { j }$ 相互条件独立，则 $p\left( { x }  _  { i },{ x }  _  { j }|y \right) =p\left( { x }  _  { i }|y \right) p\left( { x }  _  { j }|y \right) $。则有条件互信息 $I\left( { X }  _  { i },{ X }  _  { j }|Y \right) =0$，则在图中这两个特征代表的结点没有边相连。
2. 以特征为结点构建完全图，任意两个结点之间边的权重设为条件互信息 $I\left( { X }  _  { i },{ X }  _  { j }|Y \right) =0$。
3. 构建此完全图的最大带权生成树，挑选根结点（下图中根节点为节点 $\mathbf X  _  1$)，将边置为有向边。
4. 加入类别结点 $\mathbf Y$，增加 $\mathbf Y$ 到每个特征的有向边。因为所有的条件概率都是以 $y$ 为条件的。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/NaiveBayesian/tan.png?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> TAN </div>
</center>
       
## 其它讨论

