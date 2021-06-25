---
layout: post
title: 机器学习简介
categories: [MachineLearning]
description: 机器学习简介
keywords: MachineLearning
---


机器学习简介
---


## 方法概论
1. 机器学习的对象是具有一定的统计规律的数据。
2. 机器学习根据任务类型，可以划分为：
   1. 监督学习：从已标记的训练数据来训练模型。 主要分为：分类任务、回归任务、序列标注任务。
   2. 无监督学习：从未标记的训练数据来训练模型。主要分为：聚类任务、降维任务。
   3. 半监督学习：用大量的未标记训练数据和少量的已标记数据来训练模型。
   4. 强化学习：从系统与环境的大量交互知识中训练模型。
3. 机器学习根据算法类型，可以划分为：
   1. 传统统计学习<br>
      基于数学模型的机器学习方法。包括 $SVM$、逻辑回归、决策树等。
      这一类算法基于严格的数学推理，具有可解释性强、运行速度快、可应用于小规模数据集的特点。
   2. 深度学习<br>
      基于神经网络的机器学习方法。包括前馈神经网络、卷积神经网络、递归神经网络等。这一类算法基于神经网络，可解释性较差，强烈依赖于数据集规模。但是这类算法在语音、视觉、自然语言等领域非常成功。
        
4. 没有免费的午餐定理($No \ Free \ Lunch \ Theorem:NFL$)：对于一个学习算法 $A$，如果在某些问题上它比算法 $B$ 好，那么必然存在另一些问题，在那些问题中 $B$ 比 $A$ 更好。所以不存在在所有的问题上都取得最佳的性能的算法。因此要谈论算法的优劣必须基于具体的学习问题。

## 基本概念
### 特征空间
1. 输入空间：所有输入的可能取值；

   输出空间：所有输出的可能取值。
   
   特征向量：表示每个具体的输入， 所有特征向量构成特征空间。
2. 特征空间的每一个维度对应一种特征。
3. 可以将输入空间等同于特征空间，但是也可以不同。
   
   绝大多数情况下，输入空间等于特征空间。
   
   模型是定义在特征空间上的。

### 样本表示
通常输入实例用 $X$ 表示，真实标记用 $\tilde y$ 表示，模型的预测值用 $\hat y$ 表示。具体的输入取值记作 ${ X }  _  { 1 }{ ,X }  _  { 2 },\cdots,{ X }  _  { n }$；具体的标记取值记作 $\tilde y  _  1,\tilde y  _  2,\cdots $；具体的模型预测取值记作 $\hat y  _  1, \hat y  _  2,\cdots$。

所有的向量均为列向量，其中输入实例$X$的特征向量记作（假设特征空间为 $ n $ 维）：

$$
 X=\left[ \begin{matrix} \begin{matrix} { x }^{ \left( 1 \right)  } \\ { x }^{ \left( 2 \right)  } \end{matrix} \\ \begin{matrix} \vdots  \\ { x }^{ \left( n \right)  } \end{matrix} \end{matrix} \right] 
$$

这里 ${ x }^{ \left( j \right)  }$ 为 $X$ 的第 $j$ 个特征的取值。第 $i$ 个输入记作 ${ X }  _  { i }$。

训练数据由输入向量、目标值对组成。通常训练集表示为：$D=\{ \left( { X }  _  { 1 },{ \tilde { y }  }  _  { 1 } \right) ,\left( { X }  _  { 2 },{ \tilde { y }  }  _  { 2 } \right) ,\cdots ,\left( { X }  _  { N },{ \tilde { y }  }  _  { N } \right)  \} $。
1. 输入向量、目标值对又称作样本点。
2. 假设每对输入向量、目标值对是独立同分布产生的。

输入向量 $X$ 和目标值 $\tilde y$ 可以是连续的，也可以是离散的。
1. $\tilde y$ 为连续的：这一类问题称为回归问题。
2. $\tilde y$ 为离散的，且是有限的：这一类问题称之为分类问题。
3. $X$ 和 $\tilde y$ 均为序列：这一类问题称为序列标注问题。

## 监督学习
### 基本概念
1. 训练数据中的每个样本都含有标记，该标记由人工给出，所以称之为监督学习。
2. 监督学习假设输入向量 $X$ 与目标值 $\tilde y$ 遵循联合概率分布 $P\left( X,y \right) $，训练数据和测试数据以联合概率分布 $P\left( X,y \right) $ 独立同分布产生。学习过程中，假定这个联合概率分布存在，但是具体定义未知。
3. 监督学习的目的在于学习一个由输入到输出的映射，该映射由模型表示。模型属于由输入空间到输出空间的映射的集合，该集合就是解空间。解空间的确定意味着学习范围的确定。
4. 监督学习模型可分为概率模型或者非概率模型：
   1. 概率模型由条件概率分布 $P\left(y \| X \right) $ 表示。
   2. 非概率模型由决策函数 $y=f\left( X \right) $ 表示。
5. 监督学习分为学习和预测两个过程。给定训练集 $D=\{ \left( { X }  _  { 1 },{ \tilde { y }  }  _  { 1 } \right) ,\left( { X }  _  { 2 },{ \tilde { y }  }  _  { 2 } \right) ,\cdots ,\left( { X }  _  { N },{ \tilde { y }  }  _  { N } \right)  \} $，其中 ${ X }  _  { i }\in {\mathcal X} $ 为输入值，${ \tilde { y }  }  _  { i }\in {\mathcal Y}$ 是目标值。假设训练数据与测试数据是以联合概率分布 $P\left( X,y \right) $ 独立同分布的产生的。
   1. 学习过程：在给定的训练集 $D$ 上，通过学习训练得到一个模型。该模型表示为条件概率分布 $P\left( y \| X \right) $ 或者决策函数 $y=f\left( X \right) $。
   2. 预测过程：对给定的测试样本 ${ X }  _  { test }$，给出其预测结果：
      1. 对于概率模型，其预测值为：${ \hat { y }  }  _  { test }={ arg }  _  { y }\max { p\left( y \| X \right)  } $
      2. 对于非概率模型，其预测值为：${ \hat { y }  }  _  { test }=f\left( { X }  _  { test } \right) $
6. 可以通过无监督学习来求解监督学习问题 $p\left( y \| X \right) $：
   1. 首先求解无监督学习问题来学习联合概率分布 $p=\left( X,y \right) $
   2. 然后计算：$p\left( y \| X \right) =\frac { p=\left( X,y \right)  }{ \sum   _  { { y }^{ ' } }{ p=\left( X,{ y }^{ ' } \right)  }  } $。

### 生成模型和判别模型
监督学习又分为生成方法和判别方法，所用到的模型分别称为生成模型和判别模型。

#### 生成方法
1. 概念：通过数据学习联合概率分布 $p=\left( X,y \right) $，然后求出条件概率分布 $p\left( y \| X \right)$ 作为预测的模型。即生成模型为：

$$
p\left( y \| X \right) =\frac { p=\left( X,y \right)  }{ p\left( x \right)  } 
$$

2. 优点：可以还原联合概率分布 $p=\left( X,y \right) $，收敛速度快，当存在隐变量时只能用生成方法。
3. 举例：朴素贝叶斯，隐马尔可夫链。

#### 判别方法
1. 概念：直接学习决策函数 $f\left( X \right) $ 或者条件概率分布 $P\left(y \| X \right) $ 的模型。
2. 优点：直接预测，一般准确率更高，且一般比较简化问题。
3. 举例：逻辑回归，决策树。

## 机器学习三要素
机器学习三要素：模型、策略、算法。

### 模型
1. 模型定义了解空间。监督学习中，模型就是要学习的条件概率分布或者决策函数。模型的解空间包含了所有可能的条件概率分布或者决策函数，因此解空间中的模型有无穷多个。
   1. 模型为条件概率分布：<br>
      解空间为条件概率的集合：$F=\{ p|p\left( y \| X \right)  \} $。其中 ${ X }  _  { i }\in {\mathcal X} $ 为输入空间，${ \tilde { y }  }  _  { i }\in {\mathcal Y}$ 是输出空间。<br>
      通常$F$是由一个参数向量 $\theta =\left( { \theta  }  _  { 1 },\cdots ,{ \theta  }  _  { n } \right) $ 决定的概率分布族：$F=\{ p|{ p }  _  { \theta  }\left( y|X \right) ,\theta \in R \} $。其中：${ p }  _  { \theta  }$ 只与 $\theta$ 有关，称 $\theta$ 为参数空间。
   2. 模型为一个决策函数：<br>
      解空间为决策函数的集合：$F=\{ f|y=f\left( X \right)  \} $。其中 ${ X }  _  { i }\in {\mathcal X} $ 为输入空间，${ \tilde { y }  }  _  { i }\in {\mathcal Y}$ 是输出空间。<br>
      通常 $F$ 是由一个参数向量 $\theta =\left( { \theta  }  _  { 1 },\cdots ,{ \theta  }  _  { n } \right) $ 决定的函数族：$F=\{ f|y={ f }  _  { \theta  }\left( X \right) ,\theta \in R \} $。其中：${ f }  _  { \theta  }$ 只与 $\theta$ 有关，称 $\theta$ 为参数空间。
2. 解的表示一旦确定，解空间以及解空间的规模大小就确定了。
3. 可将学习过程看作一个在解空间中进行搜索的过程，搜索目标就是找到与训练集匹配的解。

### 策略
策略定义了优化目标。

#### 损失函数
1. 对于给定的输入向量 $X$，由模型预测的输出值 $\hat { y } $ 与目标值 $\tilde { y } $ 可能不一致。此时，用损失函数度量错误的程度，记作 $L\left( \tilde { y } ,\hat { y }  \right) $，也称作代价函数。
2. 常用损失函数
   1. $0-1$ 损失函数
      $$
      L\left( \tilde { y } ,\hat { y }  \right) =\begin{cases} 1,\tilde { y } =\hat { y }  \\ 0,\tilde { y } \neq \hat { y }  \end{cases}
      $$
   2. 平方损失函数$MSE$
      $$
      L\left( \tilde { y } ,\hat { y }  \right) ={ \left( \tilde { y } -\hat { y }  \right)  }^{ 2 }
      $$
   3. 绝对损失函数$MSE$
      $$
      L\left( \tilde { y } ,\hat { y }  \right) =\left| \tilde { y } -\hat { y }  \right| 
      $$
   4. 对数损失函数
      $$
      L\left( \tilde { y } ,\hat { y }  \right) =-\log { p\left( y|X \right)  } 
      $$ 
3. 训练时采用的损失函数不一定是评估时的损失函数。但通常二者是一致的因为目标是需要预测未知数据的性能足够好，而不是对已知的训练数据拟合最好。

#### 风险函数
1. 通常损失函数值越小，模型就越好。但是由于模型的输入值和目标值都是随机变量，遵从联合分布 $p=\left( X,y \right)$，因此定义风险函数为损失函数的期望：
   $$
   { R }_ { exp }={ E }_ { p }\left[ L\left( \tilde { y } ,\hat { y }  \right)  \right] =\int _ { {\mathcal X}\times {\mathcal Y} }{ L\left( \tilde { y } ,\hat { y }  \right) p\left( X,y \right) dXdy }  
   $$
   其中 ${\mathcal X} $ 和 ${\mathcal Y}$ 分别为输入空间和输出空间。
2. 学习的目标是找出使风险函数最小的模型。
3. 求 ${ R }  _  { exp }$ 的过程中要用到 $p\left( X,y \right) $，但 $ p\left( X,y \right) $ 是未知的。实际上如果它已知，则可以轻而易举求得条件概率分布，也就不需要学习。

#### 经验风险
1. 经验风险也叫经验损失。给定训练集 $D=\{ \left( { X }  _  { 1 },{ \tilde { y }  }  _  { 1 } \right) ,\left( { X }  _  { 2 },{ \tilde { y }  }  _  { 2 } \right) ,\cdots ,\left( { X }  _  { N },{ \tilde { y }  }  _  { N } \right)  \} $，模型关于 $D$ 的经验风险定义为：
   $$
   { R }_ { emp }=\frac { 1 }{ N } \sum _ { i=1 }^{ N }{ L\left( { \tilde { y }  }_ { i },{ \hat { y }  }_ { i } \right)  } 
   $$
   经验风险最小化策略认为：经验风险最小的模型就是最优的模型。即：
   $$
   \min _ { f\in F }{ \frac { 1 }{ N } \sum _  { i=1 }^{ N }{ L\left( { \tilde { y }  }_ { i },{ f\left( { X }_ { i } \right)  } \right)  }  } 
   $$
2. 经验风险是模型在训练数据集上的平均损失。根据大数定律，当 $N\rightarrow \infty $时${ R }  _  { emp }\rightarrow { R }  _  { exp }$。但是由于现实中训练集的样本数量有限，甚至很小，所以需要对经验风险进行矫正。

#### 结构风险
1. 结构风险是在经验风险上加入正则化项（或者称之为罚项）。它是为了防止过拟合而提出的。给定训练集 $D=\{ \left( { X }  _  { 1 },{ \tilde { y }  }  _  { 1 } \right) ,\left( { X }  _  { 2 },{ \tilde { y }  }  _  { 2 } \right) ,\cdots ,\left( { X }  _  { N },{ \tilde { y }  }  _  { N } \right)  \} $，模型关于 $D$ 的结构风险定义为：
   $$
   { R }_ { srm }=\frac { 1 }{ N } \sum _ { i=1 }^{ N }{ L\left( { \tilde { y }  }_ { i },{ \hat { y }  }_ { i } \right)  } +\lambda J\left( f \right) 
   $$
   其中：$J\left( f \right) $ 为模型复杂度，是定义在解空间 $F$ 上的泛函。$f$ 越复杂，则 $J\left( f \right)$ 越大。$\lambda \ge 0$ 为系数，用于权衡经验风险和模型复杂度。
2. 结构风险最小化策略认为：结构风险最小的模型是最优的模型。即：
   $$
   \min _ { f\in F }{ \frac { 1 }{ N } \sum _ { i=1 }^{ N }{ L\left( { \tilde { y }  }_ { i },{ f\left( { X }_ { i } \right)  } \right)  }  } +\lambda J\left( f \right) 
   $$
3. 结构风险最小化策略符合奥卡姆剃刀原理：能够很好的解释已知数据，且十分简单才是最好的模型。

#### 极大似然估计
1. 极大似然估计就是经验风险最小化的例子。
    
2.  已知训练集 $D=\{ \left( { X }  _  { 1 },{ \tilde { y }  }  _  { 1 } \right) ,\left( { X }  _  { 2 },{ \tilde { y }  }  _  { 2 } \right) ,\cdots ,\left( { X }  _  { N },{ \tilde { y }  }  _  { N } \right)  \} $，出现这种训练集的概率为：$\prod   _  { i=1 }^{ N }{ p\left( { \tilde { y }  }  _  { i }|X \right)  } $。根据 $D$ 出现概率最大，有：
    $$
    \max { \prod _ { i=1 }^{ N }{ p\left( { \tilde { y }  }_ { i }|X \right)  }  } \rightarrow \max { \sum _ { i=1 }^{ N }{ \log { p\left( { \tilde { y }  }_ { i }|X \right)  }  }  } \rightarrow \min { \sum _ { i=1 }^{ N }{ \left( -\log { p\left( { \tilde { y }  }_ { i }|X \right)  }  \right)  }  } 
    $$
    定义损失函数为：$L\left( \tilde { y } ,\hat { y }  \right) =-\log { p\left( y|X \right)  } $，则有：
    $$
    \min { \sum _ { i=1 }^{ N }{ \left( -\log { p\left( { \tilde { y }  }_ { i }|X \right)  }  \right)  }  } \rightarrow \min { \sum _{ i=1 }^{ N }{ L\left( { \tilde { y }  }_{ i },{ \hat { y }  }_ { i } \right)  }  } \rightarrow \min { \frac { 1 }{ N } \sum _{ i=1 }^{ N }{ L\left( { \tilde { y }  }_ { i },{ \hat { y }  }_ { i } \right)  }  } 
    $$
    即极大似然估计 = 经验风险最小化 。
    
#### 最大后验概率估计
1. 最大后验概率估计就是结构风险最小化的例子。
2. 已知训练集 $D=\{ \left( { X }  _  { 1 },{ \tilde { y }  }  _  { 1 } \right) ,\left( { X }  _  { 2 },{ \tilde { y }  }  _  { 2 } \right) ,\cdots ,\left( { X }  _  { N },{ \tilde { y }  }  _  { N } \right)  \} $，出现这种训练集的概率为：$\prod   _  { i=1 }^{ N }{ p\left( { \tilde { y }  }  _  { i }|X \right)  } g\left( \theta  \right) $。根据 $D$ 出现概率最大：
   $$
   \max { \prod _ { i=1 }^{ N }{ p\left( { \tilde { y }  }_ { i }|X \right)  } g\left( \theta  \right)  } \rightarrow \max { \sum _ { i=1 }^{ N }{ \log { p\left( { \tilde { y }  }_ { i }|X \right)  }  } +\log { g\left( \theta  \right)  }  } \rightarrow \min { \sum _{ i=1 }^{ N }{ \left( -\log { p\left( { \tilde { y }  }_{ i }|X \right)  }  \right)  }  } +\log { \frac { 1 }{ g\left( \theta  \right)  }  } 
   $$
   定义损失函数为：$L\left( \tilde { y } ,\hat { y }  \right) =-\log { p\left( y|X \right)  } $；定义模型复杂度为 $J\left( f \right) =\log { \frac { 1 }{ g\left( \theta  \right)  }  } $；定义正则化系数为 $\lambda =\frac { 1 }{ N } $。则有：
   $$
   \min { \sum _ { i=1 }^{ N }{ \left( -\log { p\left( { \tilde { y }  }_ { i }|X \right)  }  \right)  }  } +\log { \frac { 1 }{ g\left( \theta  \right)  }  } \rightarrow \min { \sum _ { i=1 }^{ N }{ L\left( { \tilde { y }  }_ { i },{ \hat { y }  }_ { i } \right)  }  } +J\left( f \right) \rightarrow \min { \frac { 1 }{ N } \sum _ { i=1 }^{ N }{ L\left( { \tilde { y }  }_ { i },{ \hat { y }  }_ { i } \right)  }  } +\lambda J\left( f \right) 
   $$
   即：最大后验概率估计 = 结构风险最小化。

### 算法
算法指学习模型的具体计算方法。通常采用数值计算的方法求解，如：梯度下降法。