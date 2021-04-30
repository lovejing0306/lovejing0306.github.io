---
layout: post
title: 优化算法之费马定理(解析解)
categories: [BasicKnowledge]
description: 优化算法之费马定理(解析解)
keywords: BasicKnowledge
---


深度学习基础知识点优化算法之费马定理(解析解)
---


## 费马定理
&emsp;&emsp;对于一个可导函数，寻找其极值的统一做法是寻找导数为 $0$ 的点，即费马定理。微积分中的这一定理指出，对于可导函数，在极值点处导数必定为 $0$：

$$
f^{ \prime  }\left( x \right) =0
$$

对于多元函数，则是梯度为0：

$$
\nabla f\left( x \right) =0
$$

导数为$0$的点称为驻点。需要注意的是，导数为 $0$ 只是函数取得极值的必要条件而不是充分条件，它只是疑似极值点。是不是极值，是极大值还是极小值，还需要看更高阶导数。

&emsp;&emsp;对于一元函数，假设$x$是驻点：
* 如果 $f^{ \prime \prime  }\left( x \right) >0$，则在该点处去极小值
* 如果 $f^{ \prime \prime  }\left( x \right) <0$，则在该点处去极大值

&emsp;&emsp;对于多元函数，假设 $x$ 是驻点：
* 如果 $Hessian$ 矩阵在梯度为零的位置上的特征值全为正时，该函数得到局部最小值
* 如果 $Hessian$ 矩阵在梯度为零的位置上的特征值全为负时，该函数得到局部最大值
* 如果 $Hessian$ 矩阵在梯度为零的位置上的特征值有正有负时，该函数得到鞍点

&emsp;&emsp;函数在梯度为零的位置上可能是局部最小值、局部最大值或者鞍点。举个例子，给定函数:

$$
f\left( x \right) ={ x }^{ 3 }
$$

我们可以找出该函数的鞍点位置。
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Optimization/saddle _ point1.jpg?raw=true"
    width="320" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">saddle _ point1</div>
</center>

再举个定义在二维空间的函数的例子，例如:

$$
f\left( x,y \right) ={ x }^{ 2 }-{ y }^{ 2 }
$$

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Optimization/saddle _ point2.jpg?raw=true"
    width="320" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">saddle _ point2</div>
</center>

我们可以找出该函数的鞍点位置。该函数看起来像一个马鞍，而鞍点恰好是马鞍上可坐区域的中心。在上图的鞍点位置，目标函数在$x$ 轴方向上是局部最小值，而在 $y$ 轴方向上是局部最大值。

&emsp;&emsp;假设一个函数的输入为 $k$ 维向量，输出为标量，那么它的 $Hessian$ 矩阵有 $k$ 个特征值。随机矩阵理论告诉我们，对于一个大的高斯随机矩阵来说，任一特征值是正或者是负的概率都是 $0.5$。那么，以上第一种情况的概率为 ${ 0.5 }^{ k }$。由于深度学习模型参数通常都是高维的($k$ 很大)，目标函数的鞍点通常比局部最小值更常见。

&emsp;&emsp;除鞍点外，最优化算法可能还会遇到另外一个问题：局部极值问题，即一个驻点是极值点，但不是全局极值。如果我们对最优化问题加以限定，可以有效的避免这两种问题。典型的是凸优化，它要求优化变量的可行域是凸集，目标函数是凸函数。

&emsp;&emsp;虽然驻点只是函数取得极值的必要条件而不是充分条件，但如果我们找到了驻点，再判断和筛选它们是不是极值点，比之前要容易多了。无论是理论结果，还是数值优化算法，一般都以找驻点作为找极值点的目标。对于一元函数，先求导数，然后解导数为$0$的方程即可找到所有驻点。对于多元函数，对各个自变量求偏导数，令它们为$0$，解方程组，即可达到所有驻点。

### 拉格朗日乘数法
&emsp;&emsp;费马定理给出的是没有约束条件下的函数极值的必要条件。对于一些实际应用问题，一般还带有等式或者不等式约束条件。对于带等式约束的极值问题，经典的解决方案是拉格朗日乘数法。

&emsp;&emsp;对于如下问题：

$$
\begin{aligned} &\min { f\left( x \right)  }  \\ &{ h }_ { i }\left( x \right) ,i=1,...,p \end{aligned}
$$

构造拉格朗日乘子函数：

$$
L\left( x,\lambda  \right) =f\left( x \right) +\sum _ { i=1 }^{ p }{ { \lambda  }_{ i }{ h }_{ i }\left( x \right)  } 
$$

在最优点处对 $x$ 和乘子变量 ${ \lambda  }  _  { i }$ 的导数都必须为 $0$：

$$
\begin{aligned} { \nabla  }_ { x }f+\sum _ { i=1 }^{ p }{ { \lambda  }_{ i }{ \nabla  }_{ x }{ h }_{ i }=0 }  \\ { h }_{ i }\left( x \right) =0 \end{aligned}
$$

解这个方程即可得到最优解。

&emsp;&emsp;机器学习中用到拉格朗日乘数法的地方有：
* 主成分分析
* 线性判别分析
* 流形学习中的拉普拉斯特征映射
* 隐马尔可夫模型


### 拉格朗日对偶
对偶也是一种最优化方法，它将一个最优化问题转换成另外一个等价的最优化问题，拉格朗日对偶是其中一个典型例子。对于如下带等式约束和不等式约束的优化问题：

$$
\begin{aligned} 
&\min { f\left( x \right)  }  \\ 
&{ g }_{ i }\left( x \right) \le 0\quad i=1,...m \\ 
&{ h }_{ i }\left( x \right) =0\quad i=1,...p \end{aligned}
$$

与拉格朗日乘数法类似，构造广义拉格朗日函数：
$$
L\left( x,\lambda ,\mu  \right) =f\left( x \right) +\sum _ { i=1 }^{ m }{ { \lambda  }_ { i }{ g }_ { i }\left( x \right)  } +\sum _ { i=1 }^{ p }{ { \mu  }_{ i }{ h }_{ i }\left( x \right)  } 
$$

$\lambda _ i$必须满足$\lambda _ i\ge 0$的约束。原问题为：

$$
\min _ x\max _ {\lambda ,\mu ,\lambda _ i\ge 0}L\left( x,\lambda ,\mu \right) 
$$

即先固定住 $x$，调整拉格朗日乘子变量，让函数 $L$ 取极大值；然后控制变量 $x$，让目标函数取极小值。

对偶问题为：
$$
\max _ {\lambda ,\mu ,\lambda _ i\ge 0}\min _ xL\left( x,\lambda ,\mu \right) 
$$

和原问题相反，这里是先控制变量 $x$，让函数 $L$ 取极小值；然后控制拉格朗日乘子变量，让函数取极大值。

一般情况下，原问题的最优解大于等于对偶问题的最优解，这称为弱对偶。在某些情况下，原问题的最优解和对偶问题的最优解相等，这称为强对偶。

强对偶成立的一种条件是 $Slater$ 条件：一个凸优化问题，如果 存在一个候选$x$ 使得所有不等式约束都是严格满足的，即对于所有的 $i$ 都有 $g_ i (x)<0$，不等式不取等号，则强对偶成立， 原问题与对偶问题等价。

> 注意，$Slater$ 条件是强对偶成立的充分条件而非必要条件。

### KKT条件
$KKT$ 条件是拉格朗日乘数法的推广（最优值必须满足 $KKT$ 条件），用于求解既带有等式约束，又带有不等式约束的函数极值。

对于如下优化问题：

$$
\begin{aligned} &\min { f\left( x \right)  }  \\ &{ g }_{ i }\left( x \right) \le 0\quad i=1,...q \\ &{ h }_{ i }\left( x \right) =0\quad i=1,...p \end{aligned}
$$

和拉格朗日对偶的做法类似，$KKT$ 条件构如下乘子函数：

$$
L\left( x,\lambda ,\mu  \right) =f\left( x \right) +\sum _ { j=1 }^{ p }{ { \lambda  }_ { j }{ h }_ { j }\left( x \right)  } +\sum _ { k=1 }^{ q }{ { \mu  }_ { k }{ g }_ { k }\left( x \right)  } 
$$

$\lambda$ 和 $\mu$ 称为 $KKT$ 乘子。在最优解处 ${ x }^{ \ast }$ 应该满足如下条件：

$$
\begin{aligned} 
{ \nabla  }_{ x }L\left( { x }^{ * } \right) =0\\ { \mu  }_{ k }\ge 0\\ { \mu  }_{ k }{ g }_{ k }\left( { x }^{ * } \right) =0\\ { h }_{ j }\left( { x }^{ * } \right) =0\\ { g }_{ k }\left( { x }^{ * } \right) \le 0
\end{aligned}
$$

等式约束 ${ h } _ { j }\left( { x }^{ * } \right) =0$ 和不等式约束 ${ g } _ { k }\left( { x }^{ * } \right) \le 0$ 是本身应该满足的约束，${ \nabla  } _ { x }L\left( { x }^{ * } \right) =0$ 和之前的拉格朗日乘数法一样。唯一多了关于 ${ g } _ { i }\left( x \right)$ 的条件：

$$
{ \mu  }_{ k }{ g }_{ k }\left( { x }^{ * } \right) =0
$$

> KKT条件只是取得极值的必要条件而不是充分条件。

#### 使用KKT条件的例子
* 支持向量机

## 打赏

如果文章对您有帮助，欢迎丢香蕉抛硬币。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/Reward/wechat.JPG?raw=true"
    width="300" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">微信</div>
</center>

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/Reward/zhifubao.JPG?raw=true"
    width="300" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">支付宝</div>
</center>



<script type="text/x-mathjax-config">
  MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/latest.js?config=TeX-MML-AM_CHTML">
</script>