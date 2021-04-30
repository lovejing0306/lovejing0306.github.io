---
layout: post
title: 优化算法之一阶优化算法(数值优化)
categories: [BasicKnowledge]
description: 优化算法之一阶优化算法(数值优化)
keywords: BasicKnowledge
---


深度学习基础知识点优化算法之一阶优化算法(数值优化)
---


## 数值优化
&emsp;&emsp;解析解方法在理论推导、某些可以得到方程组的求根公式的情况（如线性函数，正态分布的最大似然估计）中可以使用，但对绝大多数函数来说，梯度等于$0$的方程组是没法直接解出来的（如方程里面含有指数函数、对数函数之类的超越函数）。对于这种无法直接求解的方程组，只能采用近似的算法来求解，即数值优化算法。

&emsp;&emsp;数值优化算法一般都利用了目标函数的导数信息，如一阶导数和二阶导数。如果采用一阶导数，则称为一阶优化算法。如果使用了二阶导数，则称为二阶优化算法。

&emsp;&emsp;工程上实现时通常采用的是迭代法，从一个初始点 ${ x } _ { 0 }$ 开始，反复使用某种规则从 ${ x } _ { k }$ 移动到下一个点 ${ x } _ { k+1 }$，构造这样一个数列，直到收敛到梯度为 ${ 0 }$ 的点处。即有下面的极限成立：

$$
\lim _ { k\rightarrow +\infty  }{ \nabla f\left( { x }_ { k } \right)  } =0
$$

这些规则一般会利用一阶导数信息即梯度；或者二阶导数信息即$Hessian$矩阵。这样迭代法的核心是得到这样的由上一个点确定下一个点的迭代公式：

$$
{ x }_ { k+1 }=h\left( { x }_ { k } \right) 
$$

## 一阶优化算法
### 一维梯度下降
&emsp;&emsp;以简单的一维梯度下降为例来解释梯度下降算法可能降低目标函数值的原因。假设连续可导的函数 $f:\Re \rightarrow \Re$ 的输入和输出都是标量。给定绝对值足够小的数 $\epsilon$ ，其中 $\epsilon=\Delta x$ 为变化量，根据泰勒展开公式，得到以下的近似:

$$
f\left( x+\epsilon  \right) \approx f\left( x \right) +\epsilon f^{ \prime  }\left( x \right) 
$$

这里 $f^{ \prime  }\left( x \right)$ 是函数 $f$ 在 $x$ 处的梯度。一维函数的梯度是一个标量，也称导数。

可以将上式重写为以下公式：

$$
f\left( x+\epsilon  \right) -f\left( x \right) \approx \epsilon f^{ \prime  }\left( x \right) 
$$

可以继续将上式改写为以下公式：

$$
\Delta f\approx \epsilon \nabla f
$$

梯度下降优化的过程是使 $\Delta f$ 变为负的过程(变成负说明 $f\left( x \right)$ 在逐渐的减小)。

&emsp;&emsp;假设选取：

$$
\epsilon =-\eta \nabla f
$$

其中，$\eta >0$ 是一个很小的正数(称为学习率)。则有：

$$
\Delta f\approx -\eta \nabla f\cdot \nabla f=-\eta { \left\| \nabla f \right\|  }^{ 2 }
$$

由于 ${ \left\| \nabla f \right\|  }^{ 2 }\ge 0$，从而保证了 $\Delta f\le 0$，也就是说，如果按照 $\epsilon =-\eta \nabla f$ 的方式去改变$x$，那么 $f\left( x \right)$ 会一直减小，不会增加。

&emsp;&emsp;因此可以通过 $\epsilon =-\eta \nabla f$:

$$
x\leftarrow x-\eta f^{ \prime  }\left( x \right) 
$$

来迭代 $x$，函数 $f(x)$ 的值可能会降低。

&emsp;&emsp;因此在梯度下降中，我们先选取一个初始值$x$和常数 $\eta >0$，然后不断通过上式来迭代$x$，直到达到停止条件，例如 ${ f^{ \prime  }\left( x \right)  }^{ 2 }$ 的值已足够小或迭代次数已达到某个值。

&emsp;&emsp;更多解释可参见 $Neural \ Network \ and \ Deep \ Learning$ 一书

### 缺点
梯度下降法只能保证找到梯度为$0$的点，不能保证找到极小值点。迭代终止的判定依据是梯度值充分接近于$0$，或者达到最大指定迭代次数。


### 学习率
&emsp;&emsp;上述梯度下降算法中的正数 $\eta$ 通常叫做学习率。这是一个超参数，需要人工设定。如果使用过小的学习率，会导致$x$ 更新缓慢从而需要更多的迭代才能得到较好的解。下面展示了使用学习率 $\eta =0.05$ 时自变量 $x$ 的迭代轨迹。可见，迭代 10 次后，当学习率过小时，最终$x$的值依然与最优解存在较大偏差。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Optimization/lr1.jpg?raw=true"
    width="320" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">learning rate</div>
</center>

如果使用过大的学习率，$\left\| \eta f^{ \prime  }\left( x \right)  \right\|$ 可能会过大从而使前面提到的一阶泰勒展开公式不再成立：这时我们无法保证迭代 $x$ 会降低 $f(x)$ 的值。举个例子，当我们设学习率 $\eta =1.1$ 时，可以看到$x$不断越过（$overshoot$）最优解 $x-0$ 并逐渐发散。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Optimization/lr2.jpg?raw=true"
    width="320" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">learning rate</div>
</center>

### 多维梯度下降
&emsp;&emsp;接下来考虑一种更广义的情况：目标函数的输入为向量，输出为标量。假设目标函数 $f:{ \Re  }^{ d }\rightarrow \Re$ 的输入是一个$d$维向量 $x={ \left[ { x } _ { 1 },{ x } _ { 2 },...,{ x } _ { d } \right]  }^{ T }$。目标函数 $f(x)$ 有关$x$的梯度是一个由 $d$ 个偏导数组成的向量：

$$
{ \nabla  }_ { x }f\left( x \right) ={ \left[ \frac { \partial f\left( x \right)  }{ \partial { x }_ { 1 } } ,\frac { \partial f\left( x \right)  }{ \partial { x }_ { 2 } } ,...,\frac { \partial f\left( x \right)  }{ \partial { x }_ { d } }  \right]  }^{ T }
$$

为表示简洁，用 ${ \nabla  }f\left( x \right)$ 代替 ${ \nabla  } _ { x }f\left( x \right)$。梯度中每个偏导数元素 $\frac { \partial f\left( x \right)  }{ \partial { x } _ { i } }$ 代表着 $f$ 在 $x$ 有关输入 ${ x } _ { i }$ 的变化率。为了测量 $f$ 沿着单位向量 $\mu$（即 $\left\| \mu  \right\| =1$）方向上的变化率，在多元微积分中，我们定义 $f$ 在 $x$ 上沿着 $\mu$ 方向的方向导数为:

$$
{ D } _ { u }f\left( x \right) =\lim _ { h\rightarrow 0 }{ \frac { f\left( x+hu \right) -f\left( x \right) }{ h }  } 
$$

依据方向导数性质，以上的方向导数可以改写为:

$$
{ D }_ { u }f\left( x \right) =\nabla f\left( x \right)\cdot u
$$

方向导数 ${ D } _ { u }f\left( x \right)$ 给出了 $f$ 在 $x$ 上沿着所有可能方向的变化率。为了最小化 $f$，我们希望找到$f$能被降低最快的方向。因此，我们可以通过单位向量 $\mu$来最小化方向导数 ${ D } _ { u }f\left( x \right)$。

由于 ${ D } _ { u }f\left( x \right) =\left\| \nabla f\left( x \right)  \right\| \cdot \left\| \mu  \right\| \cdot \cos { \theta  }$， 其中 $\theta$ 为梯度 $\nabla f\left( x \right)$ 和单位向量 $\mu$ 之间的夹角，当 $\theta =\pi$ 时，$\cos { \theta  }$ 取得最小值 $-1$。因此，当 $\mu$ 在梯度方向 $\nabla f\left( x \right)$ 的相反方向时，方向导数 ${ D } _ { u }f\left( x \right)$ 被最小化。所以，我们可能通过梯度下降算法来不断降低目标函数 $f$ 的值：

$$
x\leftarrow x-\eta \nabla f\left( x \right)
$$

其中 $\eta$（取正数）称作学习率。

### 随机梯度下降
&emsp;&emsp;在深度学习里，目标函数通常是训练数据集中有关各个样本的损失函数的平均。设 ${ f } _ { i }\left( x \right)$ 是有关索引为 $i$ 的训练数据样本的损失函数，$n$ 是训练数据样本数，$x$ 是模型的参数向量，那么目标函数定义为:

$$
f\left( x \right) =\frac { 1 }{ n } \sum _ { i=1 }^{ n }{ { f }_ { i }\left( x \right)  } 
$$

目标函数在$x$处的梯度计算为:

$$
\nabla f\left( x \right) =\frac { 1 }{ n } \sum _ { i=1 }^{ n }{ { \nabla f } _ { i }\left( x \right)  } 
$$

如果直接使用梯度下降，每次自变量迭代的计算开销为 $O\left( n \right)$，它随着 $n$ 线性增长。因此，当训练数据样本数很大时，梯度下降每次迭代的计算开销很高。

&emsp;&emsp;随机梯度下降（$stochastic \ gradient \ descent$，简称 $SGD$ ）减少了每次迭代的计算开销。在随机梯度下降的每次迭代中，我们随机均匀采样得到一个样本索引 $i \in { 1,...,n }$ ，并计算梯度 $\nabla { f } _ { i }\left( x \right)$ 来迭代 $x$：

$$
x\leftarrow x-\eta \nabla { f }_ { i }\left( x \right) 
$$

这里 $\eta$ 同样是学习率。可以看到每次迭代的计算开销从梯度下降的 $O\left( n \right)$ 降到了常数 $O\left( 1 \right)$。值得强调的是，随机梯度 $\nabla { f } _ { i }\left( x \right)$ 是对梯度 $\nabla { f }\left( x \right)$ 的无偏估计：

$$
{ E }_ { i }\nabla { f }_ { i }\left( x \right) =\frac { 1 }{ n } \sum _ { i=1 }^{ n }{ \nabla { f }_ { i }\left( x \right)  } =\nabla f\left( x \right)
$$

这意味着，平均来说，随机梯度是对梯度的一个良好的估计。

### 小批量随机梯度下降
&emsp;&emsp;在每一次迭代中，梯度下降使用整个训练数据集来计算梯度，因此它有时也被称为批量梯度下降（$batch \ gradient \ descent$）。而随机梯度下降在每次迭代中只随机采样一个样本来计算梯度。还可以在每轮迭代中随机均匀采样多个样本来组成一个小批量，然后使用这个小批量来计算梯度。

&emsp;&emsp;设目标函数 $f:{ \Re  }^{ d }\rightarrow \Re$ 。在迭代开始前的时间步设为 $0$。该时间步的自变量记为 ${ x } _ { 0 }\in { \Re  }^{ d }$，通常由随机初始化得到。在接下来的每一个时间步$t>0$中，小批量随机梯度下降随机均匀采样一个由训练数据样本索引所组成的小批量$ { B  } _ { t } $。可以通过重复采样（$sampling \ with \ replacement$）或者不重复采样（$sampling \ without \ replacement$）得到一个小批量中的各个样本。前者允许同一个小批量中出现重复的样本，后者则不允许如此，且更常见。对于这两者间的任一种方式，都可以使用:

$$
{ g }_ { t }\leftarrow \nabla { f }_ { { B }_ { t } }\left( { x }_ { t-1 } \right) =\frac { 1 }{ B } \sum _ { i\in { B }_ { t } }^{  }{ \nabla { f }_ { i }\left( { x }_ { t-1 } \right)  } 
$$

来计算时间步$t$的小批量 ${ B  } _ { t }$ 上目标函数位于 ${ x } _ { t-1 }$ 处的梯度 ${ g } _ { t }$。这里 $B$
代表批量大小，即小批量中样本的个数，是一个超参数。同随机梯度一样，重复采样所得的小批量随机梯度 ${ g } _ { t }$ 也是对梯度 $\nabla f\left( { x } _ { t-1 } \right)$ 的无偏估计。给定学习率 ${ \eta  } _ { t }$（取正数），小批量随机梯度下降对自变量的迭代如下：

$$
{ { x }_ { t }\leftarrow { x }_ { t-1 }-\eta  }_ { t }{ g }_ { t }
$$

&emsp;&emsp;基于随机采样得到的梯度的方差在迭代过程中无法减小，因此在实际中，（小批量）随机梯度下降的学习率可以在迭代过程中自我衰减，例如 ${ \eta  } _ { t }=\eta { t }^{ \alpha  }$（通常 $\alpha =-1$ 或者 $-0.5$）、${ \eta  } _ { t }=\eta { \alpha  }^{ t }$（例如 $\alpha =0.95$）或者每迭代若干次后将学习率衰减一次。如此一来，学习率和（小批量）随机梯度乘积的方差会减小。而梯度下降在迭代过程中一直使用目标函数的真实梯度，无需自我衰减学习率。

&emsp;&emsp;小批量随机梯度下降中每次迭代的计算开销为 $O\left( B  \right)$ 。当批量大小为 $1$ 时，该算法即为随机梯度下降；当批量大小等于训练数据样本数时，该算法即为梯度下降。当批量较小时，每次迭代中使用的样本少，这会导致并行处理和内存使用效率变低。这使得在计算同样数目样本的情况下比使用更大批量时所花时间更多。当批量较大时，每个小批量梯度里可能含有更多的冗余信息。为了得到较好的解，批量较大时比批量较小时可能需要计算更多数目的样本，例如增大迭代周期数。

#### mini-batch 大小
1. 不能太大。更大的batch会使得训练更快，但是可能导致泛化能力下降。
    1. 训练更快是因为
        1. 更大的 $batch \ size$ 只需要更少的迭代步数就可以使得训练误差收敛。因为 $batch \ size$ 越大，则小批量样本来估计总体梯度越可靠，则每次参数更新沿着总体梯度的负方向的概率越大。另外，训练误差收敛速度快，并不意味着模型的泛化性能强。

        2. 更大的 $batch \ size$ 可以利用大规模数据并行的优势。
    2. 泛化能力下降是因为
        1. 更大的 $batch \ size$ 计算的梯度估计更精确，它带来更小的梯度噪声。此时噪声的力量太小，不足以将参数推出一个尖锐极小值的吸引区域。
        2. 解决方案为：提高学习率，从而放大梯度噪声的贡献。
2. 不能太小。因为对于多核架构来讲，太小的 $batch \ size$ 并不会相应地减少计算时间（考虑到多核之间的同步开销）。
3. 在有些硬件上，特定大小的效果更好。在使用 $GPU$ 时，通常使用 $2$ 的幂作为 $batch \ size$。
4. 泛化误差通常在 $batch \ size$ 大小为 $1$ 时最好，但此时梯度估计值的方差非常大，因此需要非常小的学习速率以维持稳定性。如果学习速率过大，则导致步长的变化剧烈。

## 动量法
### 梯度下降的问题
&emsp;&emsp;在每次迭代中，梯度下降法沿着自变量当前位置的梯度更新自变量。然而，自变量的迭代方向仅仅取决于自变量当前位置可能会带来一些问题。

### 简介
&emsp;&emsp;动量法是为了应对梯度下降的上述问题。设时间步$t$的自变量为$ x _ { t }$、学习率为 $\eta  _ { t }$。 在时间步 $0$，动量法创建速度变量 $v _ { 0 }$，并将其元素初始化成 $0$。在时间步 $t>0$，动量法对每次迭代的步骤做如下修改：

$$
\begin{aligned} { v }_ { t }&\leftarrow \gamma { v }_ { t-1 }+{ \eta  }_ { t }{ g }_ { t } \\ { x }_ { t }&\leftarrow { x }_ { t }-{ v }_ { t } \end{aligned}
$$

其中，动量超参数 $\gamma$ 满足 $0\le \gamma <1$。当 $\gamma =0$ 时，动量法等价于小批量随机梯度下降。

### 原因
动量方法经过加权移动平均之后，在指向极小值的方向上的速度 $v$ 不断加强，在垂直于极小值的方向上速度 $v$ 不断抵消。下图给出了非动量的方法与动量方法的路径图，红色路径表示动量方法的路径图，黑色箭头给出了在这些点非动量方法的更新方向和步长。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Optimization/momentum.png?raw=true"
    width="320" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">momentum</div>
</center>

### 指数加权移动平均
&emsp;&emsp;为了从数学上理解动量法，让我们先解释指数加权移动平均（$exponentially \ weighted \ moving \ average$）。给定超参数 $0\le \gamma <1$，当前时间步 $t$ 的变量 ${ y } _ { t }$ 是上一时间步 $t-1$ 的变量 ${ y } _ { t-1 }$ 和当前时间步另一变量 ${ x } _ { t }$ 的线性组合：

$$
{ y }_ { t }=\gamma { y }_ { t-1 }+\left( 1-\gamma  \right) { x }_ { t }
$$

我们可以对${ y } _ { t }$展开：

$$
\begin{aligned}
{ y }_ { t }&=\left( 1-\gamma  \right) { x }_ { t }+\gamma { y }_ { t-1 }\\ 
&=\left( 1-\gamma  \right) { x }_{ t }+\left( 1-\gamma  \right) \cdot \gamma { x }_{ t-1 }+{ \gamma  }^{ 2 }{ y }_{ t-2 }\\ &=\left( 1-\gamma  \right) { x }_{ t }+\left( 1-\gamma  \right) \cdot \gamma { x }_{ t-1 }+\left( 1-\gamma  \right) \cdot { \gamma  }^{ 2 }{ x }_{ t-2 }+{ \gamma  }^{ 3 }{ y }_{ t-3 }\\ &...\\ 
&=\left( 1-\gamma  \right) \sum _{ i=0 }^{ n }{ { \gamma  }^{ i } } { x }_{ t-i }
\end{aligned}
$$

令 $n=\frac { 1 }{ 1-\gamma  }$，那么 ${ \left( 1-\frac { 1 }{ n }  \right)  }^{ n }={ \gamma  }^{ \frac { 1 }{ 1-\gamma  }  }$。由于:

$$
\lim _ { n\rightarrow \infty  }{ { \left( 1-\frac { 1 }{ n }  \right)  }^{ n } } ={ e }^{ -1 }\approx 0.3679
$$

所以当 $\gamma \rightarrow 1$ 时，${ \gamma  }^{ \frac { 1 }{ 1-\gamma  }  }={ e }^{ -1 }$。例如 $0.95^{20}\approx { e }^{ -1 }$。如果把 ${ e }^{ -1 }$ 当做一个比较小的数，我们可以在近似中忽略所有含 ${ \gamma  }^{ \frac { 1 }{ 1-\gamma  }  }$ 和比 ${ \gamma  }^{ \frac { 1 }{ 1-\gamma  }  }$ 更高阶的系数的项。例如，当 $\gamma =0.95$ 时:

$$
{ y }_{ t }\approx 0.05\sum _{ i=0 }^{ 19 }{ { 0.95 }^{ i }{ x }_{ t-i } } 
$$

&emsp;&emsp;因此，在实际中，我们常常将$ { y } _ { t }$ 看作是对最近 $\frac { 1 }{ 1-\gamma  }$ 个时间步的 ${ x } _ { t }$ 值的加权平均。例如，当 $\gamma =0.95$ 时，${ y } _ { t }$ 可以被看作是对最近 $20$ 个时间步的 ${ x } _ { t }$ 值的加权平均；当 $\gamma =0.9$ 时，$y _ t$ 可以看作是对最近 $10$ 个时间步的 ${ x } _ { t }$ 值的加权平均。而且，离当前时间步 $t$ 越近的 ${ x } _ { t }$ 值获得的权重越大（越接近 1）。

### 由指数加权移动平均理解动量法
&emsp;&emsp;现在，我们对动量法的速度变量做变形：

$$
{ v }_{ t }\leftarrow \gamma { v }_{ t-1 }+\left( 1-\gamma  \right) \left( \frac { { \eta  }_{ t } }{ 1-\gamma  } { g }_{ t } \right) 
$$

由指数加权移动平均的形式可得，速度变量 ${ v } _ { t }$ 实际上对序列 $\left( \frac { { \eta  } _ { t-i } }{ 1-\gamma  } { g } _ { t-i } \right) \quad :\quad i=0,...,\frac { 1 }{ 1-\gamma  } -1$ 做了指数加权移动平均。换句话说，相比于小批量随机梯度下降，动量法在每个时间步的自变量更新量近似于将前者对应的最近 $\frac { 1 }{ 1-\gamma  }$ 个时间步的更新量做了指数加权移动平均后再除以 $1-\gamma$。所以动量法中，自变量在各个方向上的移动幅度不仅取决当前梯度，还取决于过去的各个梯度在各个方向上是否一致。

### 总结
* 动量法使用了指数加权移动平均的思想。它将过去时间步的梯度做了加权平均，且权重按时间步指数衰减。
* 动量法使得相邻时间步的自变量在更新方向上更加一致。

## Adagrad
### 背景
&emsp;&emsp;在之前介绍过的优化算法中，目标函数自变量的每一个元素在相同时间步都使用同一个学习率来自我迭代。举个例子，假设目标函数为 $f$ ，自变量为一个二维向量 ${ \left[ { x } _ { 1 },{ x } _ { 2 } \right]  }^{ T }$，该向量中每一个元素在迭代时都使用相同的学习率。例如在学习率为$\eta$的梯度下降中，元素 ${ x } _ { 1 }$和${ x } _ { 2 }$ 都使用相同的学习率 $\eta$ 来自我迭代：

$$
\begin{matrix} { x }_{ 1 }\leftarrow { x }_{ 1 }-\eta \frac { \partial f }{ \partial { x }_{ 1 } }  \\ { x }_{ 2 }\leftarrow { x }_{ 2 }-\eta \frac { \partial f }{ \partial { x }_{ 2 } }  \end{matrix}
$$

&emsp;&emsp;在“动量法”中我们看到当 ${ x } _ { 1 }$ 和 ${ x } _ { 2 }$ 的梯度值有较大差别时，需要选择足够小的学习率使得自变量在梯度值较大的维度上不发散。但这样会导致自变量在梯度值较小的维度上迭代过慢。动量法依赖指数加权移动平均使得自变量的更新方向更加一致，从而降低发散的可能。这一节介绍 $Adagrad$ 算法 ，根据自变量在每个维度上梯度值的大小来调整各个维度上的学习率，从而避免统一的学习率难以适应所有维度的问题。

### 算法
&emsp;&emsp;$Adagrad$ 的算法会使用一个小批量随机梯度 ${ g } _ { t }$ 按元素平方的累加变量 ${ s } _ { t }$。在时间步$0$，$Adagrad$ 将 ${ s } _ { 0 }$ 中每个元素初始化为 $0$。在时间步$t$，首先将小批量随机梯度 ${ g } _ { t }$ 按元素平方后累加到变量 ${ s } _ { t }$：

$$
{ s }_{ t }\leftarrow { s }_{ t-1 }+{ g }_{ t }\odot { g }_{ t }
$$

其中 $\odot$ 是按元素相乘。

> 用平方和的平方根而不是均值，因为分量可能为负值。

&emsp;&emsp;接着，我们将目标函数自变量中每个元素的学习率通过按元素运算重新调整:
$$
{ x }_{ t }\leftarrow { x }_{ t-1 }+\frac { \eta  }{ \sqrt { { s }_{ t }+\epsilon  }  } \odot { g }_{ t }
$$

其中 $\eta$ 是学习率，$\epsilon$ 是为了维持数值稳定性而添加的常数，例如 ${ 10 }^{ -6 }$。这里开方、除法和乘法的运算都是按元素进行的。这些按元素运算使得目标函数自变量中每个元素都分别拥有自己的学习率。

### 弊端
&emsp;&emsp;需要强调的是，小批量随机梯度按元素平方的累加变量 ${ s } _ { t }$ 出现在学习率的分母项中。因此，如果目标函数关于自变量中某个元素的偏导数一直都较大，那么该元素的学习率将下降较快；反之，如果目标函数关于自变量中某个元素的偏导数一直都较小，那么该元素的学习率将下降较慢。然而，由于 ${ s } _ { t }$ 一直在累加按元素平方的梯度，自变量中每个元素的学习率在迭代过程中一直在降低（或不变）。所以，当学习率在迭代早期降得较快且当前解依然不佳时，而在迭代后期由于学习率过小，可能较难找到一个有用的解。

### 总结
* $Adagrad$ 在迭代过程中不断调整学习率，让自变量中每个元素都分别拥有自己的学习率。
* 使用 $Adagrad$ 时，自变量中每个元素的学习率在迭代过程中一直在降低（或不变）。

## RMSProp
### 背景
&emsp;&emsp;在“$Adagrad$”一节里提到，由于调整学习率时分母上的变量 ${ s } _ { t }$ 一直在累加，目标函数自变量中每个元素的学习率在迭代过程中一直在降低（或不变）。所以，当学习率在迭代早期降得较快且当前解依然不佳时，$Adagrad$ 在迭代后期由于学习率过小，可能较难找到一个有用的解。为了应对这一问题，$RMSProp$ 算法对 $Adagrad$ 做了一点小小的修改。

### 算法
&emsp;&emsp;在“动量法”中介绍过指数加权移动平均。不同于$Adagrad$里状态变量 ${ s } _ { t }$ 是 截至时间步 $t$ 所有小批量随机梯度 ${ g } _ { t }$ 按元素平方和，$RMSProp$ 将这些梯度 按元素平方 做指数加权移动平均。具体来说，给定超参数 $0\le \gamma <1$，$RMSProp$ 在时间步 $t>0$ 计算:

$$
{ s }_{ t }\leftarrow \gamma { s }_{ t-1 }+{ \left( 1-\gamma  \right) g }_{ t }\odot { g }_{ t }
$$

和 $Adagrad$ 一样，$RMSProp$ 将目标函数自变量中每个元素的学习率通过按元素运算重新调整，然后更新自变量:

$$
{ x }_{ t }\leftarrow { x }_{ t-1 }+\frac { \eta  }{ \sqrt { { s }_{ t }+\epsilon  }  } \odot { g }_{ t }
$$

其中 $\eta$ 是学习率，$\epsilon$ 是为了维持数值稳定性而添加的常数，例如 ${ 10 }^{ -6 }$。因为 $RMSProp$ 的状态变量是对平方项 ${ g } _ { t }\odot { g } _ { t }$ 的指数加权移动平均，所以可以看作是最近 $\frac { 1 }{ 1-\gamma  }$ 个时间步的小批量随机梯度平方项的加权平均。如此一来，自变量每个元素的学习率在迭代过程中不再一直降低（或不变）。

### 总结
* $RMSProp$ 和 $Adagrad$ 的不同在于，$RMSProp$ 使用了小批量随机梯度 按元素平方的指数加权移动平均来调整学习率。

## Adadelta
### 背景
&emsp;&emsp;除了$RMSProp$ 以外，另一个常用优化算法$Adadelta$也针对$Adagrad$ 在迭代后期可能较难找到有用解的问题 做了改进。有意思的是，$Adadelta$ 没有学习率这一超参数。

### 算法
&emsp;&emsp;$Adadelta$ 算法也像 $RMSProp$ 一样，使用了小批量随机梯度 ${ g } _ { t }$ 按元素平方的指数加权移动平均变量 ${ s } _ { t }$。在时间步 $0$，它的所有元素被初始化为 $0$。 给定超参数 $0\le \rho <1$（对应 $RMSProp$ 中的 $\gamma$），在时间步 $t>0$，同 $RMSProp$ 一样计算:

$$
{ s }_{ t }\leftarrow \rho { s }_{ t-1 }+{ \left( 1-\rho  \right) g }_{ t }\odot { g }_{ t }
$$
 
与 $RMSProp$ 不同的是，$Adadelta$ 还维护一个额外的状态变量 $\Delta { x } _ { t }$，其元素同样在时间步 $0$ 时被初始化为 $0$。我们使用 $\Delta { x } _ { t-1 }$ 来计算自变量的变化量：

$$
{ g }_{ t }^{ \prime  }\leftarrow \sqrt { \frac { \Delta { x }_{ t-1 }+\epsilon  }{ { s }_{ t }+\epsilon  }  } \odot { g }_{ t }
$$

其中 $\epsilon$ 是为了维持数值稳定性而添加的常数，例如${ 10 }^{ -5 }$ 。接着更新自变量：

$$
{ x }_ { t }\leftarrow { x }_ { t-1 }-{ g }_ { t }^{ \prime  }
$$

最后，我们使用 $\Delta { x } _ { t }$ 来记录自变量变化量 ${ g }^{ ' }$ 按元素平方的指数加权移动平均：
$$
{ \Delta x }_ { t }\leftarrow { \rho \Delta x }_ { t-1 }+\left( 1-\rho  \right) { g }_ { t }^{ \prime  }\odot { g }_ { t }^{ \prime  }
$$

可以看到，如不考虑 $\epsilon$ 的影响，$Adadelta$ 跟 $RMSProp$ 不同之处在于使用 $\sqrt { \Delta { x } _ { t-1 } }$ 来替代超参数 $\eta$。

### 总结
* $Adadelta$ 没有学习率超参数，它通过使用有关自变量更新量按元素平方的指数加权移动平均来代替学习率。

## Adam
### 简介
&emsp;&emsp;$Adam$ 在 $RMSProp$ 基础上对小批量随机梯度也做了指数加权移动平均。

### 算法
&emsp;&emsp;$Adam$ 使用了动量变量 ${ v } _ { t }$ 和 $RMSProp$ 中小批量随机梯度按元素平方的指数加权移动平均 变量 ${ s } _ { t }$，并在时间步 $0$ 将它们中每个元素初始化为 $0$。给定超参数 $0\le { \beta  } _ { 1 }<1$（算法作者建议设为 $0.9$），时间步 $t$ 的动量变量 ${ v } _ { t }$ 即小批量随机梯度 ${ g } _ { t }$ 的指数加权移动平均：

$$
{ v }_ { t }\leftarrow { \beta  }_ { 1 }{ v }_ { t-1 }+\left( 1-{ \beta  }_{ 1 } \right) { g }_ { t }
$$

和 $RMSProp$ 中一样，给定超参数 $0\le { \beta  } _ { 2 }<1$（算法作者建议设为 $0.999$）， 将小批量随机梯度按元素平方后的项 ${ g } _ { t }\odot { g } _ { t }$ 做指数加权移动平均得到 ${ s } _ { t }$:

$$
{ s }_ { t }\leftarrow { \beta  }_ { 2 }{ s }_ { t-1 }+{ \left( 1-{ \beta  }_ { 2 } \right) g }_ { t }\odot { g }_ { t }
$$

&emsp;&emsp;由于将 ${ v } _ { 0 }$ 和 ${ s } _ { 0 }$ 中的元素都初始化为 $0$， 在时间步 $t$ 我们得到 ${ v } _ { t }=\left( 1-{ \beta  } _ { 1 } \right) \sum  _ { i=1 }^{ t }{ { \beta  } _ { 1 }^{ t-i } } { g } _ { i }$。将过去各时间步小批量随机梯度的权值相加，得到 $\left( 1-{ \beta  } _ { 1 } \right) \sum  _ { i=1 }^{ t }{ { \beta  } _ { 1 }^{ t-i } } =1-{ \beta  } _ { 1 }^{ t }$。需要注意的是，当 $t$ 较小时，过去各时间步小批量随机梯度权值之和会较小。例如当 ${ \beta  } _ { 1 }=0.9$ 时，${ v } _ { 1 }=0.1{ g } _ { 1 }$。为了消除这样的影响，对于任意时间步 $t$，我们可以将 ${ v } _ { t }$ 再除以 $1-{ \beta  } _ { 1 }^{ t }$，从而使得过去各时间步小批量随机梯度权值之和为 $1$。这也叫做偏差修正。在 $Adam$ 算法中，我们对变量 ${ v } _ { t }$ 和 ${ s } _ { t }$ 均作偏差修正：

$$
\begin{matrix} { \hat { v }  }_{ t }\leftarrow \frac { { v }_{ t } }{ 1-{ \beta  }_{ 1 }^{ t } }  \\ { \hat { s }  }_{ t }\leftarrow \frac { { s }_{ t } }{ 1-{ \beta  }_{ 2 }^{ t } }  \end{matrix}
$$

接下来，$Adam$ 算法使用以上偏差修正后的变量 ${ \hat { v }  } _ { t }$ 和${ \hat { s }  } _ { t }$，将模型参数中每个元素的学习率通过按元素运算重新调整：

$$
{ g }_ { t }^{ \prime  }\leftarrow \frac { \eta { \hat { v }  }_ { t } }{ \sqrt { { \hat { s }  }_ { t }+\epsilon  }  } 
$$

其中 $\eta$ 是学习率，$\epsilon$ 是为了维持数值稳定性而添加的常数，例如 ${ 10 }^{ -8 }$。和 $Adagrad$、$RMSProp$ 以及 $Adadelta$ 一样，目标函数自变量中每个元素都分别拥有自己的学习率。最后，使用 ${ g } _ { t }^{ \prime  }$ 迭代自变量：

$$
{ x }_ { t }\leftarrow { x }_ { t-1 }-{ g }_ { t }^{ \prime  }
$$


### 弊端
在某些情况下，$Adam$可能导致训练不收敛。主要原因是：随着时间窗口的变化，遇到的数据可能发生巨变。这就使得 $\sqrt{\widehat{s} _ t}$可能会时大时小，从而使得调整后的学习率 $\eta$ 不再是单调递减的。这种学习率的震荡可能导致模型无法收敛。

解决方案为：对 ${ s } _ { t }$ 的变化进行控制，避免其上下波动(确保 ${ s } _ { t }$ 是单调递增的)：

$$
s_t\gets \max \left( \beta _2s_{t-1}+\left( 1-\beta _ 2 \right) g_ t\odot g_ t,s_ {t-1} \right) 
$$

> $AdaDelta$、$RMSProp$ 算法也都存在这样的问题。

### 策略
虽然在训练早期$Adam$具有很好的收敛速度，但是最终模型的泛化能力并不如使用朴素的 $SGD$ 训练得到的模型好：$Adam$ 训练的模型得到的测试集误差会更大。其主要原因可能是：训练后期，$Adam$ 的更新步长过小。

一种改进策略为：在训练的初期使用 $Adam$ 来加速训练，并在合适的时期切换为$SGD$ 来追求更好的泛化性能。这种策略的缺陷是：切换的时机不好把握，需要人工经验来干预。

### 总结
* $Adam$ 在 $RMSProp$ 基础上对小批量随机梯度也做了指数加权移动平均
* $Adam$ 使用了偏差修正


## SGD 及其变种的比较

### 收敛速度
下图为 $SGD$ 及其不同的变种在代价函数的等高面中的学习过程。这里 $batch \ size$ 为全体样本大小。
1. $Adagrad$、$Adadelta$、$RMSprop$ 从最开始就找到了正确的方向并快速收敛。
2. $SGD$ 找到了正确的方向，但是收敛速度很慢。
3. $SGD$ 的动量算法、$Netsterov$ 动量算法（也叫做 $NAG$) 最初都偏离了正确方向，但最终都纠正到了正确方向。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Optimization/sgd_optimization_on_contours.gif?raw=true"
    width="320" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">sgd optimization on contours</div>
</center>

### 鞍点
下图为$SGD$及其不同的变种在代价函数的鞍点上的学习过程。这里 $batch \ size$ 为全体样本大小。
1. $SGD$、$SGD$ 的动量算法、$Netsterov$ 动量算法（也叫做 $NAG$ ) 都受到鞍点的严重影响。
    1. $SGD$ 最终停留在鞍点。如果采用 $mini-batch$，则$mini-batch$ 引入的梯度噪音，可以使得 $SGD$ 能够逃离鞍点。
    2. $SGD$ 的动量算法、$Netsterov$ 动量算法最终逃离了鞍点。
2. $Adadelta$ 、$Adagrad$、$RMSprop$  都很快找到了正确的方向。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Optimization/sgd_optimization_on_saddle.gif?raw=true"
    width="320" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">sgd optimization on saddle</div>
</center>



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