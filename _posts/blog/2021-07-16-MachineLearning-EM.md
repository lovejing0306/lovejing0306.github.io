---
layout: post
title: EM 算法
categories: [MachineLearning]
description: EM 算法
keywords: MachineLearning
---


EM 算法
---


&emsp;&emsp; $EM$ 算法用于求解含有隐变量的极大似然估计问题。由于隐变量的存在，无法直接用极大似然估计求得对数似然函数的极大值。此时通过  $jensen$  不等式构造对数似然函数的下界函数，然后优化下界函数，再用估计出的参数值构造新的下界函数，反复迭代直至收敛到局部极小值。

 $EM$  算法的每次迭代由两步组成：
1.  $E$ 步求期望。
2.  $M$ 步求极大。

所以 $EM$ 算法也称为期望极大算法。
    
## 示例
### 身高抽样问题
&emsp;&emsp;假设学校所有学生中，男生身高服从正态分布 $\mathcal N(\mu  _  1,\sigma  _  1^2)$ ，女生身高服从正态分布 $\mathcal N(\mu  _  2,\sigma  _  2^2)$ 。
现在随机抽取 $200$ 名学生，得到这些学生的身高 $\{x  _  1,x  _  2,\cdots,x  _  n\}$ ，求参数  $\{\mu  _  1,\sigma  _  1^2,\mu  _  2,\sigma  _  2^2\}$ 的估计。

&emsp;&emsp;定义隐变量为 $z$ ，其取值为 ${0,1}$，分别表示男生和女生。如果隐变量是已知的，即已知每个学生是男生还是女生 $\{z  _  1,z  _  2,\cdots,z  _  n\}$ ，则问题很好解决：
1. 统计所有男生的身高的均值和方差，得到 $\{\mu  _  1,sigma  _  1^2\}$：
    $$
    \mu_1 = \text{avg} \{x_i\mid z_i=0\}\quad \sigma_1^2 = \text{var} \{x_i\mid z_i=0\}
    $$
    其中 $\{x  _  i\mid z  _  i=0\}$ 表示满足 $z  _  i=0$ 的 $x  _  i$  构成的集合。 $\text{avg},\text{var}$ 分别表示平均值和方差。
            
2. 统计所有女生的身高的均值和方差，得到 $\{\mu  _  2,\sigma  _  2^2\}$ ：
    $$
    \mu_2 = \text{avg} \{x_i\mid z_i=1\}\quad \sigma_2^2 = \text{var} \{x_i\mid z_i=1\}
    $$
    其中 $\{x  _  i\mid z  _  i=1\}$ 表示满足 $z  _  i=1$ 的 $x  _  i$  构成的集合。 $\text{avg},\text{var}$ 分别表示平均值和方差。
            

如果已知参数 $\{\mu  _  1,\sigma  _  1^2,\mu  _  2,\sigma  _  2^2\}$ ，则任意给出一个学生的身高 $x$ ，可以知道该学生分别为男生/女生的概率。
$$
\begin{array}{c}
	p_ 1=\frac{1}{\sqrt{2\pi}\times \sigma _ 1}\exp \left( -\frac{\left( x-\mu _ 1 \right) ^2}{2\sigma _ {1}^{2}} \right)\\
	p_ 2=\frac{1}{\sqrt{2\pi}\times \sigma _ 2}\exp \left( -\frac{\left( x-\mu _ 2 \right) ^2}{2\sigma _ {2}^{2}} \right)\\
\end{array}
$$

则有： $p(z=0\mid x)=\frac{p  _  1}{p  _  1+p  _  2},p(z=1\mid x)=\frac{p  _  2}{p  _  1+p  _  2}$  。因此也就知道该学生更可能为男生，还是更可能为女生。
        
&emsp;&emsp;因此：参数 $\{\mu  _  1,\sigma  _  1^2,\mu  _  2,\sigma  _  2^2\} \Leftrightarrow$ 学生是男生或女生，这两个问题是相互依赖，相互纠缠的。
    
&emsp;&emsp;为解决该问题，通常采取下面步骤：
1. 先假定参数的初始值： $\{\mu  _  1^{<0>},\sigma  _  1^{2<0>},\mu  _  2^{<0>},\sigma  _  2^{2<0>}\}$ 。
2. 迭代： $i=0,1,\cdots$ 
    1. 根据 $\{\mu  _  1^{<i>},\sigma  _  1^{2<i>},\mu  _  2^{<i>},\sigma  _  2^{2<i>}\}$ 来计算每个学生更可能是属于男生，还是属于女生。
        > 这一步为 $E$  步（ $Expectation$ ），用于计算隐变量的后验分布 $p(z\mid x)$ 。
    2. 根据上一步的划分，统计所有男生的身高的均值和方差，得到 $\{\mu  _  1^{<i+1>},\sigma  _  1^{2<i+1>}\}$；统计所有女生的身高的均值和方差，得到 $\{\mu  _  2^{\<i+1\>},\sigma  _  2^{2\<i+1\>}\}$ 。
        > 这一步为 $M$ 步（$Maximization$），用于通过最大似然函数求解正态分布的参数。
    3. 当前后两次迭代的参数变化不大时，迭代终止。

### 三硬币模型
已知三枚硬币 $A$ ， $B$ ， $C$  ，这些硬币正面出现的概率分别为 $\pi,p,q$ 。进行如下试验：  
1. 先投掷硬币 $A$ ，若是正面则选硬币 $B$ ；若是反面则选硬币 $C$ 。
2. 然后投掷被选出来的硬币，投掷的结果如果是正面则记作 $1$ ；投掷的结果如果是反面则记作 $0$  。
3. 独立重复地 $N$ 次试验，观测结果为： $1,1,0,1,0,...0,1$  。

现在只能观测到投掷硬币的结果，无法观测投掷硬币的过程，求估计三硬币正面出现的概率。
    
&emsp;&emsp;设随机变量 $Y$ 是观测变量，表示一次试验观察到的结果，取值为 $1$ 或者 $0$ ;随机变量 $Z$ 是隐变量，表示未观测到的投掷 $A$ 硬币的结果，取值为 $1$ 或者 $0$ ; $\theta=(\pi,p,q)$ 是模型参数。则：

$$
\begin{aligned}
	P\left( Y;\theta \right) &=\sum_Z{P\left( Y,Z;\theta \right)}\\
	&=\sum_Z{P\left( Z;\theta \right) P\left( Y|Z;\theta \right)}\\
	&=\pi p^Y\left( 1-p \right) ^{1-Y}+\left( 1-\pi \right) q^Y\left( 1-q \right) ^{1-Y}\\
\end{aligned}
$$

注意：随机变量 $Y$ 的数据可以观测，随机变量 $Z$ 的数据不可观测

&emsp;&emsp;将观测数据表示为 $\mathbb Y=\{y  _  1,y  _  2,\cdots,y  _  N\}$ ，未观测数据表示为 $\mathbb Z=\{z  _  1,z  _  2,\cdots,z  _  N\}$ 。由于每次试验之间都是独立的，则有：

$$
P(\mathbb Y;\theta)=\prod_{j=1}^{N}P(Y=y_i;\theta)=\prod_{j=1}^{N}[\pi p^{y_j}(1-p)^{1-y_j}+(1-\pi)q^{y_j}(1-q)^{1-y_j}]
$$

考虑求模型参数 $\theta=(\pi,p,q)$ 的极大似然估计，即：

$$
\hat \theta=\arg\max_{\theta}\log P(\mathbb Y;\theta)
$$

这个问题没有解析解，只有通过迭代的方法求解， $EM$ 算法就是可以用于求解该问题的一种迭代算法。
    
####  $EM$ 算法求解

首先选取参数的初值，记作 $\theta^{<0>}=(\pi^{<0>},p^{<0>},q^{<0>})$ ，

然后通过下面的步骤迭代计算参数的估计值，直到收敛为止：

设第 $i$ 次迭代参数的估计值为： $\theta^{<i>}=(\pi^{<i>},p^{<i>},q^{<i>})$ ， 则 $EM$ 算法的第 $ i+1 $ 次迭代如下：
1.  $E$ 步：计算模型在参数 $\theta^{<i>}=(\pi^{<i>},p^{<i>},q^{<i>})$ 下，观测数据 $y  _  j$ 来自于投掷硬币 $B$ 的概率：
    $$
    \mu^{\<i+1\>}_ j=\frac{\pi^{<i>}(p^{<i>})^{y_j}(1-p^{<i>})^{1-y_j}}{\pi^{<i>}(p^{<i>})^{y_j}(1-p^{<i>})^{1-y_j}+(1-\pi^{<i>})(q^{<i>})^{y_j}(1-q^{<i>})^{1-y_j}}
    $$
    它其实就是 $P(Z=1\mid Y=y  _  j)$ ，即：已知观测变量 $Y=y  _  j$ 的条件下，观测数据 $y  _  j$  来自于投掷硬币 $B$  的概率。
        
2. `M` 步：计算模型参数的新估计值：
    $$
    \begin{aligned}
    	\pi ^{<i+1>}&=\frac{1}{N}\sum_ {j=1}^N{\mu _ {j}^{<i+1>}}\\
    	p^{<i+1>}&=\frac{\sum_{j=1}^N{\mu _ {j}^{<i+1>}y_j}}{\sum_{j=1}^N{\mu _ {j}^{<i+1>}}}\\
    	q^{<i+1>}&=\frac{\sum_{j=1}^N{\left( 1-\mu _ {j}^{<i+1>} \right) y_j}}{\sum_{j=1}^N{\left( 1-\mu _ {j}^{<i+1>} \right)}}\\
    \end{aligned}
    $$
    1. 第一个式子：通过后验概率 $ P(Z \mid Y) $ 估计值的均值作为先验概率 $\pi$  的估计。
    2. 第二个式子：通过条件概率 $P(Y\mid Z=1)$ 的估计来求解先验概率 $p$ 的估计。
    3. 第三个式子：通过条件概率 $P(Y\mid Z=0)$ 的估计来求解先验概率 $q$ 的估计。

####  $EM$  算法的解释
1. 初始化：随机选择三枚硬币 $A$ ， $B$ ， $C$  正面出现的概率 $\pi,p,q$ 的初始值  $\pi^{<0>},p^{<0>},q^{<0>}$ 。
2.  $E$  步：在已知概率 $\pi,p,q$ 的情况下，求出每个观测数据 $y  _  j$  是来自于投掷硬币 $B$ 的概率。即：
    $$
    p(z_j\mid y_j=1) 
    $$    
    于是对于 $N$ 次实验，就知道哪些观测数据是由硬币 $B$ 产生，哪些是由硬币 $C$  产生。
3.  $M$  步：在已知哪些观测数据是由硬币 $B$ 产生，哪些是由硬币 $C$ 产生的情况下：
    1.  $\pi$ 就等于硬币 $B$ 产生的次数的频率。
    2.  $p$ 就等于硬币 $B$ 产生的数据中，正面向上的频率。
    3.  $q$ 就等于硬币 $C$ 产生的数据中，正面向上的频率。

## EM算法

### 观测变量和隐变量
&emsp;&emsp;令 $Y =$ 表示观测随机变量， $\mathbb Y=\{y  _  1,y  _  2,\cdots,y  _  N\}$  表示对应的数据序列；令 $Z$ 表示隐随机变量， $\mathbb Z=\{z  _  1,z  _  2,\cdots,z  _  N\}$  表示对应的数据序列。 $\mathbb Y$ 和 $\mathbb Z$ 连在一起称作完全数据，观测数据 $\mathbb Y$ 又称作不完全数据。

&emsp;&emsp;假设给定观测随机变量 $Y$ ，其概率分布为 $P(Y;\theta)$ ，其中 $\theta$  是需要估计的模型参数，则不完全数据 $\mathbb Y$ 的似然函数是 $P(\mathbb Y;\theta)$ ， 对数似然函数为 $L(\theta)=\log P(\mathbb Y;\theta)$ 。
    
&emsp;&emsp;假定 $Y$ 和 $Z$ 的联合概率分布是 $ P(Y,Z;\theta)$ ，完全数据的对数似然函数是 $\log P(\mathbb Y,\mathbb Z;\theta)$ ，则根据每次观测之间相互独立，有：

$$
\begin{aligned}
	\log P\left( \mathbb{Y};\theta \right) &=\sum_i{\log P\left( Y=y_i;\theta \right)}\\
	\log P\left( \mathbb{Y},\mathbb{Z};\theta \right) &=\sum_i{\log P\left( Y=y_i,Z=z_i;\theta \right)}\\
\end{aligned}
$$

由于 $\mathbb Y$ 发生，根据最大似然估计，则需要求解对数似然函数：

$$
\begin{aligned}
	L\left( \theta \right) &=\log P\left( \mathbb{Y};\theta \right)\\
	&=\sum_{i=1}{\log P\left( Y=y_i;\theta \right)}\\
	&=\sum_{i=1}{\log \sum_Z{P\left( Y=y_i,Z;\theta \right)}}\\
	&=\sum_{i=1}{\log \left[ \sum_Z{P\left( Y=y_i|Z;\theta \right) P\left( Z;\theta \right)} \right]}\\
\end{aligned}
$$

的极大值。其中 $\sum  _  Z P(Y=y  _  i,Z;\theta)$ 表示对所有可能的Z 求和，因为边缘分布  $P(Y)=\sum  _  Z P(Y,Z)$ 。该问题的困难在于：该目标函数包含了未观测数据的的分布的积分和对数。
    

### Jensen不等式
&emsp;&emsp;回顾优化理论中的一些概念。设 $f$ 是定义域为实数的函数，如果对于所有的实数 $x$ ， $f^{''}\left( x \right) \ge 0$ ，那么 $f$ 是凸函数。当 $x$ 是向量时，如果其 $hessian$ 矩阵 $H$ 是半正定的 $H\ge 0$ ，那么 $f$ 是凸函数。如果 $f^{''}\left( x \right) >0$ 或者 $H>0$ ，那么称 $f$ 是严格凸函数。

&emsp;&emsp; $Jensen$ 不等式表述如下，如果 $f$ 是凸函数， $X$ 是随机变量，那么

$$
E\left[ f\left( X \right) \right] \ge f\left( EX \right) 
$$

特别地，如果 $f$ 是严格凸函数，那么 $E\left[ f\left( X \right) \right] = f\left( EX \right) $ 当且仅当 $P\left( x=E\left[ X \right] \right) =1$ ，也就是说 $X$ 是常量。这里我们将 $f\left( E\left[ x \right] \right) $ 简写为 $f\left( EX \right)$ 。如果用图表示会很清晰：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/EM/jensen.png?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">pca_lda</div>
</center>

图中，实线 $f$ 是凸函数， $X$ 是随机变量，有 $0.5$ 的概率是 $a$ ，有 $0.5$ 的概率是 $b$ 。（就像掷硬币一样）。 $X$ 的期望值就是 $a$ 和 $b$ 的中值了，图中可以看到 $E\left[ f\left( X \right) \right] \ge f\left( EX \right) $ 成立。

&emsp;&emsp;当 $f$ 是（严格）凹函数当且仅当 $-f$ 是（严格）凸函数。 $Jensen$ 不等式应用于凹函数时，不等号方向反向，也就是 $E\left[ f\left( X \right) \right] \le f\left( EX \right)$ 。


### EM原理
&emsp;&emsp;给定的训练样本是 $\left\{ x^{\left( 1 \right)},\cdots ,x^{\left( m \right)} \right\} $ ，我们想找到每个样例隐含的类别 $z$ ，能使得 $p(x,z)$ 最大。 $p(x,z)$ 的最大似然估计如下：

$$
\begin{aligned}
	l\left( \theta \right) &=\sum_{i=1}^m{\log p\left( x;\theta \right)}\\
	&=\sum_{i=1}^m{\log \sum_z{p\left( x,z;\theta \right)}}\\
\end{aligned}
$$

&emsp;&emsp;第一步是对极大似然取对数，第二步是对每个样例的每个可能类别 $z$ 求联合分布概率和。但是直接 $\theta $ 一般比较困难，因为有隐藏变量 $z$ 存在，但是一般确定了 $z$ 后，求解就容易了。

&emsp;&emsp; $EM$ 是一种解决存在隐含变量优化问题的有效方法。竟然不能直接最大化 $l\left( \theta \right) $ ，我们可以不断地建立 $l$ 的下界（ $E$ 步），然后优化下界（ $M$ 步）。这句话比较抽象，看下面的。

&emsp;&emsp;对于每一个样例 $i$ ，让 $Q  _  i$ 表示该样例隐含变量 $z$ 的某种分布， $Q  _  i$ 满足的条件是 $\sum  _  z{Q  _  i\left( z \right) =1,Q  _  i\left( z \right) \ge 0}$ 。（如果 $z$ 是连续性的，那么 $Q  _  i$ 是概率密度函数，需要将求和符号换做积分符号）。比如要将班上学生聚类，假设隐藏变量 $z$ 是身高，那么就是连续的高斯分布。如果按照隐藏变量是男女，那么就是伯努利分布了。可以由前面阐述的内容得到下面的公式：

$$
\begin{aligned}
	\sum_i{\log p\left( x^{\left( i \right)};\theta \right)}&=\sum_i{\log \sum_{z^{\left( i \right)}}{p\left( x^{\left( i \right)},z^{\left( i \right)};\theta \right)}}\left( 1 \right)\\
	&=\sum_i{\log \sum_{z^{\left( i \right)}}{Q_i\left( z^{\left( i \right)} \right) \frac{p\left( x^{\left( i \right)},z^{\left( i \right)};\theta \right)}{Q_i\left( z^{\left( i \right)} \right)}}}\left( 2 \right)\\
	&\ge \sum_i{\sum_{z^{\left( i \right)}}{Q_i\left( z^{\left( i \right)} \right) \log \frac{p\left( x^{\left( i \right)},z^{\left( i \right)};\theta \right)}{Q_i\left( z^{\left( i \right)} \right)}}}\left( 3 \right)\\
\end{aligned}
$$

（1）到（2）比较直接，就是分子分母同乘以一个相等的函数。（2）到（3）利用了 $Jensen$ 不等式，考虑到 $\log \left( x \right) $ 是凹函数（二阶导数小于 $0$ ），而且

$$
Q_i\left( z^{\left( i \right)} \right) \left[ \frac{p\left( x^{\left( i \right)},z^{\left( i \right)};\theta \right)}{Q_i\left( z^{\left( i \right)} \right)}\right] 
$$

就是 $\frac{p\left( x^{\left( i \right)},z^{\left( i \right)};\theta \right)}{Q  _  i\left( z^{\left( i \right)} \right)}$ 的期望（回想期望公式中的Lazy Statistician规则）:

***
设 $Y$ 是随机变量 $X$ 的函数, $Y=g\left( x \right) $ （ $g$ 是连续函数），那么
1.  $X$ 是离散型随机变量，它的分布律为 $P\left( X=x  _  k \right) =p  _  k,k=1,2,...$ 。若 $\sum  _  {k=1}^{\infty}{g\left( x  _  k \right) p  _  k}$ 绝对收敛，则有
    $$
    E\left( Y \right) =E\left[ g\left( X \right) \right] =\sum_{k=1}^{\infty}{g\left( x_k \right) p_k}
    $$
2.  $X$ 是连续型随机变量，它的概率密度为 $f\left( x \right) $ ，若 $\int  _  {-\infty}^{\infty}{g\left( x \right)}f\left( x \right) dx$ 绝对收敛，则有
    $$
    E\left( Y \right) =E\left[ g\left( X \right) \right] = \int_{-\infty}^{\infty}{g\left( x \right)}f\left( x \right) dx
    $$
***

&emsp;&emsp;对应于上述问题， $Y$ 是 $\frac{p\left( x^{\left( i \right)},z^{\left( i \right)};\theta \right)}{Q  _  i\left( z^{\left( i \right)} \right)}$ ， $X$ 是 $z^{\left( i \right)}$ ， $Q  _  i\left( z^{\left( i \right)} \right) $ 是 $p  _  k$ ， $g$ 是 $z^{\left( i \right)}$ 到 $\frac{p\left( x^{\left( i \right)},z^{\left( i \right)};\theta \right)}{Q  _  i\left( z^{\left( i \right)} \right)}$ 的映射。这样解释了式子（2）中的期望，再根据凹函数时的 $Jensen$ 不等式：

$$
f\left( E_{z^{\left( i \right)}\thicksim Q_i}\left[ \frac{p\left( x^{\left( i \right)},z^{\left( i \right)};\theta \right)}{Q_i\left( z^{\left( i \right)} \right)} \right] \right) \ge E_{z^{\left( i \right)}\thicksim Q_i}\left[ f\left( \frac{p\left( x^{\left( i \right)},z^{\left( i \right)};\theta \right)}{Q_i\left( z^{\left( i \right)} \right)} \right) \right] 
$$

可以得到（3）。

&emsp;&emsp;这个过程可以看作是对 $ l\left( \theta \right) $ 求了下界。对于 $Q  _  i$ 的选择，有多种可能，那种更好的？假设 $\theta$ 已经给定，那么 $ l\left( \theta \right) $ 的值就决定于 $Q  _  i\left( z^{\left( i \right)} \right) $ 和 $p\left( x^{\left( i \right)},z^{\left( i \right)} \right)$ 了。我们可以通过调整这两个概率使下界不断上升，以逼近 $ l\left( \theta \right) $ 的真实值，那么什么时候算是调整好了呢？当不等式变成等式时，说明我们调整后的概率能够等价于 $ l\left( \theta \right) $ 了。按照这个思路，我们要找到等式成立的条件。根据 $Jensen$ 不等式，要想让等式成立，需要让随机变量变成常数值，这里得到：

$$
\frac{p\left( x^{\left( i \right)},z^{\left( i \right)};\theta \right)}{Q_i\left( z^{\left( i \right)} \right)} = c
$$

&emsp;&emsp; $c$ 为常数，不依赖于 $z^{\left( i \right)}$ 。对此式子做进一步推导，我们知道 $\sum  _  z{Q  _  i\left( z^{\left( i \right)} \right)}=1$ ，那么也就有 $\sum  _  z{p\left( x^{\left( i \right)},z^{\left( i \right)};\theta \right)}=c$ ，（多个等式分子分母相加不变，这个认为每个样例的两个概率比值都是 $c$ ），那么有下式：

$$
\begin{aligned}
	Q_i\left( z^{\left( i \right)} \right) &=\frac{p\left( x^{\left( i \right)},z^{\left( i \right)};\theta \right)}{\sum_z{p\left( x^{\left( i \right)},z;\theta \right)}}\\
	&=\frac{p\left( x^{\left( i \right)},z^{\left( i \right)};\theta \right)}{p\left( x^{\left( i \right)};\theta \right)}\\
	&=p\left( z^{\left( i \right)}|x^{\left( i \right)};\theta \right)\\
\end{aligned}
$$

&emsp;&emsp;至此，我们推出了在固定其他参数 $\theta$ 后， $Q  _  i\left( z^{\left( i \right)} \right)$ 的计算公式就是后验概率，解决了 $Q  _  i\left( z^{\left( i \right)} \right)$ 如何选择的问题。这一步就是 $E$ 步，建立 $ l\left( \theta \right) $ 的下界。接下来的M步，就是在给定 $Q  _  i\left( z^{\left( i \right)} \right)$ 后，调整 $\theta$ ，去极大化 $ l\left( \theta \right) $ 的下界（在固定 $Q  _  i\left( z^{\left( i \right)} \right)$ 后，下界还可以调整的更大）。那么一般的 $EM$ 算法的步骤如下：
***
循环重复直到收敛 

$E$ 步：对于每一个 $i$ ，计算

$$
Q_i\left( z^{\left( i \right)} \right) =p\left( z^{\left( i \right)}|x^{\left( i \right)};\theta \right) 
$$

$M$ 步：计算

$$
\theta =arg\underset{\theta}{\max}\sum_i{\sum_{z^{\left( i \right)}}{Q_i\left( z^{\left( i \right)} \right) \log \frac{p\left( x^{\left( i \right)},z^{\left( i \right)};\theta \right)}{Q_i\left( z^{\left( i \right)} \right)}}}
$$
***

&emsp;&emsp;那么究竟怎么确保 $EM$ 收敛？假定 $\theta ^{\left( t \right)}$ 和 $\theta ^{\left( t+1 \right)}$ 是 $EM$ 第 $t$ 次和 $t+1$ 次迭代后的结果。如果我们证明了 $l\left( \theta ^{\left( t \right)} \right) \le l\left( \theta ^{\left( t+1 \right)} \right) $ ，也就是说极大似然估计单调增加，那么最终我们会到达最大似然估计的最大值。下面来证明，选定 $\theta ^{\left( t \right)}$ 后，我们得到 $E$ 步:

$$
Q_{i}^{\left( t \right)}\left( z^{\left( i \right)} \right) =p\left( z^{\left( i \right)}|x^{\left( i \right)};\theta ^{\left( t \right)} \right) 
$$

这一步保证了在给定 $\theta ^{\left( t \right)}$ 时， $Jensen$ 不等式中的等式成立，也就是

$$
l\left( \theta ^{\left( t \right)} \right) =\sum_i{\sum_{z^{\left( i \right)}}{Q_{i}^{\left( t \right)}\left( z^{\left( i \right)} \right) \log \frac{p\left( x^{\left( i \right)},z^{\left( i \right)};\theta ^{\left( t \right)} \right)}{Q_{i}^{\left( t \right)}\left( z^{\left( i \right)} \right)}}}
$$

然后进行 $M$ 步，固定 $ Q  _  {i}^{\left( t \right)}\left( z^{\left( i \right)} \right) $ ，并将 $\theta ^{\left( t \right)}$ 视作变量，对上面的 $l\left( \theta ^{\left( t \right)} \right)$ 求导后，得到 $\theta ^{\left( t+1 \right)}$ ，这样经过一些推导会有以下式子成立：

$$
\begin{aligned}
	l\left( \theta ^{\left( t+1 \right)} \right) &\ge \sum_i{\sum_{z^{\left( i \right)}}{Q_{i}^{\left( t \right)}\left( z^{\left( i \right)} \right) \log \frac{p\left( x^{\left( i \right)},z^{\left( i \right)};\theta ^{\left( t+1 \right)} \right)}{Q_{i}^{\left( t \right)}\left( z^{\left( i \right)} \right)}}}\left( 4 \right)\\
	&\ge \sum_i{\sum_{z^{\left( i \right)}}{Q_{i}^{\left( t \right)}\left( z^{\left( i \right)} \right) \log \frac{p\left( x^{\left( i \right)},z^{\left( i \right)};\theta ^{\left( t \right)} \right)}{Q_{i}^{\left( t \right)}\left( z^{\left( i \right)} \right)}}}\left( 5 \right)\\
	&=l\left( \theta ^{\left( t \right)} \right) \left( 6 \right)\\
\end{aligned}
$$

解释第（4）步，得到 $\theta ^{\left( t+1 \right)}$ 时，只是最大化 $l\left( \theta ^{\left( t \right)} \right)$ ，也就是 $l\left( \theta ^{\left( t+1 \right)} \right)$ 的下界，而没有使等式成立，等式成立只有是在固定 $\theta$ ，并按 $E$ 步得到 $Q  _  i$ 时才能成立。

&emsp;&emsp;况且根据我们前面得到的下式，对于所有的 $Q  _  i$ 和 $\theta$ 都成立:

$$
l\left( \theta ^{\left( t \right)} \right) \ge \sum_i{\sum_{z^{\left( i \right)}}{Q_{i}\left( z^{\left( i \right)} \right) \log \frac{p\left( x^{\left( i \right)},z^{\left( i \right)};\theta \right)}{Q_{i}^{\left( t \right)}\left( z^{\left( i \right)} \right)}}}
$$

第（5）步利用了 $M$ 步的定义， $M$ 步就是将 $\theta ^{\left( t \right)}$ 调整到 $\theta ^{\left( t+1 \right)}$ ，使得下界最大化。因此（5）成立，（6）是之前的等式结果。

这样就证明了 $l\left( \theta ^{\left( t \right)} \right)$ 会单调增加。一种收敛方法是 $l\left( \theta ^{\left( t \right)} \right)$ 不再变化，还有一种就是变化幅度很小。

&emsp;&emsp;再次解释一下（4）、（5）、（6）。首先（4）对所有的参数都满足，而其等式成立条件只是在固定 $\theta$ ，并调整好 $Q$ 时成立，而第（4）步只是固定 $Q$ ，调整 $\theta$ ，不能保证等式一定成立。（4）到（5）就是 $M$ 步的定义，（5）到（6）是前面 $E$ 步所保证等式成立条件。也就是说 $E$ 步会将下界拉到与 $l\left( \theta ^{\left( t \right)} \right)$ 一个特定值（这里 $\theta ^{\left( t \right)}$ ）一样的高度，而此时发现下界仍然可以上升，因此经过 $M$ 步后，下界又被拉升，但达不到与 $l\left( \theta ^{\left( t \right)} \right)$ 另外一个特定值一样的高度，之后 $E$ 步又将下界拉到与这个特定值一样的高度，重复下去，直到最大值。如果我们定义:

$$
J\left( Q,\theta \right) =\sum_i{\sum_{z^{\left( i \right)}}{Q_i\left( z^{\left( i \right)} \right) \log \frac{p\left( x^{\left( i \right)},z^{\left( i \right)};\theta \right)}{Q_i\left( z^{\left( i \right)} \right)}}}
$$

从前面的推导中我们知道 $l\left( \theta \right) \ge J\left( Q,\theta \right) $ ， $EM$ 可以看作是 $J$ 的坐标上升法， $E$ 步固定 $\theta$ ，优化 $Q$ ； $M$ 步固定 $Q$ 优化 $\theta$ 。


### 伪码
输入：
1. 观测变量数据 $\mathbb Y=\{y  _  1,y  _  2,\cdots,y  _  N\}$ 
2. 联合分布 $P(Y,Z ;\theta)$ ，以及条件分布 $P( Z \mid Y;\theta)$ 
   > 联合分布和条件分布的形式已知（比如说高斯分布等），但是参数未知（比如均值、方差）

输出：模型参数 $\theta$ 

算法步骤：
1. 选择参数的初值 $\theta^{<0> }$ ，开始迭代。
2.  $E$ 步：记 $\theta^{<i>}$ 为第 $i$ 次迭代参数 $\theta$ 的估计值，在第 $i+1$  次迭代的 $E$ 步，计算：
    $$
    \begin{aligned}
    	Q\left( \theta ,\theta ^{<i>} \right) &=\sum_{j=1}^N{\mathbb{E}_{P\left( Z|Y=y_j;\theta ^{<i>} \right)}\log P\left( Y=y_j,Z;\theta \right)}\\
    	&=\sum_{j=1}^N{\left( \sum_Z{P\left( Z|Y=y_j;\theta ^{<i>} \right) \log P\left( Y=y_{j,}Z;\theta \right)} \right)}\\
    \end{aligned}
    $$
    其中 $\mathbb E  _  {P(Z\mid Y=y  _  j;\theta^{<i>})}\log P(Y=y  _  j,Z ;\theta)$ 表示：对于观测点 $Y=y  _  j$ ， $\log P(Y=y  _  j,Z ;\theta)$ 关于后验概率 $P(Z\mid Y=y  _  j;\theta^{<i>})$ 的期望。
3.  $M$ 步：求使得 $Q(\theta,\theta^{<i>})$ 最大化的 $\theta$ ，确定 $i+1$  次迭代的参数的估计值 $\theta^{<i+1>}$ 
    $$
    \theta^{<i+1>}=\arg\max_\theta Q(\theta,\theta^{<i>})
    $$
4. 重复上面两步，直到收敛。

### 收敛
通常收敛的条件是：给定较小的正数  $\varepsilon  _  1,\varepsilon  _  2$ ，满足： $||\theta^{<i+1>}-\theta^{<i>}|| \lt \varepsilon  _  1$ 或者 $||Q(\theta^{<i+1>},\theta^{<i>})-Q(\theta^{<i>},\theta^{<i>})|| \lt \varepsilon  _  2$ 。

### 核心    
$Q(\theta,\theta^{<i>})$ 是算法的核心，称作 $Q$ 函数。其中：
1. 第一个符号表示要极大化的参数（未知量）
2. 第二个符号表示参数的当前估计值（已知量）

### 解释
&emsp;&emsp; $EM$ 算法的直观理解： $EM$ 算法的目标是最大化对数似然函数 $L(\theta)=\log P(\mathbb Y)$ 。直接求解这个目标是有问题的。因为要求解该目标，首先要得到未观测数据的分布 $P(Z \mid Y;\theta)$  。如：身高抽样问题中，已知身高，需要知道该身高对应的是男生还是女生。
但是未观测数据的分布就是待求目标参数 $\theta$ 的解的函数。这是一个“鸡生蛋-蛋生鸡” 的问题。

&emsp;&emsp; $EM$ 算法试图多次猜测这个未观测数据的分布 $P(Z \mid Y;\theta)$ 。每一轮迭代都猜测一个参数值  $\theta^{<i>}$ ，该参数值都对应着一个未观测数据的分布 $ P(Z \mid Y;\theta^{<i>})$ 。如：已知身高分布的条件下，男生或女生的分布。

&emsp;&emsp;然后通过最大化某个变量来求解参数值。这个变量就是 $B(\theta,\theta^{<i>})$  变量，它是真实的似然函数的下界 。  
1. 如果猜测正确，则 B 就是真实的似然函数。
2. 如果猜测不正确，则 B 就是真实似然函数的一个下界。

### 隐变量求解
&emsp;&emsp;隐变量估计问题也可以通过梯度下降法等算法求解，但由于求和的项数随着隐变量的数目以指数级上升，因此代价太大。 $EM$ 算法可以视作一个非梯度优化算法。无论是梯度下降法，还是 $EM$ 算法，都容易陷入局部极小值。

### 收敛性定理

**定理一**：设 $P(\mathbb Y;\theta)$ 为观测数据的似然函数， $\theta^{<i>}$  为 $EM$ 算法得到的参数估计序列， $P(\mathbb Y;\theta^{<i>})$  为对应的似然函数序列，其中 $i=1,2,\cdots$ 。则： $P(\mathbb Y;\theta^{<i>})$  是单调递增的，即：
$$
P(\mathbb Y;\theta^{<i+1>}) \ge P(\mathbb Y;\theta^{<i>}) 
$$

    
**定理二**：设 $L(\theta)=\log P(\mathbb Y;\theta)$ 为观测数据的对数似然函数，  $\theta^{<i>}$ 为 $EM$ 算法得到的参数估计序列， $L(\theta^{<i>})$  为对应的对数似然函数序列，其中 $i=1,2,\cdots$ 。
1. 如果 $P(\mathbb Y;\theta)$ 有上界，则 $L(\theta^{<i>})$ 会收敛到某一个值 $L^{ \ * }$ 。
2. 在函数 $Q(\theta,\theta^{<i>})$ 与 $L(\theta)$ 满足一定条件下，由 $EM$  算法得到的参数估计序列 $\theta^{<i>}$ 的收敛值 $\theta^{ \ * }$ 是 $L(\theta)$  的稳定点。
> 关于 “满足一定条件”：大多数条件下其实都是满足的。

定理二只能保证参数估计序列收敛到对数似然函数序列的稳定点 $L^{ \ * }$  ，不能保证收敛到极大值点。
    
$EM$ 算法的收敛性包含两重意义：
1. 关于对数似然函数序列 $L(\theta^{<i>})$ 的收敛。
2. 关于参数估计序列 $\theta^{<i>}$ 的收敛。

前者并不蕴含后者。
    
实际应用中， $EM$ 算法的参数的初值选择非常重要。
1. 参数的初始值可以任意选择，但是 $EM$  算法对初值是敏感的，选择不同的初始值可能得到不同的参数估计值。
2. 常用的办法是从几个不同的初值中进行迭代，然后对得到的各个估计值加以比较，从中选择最好的（对数似然函数最大的那个）。

 $EM$ 算法可以保证收敛到一个稳定点，不能保证得到全局最优点。其优点在于：简单性、普适性。
    

## EM算法与高斯混合模型

### 高斯混合模型
&emsp;&emsp;高斯混合模型( $Gaussian mixture model,GMM$ )：指的是具有下列形式的概率分布模型：
$$
P(y;\theta)=\sum_{k=1}^{K}\alpha_k\phi(y;\theta_k)
$$    
其中 $\alpha  _  k$ 是系数，满足 ：
1.  $\alpha  _  k \ge 0,\sum  _  {k=1}^K \alpha  _  k=1$ 。
2.  $\phi(y;\theta  _  k)$ 是高斯分布密度函数，称作第 $k$ 个分模型， $\theta  _  k=(\mu  _  k,\sigma  _  k^{2})$ ：
    $$
    \phi(y;\theta_k)=\frac{1}{\sqrt{2\pi}\sigma_k}\exp\left(-\frac{(y-\mu_k)^{2}}{2\sigma_k^{2}}\right)
    $$

如果用其他的概率分布密度函数代替上式中的高斯分布密度函数，则称为一般混合模型。
    

### 参数估计
&emsp;&emsp;假设观察数据 $\mathbb Y=\{y  _  1,y  _  2,\cdots,y  _  N\}$ 由高斯混合模型 $ P(y;\theta)=\sum  _  {k=1}^{K}\alpha  _  k\phi(y;\theta  _  k)$ 生成，其中  $\theta=(\alpha  _  1,\alpha  _  2,\cdots,\alpha  _  K;\theta  _  1,\theta  _  2,\cdots,\theta  _  K)$ 。可以通过 $EM$ 算法估计高斯混合模型的参数 $\theta$ 。

&emsp;&emsp;可以设想观察数据 $y  _  j$ 是这样产生的：  
1. 首先以概率 $\alpha  _  k$ 选择第 $k$ 个分模型 $\phi(y;\theta  _  k)$ 。
2. 然后以第 $k$ 个分模型的概率分布 $\phi(y;\theta  _  k)$ 生成观察数据 $y  _  j$ 。

这样，观察数据 $y  _  j$ 是已知的，观测数据 $y  _  j$ 来自哪个分模型是未知的。对观察变量  $y$ ，定义隐变量 $z$ ，其中 $p(z=k)=\alpha  _  k$ 。
    
&emsp;&emsp;完全数据的对数似然函数为：

$$
P(y=y_j,z=k;\theta)=\alpha_k\frac{1}{\sqrt{2\pi}\sigma_k}\exp\left(-\frac{(y_j-\mu_k)^{2}}{2\sigma_k^{2}}\right)
$$

其对数为：

$$
\log P(y=y_j,z=k;\theta)=\log \alpha_k-\log\sqrt{2\pi}\sigma_k -\frac{(y_j-\mu_k)^{2}}{2\sigma_k^{2}}
$$

后验概率为：

$$
P(z=k\mid y=y_j;\theta^{<i>})=\frac{\alpha_k\frac{1}{\sqrt{2\pi}\sigma_k^{<i>}}\exp\left(-\frac{(y_j-\mu_k^{<i>})^{2}}{2\sigma_k^{^{<i>}2}}\right)}{\sum_{t=1}^K\alpha_t\frac{1}{\sqrt{2\pi}\sigma_t^{<i>}}\exp\left(-\frac{(y_j-\mu_t^{<i>})^{2}}{2\sigma_t^{^{<i>}2}}\right)}
$$

即： $P(z=k\mid y=y  _  j;\theta^{<i>})=\frac{P(y=y  _  j,z=k;\theta^{<t>})}{\sum  _  {t=1}^KP(y=y  _  j,z=t;\theta^{})}$ 。则 $Q$ 函数为：

$$
\begin{aligned}
	Q\left( \theta ,\theta ^{<i>} \right) &=\sum_{j=1}^N{\left( \sum_Z{P\left( z|y=y_j;\theta ^{<i>} \right) \log P\left( y=y_j,z;\theta \right)} \right)}\\
	&=\sum_{j=1}^N{\sum_{k=1}^K{P\left( z=k|y=y_j;\theta ^{<i>} \right) \left( \log \alpha _ k-\log \sqrt{2\pi}\sigma _ k-\frac{\left( y_j-\mu _ k \right) ^2}{2\sigma _ {k}^{2}} \right)}}\\
\end{aligned}
$$
    
&emsp;&emsp;求极大值： $\theta^{<i+1>}=\arg\max  _  {\theta}Q(\theta,\theta^{<i>})$ 。根据偏导数为 $0$ ，以及 $\sum  _  {k=1}^{K}\alpha  _  k=1$ 得到：
    
1.$\alpha  _  k$ ：
    $$
    \alpha_k^{<i+1>}=\frac{n_k}{N}
    $$
    其中： $n  _  k=\sum  _  {j=1}^NP(z=k\mid y=y  _  j;\theta^{<i>})$  ，其物理意义为：所有的观测数据 $\mathbb Y$ 中，产生自第 $k$  个分模型的观测数据的数量。
        
2.$\mu  _  k$ ：
    $$
    \mu_k^{<i+1>}=\frac{\overline {Sum}_ k}{n_k}
    $$

    其中： $\overline {Sum}  _  k=\sum  _  {j=1}^N y  _  j P(z=k\mid y=y  _  j;\theta^{<i>})$  ，其物理意义为：所有的观测数据 $\mathbb Y$ 中，产生自第 $k$  个分模型的观测数据的总和。
        
3.$\sigma^2$ ：
    $$
    \sigma_k^{<i+1>2}=\frac{\overline {Var}_ k}{n_k}
    $$   
    其中：$\overline {Var}  _  k=\sum  _  {j=1}^N (y  _  j-\mu  _  k^{<i>})^2P(z=k\mid y=y  _  i;\theta^{<i>})$ ，其物理意义为：所有的观测数据 $\mathbb Y$ 中，产生自第 $k$ 个分模型的观测数据，偏离第 $k$ 个模型的均值 $\mu  _  k^{<i>}$ 的平方和。
        
#### 伪码
输入：
1. 观察数据 $\mathbb Y=\{y  _  1,y  _  2,\cdots,y  _  N\}$ 
2. 高斯混合模型的分量数 $K$ 

输出：高斯混合模型参数 $\theta=(\alpha  _  1,\alpha  _  2,\cdots,\alpha  _  K;\mu  _  1,\mu  _  2,\cdots,\mu  _  K;\sigma^2  _  1,\sigma^2  _  2,\cdots,\sigma^2  _  K)$ 

算法步骤：
1. 随机初始化参数 $\theta^{<0>}$ 。
2. 根据 $\theta^{<i>}$ 迭代求解 $\theta^{<i+1>}$ ，停止条件为：对数似然函数值或者参数估计值收敛。
    $$
    \alpha_k^{<i+1>}=\frac{n_k}{N},\;\mu_k^{<i+1>}=\frac{\overline {Sum}_ k}{n_k},\;\sigma_k^{<i+1>2}=\frac{\overline {Var}_ k}{n_k}
    $$
    其中：
    1.  $n  _  k=\sum  _  {j=1}^NP(z=k\mid y=y  _  j;\theta^{<i>})$ 。其物理意义为：所有的观测数据 $\mathbb Y$ 中，产生自第  $k$ 个分模型的观测数据的数量。         
    2.  $\overline {Sum}  _  k=\sum  _  {j=1}^N y  _  j P(z=k\mid y=y  _  j;\theta^{<i>})$ 。其物理意义为：所有的观测数据 $\mathbb Y$ 中，产生自第  $k$ 个分模型的观测数据的总和。       
    3.  $\overline {Var}  _  k=\sum  _  {j=1}^N (y  _  j-\mu  _  k^{<i>})^2P(z=k\mid y=y  _  i;\theta^{<i>})$ 。其物理意义为：所有的观测数据 $\mathbb Y$ 中，产生自第  $k$ 个分模型的观测数据，偏离第 $k$ 个模型的均值 $\mu  _  k^{<i>}$ 的平方和。
                

## EM 算法与 kmeans 模型
&emsp;&emsp; $kmeans$ 算法：给定样本集 $\mathbb D=\{x  _  1,x  _  2,\cdots,N  _  N\}$ ，针对聚类所得簇划分 $C=\{\mathbb C  _  1,\mathbb C  _  2,\cdots,\mathbb C  _  K\}$ ，最小化平方误差：

$$
\min_{C} \sum_{k=1}^{K}\sum_{x_i \in \mathbb C_k}||x_i-\mu_k||_ 2^{2}
$$

其中 $\mu  _  k=\frac {1}{|\mathbb C  _  k|}\sum  _  {x  _  i \in \mathbb C  _  k}x  _  i$ 是簇 $\mathbb C  _  k$ 的均值向量。
    
&emsp;&emsp;定义观测随机变量为 $x$ ，观测数据为 $\mathbb D$ 。定义隐变量为 $z$ ，它表示 $x$ 所属的簇的编号。设参数 $\theta= (\mathbf {\mu}  _  1,\mathbf {\mu}  _  2,\cdots,\mathbf {\mu}  _  K)$ ，则考虑如下的生成模型：

$$
P(x,z\mid\theta) \propto \begin{cases}\exp(-||x-\mathbf {\mu}_ z||_ 2^2)\quad &||x-\mathbf {\mu}_z||_2^2=\min_ {1\le k\le K}||x-\mathbf {\mu}_ k||_ 2^2\\ 0\quad &||x-\mathbf {\mu}_ z||_ 2^2\gt \min_ {1\le k\le K}||x-\mathbf {\mu}_ k||_ 2^2 \end{cases}
$$

其中 $\min  _  {1\le k\le K}||x-\mathbf {\mu}  _  k||  _  2^2$ 表示距离 $\mathbf{\vec x}$ 最近的中心点所在的簇编号。即：
1. 若 $x$ 最近的簇就是 $\mathbf{\mu}  _  z$ 代表的簇，则生成概率为 $\exp(-||x-\mathbf {\mu}  _  z||  _  2^2)$ 。
2. 若 $x$ 最近的簇不是 $\mathbf{\mu}  _  z$ 代表的簇，则生成概率等于 $0$ 。

&emsp;&emsp;计算后验概率：

$$
P(z\mid x,\theta^{<i>})\propto \begin{cases} 1\quad &||x_i-\mathbf {\mu}_ z||_ 2^2=\min_ {1\le k\le K}||x-\mathbf {\mu}_ k^{<i>}||_ 2^2\\ 0\quad &||x_i-\mathbf {\mu}_ z||_ 2^2\gt \min_ {1\le k\le K}||x-\mathbf {\mu}_ k^{<i>}||_ 2^2 \end{cases}
$$

即：
1. 若 $x$ 最近的簇就是 $\mathbf{\mu}  _  z$ 代表的簇，则后验概率为 $1$ 。
2. 若 $x$ 最近的簇不是 $\mathbf{\mu}  _  z$ 代表的簇，则后验概率为 $0$ 。

&emsp;&emsp;计算 Q 函数：

$$
 Q(\theta,\theta^{<i>})=\sum_{j=1}^N\left(\sum_z P(z\mid x=x_j;\theta^{<i>})\log P(x=x_j,z;\theta) \right)
$$

设距离 $x  _  j$ 最近的聚类中心为 $\mathbf{\mu}  _  {t  _  j}^{<i>}$ ，即它属于簇 $t  _  j$ ，则有：

$$
Q(\theta,\theta^{<i>})=\text{const}-\sum_{j=1}^N ||x_j-\mu_{t_j}||_ 2^2
$$

则有：

$$
\theta^{<i+1>}=\arg\max_\theta Q(\theta,\theta^{<i>})=\arg\min_\theta \sum_{j=1}^N ||x_j-\mu_{t_j}||_ 2^2
$$

定义集合 $\mathbb I  _  k=\{j\mid t  _  j=k\},\quad k=1,2\cdots,K$ ，它表示属于簇 $k$ 的样本的下标集合。则有：

$$
\sum_{j=1}^N ||x_j-\mu_{t_j}||_ 2^2=\sum_{k=1}^K\sum_{j\in \mathbb I_k} ||x_j-\mu_k||_ 2^2
$$

则有：

$$
 \theta^{<i+1>}=\arg\min_\theta\sum_{k=1}^K\sum_{j\in \mathbb I_k} ||x_j-\mu_k||_ 2^2
$$

这刚好就是 $k-means$ 算法的目标：最小化平方误差。
    
&emsp;&emsp;由于求和的每一项都是非负的，则当每一个内层求和 $\sum  _  {j\in \mathbb I  _  k}||x  _  j-\mathbf{\mu}  _  {k}||  _  2^2$ 都最小时，总和最小。即：

$$
\mu^{<i+1>}_ k=\arg\min_{\mu_k}\sum_{j\in \mathbb I_k}||x_j-\mathbf{\mu}_ {k}||_ 2^2
$$

得到： $\vec \mu  _  k^{<i+1>}=\frac {1}{|\mathbb I  _  k|}\sum  _  {j \in \mathbb I  _  k}x  _  j$ ，其中 $|\mathbb I  _  k|$ 表示集合 $|\mathbb I  _  k|$ 的大小。这就是求平均值来更新簇中心。
    

## EM 算法的推广

### F 函数
$F$ 函数：假设隐变量 $Z$ 的概率分布为 $\tilde P( Z)$ ，定义分布 $\tilde P( Z )$ 与参数 $\theta$ 的函数 $F(\tilde P,\theta)$ 为：

$$
F(\tilde P,\theta)=\mathbb E_{\tilde P}[\log P( Y, Z ;\theta)]+H(\tilde P)
$$

其中 $H(\tilde P)=-\mathbb E  _  {\tilde P}\log \tilde P$ 是分布 $\tilde P( Z )$ 的熵。
> 通常假定 $P( Y,Z ;\theta)$ 是 $\theta$ 的连续函数，因此 $F(\tilde P,\theta)$ 为 $\tilde P( Z )$ 和 $\theta$ 的连续函数。
    
函数 $F(\tilde P,\theta)$ 有下列重要性质：
1. 对固定的 $\theta$ ，存在唯一的分布 $\tilde P  _  {\theta}( Z )$ 使得极大化 $F(\tilde P,\theta)$ 。此时 $\tilde P  _  {\theta}( Z )=P( Z \mid Y;\theta)$ ，并且 $\tilde P  _  {\theta}$ 随着 $\theta$ 连续变化。
2. 若 $\tilde P  _  {\theta}( Z )=P( Z \mid Y;\theta)$ ， 则 $F(\tilde P,\theta)=\log P( Y;\theta)$ 。

定理一：设 $L(\theta)=\log P(\mathbb Y;\theta)$ 为观测数据的对数似然函数， $\theta^{<i>}$ 为 $EM$ 算法得到的参数估计序列，函数 $F(\tilde P,\theta)=\sum  _  Y\mathbb E  _  {\tilde P}[\log P(Y,Z ;\theta)]+H(\tilde P)$ ，则：
1. 如果 $F(\tilde P,\theta)$ 在 $\tilde P^{ \ * }(Z )$ 和 $\theta^{ \ * }$ 有局部极大值，那么 $L(\theta)$ 也在 $\theta^{ \ * }$ 有局部极大值。
2. 如果 $F(\tilde P,\theta)$ 在 $\tilde P^{ \ * }( Z )$ 和 $\theta^{ \ * }$ 有全局极大值，那么 $L(\theta)$ 也在 $\theta^{ \ * }$ 有全局极大值。

定理二： $EM$ 算法的一次迭代可由 $F$ 函数的极大-极大算法实现：设 $\theta^{<i>}$ 为第 $i$ 次迭代参数 $\theta$ 的估计， $\tilde P^{<i>}$ 为第 $i$ 次迭代函数 $\tilde P(Z )$ 的估计。在第 $i+1$ 次迭代的两步为：
1. 对固定的 $\theta^{<i>}$ ，求 $\tilde P^{<i+1>}$ 使得 $F(\tilde P,\theta^{<i>})$ 极大化。
2. 对固定的 $\tilde P^{<i+1>}$ ，求 $\theta^{<i+1>}$ 使得 $F(\tilde P^{<i+1>},\theta)$ 极大化。

### GEM算法1
该算法的问题是，有时候求 $F(\tilde P^{<i+1>},\theta)$ 极大化很困难。

#### 伪码
输入：
1. 观测数据 $\mathbb Y=\{y  _  1,y  _  2,\cdots\}$ 
2.  $F$ 函数

输出：模型参数

算法步骤：
1. 初始化参数 $\theta^{<0>}$ ，开始迭代。
2. 第 $i+1$ 次迭代：
    1. 记 $\theta^{<i>}$ 为参数 $\theta$ 的估计值， $\tilde P^{<i>}$ 为函数 $\tilde P$ 的估计值。求 $\tilde P^{<i+1>}$ 使得 $F(\tilde P,\theta^{<i>})$ 极大化。
    2. 求 $\theta^{<i+1>}$ 使得 $F(\tilde P^{<i+1>},\theta)$ 极大化。
    3. 重复上面两步直到收敛。

### GEM算法2
此算法不需要求 $Q(\theta,\theta^{<i>})$ 的极大值，只需要求解使它增加的 $\theta^{<i+1>}$ 即可。

#### 伪码
输入：
1. 观测数据 $\mathbb Y=\{y  _  1,y  _  2,\cdots\}$ 
2.  $Q$ 函数

输出：模型参数

算法步骤：
1. 初始化参数 $\theta^{<0>}$ ，开始迭代。
2. 第 $i+1$ 次迭代：
    1. 记 $\theta^{<i>}$ 为参数 $\theta$ 的估计值，计算
        $$
        Q(\theta,\theta^{<i>})=\sum_{j=1}^N\left(\sum_Z P(Z\mid Y=y_j;\theta^{<i>})\log P(Y=y_j,Z;\theta) \right)
        $$
    2. 求 $\theta^{<i+1>}$ 使得 $Q(\theta^{<i+1>},\theta^{<i>}) \gt Q(\theta^{<i>},\theta^{<i>})$ 
    3. 重复上面两步，直到收敛。


### GEM算法3
该算法将 $EM$ 算法的 $M$ 步分解为 $d$ 次条件极大化，每次只需要改变参数向量的一个分量，其余分量不改变。

#### 伪码
输入：
1. 观测数据 $\mathbb Y=\{y  _  1,y  _  2,\cdots\}$ 
2.  $Q$ 函数

输出：模型参数

算法步骤：
1. 初始化参数 $\theta^{<0>}=(\theta  _  1^{<0>},\theta  _  2^{<0>},\cdots,\theta  _  d^{<0>})$ ，开始迭代
2. 第 $i+1$ 次迭代：
    1. 记 $\theta^{<i>}=(\theta  _  1^{<i>},\theta  _  2^{<i>},\cdots,\theta  _  d^{<i>})$ 为参数 $\theta=(\theta  _  1,\theta  _  2,\cdots,\theta  _  d)$ 的估计值，计算
        $$
        Q(\theta,\theta^{<i>})=\sum_{j=1}^N\left(\sum_Z P(Z\mid Y=y_j;\theta^{<i>})\log P(Y=y_j,Z;\theta) \right)
        $$
    2. 进行 d 次条件极大化：
        1. 首先在 $\theta  _  2^{<i>},\cdots,\theta\  _  d^{<i>}$ 保持不变的条件下求使得 $Q(\theta,\theta^{<i>})$ 达到极大的 $\theta  _  1^{<i+1>}$ 
        2. 然后在 $\theta  _  1=\theta  _  1^{<i+1>},\theta  _  j=\theta  _  j^{<i>},j=3,\cdots,d$ 的条件下求使得 $Q(\theta,\theta^{<i>})$ 达到极大的 $\theta  _  2^{<i+1>}$ 
        3. 如此继续，经过 $d$ 次条件极大化，得到 $\theta^{<i+1>}=(\theta  _  1^{<i+1>},\theta  _  2^{<i+1>},\cdots,\theta  _  d^{<i+1>})$ ，使得 $Q(\theta^{<i+1>},\theta^{<i>}) \gt Q(\theta^{<i>},\theta^{<i>})$ 
    3. 重复上面两步，直到收敛。
