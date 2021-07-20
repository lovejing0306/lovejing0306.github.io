---
layout: post
title: 最大熵算法
categories: [MachineLearning]
description: 最大熵算法
keywords: MachineLearning
---


最大熵算法
---


## 最大熵模型MEM
&emsp;&emsp;设随机变量 $X$ 的概率分布为 $P(X)$ ，熵为：

$$
H(P)=-\sum_X P(X)\log P(X)
$$

可以证明：

$$
0 \le H(P) \le \log |X| 
$$

其中 $\ | X \ | $ 为 $X$ 的取值的个数。
> 当且仅当 $X$ 的分布为均匀分布时有 $H(P)=\log |X|$ 。即 $P(X)=\frac{1}{|X|}4$ 时熵最大。
    

### 最大熵原理
学习概率模型时，在所有可能的概率模型（即概率分布）中，熵最大的模型是最好的。
* 通常还有其他已知条件来确定概率模型的集合，因此最大熵原理为：在满足已知条件的情况下，选取熵最大的模型。
* 在满足已知条件前提下，如果没有更多的信息，则那些不确定部分都是“等可能的”。而等可能性通过熵最大化来刻画。

最大熵原理选取熵最大的模型，而决策树的划分目标选取熵最小的划分。原因在于：
* 最大熵原理认为在满足已知条件之后，选择不确定性最大（即：不确定的部分是等可能的）的模型。也就是不应该再施加任何额外的约束。因此这是一个求最大不确定性的过程，所以选择熵最大的模型。
* 决策树的划分目标是为了通过不断的划分从而不断的降低实例所属的类的不确定性，最终给实例一个合适的分类。因此这是一个不确定性不断减小的过程，所以选取熵最小的划分。
        

### 期望的约束
&emsp;&emsp;一种常见的约束为期望的约束：

$$
\mathbb E[f(X)]=\sum_XP(X)f(X)=\tau
$$

其中 $f(\cdot)$ 代表随机变量 $X$ 的某个函数（其结果是另一个随机变量）。
1. 其物理意义为：随机变量 $\tilde X=f(X)$ 的期望是一个常数。
2. 示例：当 $f(X)=X$ 时，约束条件为： $\mathbb E[X]=\tau$ ，即随机变量 $X$ 的期望为常数。

&emsp;&emsp;如果有多个这样的约束条件：

$$
\begin{array}{c}
	\mathbb{E}\left[ f_1\left( X \right) \right] =\sum_X{P\left( X \right) f_1\left( X \right)}=\tau _ 1\\
	\vdots\\
	\mathbb{E}\left[ f_k\left( X \right) \right] =\sum_X{P\left( X \right) f_k\left( X \right)}=\tau _ k\\
\end{array}
$$

则需要求解约束最优化问题：

$$
\begin{array}{c}
\max_{P(X)} -\sum_X P(X)\log P(X)\\
s.t. \sum_XP(X)=1,0\le P\le 1\\
\mathbb E[f_1(X)]=\sum_XP(X)f_1(X)=\tau_1\\ 
\vdots\\
\mathbb E[f_k(X)]=\sum_XP(X)f_k(X)=\tau_k
\end{array}
$$
    
&emsp;&emsp;给出拉格朗日函数：

$$
L(P)= -\sum_X P(X)\log P(X)+\lambda_0\left(\sum_XP(X)-1\right)+\lambda_1\left(\sum_XP(X)f_1(X)-\tau_1\right)\ +\cdots+\lambda_k\left(\sum_XP(X)f_k(X)-\tau_k\right)
$$

可以求得：

$$
P(X)=\frac 1Z\exp\left(-\sum_{i=1}^k\lambda_if_i(X)\right),\quad Z=\sum_X\exp\left(-\sum_{i=1}^k\lambda_if_i(X)\right)
$$

将 $P(X)$ 代入，有：

$$
\begin{array}{c}
 \mathbb E[f_1(X)]=\sum_X\frac {f_1(X)}{Z}\exp\left(-\sum_{i=1}^k\lambda_if_i(X)\right)=\tau_1\\ 
 \vdots\\ 
 \mathbb E[f_k(X)]=\sum_X\frac {f_k(X)}{Z}\exp\left(-\sum_{i=1}^k\lambda_if_i(X)\right)=\tau_k
\end{array}
$$

则可以求解出各个 $\lambda  _  i$ 。该式子并没有解析解，而且数值求解也相当困难。

&emsp;&emsp;当只有一个约束 $f(X)=X$ 时，表示约束了变量的期望，即 $\sum  _  XP(X)X=\tau$ 。此时有：

$$
P(X)=\frac{\exp(-\lambda X)}{\sum_X\exp(-\lambda X)}
$$

代入 $\sum  _  XP(X)X=\tau$ ，解得：

$$
P(X)=\frac 1\tau \exp\left(-\frac X\tau\right) 
$$

即：约束了随机变量期望的分布为指数分布。
    
&emsp;&emsp;当有两个约束 $f  _  1(X)=X,\quad f  _  2(X)=X^2$ 时，表示约束了变量的期望和方差。即：

$$
\sum_X P(X)X=\tau_1,\quad\sum_X P(X) X^2=\tau_2
$$

此时有：

$$
P(X)=\frac{\exp(-\lambda_1 X-\lambda_2 X)}{\sum_X\exp(-\lambda_1 X-\lambda_2 X)}
$$

代入约束可以解得：

$$
P(X)=\sqrt{\frac {1}{2\pi(\tau_2-\tau_1^2)}}\exp\left(-\frac{(X-\tau_1)^2}{2(\tau_2-\tau_1^2)}\right)
$$

它是均值为 $\tau  _  1$ ，方差为 $\tau  _  2-\tau  _  1^2$ 的正态分布。即：约束了随机变量期望、方差的分布为正态分布。
    

## 分类任务最大熵模型
&emsp;&emsp;设分类模型是一个条件概率分布 $P(Y\mid X=x),X \in \mathcal X \subseteq \mathbb R^{n}$  为输入， $Y \in \mathcal Y$ 为输出。给定一个训练数据集 $\mathbb D=\{(x  _  1,y  _  1),(x  _  2,y  _  2),\cdots,(x  _  N,y  _  N)\}$ ，学习的目标是用最大熵原理选取最好的分类模型。
    
### 最大熵模型
&emsp;&emsp;根据训练集 $\mathbb D$ ，可以得到联合分布 $P(X,Y)$ 的经验分布 $\tilde P(X,Y)$ 和 $P(X)$ 的经验分布 $\tilde P(X)$ ：

$$
\begin{aligned}
&\tilde P(X=x,Y=y)=\frac{\upsilon(X=x, Y=y)}{N} ,\quad x \in\mathcal X,y\in \mathcal Y\\
&\tilde P(X)=\frac{\upsilon(X=x)}{N} ,\quad x \in\mathcal X
\end{aligned}
$$

其中 $N$ 为样本数量， $\upsilon$ 为频数。
    
&emsp;&emsp;用特征函数 $f(x,y)$ 描述输入 $x$ 和输出 $y$ 之间的某个事实：

$$
f(x,y)= \begin{cases} 1, & \text{if $x,y$ statisfy the fact.} \\
0, & \text{or else.} \end{cases}
$$

特征函数是一个二值函数，但是理论上它也可以取任意值。
        
&emsp;&emsp;特征函数 $f(x,y)$ 关于经验分布 $\tilde P(X,Y)$ 的期望定义为

$$
\mathbb E_{\tilde P}[f]:\mathbb E_{\tilde P}[f]=\sum_{x,y}\tilde P(x,y)f(x,y)
$$

这个期望其实就是约束 $f$ 在训练集上的统计结果的均值（也就是约束 $f$ 出现的期望的估计量）。
1. 如果 $f$ 取值为二值 $0,1$ ，则表示约束 $f$ 在训练集上出现的次数的均值。
2. 如果 $f$ 取值为任意值，则表示约束 $f$ 在训练集上累计的结果的均值。

特征函数 $f(x,y)$ 关于模型 $P(Y\mid X)$ 与经验分布 $\tilde P(X)$ 的期望用 $\mathbb E  _  {P}[f]$ 表示：

$$
\mathbb E_{P}[f]=\sum_{x,y}\tilde P(x)P(y\mid x)f(x,y)
$$

理论上 $\mathbb E  _  {P}[f]=\sum  _  {x,y} P(x)P(y\mid x)f(x,y)$ ，这里使用 $\tilde P(x)$ 作为 $P(x)$ 的估计。可以假设这两个期望相等，即： $\mathbb E  _  {\tilde P}[f]=\mathbb E  _  {P}[f]$ 。
1.  $\tilde P(x,y)$ 在 $(x,y)\not\in\mathbb D$ 时为 $0$ ，在 $(x,y) \in\mathbb D$ 才有可能非 $0$ 。因此  $\mathbb E  _  {\tilde P}[f]=\sum  _  {x,y}\tilde P(x,y)f(x,y)$ 仅仅在 $(x,y) \in\mathbb D$ 上累加。
2.  $\tilde P(x)$  在 $x \not\in\mathbb D$ 时为 $0$ ，在 $x \in\mathbb D$ 才有可能非 $0$ 。因此 $\mathbb E  _  {P}[f]=\sum  _  {x,y}\tilde P(x)P(y\mid x)f(x,y)$ 仅在 $x \in \mathbb D$ 上累加。

&emsp;&emsp;理论上，由于 $P(y\mid x)=\frac{P(x,y)}{P(x)}$ ，看起来可以使用 $\frac{\tilde P(x,y)}{\tilde P(x)}$ 作为  $P(y\mid x)$ 的一个估计。但是这个估计只考虑某个点 $(x,y)$ 上的估计，并未考虑任何约束。所以这里通过特征函数的两种期望相等来构建在数据集整体上的最优估计。
    
&emsp;&emsp;最大熵模型：假设有 $n$ 个约束条件 $f  _  i(x,y),i=1,2,\cdots,n$ ，满足所有约束条件的模型集合为： $C=\{P \in \mathcal P \mid\mathbb E  _  P[f  _  i]=\mathbb E  _  {\tilde P}[f  _  i],i=1,2,\cdots,n\}$ 。定义在条件概率分布 $P(Y\mid X)$ 上的条件熵为：

$$
H(P)=-\sum_{x,y}\tilde P(x)P(y\mid x)\log P(y\mid x)
$$

则模型集合 $C$ 中条件熵最大的模型称为最大熵模型。
    

### 模型求解
&emsp;&emsp;对给定的训练数据集 $\mathbb D=\{(x  _  1,y  _  1),(x  _  2,y  _  2),\cdots,(x  _  N,y  _  N)\}$ ，以及特征函数 $f  _  i(x,y),i=1,2,\cdots,n$  ，最大熵模型的学习等价于约束最优化问题：

$$
\begin{array}{c}
\max_{P \in \mathcal C} H(P)=-\sum_{x,y}\tilde P(x)P(y\mid x)\log P(y\mid x)\\
s.t.\mathbb E_P[f_i]=\mathbb E_{\tilde P}[f_i],i=1,2,\cdots,n\\
\sum_y P(y\mid x)=1
\end{array}
$$

将其转化为最小化问题：

$$
\begin{array}{c}
\min_{P\in \mathcal C} -H(P)=\sum_{x,y}\tilde P(x)P(y\mid x)\log P(y\mid x)\\
s.t.\mathbb E_P[f_i]-\mathbb E_{\tilde P}[f_i]=0,i=1,2,\cdots,n\\ 
\sum_y P(y\mid x)=1
\end{array}
$$

其中：
1. $\tilde P(x),\mathbb E  _  {\tilde P}[f  _  i]=\sum  _  {x,y}\tilde P(x,y)f  _  i(x,y)$ 是已知的。
2. $P(y\mid x),\mathbb E  _  P[f  _  i]=\sum  _  {x,y}\tilde P(x)P(y\mid x)f  _  i(x,y)$ 是未知的。

&emsp;&emsp;将约束最优化的原始问题转换为无约束最优化的对偶问题，通过求解对偶问题来求解原始问题。引入拉格朗日乘子  $w  _  0,w  _  1,\cdots,w  _  n$ ，定义拉格朗日函数 $ L(P,w)$ ：

$$
\begin{aligned}
L(P,w)&=-H(P)+w_0(1-\sum_y P(y\mid x))+\sum_{i=1}^{n}w_i(\mathbb E_{\tilde P}[f_i]-E_P(f_i))\\ &=\sum_{x,y}\tilde P(x)P(y\mid x)\log P(y\mid x)+w_0\left(1-\sum_y P(y\mid x)\right)\ +\sum_{i=1}^{n}w_i\left(\sum_{x,y}\tilde P(x,y)f_i(x,y)-\sum_{x,y}\tilde P(x)P(y\mid x)f_i(x,y)\right)
\end{aligned}
$$

最优化的原始问题是：

$$
\min_{P \in C} \max_{w} L(P,w) 
$$

对偶问题是 

$$
\max_{w} \min_{P \in C} L(P, w) 
$$

由于拉格朗日函数 $L(P,w)$ 是凸函数，因此原始问题的解与对偶问题的解是等价的。
> 求解对偶问题：先求解内部的极小化问题，之后求解对偶问题外部的极大化问题。

&emsp;&emsp;先求解内部的极小化问题：

$$
\min_{P \in C} L(P,w)
$$

它是一个 $w$ 的函数，将其记作：

$$
\Psi(w)=\min_{P \in C} L(P,w)=L(P_w,w) 
$$

先用 $L(P,w)$ 对 $P(y\mid x)$ 求偏导数：

$$
\begin{aligned}
\frac{\partial L(P,w)}{\partial P(y\mid x)}&=\sum_{x,y}\tilde P(x)(\log P(y\mid x)+1)-\sum_y w_0-\sum_{x,y}\left(\tilde P(x)\sum_{i=1}^{n}w_if_i(x,y)\right)\\
&=\sum_{x,y} \tilde P(x)\left(\log P(y\mid x)+1-w_0-\sum_{i=1}^{n}w_i f_i(x,y)\right)
\end{aligned}
$$

令偏导数为 $0$ 。在 $\tilde P(x) \gt 0$ 时，解得：

$$
\begin{aligned}
P(y\mid x)&=\exp\left(\sum_{i=1}^{n}w_i f_i(x,y)+w_0-1\right) \\
&=\frac{\exp\left(\sum_{i=1}^{n}w_i f_i(x,y)\right)}{\exp(1-w_0)}
\end{aligned}
$$

由于 $\sum  _  y P(y\mid x)=1$ ，则有： $\sum  _  y \frac{\exp(\sum  _  {i=1}^{n}w  _  i f  _  i(x,y))}{\exp(1-w  _  0)}=1$  。因此有：

$$
\exp(1-w_0)=\sum_y \exp\left(\sum_{i=1}^{n}w_i f_i(x,y)\right)
$$

定义 $Z  _  w(x)=\sum  _  y \exp\left(\sum  _  {i=1}^{n}w  _  i f  _  i(x,y)\right)$ 为规范因子，则：

$$
P_w(y\mid x)=\frac{1}{Z_w(x)} \exp\left(\sum_{i=1}^{n}w_i f_i(x,y)\right)
$$

由该式表示的模型 $P  _  w=P  _  w(y\mid x)$ 就是最大熵模型。
        
&emsp;&emsp;再求解对偶问题外部的极大化问题：

$$
\max_w \Psi(w)
$$

将其解记作 $w^{ \ * }$ ，即：

$$
w^{ \ * }=\arg\max_w \Psi(w)
$$

求得 $w^{ \ * }$ 之后，用它来表示 $P  _  w=P  _  w(y\mid x)$ ，得到 $P^{ \ * }=P  _  {w^{ \ * }}=P  _  {w^{ \ * }}(y\mid x)$ ，即得到最大熵模型。

&emsp;&emsp;上述过程总结为：
1. 先求对偶问题的内部极小化，得到 $\Psi(w)$ 函数，以及极值点 $P  _  w(y\mid x)$ 。
2. 再求 $\Psi(w)$ 函数的极大值，得到 $w^{ \ * }$ 。
3. 最后将 $w^{ \ * }$ 代入 $P  _  w(y\mid x)$ 得到最终模型 $P^{ \ * }$ 。

&emsp;&emsp; $\Psi(w)$ 函数的最大化，等价于最大熵模型的极大似然估计。证明如下：已知训练数据 $\mathbb D$ 中， $(x,y)$ 出现的频次为 $k  _  {x,y}$ 。则条件概率分布 $P(y\mid x)$ 的对数似然函数为：

$$
\log \prod_{x,y} P(y\mid x)^{k_{x,y}}=\sum_{x,y}k_{x,y}\log P(y\mid x)
$$

将对数似然函数除以常数 $N$ ，考虑到 $\frac{k  _  {x,y}}{N}=\tilde P(x,y)$ ，其中 $\tilde P(x,y)$ 为经验概率分布。则 $P(y\mid x)$ 的对数似然函数为：

$$
\sum_{x,y}\tilde P(x,y) \log P(y\mid x)
$$

再利用：

$$
P_w(y\mid x)=\frac{1}{Z_w(x)} \exp\left(\sum_{i=1}^{n}w_i f_i(x,y)\right)
$$

代入，最后化简合并，最终发现它就是 $\Psi(w)$ 
    

### 最大熵与逻辑回归
&emsp;&emsp;设 $x=(x  _  1,x  _  2,\cdots,x  _  n)^T$ 为 $n$ 维变量，对于二类分类问题，定义 $n$ 个约束：

$$
f_i(x,y)=\begin{cases} x_i,&y=1\\ 
0,&y=0 \end{cases},\quad i=1,2,\cdots,n
$$

根据最大熵的结论，有：

$$
\begin{aligned}
Z_w(x)&=\sum_y \exp\left(\sum_{i=1}^{n}w_i f_i(x,y)\right) \\
&=\exp\left(\sum_{i=1}^{n}w_i f_i(x,y=0)\right)+\exp\left(\sum_{i=1}^{n}w_i f_i(x,y=1)\right)\\ 
&=1+\exp\left(\sum_{i=1}^{n}w_i x_i\right) \\
&=1+\exp(w\cdot x)
\end{aligned}
$$

以及：
$$
\begin{aligned}
P_w(y\mid x)&=\frac{1}{Z_w(x)} \exp\left(\sum_{i=1}^{n}w_i f_i(x,y)\right) \\
&=\frac{1}{1+\exp(w\cdot x)} \exp\left(\sum_{i=1}^{n}w_i f_i(x,y)\right)
\end{aligned}
$$

当 $y=1$ 时有：

$$
\begin{aligned}
 P_w(y=1\mid x)&=\frac{1}{1+\exp(w\cdot x)} \exp\left(\sum_{i=1}^{n}w_i f_i(x,y=1)\right)\\ 
 &=\frac{1}{1+\exp(w \cdot x)} \exp\left(\sum_{i=1}^{n}w_i x_i\right) \\
 &=\frac{\exp(w\cdot x)}{1+\exp(w\cdot x)}
\end{aligned}
$$

当 $y=0$ 时有：

$$
\begin{aligned}
P_w(y=0\mid x)&=\frac{1}{1+\exp(w \cdot x)} \exp\left(\sum_{i=1}^{n}w_i f_i(x,y=0)\right) \\
&=\frac{1}{1+\exp(w\cdot x)}
\end{aligned}
$$

最终得到：

$$
\log \frac{P_w(y=1\mid x)}{P_w(y=0\mid x)}=w\cdot x
$$

这就是逻辑回归模型。
    

## 最大熵的学习
1. 最大熵模型的学习就是在给定训练数据集 $\mathbb D$ 时，对模型进行极大似然估计或者正则化的极大似然估计。
2. 最大熵模型与 $logistic$ 回归模型有类似的形式，它们又称为对数线性模型。
    1. 它们的目标函数具有很好的性质：光滑的凸函数。因此有多种最优化方法可用，且保证能得到全局最优解。
    2. 最常用的方法有：改进的迭代尺度法、梯度下降法、牛顿法、拟牛顿法。

### 改进的迭代尺度法
&emsp;&emsp;改进的迭代尺度法( $Improved Iterative Scaling:IIS$ )是一种最大熵模型学习的最优化算法。已知最大熵模型为：

$$
P_w(y\mid x)=\frac{1}{Z_w(x)} \exp\left(\sum_{i=1}^{n}w_i f_i(x,y)\right)
$$

其中  

$$
Z_w(x)=\sum_y \exp\left(\sum_{i=1}^{n}w_i f_i(x,y)\right)
$$

对数似然函数为：

$$
\begin{aligned}
L(x)=\log \prod_{x,y}P_w(y\mid x)^{\tilde P(x,y)}\\
=\sum_{x,y}[\tilde P(x,y) \log P_w(y\mid x)]\\ 
=\sum_{x,y}\left(\tilde P(x,y)\sum_{i=1}^{n}w_i f_i(x,y)\right)-\sum_{x}\left(\tilde P(x)\log Z_w(x)\right)
\end{aligned}
$$

最大熵模型的目标是：通过极大化似然函数学习模型参数，求出使得对数似然函数最大的参数 $\hat{w}$ 。
    
&emsp;&emsp; $IIS$ 原理：假设最大熵模型当前的参数向量是 $w=(w  _  1,w  _  2,\cdots,w  _  n)^{T}$ ，希望找到一个新的参数向量 $w +\delta=(w  _  1+\delta  _  1,w  _  2+\delta  _  2,\cdots,w  _  n+\delta  _  n)^{T}$ ，使得模型的对数似然函数值增大。若能找到这样的新参数向量，则更新 

$$
w \leftarrow w+\delta 
$$

重复这一过程，直到找到对数似然函数的最大值。

&emsp;&emsp;对于给定的经验分布 $\tilde P(x,y)$ ，模型参数从 $w$ 到 $w+ \delta$ 之间，对数似然函数的改变量为：

$$
 L(w+ \delta)-L(w)=\sum_{x,y}\left(\tilde P(x,y)\sum_{i=1}^{n}\delta_i f_i(x,y)\right)-\sum_{x}\left(\tilde P(x)\log \frac{Z_{w+\delta}(x)}{Z_w(x)}\right)
$$

利用不等式：当 $\alpha \gt 0$ 时 $-\log \alpha \ge 1-\alpha$ , 有：

$$
L(w+\delta)-L(w) \ge \sum_{x,y}\left(\tilde P(x,y)\sum_{i=1}^{n}\delta_i f_i(x,y)\right)+\sum_{x}\left[\tilde P(x)\left(1-\frac{Z_{w+\delta}(x)}{Z_w(x)}\right)\right]\ =\sum_{x,y}\left(\tilde P(x,y)\sum_{i=1}^{n}\delta_i f_i(x,y)\right)+\sum_{x}\tilde P(x)-\sum_{x}\left(\tilde P(x) \frac{Z_{w+\delta}(x)}{Z_w(x)} \right)
$$

考虑到 $\sum  _  {x}\tilde P(x) =1$ ，以及：

$$
\begin{aligned}
\frac{Z_{w+\delta}(x)}{Z_w(x)} &=\frac{\sum_y \exp\left(\sum_{i=1}^{n}(w_i+\delta_i) f_i(x,y)\right)}{Z_w(x)}\\ 
&=\frac {1}{Z_w(x)}\sum_y \left[\exp\left(\sum_{i=1}^{n}w_i f_i(x,y)\right)\cdot \exp\left(\sum_{i=1}^{n} \delta_i f_i(x,y)\right)\right]\\ 
&=\sum_y \left[\frac {1}{Z_w(x)} \cdot \exp\left(\sum_{i=1}^{n}w_i f_i(x,y)\right)\cdot \exp\left(\sum_{i=1}^{n} \delta_i f_i(x,y)\right)\right]
\end{aligned}
$$

根据 $P  _  w(y\mid x)=\frac{1}{Z  _  w(x)} \exp\left(\sum  _  {i=1}^{n}w  _  i f  _  i(x,y)\right)$ 有：

$$
\frac{Z_{w+\delta}(x)}{Z_w(x)} =\sum_y \left[P_w(y\mid x)\cdot \exp\left(\sum_{i=1}^{n} \delta_i f_i(x,y)\right)\right]
$$

则有：

$$
 L(w+\delta)-L(w) \ge \sum_{x,y}\left(\tilde P(x,y)\sum_{i=1}^{n}\delta_i f_i(x,y)\right)+1\ -\sum_x \left[\tilde P(x) \sum_y\left(P_w(y\mid x)\exp\sum_{i=1}^{n}\delta_if_i(x,y)\right)\right]
$$

令

$$
A(\delta\mid w)=\sum_{x,y}\left(\tilde P(x,y)\sum_{i=1}^{n}\delta_i f_i(x,y)\right)+1\ -\sum_x \left[\tilde P(x) \sum_y \left(P_x(y\mid x)\exp\sum_{i=1}^{n}(\delta_i f_i(x,y))\right)\right)]
$$

则 $L(w+\delta)-L(w) \ge A(\delta\mid w)$ 。
        
&emsp;&emsp;如果能找到合适的 $\delta$ 使得 $A(\delta\mid w)$ 提高，则对数似然函数也会提高。但是 $\delta$  是个向量，不容易同时优化。
1. 一个解决方案是：每次只优化一个变量 $\delta  _  i$ 。
2. 为达到这个目的，引入一个变量 $f^{o}(x,y)=\sum  _  {i=1}^{n}f  _  i(x,y)$ 。

&emsp;&emsp; $A(\delta\mid w)$ 改写为：

$$
A(\delta\mid w)=\sum_{x,y}\left(\tilde P(x,y)\sum_{i=1}^{n}\delta_i f_i(x,y)\right)+1\ -\sum_x \left[\tilde P(x) \sum_y \left(P_w(y\mid x)\exp \left(f^{o}(x,y)\sum_{i=1}^{n}\frac{\delta_if_i(x,y)}{f^{o}(x,y)}\right)\right)\right]
$$

利用指数函数的凸性，根据

$$
 \frac{f_i(x,y)}{f^{o}(x,y)} \ge 0,\quad \sum_{i=1}^{n}\frac{f_i(x,y)}{f^{o}(x,y)}=1
$$

以及 $Jensen$ 不等式有：

$$
\exp\left(f^{o}(x,y)\sum_{i=1}^{n}\frac{\delta_if_i(x,y)}{f^{o}(x,y)}\right) \le \sum_{i=1}^{n}\left(\frac{f_i(x,y)}{f^{o}(x,y)}\exp(\delta_i f^{o}(x,y))\right)
$$

于是：

$$
A(\delta\mid w) \ge \sum_{x,y}\left(\tilde P(x,y)\sum_{i=1}^{n}\delta_i f_i(x,y)\right)+1\ -\sum_x \left[\tilde P(x) \sum_y \left(P_w(y\mid x)\sum_{i=1}^{n}\left(\frac{f_i(x,y)}{f^{o}(x,y)}\exp(\delta_i f^{o}(x,y))\right)\right)\right]
$$

令

$$
B(\delta\mid w)=\sum_{x,y}\left(\tilde P(x,y)\sum_{i=1}^{n}\delta_i f_i(x,y)\right)+1\ -\sum_x \left[\tilde P(x) \sum_y \left(P_w(y\mid x)\sum_{i=1}^{n}\left(\frac{f_i(x,y)}{f^{o}(x,y)}\exp(\delta_i f^{o}(x,y))\right)\right)\right]
$$

则： $L(w+\delta)-L(w)\ge B\delta\mid w)$ 。这里 $(\delta\mid w)$ 是对数似然函数改变量的一个新的（相对不那么紧）的下界。

&emsp;&emsp;求 $B(\delta\mid w)$ 对 $\delta  _  i$ 的偏导数：

$$
\frac{\partial B(\delta\mid w)}{\partial \delta_i}=\sum_{x,y}[\tilde P(x,y)f_i(x,y)]-\sum_{x}\left(\tilde P(x)\sum_y[P_{w}(y\mid x)f_i(x,y)\exp(\delta_if^{o}(x,y))]\right) =0
$$

令偏导数为 $0$ 即可得到 $\delta  _  i$ ：

$$
\sum_{x}\left(\tilde P(x)\sum_y[P_{w}(y\mid x)f_i(x,y)\exp(\delta_if^{o}(x,y))]\right)=\mathbb E_{\tilde P}[f_i]
$$

最终根据 $\delta  _  1,\cdots,\delta  _  n$ 可以得到 $\delta$ 。
    
#### 伪码
输入：
1. 特征函数 $f  _  1,f  _  2,\cdots,f  _  n$ 
2. 经验分布 $\tilde P(x,y), \tilde P(x)$ 
3. 模型 $P  _  \mathbf{w}(y\mid x)$ 

输出：
1. 最优参数 $w  _  i^{ \ * }$ 
2. 最优模型 $P  _  {w^{ \ * }}(y\mid x)$ 

算法步骤：
1. 初始化：取 $w  _  i=0,i=1,2,\cdots,n$ 。
2. 迭代，迭代停止条件为：所有 $w  _  i$ 均收敛。迭代步骤为：
    1. 求解 $\delta=(\delta  _  1,\cdots,\delta  _  n)^T$ ，求解方法为：对每一个 $i,i=1,2,\cdots,n$ ：
        1. 求解 $\delta  _  i$ 。其中 $\delta  _  i$ 是方程： $\sum  _  {x}\left(\tilde P(x)\sum  _  y[P  _  {w}(y\mid x)f  _  i(x,y)\exp(\delta  _  i,f^{o}(x,y))]\right)=\mathbb E  _  {\tilde P}[f  _  i]$ 的解，其中： $ f^{o}(x,y)=\sum  _  {i=1}^{n}f  _  i(x,y)$ 。
        2. 更新 $w  _  i \leftarrow w  _  i + \delta  _  i$ 。
    2. 判定迭代停止条件。若不满足停止条件，则继续迭代。
            
### 拟牛顿法
&emsp;&emsp;若对数似然函数 $L(w)$ 最大，则 $-L(w)$ 最小。令 $F(w)=-L(w)$ ，则最优化目标修改为：

$$
 \min_{w \in \mathbb R^{n}}F(w)= \min_{w \in \mathbb R^{n}} \sum_{x}\left(\tilde P(x)\log\sum_y \exp\left(\sum_{i=1}^{n}w_i f_i(x,y)\right)\right) -\sum_{x,y}\left(\tilde P(x,y)\sum_{i=1}^{n}w_i f_i(x,y)\right)
$$

计算梯度:

$$
\vec g(w)=\left(\frac{\partial F(w)}{\partial w_1},\frac{\partial F(w)}{\partial w_2},\cdots,\frac{\partial F(w)}{\partial w_n}\right)^{T},\ \frac{\partial F(w)}{\partial w_i}=\sum_{x}[\tilde P(x)P_{w}(y\mid x)f_i(x,y)]- \mathbb E_{\tilde P}[f_i],\quad i=1,2,\cdots,n
$$
    
#### 最大熵模型学习的 BFGS 算法：
输入：
1. 特征函数 $f  _  1,f  _  2,\cdots,f  _  n$ 
2. 经验分布 $ \tilde P(x,y), \tilde P(x)$ 
3. 目标函数 $F(w)$ 
4. 梯度 $g(w)=\nabla F(w)$ 
5. 精度要求 $\varepsilon$ 

输出：
1. 最优参数值 $w^{ \ * }$ 
2. 最优模型 $P  _  {w^{ \ * }}(y\mid w)$ 

算法步骤：
1. 选定初始点 $w^{ \ < 0 \ > }$ ，取 $\mathbf B  _  0$ 为正定对阵矩阵，迭代计数器 $k=0$ 。
2. 计算 $g  _  k=g(w^{ \ < k \ > })$ ：
    1. 若 $|g  _  k| \lt \varepsilon$ ，停止计算，得到 $w^{ \ * }=w^{ \ < k \ > }$ 
    2. 若 $|g  _  k| \ge \varepsilon$ ：
        1. 由 $\mathbf B  _  k p  _  k=-g  _  k$ 求得 $p  _  k$ 
        2. 一维搜索：求出 $\lambda  _  k:\lambda  _  k=\arg\min  _  {\lambda \ge 0}F(w^{ \ < k \ > }+\lambda  _  k p  _  k)$ 
        3. 置 $w^{ \ < k+1 \ > }=w^{ \ < k \ > }+\lambda  _  k p  _  k$ 
        4. 计算 $g  _  {k+1}=g(w^{ \ < k+1 \ > })$ 。 若 $|g  _  {k+1}| \lt \varepsilon$ ，停止计算，得到 $w^{ \ * }=w^{ \ < k+1 \ > }$ 。
        5. 否则计算 $\mathbf B  _  {k+1}$ ：
            $$
            \mathbf B_{k+1}=\mathbf B_k+\frac{y_k y_k^{T}}{y_k^{T} \delta_k}-\frac{\mathbf B_k \delta_k \delta_k^{T}\mathbf B_k}{\delta_k^{T}\mathbf B_k \delta_k}
            $$
            其中： $y  _  k=g  _  {k+1}-g  _  k$ ,  $\delta  _  k=w^{ \ < k+1 \ > }-w^{ \ < k \ > }$ 。
        6. 置 $k=k+1$ ，继续迭代。