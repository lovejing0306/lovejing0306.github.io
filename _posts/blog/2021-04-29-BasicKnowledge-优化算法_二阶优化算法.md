---
layout: post
title: 优化算法之二阶优化
categories: [BasicKnowledge]
description: 优化算法之二阶优化
keywords: BasicKnowledge
---


深度学习基础知识点优化算法之二阶优化
---


## 二阶优化
牛顿法和拟牛顿法是无约束最优化问题的常用法，有收敛速度快的优点。牛顿法是迭代算法，每一步需要求解目标函数的海赛矩阵的逆矩阵，计算比较复杂。拟牛顿法通过正定矩阵近似海赛矩阵的逆矩阵或海赛矩阵，来简化这一计算过程。

## 牛顿法
考虑无约束最优化的问题

$$
\underset{x\in R^n}{\min}f\left( x \right) (1)
$$

其中，$x^{\*}$为目标函数的极小值。

假设 $f\left( x \right)$ 具有二阶连续偏导数，若第$k$次迭代值为$x^{\left( k \right)}$，则可将$f\left( x \right) $在$x^{\left( k \right)}$附近进行二阶泰勒展开：

$$
f\left( x \right) =f\left( x^{\left( k \right)} \right) +g_{k}^{T}\left( x-x^{\left( k \right)} \right) +\frac{1}{2}\left( x-x^{\left( k \right)} \right) ^TH\left( x^{\left( k \right)} \right) \left( x-x^{\left( k \right)} \right) (2)
$$

其中，$g_k=g\left( x^{\left( k \right)} \right) =\nabla f\left( x^{\left( k \right)} \right)$ 是 $f\left( x \right) $的梯度向量在点$x^{\left( k \right)}$ 的值，$H\left( x^{\left( k \right)} \right)$ 是 $f\left( x \right)$ 的海赛矩阵。

$$
H\left( x \right) =\left[ \frac{\partial ^2f}{\partial x_i\partial x_j} \right] _ {n\times n} (3)
$$

在点 $x^{\left( k \right)}$ 的值。

函数 $f\left( x \right)$ 有极值的必要条件是在极值点处的一阶导数为 $0$，即梯度向量为 $0$。特别是当 $H\left( x^{\left( k \right)} \right)$ 是正定矩阵时，函数 $f\left( x \right)$ 的极值为极小值。

牛顿法利用极小值的必要条件：

$$
\nabla f\left( x \right) =0 (4)
$$

每次迭代中从点 $x^{\left( k \right)}$ 开始，求目标函数的极小值，作为第 $k+1$ 次迭代值 $x^{\left( k+1 \right)}$。具体的，假设 $x^{\left( k+1 \right)}$ 满足：

$$
\nabla f\left( x^{\left( k+1 \right)} \right) =0 (5)
$$

由公式(2)有：

$$
\nabla f\left( x \right) =g_k+H_k\left( x-x^{\left( k \right)} \right) (6) 
$$

其中，$H_k=H\left( x^{\left( k \right)} \right)$.公式(5)成为：

$$
g_k+H_k\left( x^{\left( k+1 \right)}-x^{\left( k \right)} \right) =0 (7)
$$

因此，

$$
x^{\left( k+1 \right)}=x^{\left( k \right)}-H_{k}^{-1}g_k (8)
$$

或者

$$
x^{\left( k+1 \right)}=x^{\left( k \right)}+p_k (9)
$$

其中，

$$
H_kp_k=-g_k (10)
$$

用公式(8)作为迭代式的算法就是牛顿法。

## 拟牛顿法
牛顿法迭代中需要计算海赛矩阵的逆矩阵 $H^{-1}$，考虑到这一计算过程较为复杂，因此使用一个 $n$ 阶矩阵 $G_k=G\left( x^{\left( k \right)} \right)$ 来近似代替 $H_{k}^{-1}=H^{-1}\left( x^{\left( k \right)} \right)$。

牛顿法迭代中海赛矩阵 $H_k$ 满足的条件。首先，$H_k$ 满足以下关系。在式(6)中取 $x=x^{\left( k+1 \right)}$，得：

$$
g_{k+1}-g_k=H_k\left( x^{\left( k+1 \right)}-x^{\left( k \right)} \right) (11)
$$

记 $y_k=g_{k+1}-g_k$，$\delta _ k=x^{\left( k+1 \right)}-x^{\left( k \right)}$，则：

$$
y_k=H_k\delta_ k (12)
$$

或

$$
H_{k}^{-1}y_ k=\delta_ k (13)
$$

公式(12)或(13)称为拟牛顿条件。

如果 $H_k$ 是正定的（$H^{-1}$ 也是正定的），那么可以保证牛顿法搜索方向 $p_k$ 是下将降方向。这是由于搜索方向是 $p_k=-\lambda g_ k$，由式(8)有：

$$
x=x^{\left( k \right)}+\lambda p_k=x^{\left( k \right)}-\lambda H_{k}^{-1}g_k (14)
$$

所以 $f\left( x \right) $在$x^{\left( k \right)}$ 的泰勒展开式(2)可以近似写成：

$$
f\left( x \right) =f\left( x^{\left( k \right)} \right) -\lambda g_{k}^{T}H_{k}^{-1}g_k (15)
$$

因为 $H_{k}^{-1}$ 是正定的，故有 $g_{k}^{T}H_{k}^{-1}g_k>0$。当 $\lambda$ 为一个充分小的正数时，总有 $f\left( x \right) < f\left( x^{\left( k \right)} \right)$，也就是说 $p_ k$ 是下降方向。


拟牛顿法将 $G_ k$ 作为 $H_{k}^{-1}$ 的近似，要求矩阵 $G_k$ 满足同样的条件。首先，每次迭代矩阵 $G_k$ 是正定的。同时，$G_k$ 满足下面的拟牛顿条件：
$$
G_{k+1}y_ k=\delta_ k (16)
$$

按照拟牛顿条件选择 $G_k$ 作为 $H_{k}^{-1}$ 或选择 $B_k$ 作为 $H_k$ 的近似的算法称为拟牛顿法。

按照拟牛顿条件，在每次迭代中可以选择更新矩阵$G_{k+1}$：

$$
G_{k+1}=G_k+\varDelta G_k (17)
$$

### DFP(Davidon-Fletcher-Powell)
DFP 算法选择 $G_{k+1}$ 的方法是，假设每一步迭代中矩阵 $G_{k+1}$ 是由 $G_{k}$ 加上两个附加项构成，即：

$$
G_{k+1}=G_k+P_k+Q_k (18)
$$

其中，$P_k$ 和 $Q_k$ 是待定矩阵。这时：

$$
G_{k+1}y_k=G_ky_k+P_ky_k+Q_ky_k (19)
$$

为使 $G_{k+1}$ 满足拟牛顿条件，可使 $P_k$ 和 $Q_k$ 满足：

$$
\begin{array}{c}
	P_ky_k=\delta_ k\\
	Q_ky_k=-G_ky_ k\\
\end{array} (20)
$$

其中，$P_ k$ 和 $Q_ k$ 可以有如下选择：
$$
\begin{array}{c}
	P_k=\frac{\delta _k\delta _{k}^{T}}{\delta _{k}^{T}y_k}\\
	Q_k=-\frac{G_ky_ky_{k}^{T}G_k}{y_{k}^{T}G_ky_k}\\
\end{array} (21)
$$

于是矩阵 $G_ {k+1}$ 的迭代公式：

$$
G_{k+1}=G_k+\frac{\delta _k\delta _ {k}^{T}}{\delta _ {k}^{T}y_k}-\frac{G_ky_ky_{k}^{T}G_k}{y_{k}^{T}G_ky_k} (22)
$$

称为 DFP 算法。

### BFGS (Broyden-Fletcher-Goldfarb-Shanno)
可以考虑用 $G_ {k}$ 逼近海赛矩阵 $H^{-1}$ 的逆矩阵，也可以考虑用 $B_{k}$ 逼近海赛矩阵 $H$。

这时，相应的拟牛顿条件是：
$$
B_{k+1}\delta_ k=y_ k (23)
$$

可以用同样的方法得到另一迭代公式。首先令：

$$
B_{k+1}=B_k+P_k+Q_k (24)

B_{k+1}\delta_ k=B_k\delta _k+P_k\delta_ k+Q_ k\delta _ k (25)
$$

考虑使 $P_k$ 和 $Q_k$ 满足：
$$
\begin{array}{c}
	P_k\delta _ k=y_k \\ 
	Q_k\delta _ k=-B_k\delta_  k\\
\end{array} (26)
$$

找出适合条件的 $P_k$ 和 $Q_k$，得到 BFGS 算法矩阵 $B_{k+1}$ 的迭代公式：

$$
B_{k+1}=B_k+\frac{y_ky_{k}^{T}}{y_{k}^{T}\delta _ k}-\frac{B_k\delta _ k\delta _{k}^{T}B_k}{\delta _ {k}^{T}B_ k\delta _ k}
$$
