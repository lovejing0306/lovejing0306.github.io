---
layout: post
title: HighwayNet
categories: [Classification]
description: HighwayNet
keywords: Classification
---


分类模型 HighwayNet
---



## 背景
&emsp;&emsp;随着神经网络的发展，网络的深度逐渐加深(更深的层数以及更小的感受野，能够提高网络分类的准确性)，网络的训练也就变得越来越困难。$Highway \ Networks$ 是一种解决深层网络训练困难的网络框架。

&emsp;&emsp;$Highway \ Networks$ 受 $LSTM$ 启发，可以使用传统基于梯度的方法快速训练深度网络（几百层的）。即使不需要大的深度，高速网络也可以自适应表示合适的特征变换。

&emsp;&emsp;$Highway \ Networks$ 是一种可学习的门限机制，在此机制下，一些信息流没有衰减的通过一些网络层，适用于 $SGD$ 法。

## 原理
&emsp;&emsp;传统的神经网络每层网络对输入进行一个非线性映射变换，可以表达如下 （忽略偏置和层索引）：
$$
y=H\left( x,{ W }_ { H } \right) 
$$

其中，$H$ 为非线性函数，$W$ 权重，$x$ 输入，$y$ 输出。

&emsp;&emsp;受 $LSTM$ 门机制的启发，$Highway \ Networks$ 神经网络，增加了两个非线性转换层，T（transform gate） 和C（carry gate），表达如下：

$$
y=H\left( x,{ W }_ { H } \right) \cdot T\left( x,{ W }_ { T } \right) +x\cdot C\left( x,{ W }_ { C } \right) 
$$

其中， $T$ 表示输入信息经过 $convolutional$ 或者是 $recurrent$ 的信息被转换的部分， $C$ 表示的是原始输入信息 $x$ 保留的部分 ，$T=sigmoid(wx + b)$ 。

&emsp;&emsp;为了计算方便，这里定义了$C = 1 - T$ ：

$$
y=H\left( x,{ W }_ { H } \right) \cdot T\left( x,{ W }_ { T } \right) +x\cdot \left( 1-T\left( x,{ W }_ { T } \right)  \right) 
$$

需要注意的是 $x$，$y$，$H$， $T$ 的维度必须一致，要想保证其维度一致，可以采用 $sub-sampling$ 或者 $zero-padding$ 策略，也可以使用普通的线性层改变维度，使其一致。

&emsp;&emsp;考虑一下特殊的情况，$T= 0$ 的时候，$y=x$，原始输入信息全部保留，不做任何的改变，$T= 1$ 的时候，$Y = H$，原始信息全部转换，不在保留原始信息，仅仅相当于一个普通的神经网络。

$$
y=\begin{cases} x\quad &if\quad T\left( x,{ W }_ { T } \right) =0 \\ H\left( x,{ W }_ { H } \right) \quad &if\quad T\left( x,{ W }_ { T } \right) =1 \end{cases}
$$
