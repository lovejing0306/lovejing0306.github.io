---
layout: post
title: 激活函数Swish
categories: [BasicKnowledge]
description: 激活函数Swish
keywords: BasicKnowledge
---


深度学习基础知识点激活函数Swish
---

## 简介

受到 $LSTM$ 和 $highway \ network$ 中使用 $sigmoid$ 函数进行门控的启发，谷歌提出 $Swish$ 激活函数。

## Swish 激活函数

$swish$ 激活函数的原始形式：

$$
f\left( x \right) =x\cdot \sigma \left( x \right) 
$$

其中，$\sigma$ 是 $sigmoid$ 函数。$swish$ 激活函数的图形如下：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/swish.jpg?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> swish 激活函数</div>
</center>

$swish$ 激活函数的一阶导数如下：

$$
\begin{array}{c}
	f^{'}\left( x \right) =\sigma \left( x \right) +x\cdot \sigma \left( x \right) \left( 1-\sigma \left( x \right) \right)\\
	=\sigma \left( x \right) +x\cdot \sigma \left( x \right) -x\cdot \sigma \left( x \right) ^2\\
	=x\cdot \sigma \left( x \right) +\sigma \left( x \right) \left( 1-x\cdot \sigma \left( x \right) \right)\\
	=f\left( x \right) +\sigma \left( x \right) \left( 1-f\left( x \right) \right)\\
\end{array}
$$

$swish$ 激活函数的一阶和二阶导数的图像如下：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/swish-derivative.jpg?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> swish 导数</div>
</center>

超参数版 $swish$ 激活函数：

$$
f\left( x \right) =x\cdot \sigma \left( \beta x \right) 
$$

其中，$\beta$ 是超参数。超参数版 $swish$ 激活函数的图形如下：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/swish-beta.jpg?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> swish 超参数</div>
</center>

## 优点
* 当 $x>0$ 时，不存在梯度消失的情况；当 $x<0$ 时，神经元也不会像 $ReLU$ 一样出现死亡的情况。
* $swish$ 处处可导，连续光滑。
* $swish$ 并非一个单调的函数。
* 提升了模型的性能。

## 缺点
* 计算量大

## 参考
[Swish: a Self-Gated Activation Function](https://arxiv.org/abs/1710.05941v1)

[Searching for Activation Functions](https://arxiv.org/abs/1710.05941)


<script type="text/x-mathjax-config">
  MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/latest.js?config=TeX-MML-AM_CHTML">
</script>