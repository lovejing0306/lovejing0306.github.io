---
layout: post
title: 卷积之Separable Convolutions
categories: [BasicKnowledge]
description: 卷积之Separable Convolutions
keywords: BasicKnowledge
---


深度学习基础知识点卷积之Separable Convolutions
---


可分离卷积可以分为空间可分离卷积 $(Spatially \ Separable \ Convolutions)$和深度可分离卷积 $(depthwise \ separable \ convolution)$。

## Spatially Separable Convolutions

普通的 $3 \times 3$ 卷积在一个 $5 \times 5$的 $feature \ map$上是如下图这样进行计算：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Convolution/SpatiallySeparableConvolutions-1.jpg?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Spatially Separable Convolutions 1</div>
</center>

每个位置需要 $9$ 次乘法，一共有 $9$ 个位置，所以整个操作下来就是 $9 \times 9 = 81$ 次乘法操作。


空间可分离卷积是将一个卷积分解为两个单独的运算，做一次 $3 \times 3$ 卷积，等价于先做一次 $3 \times 1$ 卷积再做一次 $1 \times 3$ 卷积。空间可分离卷积的计算如下图所示：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Convolution/SpatiallySeparableConvolutions2.png?raw=true"
    width="520" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Spatially Separable Convolutions 2</div>
</center>

* 第一步先使用 $3 \times 1$ 的 $filter$，所需计算量为：$15 \times 3=45$
* 第二步使用 $1 \times 3$ 的 $filter$，所需计算量为：$9 \times 3 = 27$

总共需要 $72$ 次乘法就可以得到最终结果，要小于普通卷积的 $81$ 次乘法。

## Depthwise Separable Convolutions

深度可分离卷积的步骤 $(10275 \ multiplications)$：
* 先变小 $(depthwise \ convolution)$：将 $Layer$ 和 $Kernel$ 在深度方向上分离，做一次 $2D$ 卷积，然后在通道方向上 $concat$，得到深度不变、宽长变小的 $Layer$
* 再加深$(1\times1 \ convolution)$：做 $128$ 次 $1 \times 1$ 的卷积得到宽长不变、深度为 $128$ 的 $Layer$

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Convolution/DepthwiseSeparableConvolutions-1.png?raw=true"
    width="520" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Depthwise Separable Convolutions 1</div>
</center>

对比普通的卷积$(86400 multiplications)$：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Convolution/DepthwiseSeparableConvolutions-2.png?raw=true"
    width="520" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Depthwise Separable Convolutions 2</div>
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