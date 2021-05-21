---
layout: post
title: 卷积之Grouped Convolutions
categories: [BasicKnowledge]
description: 卷积之Grouped Convolutions
keywords: BasicKnowledge
---


深度学习基础知识点卷积Grouped Convolutions
---


## 背景
组卷积最初是在 $AlexNet$ 中提出的，之后被大量应用在 $ResNeXt$ 网络结构中，提出的动机就是通过将 $feature$ 划分为不同的组来降低模型计算复杂度。


## 卷积过程
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Convolution/GroupedConvolution.png?raw=true"
    width="520" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Grouped Convolutions</div>
</center>

分组卷积就是将输入 $feature \  map$ 的通道进行分组，然后每个组内部进行卷积操作，最终将得到的组卷积的结果 $Concate$ 到一起，得到输出的 $feature \ map$。

## 优点
* 训练效率高：由于卷积被分为几个不同的组，每个组的计算就可以分配给不同的 $GPU$ 核心来进行计算。这种结构的设计更符合 $GPU$ 并行计算的要求，这也能解释为何 $ResNeXt$ 在 $GPU$ 上效率要高于 $Inception$ 模块。
* 模型效率高：模型参数随着组数或者基数的增加而减少。
* 效果好：分组卷积可能能够比普通卷积组成的模型效果更优，这是因为滤波器之间的关系是稀疏的，而划分组以后对模型可以起到一定正则化的作用。从 $COCO$ 数据集榜单就可以看出来，有很多是 $ResNeXt101$ 作为 $backbone$ 的模型在排行榜非常靠前的位置。


## 缺点
$information$ 只是在分组的区域内交互，并没有在所有 $channels$ 上都进行交互融合，导致学得的特征也具有分组的特性，阻挡了不同分组之间的“信息流”。



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