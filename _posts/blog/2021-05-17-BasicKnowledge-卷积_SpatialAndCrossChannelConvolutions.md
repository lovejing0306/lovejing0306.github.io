---
layout: post
title: 卷积之Spatial and Cross-Channel Convolutions
categories: [BasicKnowledge]
description: 卷积之Spatial and Cross-Channel Convolutions
keywords: BasicKnowledge
---


深度学习基础知识点卷积之 Spatial and Cross-Channel Convolutions
---


## 背景
最初被使用在Inception模块中，主要是将跨通道相关性和空间相关性的操作拆分为一系列独立的操作。

## 原理
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Convolution/SpatialAndCross-ChannelConvolutions.jpg?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Spatial and Cross-Channel Convolutions</div>
</center>

先使用 $1 \times 1 \ Convolution$ 来约束通道个数，降低计算量，然后每个分支都是用 $3 \times 3$ 卷积，最终使用 $concat$ 的方式融合特征。

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