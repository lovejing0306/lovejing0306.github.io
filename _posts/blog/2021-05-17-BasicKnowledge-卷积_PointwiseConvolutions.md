---
layout: post
title: 卷积之Pointwise Convolutions
categories: [BasicKnowledge]
description: 卷积之Pointwise Convolutions
keywords: BasicKnowledge
---


深度学习基础知识点卷积之Pointwise Convolutions
---


## 简介
最初 $1 \times 1$ 卷积是在 $Network in Network$ 中提出的，之后 $1 \times 1 \ convolution$ 最初在 $GoogLeNet$ 中大量使用，$1 \times 1$ 卷积有以下几个特点：
* 用于降维或者升维，可以灵活控制特征图个数
* 减少参数量，特征图少了，参数量也会减少，计算更高效
* 在卷积之后增加了非线性特征（添加激活函数）

## 卷积方式
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Convolution/PointwiseConvolutions.png?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Pointwise Convolutions</div>
</center>
