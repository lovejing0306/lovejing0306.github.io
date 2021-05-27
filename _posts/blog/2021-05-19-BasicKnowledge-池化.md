---
layout: post
title: 池化
categories: [BasicKnowledge]
description: 池化
keywords: BasicKnowledge
---


深度学习基础知识点池化
---


## 目的

* 在语义上把相似的特征合并起来
* 选出重要特征
* 增加模型的鲁棒性

## 特性

* 保证了图像的平移不变性，使得模型不受位置变化的影响；
* 池化操作使网络拥有更大的感受野，从而能够接受更大的输入。感受野的增大，将允许网络在更深层学习到更加抽象的特征表征；

## 池化层输出尺寸计算公式
* 输入张量的尺寸 ${ W }  _  { 1 }\times { H }  _  { 1 }\times { D }  _  { 1 }$
* 两个超参数
 * 空间大小 $F$
 * 步长 $S$
* 输出张量的尺寸 ${ W }  _  { 2 }\times { H }  _  { 2 }\times { D }  _  { 2 }$

$$
\begin{aligned}
{ W }_{ 2 } &={ ({ W }_{ 1 }-F) }/{ S }+1 \\ 
{ H }_{ 2 } &={ ({ H }_{ 1 }-F) }/{ S }+1 \\ 
{ D }_{ 2 } &={ D }_{ 1 }
\end{aligned}
$$

## 池化方式
* 平均池化($Mean Pooling$)
  * 操作方式：对邻域内特征点求取平均值
  * 目的：综合考虑周围像素的特征
  * 优点：对背景保留更好
* 最大池化($Max Pooling$)
  * 操作方式：对邻域内特征点取最大值
  * 目的：获取相邻像素间最重要的特征信息，避免模型学习到一些无关紧要的特征
  * 优点：对纹理提取更好
* 重叠池化($Overlapping \ Pooling$)
  * 重叠池化的相邻池化窗口之间会有重叠区域
* 空间金字塔池化($Spatial \ Pyramid \ Pooling$)
  * 空间金字塔池化拓展了卷积神经网络的实用性，使它能够以任意尺寸的图片作为输入

## 池化过程

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Pooling/pooling.jpg?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">pooling</div>
</center>
