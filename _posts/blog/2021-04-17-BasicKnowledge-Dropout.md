---
layout: post
title: Dropout
categories: [BasicKnowledge]
description: Dropout
keywords: BasicKnowledge
---


深度学习基础知识点 Dropout
---

## 简介
&emsp;&emsp;$Dropout$ 由 $Hintion$ 提出，是一种用于防止过拟合和提供有效近似联结指数级不同神经网络结构的方法。如下图所示，$dropout$ 中的 $drop$ 指随机“丢弃”网络层中的某些节点，对一个网络使用 $dropout$ 相当于从网络中采样一个“稀疏”的网络，这个“稀疏”的网络包含所有节点（不管是存活还是被丢弃）。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Normalization/Dropout/dropout1.png?raw=true"
    width=480 height=
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">dropout</div>
</center>

<center>
<img src="https://github.com/lovejing0306/TensorFlow/blob/master/project/object_detection/yolo_v2/yolo2_data/detection.jpg?raw=true" width=256 height=256  border=4  alt="图片名称" />
</center>

## 原理
&emsp;&emsp;$Dropout$ 可以看做是模型平均，所谓模型平均，是把来自不同模型的估计或者预测通过一定的权重平均起来，在一些文献中也称为模型组合，它一般包括组合估计和组合预测。

## 训练过程
### 训练
* $Dropout$ 是在标准的反向传播的网络结构上，使隐藏层的激活值，以一定的比例 $p$ 变为 $0$，即按照一定比例 $p$，随机地让一部分隐藏层节点失效。
* 演示图

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Normalization/Dropout/dropout2.png?raw=true"
    width=320 height=
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">dropout</div>
</center>

### 测试
* 在测试时使用一个没有 $dropout$ 的网络，该网络的权值是训练时的网络权值的缩小版，即，如果一个隐藏层单元在训练过程中以概率 $p$ 被保留，那么该单元的输出权重在测试时乘以 $p$。
* 这样共享权值的 ${ 2 }^{ n }$ 个训练网络就可以在测试时近似联结成一个网络，因此能有效降低泛化误差。
* 演示图

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Normalization/Dropout/dropout3.png?raw=true"
    width=768 height=
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">dropout</div>
</center>


## 作用
* 防止模型过拟合
    * $Dropout$ 使得神经网络的训练效果近乎于对 ${ 2 }^{ n }$ 个子网络的平均
* 降低神经元之间复杂的共适应关系
    * 神经网络（尤其是深度神经网络）在训练过程中，神经元之间会产生复杂的共适应关系，但是我们更希望的是神经元能够自己表达出数据中共同的本质特征。
    * 使用 $Dropout$ 后，两个神经元不一定每次都出现在同一个网络中，这样网络中的权值更新不再依赖于具有固定关系的神经元节点之间的共同作用，使得网络更加 $robust$。


<script type="text/x-mathjax-config">
  MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/latest.js?config=TeX-MML-AM_CHTML">
</script>