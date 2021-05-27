---
layout: post
title: 激活函数FReLU
categories: [BasicKnowledge]
description: 激活函数FReLU
keywords: BasicKnowledge
---


深度学习基础知识点激活函数FReLU
---

## 背景

作者提出了一种适用于视觉领域，简单但有效的激活函数 $FReLU$，它通过增加少量的空间开销，将 $ReLU$ 和 $PReLU$ 扩展为二维激活函数。作者在 $ImageNet$、$COCO$ 检测和语义分割任务上进行了实验，显示了 $FReLU$ 在视觉识别任务上有很大的改进和鲁棒性。

## 简介
作者认为在卷积神经网络中最主要的两个层是卷积层和非线性激活层。

卷积层中为了更好的自适应的捕获空间依赖关系，许多更复杂、更有效的卷积技术已经被提出，特别是在密集预测任务(如语义分割、目标检测)方面取得了巨大的成功。针对复杂卷积技术的低效实现，作者提出一个问题：规则的卷积能否达到类似的精度？

非线性激活层对卷积层的线性输出进行非线性变换，受卷积层和激活层不同作用的驱动，另一个问题出现了：能否设计一个专门针对视觉任务的激活函数?

作者认为激活中的空间不敏感是阻碍视觉任务取得显著改善的主要障碍，为此作者提出了一种新的视觉激活函数 $FReLU$ 来消除这一障碍。

## Funnel 激活函数
$ReLU$ 的条件是手工设计的 $0$，$PReLU$ 的条件是参数化的 $px$，$FReLU$ 专门为视觉任务设计，作者根据空间上下文将其修改为二维漏斗式条件。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/FReLU_Fig2.jpg?raw=true"
    width="560" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig2</div>
</center>


### ReLU
$ReLU$ 的形式为：

$$
f(x)=max(x, 0)
$$

使用 $max()$ 作为非线性，使用手工设计的 $0$ 作为条件。

### PReLU
作为 $ReLU$ 的高级变体，PReLU的形式为：

$$
f(x) = max(x, px)   (p<1)
$$

其中 $p$ 是一个可学习的参数，初始化为 $0.25$。

### FReLU
$FReLU$ 采用相同的 $max()$ 作为简单的非线性函数。对于条件部分，$FReLU$ 将其扩展为依赖于每个像素的空间上下文的 $2D$ 条件。

形式上，定义的漏斗条件为 $T(x)$。为了实现空间条件，使用一个参数池窗口（$Parametric \ Pooling \ Window$）来创建空间依赖，具体来说，激活函数的形式为：

$$
\begin{array}{c}
	f\left( x_{c,i,j} \right) =\max \left( x_{c,i,j},T\left( x_{c,i,j} \right) \right)\\
	T\left( x_{c,i,j} \right) =x_{c,i,j}^{w}\cdot p_{c}^{w}\\
\end{array}
$$

其中，$x_{c,i,j}$ 为非线性激活 $f(\cdot )$ 在第 $c$ 个通道上的二维空间位置 $(i,j)$ 的输入像素；函数 $T(\cdot )$ 表示漏斗条件；$x_{c,i,j}^{w}$ 代表一个 $k_{h}\times k_{w}$ 的参数池窗口，中心点为 $x_{c,i,j}$； $p_{c}^{w}$ 表示为该窗口在同一通道中共享的权重。

### Pixel-wise modeling capacity
像素级条件使网络具有像素级建模能力，函数 $max(\cdot )$ 为每个像素提供了查看空间上下文或不查看空间上下文的选择。

## 实现细节
所有区域 $x_{c,i,j}^{w}$ 在同一通道共享相同的权重 $p_{c}^{w}$，因此，它只是增加了一些额外的参数的数量。$x_{c,i,j}^{w}$ 是一个滑动窗口，大小为 $3 \times 3$ 的滑动窗口，将二维填充设置为1，在这种情况下:

$$
 x_{c,i,j}^{w}\bullet p_{c}^{w}=\sum_{i-1\leqslant h\leqslant i+1,j-1\leqslant w\leqslant j+1}x_{c,h,w}\bullet p_{c,h,w}
$$

### 参数初始化
使用高斯初始化来初始化超参数。因此，得到的条件值接近于零，这不会太多地改变原始网络的性质。作者也调查了没有参数的情况，如最大池化、平均池化等，它们没有显示出改进，这显示了附加参数的重要性。


### 参数计算
假设有一个 $K_{h}^{'} \times K_{w}^{'}$ 的卷积，输入特征图尺寸为 $C \times H \times W$，输出特质图尺寸为 $C \times H^{'} \times W^{'}$，则计算得到的参数数量为 $CCK_{h}^{'}K_{w}^{'}$，$FLOPs$（浮点操作）为 $CCK_{h}^{'}K_{w}^{'}HW$。

在此基础上，添加了带有 $K_{h}\times K_{w}$ 窗口大小的漏斗条件，额外的参数数量 $CCK_{h}K_{w}$，和额外的浮点数操作 $CCK_{h}K_{w}HW$。为了简化计算，假设 $K=K_{h}=K_{w}$，$K^{'}=K_{h}^{'}=K_{w}^{'}$。

## 实验
### 分类
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/FReLU_Table1.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Table 1</div>
</center>


<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/FReLU_Table2.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Table 2</div>
</center>

### 检测
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/FReLU_Table3.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Table 3</div>
</center>

### 分割
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/FReLU_Table4.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Table 4</div>
</center>

## 参考
[Funnel Activation for Visual Recognition](https://arxiv.org/abs/2007.11824)

[用于视觉识别的Funnel激活函数](https://blog.csdn.net/qq_33384379/article/details/107611108)

[一种新的激活函数~Funnel Activation](https://blog.csdn.net/weixin_44402973/article/details/107864585)

[在视觉任务上大幅超越ReLU的新型激活函数](https://mp.weixin.qq.com/s/1d3Uv4_wjIdxJ72iEoyvHA?st=6BAFD7B89681D5533FD9901D0F9E5E897FEA181D4F4D718FE05130CDD100145A4CC31B6E546BCE8D38DEBD4DE3442CFB4B70CB06698661D73459FF37E8324113507BAECBC8C9B4A0B3EB2E438FBDB572A4928DCA7C0641CC12EC89D475A6E8387271ACA37402303B7F807B137BAF75A74C7DE4713686BA6DF83569BC4124533FC87F72DF45C08F6D31F155333D22F210&vid=1688853915012988&cst=972381F8D188DB4CA99F0AEE30B5EF4389FFA36C58D5A4FC10AA7BC3B4B3199C65E1114DEED5F02C29FE70E4EAA15DF0&deviceid=a1f13f9f-9e82-44f0-801a-e4b645223a5e&version=3.0.36.2330&platform=mac)

[FunnelAct](https://github.com/megvii-model/FunnelAct/blob/master/resnet/frelu.py)

