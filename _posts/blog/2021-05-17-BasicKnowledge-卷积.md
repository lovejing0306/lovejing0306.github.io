---
layout: post
title: 卷积
categories: [BasicKnowledge]
description: 卷积
keywords: BasicKnowledge
---


深度学习基础知识点卷积
---


## 目的
   * 是从输入图像中提取特征

## 特性
   * 权值共享
     * 该特性可大大减少训练过程使用的参数量

## 卷积层输出尺寸计算公式

输入张量的尺寸 ${ W }  _  { 1 }\times { H }  _  { 1 }\times { D }  _  { 1 }$

4个超参数
* 滤波器的数量$K$
* 滤波器的空间尺寸$F$
* 步长$S$
* 零填充数量$P$

输出张量的尺寸 ${ W }  _  { 2 }\times { H }  _  { 2 }\times { D }  _  { 2 }$

$$
\begin{aligned}
{ W }_ { 2 } &={ ({ W }_{ 1 }-F+2P) }/{ S }+1 \\ 
{ H }_ { 2 } &={ ({ H }_{ 1 }-F+2P) }/{ S }+1 \\ 
{ D }_ { 2 } &=K 
\end{aligned}
$$

## 卷积操作演示

### no padding
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Convolution/no_padding.gif?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">no padding</div>
</center>

### padding

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Convolution/padding.gif?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">padding</div>
</center>

### striding(大步长)

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Convolution/striding.gif?raw=true"
    width="304" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">striding</div>
</center>

### 多通道版本
* 滤波器的每个卷积核在各自的输入通道上滑动

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Convolution/filter.gif?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">multi channel</div>
</center>

* 每个通道处理的结果汇在一起形成一个通道

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Convolution/merge_channel.gif?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">merge channel</div>
</center>

* 加上偏置

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Convolution/bias.gif?raw=true"
    width="172" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">bias</div>
</center>

# Atrous/Dilated 卷积

## 简介
* $Atrous$卷积，就是带洞卷积，其卷积核是稀疏的，起源于语义分割。

## 思想
* 紧密相邻的像素几乎相同，全部纳入属于冗余，不如跳$H$($hole \ size$)个取一个。

## 卷积方式
* 图 $(a)$ 对应 $3 \times 3$ 的 $1-dilated \ conv$，和普通的卷积操作一样
* 图 $(b)$ 对应 $3\times3$ 的 $2-dilated \ conv$，实际的卷积 $kernel \ size$ 还是 $3 \times 3$，但是空洞为 $1$，也就是对于一个 $7\times7$ 的图像$patch$，只有 $9$ 个红色的点和 $3\times3$ 的 $kernel$ 发生卷积操作，其余的点略过。也可以理解为 $kernel$ 的 $size$ 为 $7\times7$，但是只有图中的 $9$ 个点的权重不为 $0$，其余都为 $0$。 可以看到虽然 $kernel \ size$ 只有 $3\times3$，但是这个卷积的感受野已经增大到了 $7\times7$（如果考虑到这个 $2-dilated \ conv$ 的前一层是一个 $1-dilated \ conv$ 的话，那么每个红点就是 $1-dilated$ 的卷积输出，所以感受野为 $3\times3$，所以 $1-dilated$ 和 $2-dilated$ 合起来就能达到 $7\times7$ 的 $conv$）
* 图$(c)$ 是 $4-dilated \ conv$ 操作，同理跟在两个 $1-dilated$ 和 $2-dilated \ conv$ 的后面，能达到 $15\times15$ 的感受野。对比传统的 $conv$ 操作，$3$ 层 $3\times3$ 的卷积加起来，$stride$ 为 $1$ 的话，只能达到 $(kernel-1) \times layer+1=7$ 的感受野，也就是和层数 $layer$ 成线性关系，而 $dilated \ conv$ 的感受野是指数级的增长。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Convolution/atrous-conv.png?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">atrous_conv1</div>
</center>

## 卷积过程

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Convolution/atrous_conv.gif?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">atrous_conv2</div>
</center>

## 优点
* 可以不增加参数量的同时增大感受野。
* 每个卷积输出都包含较大范围的信息。

# 反卷积(转置卷积)

## 目的
* 增加特征图的尺寸，重建先前的空间分辨率

## 实现方式
* 上采样->卷积

## 原理
* 卷积操作的逆过程

## 反卷积输出尺寸计算公式
* 输入张量的尺寸${ W }  _  { 1 }\times { H }  _  { 1 }\times { D }  _  { 1 }$
* 4个超参数
  * 滤波器的数量$K$
  * 滤波器的空间尺寸$F$
  * 步长$S$，由${ W }  _  { 1 }$变成${ W }  _  { 2 }$需要的步长$S$
  * 零填充数量$P$，由${ W }  _  { 1 }$变成${ W }  _  { 2 }$需要的填充$P$
* 输出张量的尺寸${ W }  _  { 2 }\times { H }  _  { 2 }\times { D }  _  { 2 }$
  $$
  \begin{aligned} { W }_ { 2 } & =S{ ({ W }_ { 1 }-1) }+F-2P \\ { H }_ { 2 } & =S{ ({ H }_ { 1 }-1) }+F-2P \\ { D }_ { 2 } & =K \end{aligned}
  $$
## 反卷积过程
* 如下图所示为一个参数为${ w }  _  { 2 }=4,k=3,s=1,p=0$

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Convolution/deconvolution1.gif?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">deconvolution1</div>
</center>

* 如下图所示为一个参数为${ w }  _  { 2 }=5,k=3,s=2,p=1$

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Convolution/deconvolution2.gif?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">deconvolution2</div>
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
