---
layout: post
title: AlexNet
categories: [Classification]
description: AlexNet
keywords: Classification
---


分类模型 AlexNet
---


## 背景
&emsp;&emsp;$AlexNet$ 是 $2012$ 年 $Alex$ 和 $Hinton$ 参加 $ILSVRC$ 比赛时使用的网络，并赢得了 $2012$ 届 $ImageNet$ 大赛的冠军，使得 $CNN$ 成为图像分类的核心算法模型。

## 网络结构
&emsp;&emsp;$AlexNet$ 整体网络结构：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/AlexNet/alexnet1.jpg?raw=true"
    width="640" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">alexnet1</div>
</center>

&emsp;&emsp;该网络包括八个带权层（第一层：卷积+池化；第二层：卷积+池化；第三层：卷积；第四层：卷积；第五层：卷积+池化；第六层：全连接层；第七层：全连接层；第八层：全连接层；），最后一个全连接层的输出被送到一个 $1000-way$ 的 $softmax$ 层，产生一个覆盖 $1000$ 类标签的分布。

> conv1阶段DFD(data flow diagram)

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/AlexNet/alexnet2.png?raw=true"
    width="640" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">alexnet2</div>
</center>

```
卷积
输入大小：227*227*3
卷积核大小：11*11*3
步长：4
卷积核个数：96
输出大小：55*55*96  (227-11)/4+1=55
偏置个数：96
激活函数：relu
说明：96个卷积核分成2组，每组48个卷积核。对应生成2组55*55*48的卷积后的像素层数据。
这些像素层经过relu1单元的处理，生成激活像素层，尺寸仍为2组55*55*48的像素层数据。
96层像素层分为2组，每组48个像素层，每组在一个独立的GPU上进行运算。

池化
输入大小：55*55*96
卷积核大小：3*3
步长：2
输出大小：27*27*96  (55-3)/2+1=27
```

> conv2阶段DFD(data flow diagram)

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/AlexNet/alexnet3.png?raw=true"
    width="640" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">alexnet3</div>
</center>

```
卷积
输入大小：27*27*96
填充：2个像素
卷积核大小：5*5*48
步长：1
卷积核个数：256
输出大小：27*27*256  (27-5+2*2)/1+1=27
偏置个数：256
激活函数：relu
说明：

池化
输入大小：27*27*256   
卷积核大小：3*3
步长：2
输出大小：13*13*256  (57-3)/2+1=13
```

> conv3阶段DFD(data flow diagram)

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/AlexNet/alexnet4.png?raw=true"
    width="640" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">alexnet4</div>
</center>

```
卷积
输入大小：13*13*256
填充：1个像素
卷积核大小：3*3*256
步长：1
卷积核个数：384
输出大小：13*13*384  (13-3+1*2)/1+1=13
偏置个数：384
激活函数：relu
说明：
```

> conv4阶段DFD(data flow diagram)

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/AlexNet/alexnet5.png?raw=true"
    width="640" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">alexnet5</div>
</center>

```
卷积
输入大小：13*13*384
填充：1个像素
卷积核大小：3*3*384
步长：1
卷积核个数：384
输出大小：13*13*384  (13-3+1*2)/1+1=13
偏置个数：384
激活函数：relu
说明：
```

> conv5阶段DFD(data flow diagram)

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/AlexNet/alexnet6.png?raw=true"
    width="640" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">alexnet6</div>
</center>

```
卷积
输入大小：13*13*384
填充：1个像素
卷积核大小：3*3*384
步长：1
卷积核个数：256
输出大小：13*13*256  (13-3+1*2)/1+1=13
偏置个数：256
激活函数：relu
说明：

池化
输入大小：13*13*256  
卷积核大小：3*3
步长：2
输出大小：6*6*256  (13-3)/2+1=6
```

> fc6阶段DFD(data flow diagram)

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/AlexNet/alexnet7.png?raw=true"
    width="640" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">alexnet7</div>
</center>

```
卷积
输入大小：6*6*256
卷积核大小：6*6*256
卷积核个数：4096
输出大小：4096
偏置个数：4096
激活函数：relu
说明：使用了dropout
```
> fc7阶段DFD(data flow diagram)

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/AlexNet/alexnet8.png?raw=true"
    width="640" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">alexnet8</div>
</center>

```
全连接
输入大小：4096
输出大小：4096
偏置个数：4096
激活函数：relu
说明：使用了dropout
```

> fc8阶段DFD(data flow diagram)

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/AlexNet/alexnet9.png?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">alexnet</div>
</center>

```
全连接
输入大小：4096
输出大小：1000
偏置个数：1000
```

## 网络技巧

### ReLu激活函数
&emsp;&emsp;模拟神经元输出的标准函数一般是：$tanh(x)$ 或 $sigmoid(x)$ 函数，此类函数在 $x$ 非常大或非常小时，函数输出基本不变，所以此类函数成为饱和函数。

&emsp;&emsp;$f(x)=max(0,x)$，扭曲线性函数，是一种非线性的非饱和函数。训练期间非饱和函数比饱和函数训练更快，并且这种扭曲线性函数，不但保留了非线性的表达能力，而且由于其具有线性性质（正值部分），相比 $tanh$ 和 $sigmoid$ 函数在误差反向传递时，不会有由于非线性引起的梯度弥散现象。$ReLU$ 的这些性质可以训练更深的网络。

### 多GPU训练
&emsp;&emsp;单个GTX 580 $GPU$ 只有 $3GB$ 内存，限制了可以在其上训练的网络的最大规模。事实证明，$120$ 万个训练样本才足以训练网络，网络太大了，不适合在一个 $GPU$ 上训练。因此将网络分布在两个 $GPU$ 上。

### 局部响应归一化
#### 简介
&emsp;&emsp;$LRN$ 归一化技术首次在 $AlexNet$ 模型中被提出，其一般跟在激活层或池化层后。

#### 原理
&emsp;&emsp;在神经生物学中有一个概念叫做“侧抑制”($lateral inhibitio$)，指的是被激活的神经元抑制相邻神经元。$LRN$ 就是借鉴“侧抑制”的思想来实现局部抑制，尤其当使用 $relu$ 的时候这种“侧抑制”很管用。

#### 作用
&emsp;&emsp;有利于提高模型的泛化能力(如何提高？$LRN$ 模仿生物神经系统的侧抑制机制，对局部神经元的活动创建竞争机制，使响应比较大的值相对更大，从而提升模型的泛化能力。)

### 重叠池化
****

### 减小过拟合策略
#### 数据扩充
* 图像平移和翻转
* 调整 $RGB$ 像素值

#### Dropout
&emsp;&emsp;运用了 $dropout$ 的训练过程，相当于训练了很多个只有半数隐层单元的神经网络，每一个这样的半数网络，都可以给出一个分类结果，这些结果有的是正确的，有的是错误的。随着训练的进行，大部分半数网络都可以给出正确的分类结果，那么少数的错误分类结果就不会对最终结果造成大的影响。

## 总结
* 输入图像的大小 $227 \times 227 \times 3$
* 卷积核大小 $11 \times 11$、$5 \times 5$、$3 \times 3$
* 池化核大小 $3 \times 3$
* 激活函数使用了 $relu$
* 输出层使用了 $softmax$
* 网络总共为 $9$ 层（将卷积和池化合并为一层）
* 局部响应归一化
* $dropout$
* 多 $GPU$ 训练
* 减少池化次数