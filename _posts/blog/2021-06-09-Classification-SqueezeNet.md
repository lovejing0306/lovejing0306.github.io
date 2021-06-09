---
layout: post
title: SqueezeNet
categories: [Classification]
description: SqueezeNet
keywords: Classification
---


分类模型 SqueezeNet(压缩网络)
---


## 背景
&emsp;&emsp;$SqueezeNet$ 是 $Han$ 等提出的一种轻量且高效的 $CNN$ 模型，它的参数比 $AlexNet$ 少 $50$ 倍，但模型性能（$accuracy$）与 $AlexNet$ 接近。

&emsp;&emsp;$SqueezeNet$ 设计目标不是为了得到最佳的识别精度，而是希望简化网络复杂度，同时达到 $public$ 网络的识别精度。所以 $SqueezeNet$ 主要是为了降低 $CNN$ 模型参数数量而设计的。

## 网络结构
### 设计原则
1. 替换 $3 \times 3$ 的卷积 $kernel$ 为 $1 \times 1$ 的卷积 $kernel$

    替换 $3 \times 3$ 的卷积 $kernel$ 为 $1 \times 1$ 的卷积 $kernel$ 可以让参数缩小 $9$ 倍。但是为了不影响识别精度，并不是全部替换，而是一部分用 $3 \times 3$，一部分用 $1 \times 1$。
2. 减少输入 $3 \times 3$ 卷积核的输入通道的数量

    如果是 $conv1-conv2$ 这样的直连，是没有办法减少 $conv2$ 的 $input \ channel$ 的数量。所以作者巧妙地把原本一层 $conv$ 分解为两层，并且封装为一个 $Fire \ Module$。
3. 减少 $pooling$

    这个观点在很多其他工作中都已经有体现了，比如 $GoogleNet$ 以及 $Deep \ Residual \ Learning$。

### 主体结构
&emsp;&emsp;整个 $SqueezeNet$ 使用 $Fire \ Module$ 堆积而成的，网络结构如下图所示，其中左图是标准的 $SqueezeNet$，其开始是一个卷积层，后面是 $Fire \ Module$ 的堆积，值得注意的是其中穿插着 $stride=2$ 的 $maxpool$ 层，其主要作用是下采样，并且采用延迟的策略，尽量使前面层拥有较大的 $feature \ map$。中图和右图分别是引入了不同 “$skip \  connection$” 机制的 $SqueezeNet$，这是借鉴了 $ResNet$ 的结构。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/SqueezeNet/squeezenet1.jpg?raw=true"
    width="720" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">squeezenet1</div>
</center>

### 参数配置
&emsp;&emsp;每层采用的参数信息如表所示：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/SqueezeNet/squeezenet2.jpg?raw=true"
    width="720" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">squeezenet2</div>
</center>

### 实现细节
1. 在 $Fire \ Module$ 中，$expand$ 层采用了混合卷积核 $1 \times 1$ 和 $3 \times 3$，其 $stride$ 均为 $1$，对于 $1 \times 1$ 卷积核，其输出 $feature \ map$ 与原始一样大小，但是由于它要和 $3 \times 3$ 得到的 $feature \ map$ 做 $concat$，所以 $3 \times 3$ 卷积进行了 $padding=1$ 的操作；
2. $Fire$ 模块中所有卷积层的激活函数采用 $ReLU$；
3. $Fire9$ 层后采用了 $dropout$，其中 $keep  _  {prob}=0.5$；
4. $SqueezeNet$ 没有全连接层，而是采用了全局的 $avgpool$ 层，即 $pool \ size$ 与输入 $feature \ map$ 大小一致；
5. 训练采用线性递减的学习速率，初始学习速率为 $0.04$。

## 网络技巧
### Fire Module
&emsp;&emsp;$Fire \ Module$ 是 $SqueezeNet$ 的核心构件，$Fire \ Module$ 将原来简单的一层 $conv$ 层变成两层：$squeeze$ 层 + $expand$ 层。其中 $squeeze$ 层采用 $1 \times 1$ 卷积核，$expand$ 层混合使用 $1 \times 1$ 和 $3 \times 3$ 卷积核，$Fire \ Module$ 的基本结构如下图所示：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/SqueezeNet/squeezenet3.jpg?raw=true"
    width="400" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">squeezenet3</div>
</center>

在 $squeeze$ 层卷积核数记为 ${ s }  _  { 1 \times 1 }$，在 $expand$ 层，记 $1 \times 1$ 卷积核数为 ${ e }  _  { 1 \times 1 }$，而 $3 \times 3$ 卷积核数为 ${ e }  _  { 3 \times 3 }$。为了尽量降低 $3 \times 3$ 的输入通道数，这里让 ${ s }  _  { 1 \times 1 }$ 的值小于 ${ e }  _  { 1 \times 1 }$ 与 ${ e }  _  { 3 \times 3 }$ 的和，即满足设计原则中的第二条。

### 模型压缩
1. $SVD$
2. 网络剪枝

    网络剪枝就是在 $weight$ 中设置一个阈值，低于这个阈值就设为 $0$，从而将 $weight$ 变成稀疏矩阵，可以采用比较高效的稀疏存储方式，进而降低模型大小。
3. 量化
   
    对参数降低位数，比如从 $float32$ 变成 $int8$，这样是有道理，因为训练时采用高位浮点是为了梯度计算，而真正做 $inference$ 时也许并不需要这么高位的浮点，$TensorFlow$ 中是提供了量化工具的，采用更低位的存储不仅降低模型大小，还可以结合特定硬件做 $inference$ 加速。