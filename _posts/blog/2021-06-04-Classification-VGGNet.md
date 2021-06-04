---
layout: post
title: VGGNet
categories: [Classification]
description: VGGNet
keywords: Classification
---


分类模型 VGGNet
---


## 背景
&emsp;&emsp;$2014$ 年，牛津大学计算机视觉组（$Visual Geometry Group$）和 $Google \ DeepMind$ 公司的研究员一起研发出了新的深度卷积神经网络：$VGGNet$，并取得了 $ILSVRC2014$ 比赛分类项目的第二名（第一名是 $GoogLeNet$，也是同年提出的）和定位项目的第一名。

&emsp;&emsp;$VGGNet$ 探索了卷积神经网络的深度与其性能之间的关系，成功地构筑了 `16~19 `层深的卷积神经网络，证明了增加网络的深度能够在一定程度上影响网络最终的性能，使错误率大幅下降，同时拓展性又很强，迁移到其它图片数据上的泛化性也非常好。

&emsp;&emsp;$VGGNet$ 可以看成是 $AlexNet$ 的加深版本，都是由卷积层、全连接层两大部分构成。到目前为止，$VGG$ 仍然被用来提取图像特征。

## VGGNet与AlexNet网络结构比对

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/VGGNet/vggnet4.jpg?raw=true"
    width="400" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">VGGNet AlexNet</div>
</center>

## 网络结构
&emsp;&emsp;论文中所有网络的结构：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/VGGNet/vggent1.jpg?raw=true"
    width="400" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">vggnet</div>
</center>

&emsp;&emsp;在论文中分别使用了 $A$、$A-LRN$、$B$、$C$、$D$、$E$ 这 $6$ 种网络结构进行测试，$6$ 种网络结构相似，都是由 $5$ 层卷积层、$3$ 层全连接层组成，其中区别在于每个卷积层的子层数量不同，从 $A$ 至 $E$ 依次增加（子层数量从 $1$ 到 $4$），总的网络深度从 $11$ 层到 $19$ 层（添加的层以粗体显示），表格中的卷积层参数表示为“$conv$⟨感受野大小⟩-通道数”，例如 $con3-128$，表示使用 $3 \times 3$ 的卷积核，通道数为 $128$。为了简洁起见，在表格中不显示 $ReLU$ 激活功能。

&emsp;&emsp;其中，网络结构 $D$ 就是著名的 $VGG16$，网络结构 $E$ 就是著名的 $VGG19$。以网络结构 $D$（$VGG16$）为例，介绍其处理过程如下：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/VGGNet/vggnet3.png?raw=true"
    width="400" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">vggnet</div>
</center>

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/VGGNet/vggnet2.jpg?raw=true"
    width="128" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">vggnet</div>
</center>

> Conv1阶段

```
卷积
输入大小：224x224x3
卷积核大小：3x3
步长：1
填充：1
卷积核个数：64
输出大小：224x224x64
偏置个数：64
激活函数：relu

卷积
输入大小：224x224x64
卷积核大小：3x3
步长：1
填充：1
卷积核个数：64
输出大小：224x224x64
偏置个数：64
激活函数：relu

池化
输入大小：224x224x64
卷积核大小：2x2
步长：2
输出大小：112x112x64
```
> Conv2阶段

```
卷积
输入大小：112x112x64
卷积核大小：3x3
步长：1
填充：1
卷积核个数：128
输出大小：112x112x128
偏置个数：128
激活函数：relu

卷积
输入大小：112x112x128
卷积核大小：3x3
步长：1
填充：1
卷积核个数：128
输出大小：112x112x128
偏置个数：128
激活函数：relu

池化
输入大小：112x112x128
卷积核大小：2x2
步长：2
输出大小：56x56x128
```
> Conv3阶段

```
卷积
输入大小：56x56x128
卷积核大小：3x3
步长：1
填充：1
卷积核个数：256
输出大小：56x56x256
偏置个数：256
激活函数：relu

卷积
输入大小：56x56x256
卷积核大小：3x3
步长：1
填充：1
卷积核个数：256
输出大小：56x56x256
偏置个数：256
激活函数：relu

卷积
输入大小：56x56x256
卷积核大小：3x3
步长：1
填充：1
卷积核个数：256
输出大小：56x56x256
偏置个数：256
激活函数：relu

池化
输入大小：56x56x256
卷积核大小：2x2
步长：2
输出大小：28x28x256
```
> Conv4阶段

```
卷积
输入大小：28x28x256
卷积核大小：3x3
步长：1
填充：1
卷积核个数：512
输出大小：28x28x512
偏置个数：512
激活函数：relu

卷积
输入大小：28x28x512
卷积核大小：3x3
步长：1
填充：1
卷积核个数：512
输出大小：28x28x512
偏置个数：512
激活函数：relu

卷积
输入大小：28x28x512
卷积核大小：3x3
步长：1
填充：1
卷积核个数：512
输出大小：28x28x512
偏置个数：512
激活函数：relu

池化
输入大小：28x28x512
卷积核大小：2x2
步长：2
输出大小：14x14x512
```
> Conv5阶段

```
卷积
输入大小：14x14x512
卷积核大小：3x3
步长：1
填充：1
卷积核个数：512
输出大小：14x14x512
偏置个数：512
激活函数：relu

卷积
输入大小：14x14x512
卷积核大小：3x3
步长：1
填充：1
卷积核个数：512
输出大小：14x14x512
偏置个数：512
激活函数：relu

卷积
输入大小：14x14x512
卷积核大小：3x3
步长：1
填充：1
卷积核个数：512
输出大小：14x14x512
偏置个数：512
激活函数：relu

池化
输入大小：14x14x512
卷积核大小：2x2
步长：2
输出大小：7x7x512
```
> FC6阶段

```
输入大小：7x7x512
偏置个数：4096
输出大小：4096
激活函数：relu
```
> FC7阶段

```
输入大小：4096
偏置个数：4096
输出大小：4096
激活函数：relu
```
> FC8阶段

```
输入大小：4096
偏置个数：1000
输出大小：1000
```

## 网络技巧
### 结构简洁
&emsp;&emsp;$VGG$ 由 $5$ 层卷积层、$3$ 层全连接层、$softmax$ 输出层构成，层与层之间使用 $max-pooling$（最大化池）分开，所有隐层的激活单元都采用 $ReLU$ 函数。

### 小卷积核
&emsp;&emsp;$VGG$ 使用多个小卷积核（$3 \times 3$）的卷积层代替一个卷积核较大的卷积层，一方面可以减少参数，另一方面相当于进行了更多的非线性映射，可以增加网络的拟合/表达能力。

&emsp;&emsp;小卷积核是 $VGG$ 的一个重要特点，虽然 $VGG$ 是在模仿 $AlexNet$ 的网络结构，但没有采用 $AlexNet$ 中比较大的卷积核尺寸（如 $7 \times 7$），而是通过降低卷积核的大小（$3 \times 3$），增加卷积子层数来达到同样的性能（$VGG$：从 $1$ 到 $4$ 卷积子层，$AlexNet$：$1$ 子层）。

&emsp;&emsp;$VGG$ 的作者认为两个 $3 \times 3$ 的卷积堆叠获得的感受野大小，相当一个 $5 \times 5$ 的卷积；而 $3$ 个 $3 \times 3$ 卷积的堆叠获取到的感受野，相当于一个 $7 \times 7$ 的卷积。这样可以增加非线性映射，也能很好地减少参数（例如 $7 \times 7$ 的参数为 $49$ 个，而 $3$ 个 $3 \times 3$ 的参数为 $27$）。

### 小池化核
&emsp;&emsp;相比 $AlexNet$ 的 $3 \times 3$ 的池化核，$VGG$ 全部采用 $2 \times 2$ 的池化核。

### 通道数增多
&emsp;&emsp;$VGG$网络第一层的通道数为$64$，后面每层都进行了翻倍，最多到$512$个通道，通道数的增加，使得更多的信息可以被提取出来。

### 层数更深、特征图更宽
&emsp;&emsp;由于卷积核专注于扩大通道数、池化专注于缩小宽和高，使得模型架构上更深更宽的同时，控制了计算量的增加规模。

### 全连接转卷积（测试阶段）
&emsp;&emsp;这也是 $VGG$ 的一个特点，在网络测试阶段将训练阶段的三个全连接替换为三个卷积，使得测试得到的全卷积网络因为没有全连接的限制，因而可以接收任意大小的输入。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/VGGNet/vggnet6.jpeg?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">vggnet</div>
</center>

&emsp;&emsp;上图是 $VGG$ 网络最后三层的替换过程，上半部分是训练阶段，此时最后三层都是全连接层（输出分别是 $4096$、$4096$、$1000$），下半部分是测试阶段（输出分别是 $1 \times 1 \times 4096$、$1 \times 1 \times 4096$、$1 \times 1 \times 1000$），最后三层都是卷积层。

&emsp;&emsp;训练阶段，有 $4096$ 个输出的全连接层 $FC6$ 的输入是一个 $7 \times 7 \times 512$ 的 $feature \ map$，因为全连接层的缘故，不需要考虑局部性， 可以把 $7 \times 7 \times 512$ 看成一个整体，$7 \times 7 \times 512=25508$ 个输入的每个元素都会与输出的每个元素（或者说是神经元）产生连接，所以每个输入都会有$4096$ 个系数对应 $4096$ 个输出，所以网络的参数（也就是两层之间连线的个数，也就是每个输入元素的系数个数）规模就是 $7 \times 7 \times 512  \times 4096$。对于 $FC7$，输入是 $4096$ 个，输出是 $4096$ 个，因为每个输入都会和输出相连，即每个输出都有 $4096$ 条连线（系数），那么 $4096$ 个输入总共有 $4096 \times 4096$ 条连线（系数），最后一个 $FC8$ 计算方式一样。

&emsp;&emsp;测试阶段，由于换成了卷积，第一个卷积后要得到 $4096$（或者说是 $1 \times 1 \times 4096$）的输出，那么就要对输入的 $7 \times 7 \times 512$ 的 $feature \ map$ 的宽高（即 $width$、$height$ 维度）进行降维，同时对深度（即 $Channel/depth$ 维度）进行升维。要把 $7 \times 7$ 降维到 $1 \times 1$，那么干脆直接一点，就用 $7 \times 7$ 的卷积核就行，另外深度层级的升维，因为 $7 \times 7$ 的卷积把宽高降到 $1 \times  1$，那么刚好就升高到 $4096$ 就好了，最后得到了 $1 \times 1 \times 4096$ 的 $feature \ map$。这其中卷积的参数量上，把 $7 \times 7 \times 512$ 看做一组卷积参数，因为该层的输出是 $4096$，那么相当于要有 $4096$ 组这样 $7 \times 7 \times 512$ 的卷积参数，那么总共的卷积参数量就是：$[7 \times 7 \times 512] \times 4096$，这里将 $7 \times 7 \times 512$ 用中括号括起来，目的是把这看成是一组，就不会懵。第二个卷积依旧得到 $1 \times 1 \times 4096$ 的输出，因为输入也是 $1 \times 1 \times 4096$，三个维度（宽、高、深）都没变化，可以很快计算出这层的卷积的卷积核大小也是 $1 \times 1$，而且，通道数也是 $4096$，因为对于输入来说，$1 \times 1 \times 4096$ 是一组卷积参数，即一个完整的 $filter$，那么考虑所有 $4096$ 个输出的情况下，卷积参数的规模就是 $[1 \times 1 \times 4096] \times 4096$。第三个卷积的计算一样。

&emsp;&emsp;$VGG$ 的作者把训练阶段的全连接替换为卷积是参考了 $OverFeat$ 的工作，如下图是 $OverFeat$ 将全连接换成卷积后，带来可以处理任意分辨率（在整张图）上计算卷积，而无需对原图 $resize$ 的优势。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/VGGNet/vggnet5.jpg?raw=true"
    width="512" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">vggnet</div>
</center>

&emsp;&emsp;可以看到，训练阶段用的是 $crop$ 或者 $resize$ 到 $14 \times 14$ 的输入图像，而测试阶段可以接收任意维度，如果使用未经 $crop$ 的原图作为输入（假设原图比 $crop$ 或者 $resize$ 到训练尺度的图像要大），这会带来一个问题：$feature \ map$ 变大了。比方 $VGG$ 训练阶段用 $224 \times 224 \times 3$ 的图作为模型输入，经过 $5$ 组卷积和池化，最后到 $7 \times 7 \times 512$ 维度，最后经过无论是三个卷积或者三个全连接，维度都会到 $1 \times 1 \times 4096->1 \times 1 \times 4096->1 \times 1 \times 1000$，而使用$ 384 \times 384 \times 3$ 的图做模型输入，到五组卷积和池化做完（即），那么 $feature \ map$ 变为 $12 \times 12 \times 512$，经过三个由全连接变的三个卷积，即 $feature \ map$ 经历了 $6 \times 6 \times 4096->6 \times 6 \times 4096->6 \times 6 \times 1000$ 的变化过程后，再把 $6 \times 6 \times 1000$的$feature \  map$ 进行 $average$，最终交给 $SoftMax$ 的是 $1 \times 1 \times 1000$ 的 $feature \ map$ 进行分类。

### 过拟合
* $L2$ 正则化
* $dropout$

## 总结
* 输入图像的大小 $224 \times 224 \times 3$
* 通过增加深度能有效地提升性能
* 最佳模型：$VGG16$，从头到尾只有 $3 \times 3$ 卷积与 $2 \times 2$ 池化，简洁优美
* 卷积可替代全连接，可适应各种尺寸的图片
