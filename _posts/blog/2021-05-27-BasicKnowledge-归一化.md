---
layout: post
title: 归一化
categories: [BasicKnowledge]
description: 归一化
keywords: BasicKnowledge
---


深度学习基础知识点归一化
---


## 局部响应归一化(LRN, Local Response Normalization)

### 简介
&emsp;&emsp;$LRN$ 归一化技术首次 在 $AlexNet$ 模型中被提出，其一般跟在激活层或池化层后。

### 原理
&emsp;&emsp;在神经生物学中有一个概念叫做“侧抑制”($lateral \ inhibitio$)，指的是被激活的神经元抑制相邻神经元。$LRN$ 借鉴“侧抑制”的思想来实现局部抑制，尤其当使用 $relu$ 的时候这种“侧抑制”很管用。

### 作用
&emsp;&emsp;有利于提高模型的泛化能力(如何提高？$LRN$ 模仿生物神经系统的侧抑制机制，对局部神经元的活动创建竞争机制，使响应比较大的值相对更大，从而提升模型的泛化能力。)

### 计算公式
* 参数说明
  * $i$ 是通道的位置，表示更新第几个通道的值
  * $x$ 和 $y$ 表示更新像素的位置
  * ${ b }  _  { x,y }^{ i }$ 是归一化的值
  * ${ a }  _  { x,y }^{ i }$ 是输入值，是激活函数的输出值
  * $k$、$\alpha$、$\beta$、${ n }/{ 2 }$ 都是自定义系数
  * $N$是总的通道数，不同通道累加的平方和
* 公式

$$
{ b }_{ x,y }^{ i }={ { a }_{ x,y }^{ i } }/{ { \left( k+\alpha \sum _ { j=\max { \left( 0,i-{ n }/{ 2 } \right)  }  }^{ \min { \left( N-1,i+{ n }/{ 2 } \right)  }  }{ { \left( { a }_ { x,y }^{ j } \right)  }^{ 2 } }  \right)  }^{ \beta  } }
$$

* 说明
  * 总的来说，是对输入值 ${ a }  _  { x,y }^{ i }$ 除以一个数达到归一化的目的
  * 累加多少个通道的像素值？取决于自定义系数 ${ n }/{ 2 }$

### 后期争议
&emsp;&emsp;在2015年 Very Deep Convolutional Networks for Large-Scale Image Recognition一文中提到 $LRN$ 基本没什么用。


## 批量归一化(BN, Batch Normalization)

### Internal Covariate Shift
&emsp;&emsp;训练深层网络的过程中，由于网络中参数变化而引起内部结点数据的分布发生变化的这一过程被称作内部神经元分布转移 “$Internal \ Covariate \ Shift$”。

&emsp;&emsp;我们定义每一层的线性变换为 ${ Z }^{ l }={ W }^{ l }\times input+{ b }^{ l }$，其中 $l$ 代表层数；非线性变换为 ${ A }^{ l }={ g }^{ l }\left( { Z }^{ l } \right)$，其中 ${ g }^{ l }$ 为第 $l$ 层的激活函数。随着梯度下降的进行，每一层的参数 $ W^{l}$ 与 $b^{l}$ 都会被更新，那么 ${ Z }^{ l }$ 的分布也就发生了改变，进而 $A^{l}$ 也同样出现分布的改变。而 $A^{l} $ 作为第 $l+1$ 层的输入，意味着 $l+1$ 层就需要去不停适应这种数据分布的变化，这一过程就被叫做 “$Internal \ Covariate \ Shift$”。

### Internal Covariate Shift 带来的问题
* 上层网络需要不停调整来适应输入数据分布的变化，导致网络学习速度的降低
  * 梯度下降的过程会让每一层的参数 $W^{l}$和$b^{l}$ 发生变化，进而使得每一层的线性与非线性计算结果分布产生变化。
  * 上层网络就要不停地去适应这种分布变化，这个时候就会使得整个网络的学习速率过慢。
* 网络的训练过程容易陷入梯度饱和区，减缓网络收敛速度
  * 神经网络中采用饱和激活函数时，例如 $sigmoid$，$tanh$ 激活函数，很容易使得模型训练陷入梯度饱和区。
    > 原因：随着模型训练的进行，我们的参数 $W^{l}$ 会逐渐更新并变大，此时 ${ Z }^{ l }={ W }^{ l }\times { Z }^{ l-1 }+{ b }^{ l }$ 就会随之变大，${ Z }^{ l }$ 还受到更底层网络参数 $W^{1}$，$ W^{2}$，···，$ W^{l-1}$ 的影响。随着网络层数的加深，${ Z }^{ l }$ 很容易陷入梯度饱和区，此时梯度会变得很小甚至接近于 $0$，参数的更新速度就会减慢，进而就会放慢网络的收敛速度。
  * 对于激活函数梯度饱和问题，有两种解决思路。
    > 一种是更换为非饱和性激活函数，例如 $ReLU$ 可以在一定程度上解决训练进入梯度饱和区的问题。另一种是让激活函数的输入分布保持在一个稳定状态来尽可能避免它们陷入梯度饱和区，这也就是 $Normalization$ 的思路。

### 减缓 Internal Covariate Shift 的方法
&emsp;&emsp;$ICS$ 产生的原因是由于参数更新使网络中每一层输入值的分布发生改变，并且随着网络层数的加深而变得更加严重，因此可以通过固定网络中每一层输入值的分布来对减缓$ICS$问题。
* 白化($Whitening$)
  * 白化是机器学习里面常用的一种规范化数据分布的方法，主要是 $PCA$ 白化与 $ZCA$ 白化。白化是对输入数据分布进行变换，进而达到以下两个目的：
    * 使得输入特征分布具有相同的均值与方差
      > $PCA$ 白化保证了所有特征分布均值为 $0$，方差为 $1$；$ZCA$ 白化则保证了所有特征分布均值为 $0$，方差相同；

    * 去除特征之间的相关性
  * 白化操作可以减缓 $ICS$ 的问题，进而固定了每一层网络输入分布，加速网络训练过程的收敛
* $Batch \ Normalization$ 的提出
  * 白化主要有以下两个问题：
    * 白化过程计算成本太高，并且在每一轮训练中的每一层都需要做如此高成本计算的白化操作；
    * 白化过程改变了网络每一层的分布，使底层网络学习到的参数信息被白化操作丢失掉。
  * 提出的 $normalization$ 方法要能够简化计算过程；另一方面经过规范化处理后要让数据尽可能保留原始的表达能力。于是就有了简化+改进版的白化—— $Batch \ Normalization$。

### Batch 的来路
&emsp;&emsp;在深度学习中，由于采用 $full \ batch$ 的训练方式对内存要求较大，且每一轮训练时间过长；我们一般都会采用对数据做划分，用 $mini-batch$ 对网络进行训练。因此，$Batch \ Normalization$ 也就在 $mini-batch$ 的基础上进行计算。

### 计算过程
#### 参数说明
* $M$：训练样本的数量
* $N$：训练样本的特征数
* $X$：训练样本集， $X=\{x^{(1)},x^{(2)},\cdots,x^{(M)}\}$，$X\in \mathbb{R}^{N\times M}$ （注意这里$X$的一列是一个样本）
* $m$：batch size，即每个 batch 中样本的数量
* $\chi^{(i)}$ ：第 $i$ 个mini-batch的训练数据，$X= \{\chi^{(1)},\chi^{(2)},\cdots,\chi^{(k)}\}$ ，其中 $\chi^{(i)}\in \mathbb{R}^{N\times m}$
* $l$：网络中的层标号
* $L$：网络中的最后一层或总层数
* $d  _  l$：第 $l$ 层的维度，即神经元结点数
* $W^{l}$：第 $l$ 层的权重矩阵，$W^{l}\in \mathbb{R}^{d  _  l\times d  _  {l-1}}$
* $b^{l}$：第 $l$ 层的偏置向量，$b^{l}\in \mathbb{R}^{d  _  l\times 1}$
* $Z^{l}$：第 $l$ 层的线性计算结果，$Z^{l}=W^{l}\times input+b^{l}$
* $g^{l}(\cdot)$：第 $l$ 层的激活函数
* $A^{l}$：第 $l$ 层的非线性激活结果，$A^{l}=g^{l}(Z^{l})$

#### 算法步骤
  * 对每个特征进行独立的 $normalization$。我们考虑一个 $batch$ 的训练，传入 $m$ 个训练样本，并关注网络中的某一层，忽略上标 $l$。
  * 关注当前层的第 $j$ 个维度，也就是第 $j$ 个神经元结点，则有 $Z  _  j\in \mathbb{R}^{1\times m}$。当前维度进行规范化(其中 $\epsilon$ 是为了 防止方差为 $0$ 产生无效计算)：

    $$
    \begin{aligned} { \mu  }_ { j }&=\frac { 1 }{ m } \sum _ { i=1 }^{ m }{ { Z }_ { j }^{ (i) } }  \\ { \sigma  }_ { j }^{ 2 }&=\frac { 1 }{ m } \sum _{ i=1 }^{ m }{ { \left( { Z }_{ j }^{ (i) }-{ \mu  }_ { j } \right)  }^{ 2 } }  \\ { \hat { Z }  }_ { j }&=\frac { { Z }_ { j }-{ \mu  }_ { j } }{ \sqrt { { \sigma  }_ { j }^{ 2 }+\epsilon  }  }  \end{aligned}
    $$

  * 示例演示
    * 只关注第 $l$ 层的计算结果，左边的矩阵是 $Z^{l}=W^{l}A^{l-1}+b^{l}$ 线性计算结果，还未进行激活函数的非线性变换。此时每一列是一个样本，图中可以看到共有$8$列，代表当前训练样本的 $batch$ 中共有 $8$ 个样本，每一行代表当前 $l$ 层神经元的一个节点，可以看到当前 $l$ 层共有 $4$ 个神经元结点，即第 $l$ 层维度为 $4$。可以看到，每行的数据分布都不同。

    ![bn1](https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Normalization/BatchNormalization/bn1.jpg?raw=true)

    * 对于第一个神经元，我们求得 $\mu  _  1=1.65$ ， $\sigma^2  _  1=0.44$ （其中 $\epsilon=10^{-8}$），此时我们利用 $\mu  _  1$，$\sigma^2  _  1$ 对第一行数据（第一个维度）进行 $normalization$ 得到新的值 $[-0.98,-0.23,-0.68,-1.13,0.08,0.68,2.19,0.08]$ 。同理可以计算出其他输入维度归一化后的值，如下图：
    
    ![bn2](https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Normalization/BatchNormalization/bn2.jpg?raw=true)
    
    * 通过上面的变换，解决了第一个问题，用更加简化的方式来对数据进行规范化，使得第 $l$ 层的输入每个特征的分布均值为 $0$，方差为 $1$。
    
    * 虽然 $Normalization$ 操作缓解了 $ICS$ 问题，让每一层网络的输入数据分布都变得稳定，但却导致了数据表达能力的缺失，使得底层网络学习到的参数信息丢失。
    * 因此，$BN$ 又引入了两个可学习的参数 $\gamma$ 与 $\beta$ 。这两个参数的引入是为了恢复数据本身的表达能力，对规范化后的数据进行线性变换，即 $\tilde{Z  _  j}=\gamma  _  j \hat{Z}  _  j+\beta  _  j$。
    * 特别地，当 $\gamma^2=\sigma^2$，$\beta=\mu$ 时，可以实现等价变换（$identity \ transform$）并且保留了原始输入特征的分布信息。
    * 通过上面的步骤，就在一定程度上保证了输入数据的表达能力。

#### 公式
对于神经网络中的第 $l$ 层，有：

$$
\begin{aligned} { Z }^{ l }&={ W }^{ l }{ A }^{ l-1 }+{ b }^{ l }\\ \mu &=\frac { 1 }{ m } \sum _ { i=1 }^{ m }{ { Z }^{ li } } \\ { \sigma  }^{ 2 }&=\frac { 1 }{ m } \sum _ { i=1 }^{ m }{ { \left( { Z }^{ li }-\mu  \right)  }^{ 2 } } \\ { \tilde { Z }  }^{ l }&=\gamma \cdot \frac { { Z }^{ l }-\mu  }{ \sqrt { { \sigma  }^{ 2 }+\epsilon  }  } +\beta \\ { A }^{ l }&={ g }^{ l }\left( { \tilde { Z }  }^{ l } \right) \end{aligned}
$$

### 测试阶段使用 Batch Normalization
&emsp;&emsp;$BN$ 在每一层计算的 $\mu$ 与 $\sigma^2$ 都是基于当前 $batch$中 的训练数据，但是这就带来了一个问题：在预测阶段，有可能只需要预测一个样本或很少的样本，没有像训练样本中那么多的数据，此时 $\mu$ 与 $\sigma^2$ 的计算一定是有偏估计，这个时候该如何进行计算呢？

&emsp;&emsp;利用 $BN$ 训练好模型后，保留了每组 $mini-batch$ 训练数据在网络中每一层的 $\mu  _  {batch}$ 与 $\sigma^2  _  {batch}$ 。此时我们使用整个样本的统计量来对测试数据进行归一化，具体来说使用均值与方差的无偏估计：

$$
\begin{aligned} { \mu  }_{ test }&=E\left( { \mu  }_{ batch } \right)  \\ { \sigma  }_{ test }^{ 2 }&=\frac { m }{ m-1 } E\left( { \sigma  }_{ batch }^{ 2 } \right)  \end{aligned}
$$

得到每个特征的均值与方差的无偏估计后，对测试数据采用同样的 $normalization$ 方法：
$$
BN({ X })=\gamma \cdot \frac { { X }_ { test }-{ \mu  }_ { test } }{ \sqrt { { { \sigma  }_ { test }^{ 2 }+\epsilon  } }  } +\beta 
$$

> 原文称该方法为移动平均

> 为什么训练时不采用移动平均？
> $BN$ 的作者认为在训练时采用移动平均可能会与梯度优化存在冲突

&emsp;&emsp;另外，除了采用整体样本的无偏估计外。吴恩达在 $Coursera$ 上的 $Deep Learning$ 课程指出可以对 $train$ 阶段每个 $batch$ 计算的 $mean/variance$ 采用指数加权平均来得到 $test$ 阶段 $mean/variance$ 的估计。

### 使用地方
&emsp;&emsp;论文中特别指出在 $CNN$ 中，$BN$ 应作用在激活函数前，即对 $x=Wu+b$ 做规范化。

### 作用
&emsp;&emsp;$Batch \ Normalization$ 在实际工程中被证明了能够缓解神经网络难以训练的问题，$BN$ 具有的有事可以总结为以下几点：

* $BN$ 使网络中每层输入数据的分布相对稳定，加速模型学习速度
  * $BN$ 通过规范化与线性变换使得每一层网络的输入数据的均值与方差都在一定范围内，使得上层网络不必不断去适应底层网络中输入的变化。
  * 从而实现了网络中层与层之间的解耦，允许每一层进行独立学习，有利于提高整个神经网络的学习速度。
* $BN$ 使模型对网络中的参数不那么敏感，简化调参过程，使得网络学习更加稳定
  * 在神经网络中，我们经常会谨慎地采用一些权重初始化方法（例如$Xavier$）或者合适的学习率来保证网络稳定训练。
  * 当学习率设置太高时，会使得参数更新步伐过大，容易出现震荡和不收敛。
  * 但是使用 $BN$ 的网络将不会受到参数数值大小的影响。
* $BN$ 允许网络使用饱和性激活函数（例如 $sigmoid$，$tanh$等），缓解梯度消失问题
  * 在不使用 $BN$ 层的时候，由于网络的深度与复杂性，很容易使得底层网络变化累积到上层网络中，导致模型的训练很容易进入到激活函数的梯度饱和区；
  * 通过 $normalize$ 操作可以让激活函数的输入数据落在梯度非饱和区，缓解梯度消失的问题；
  * 另外通过自适应学习 $\gamma$ 与 $\beta$ 又让数据保留更多的原始表达能力。
* $BN$ 具有一定的正则化效果 （一定程度可以防止过拟合）
  * 在 $Batch \ Normalization$ 中，由于我们使用 $mini-batch$ 的均值与方差作为对整体训练样本均值与方差的估计
  * 尽管每一个 $batch$ 中的数据都是从总体样本中抽样得到，但不同 $mini-batch$ 的均值与方差会有所不同，这就为网络的学习过程中增加了随机噪音
  * 与 $Dropout$ 通过关闭神经元给网络训练带来噪音类似，在一定程度上对模型起到了正则化的效果
* 另外，作者证明了加入$BN$后，可以丢弃 $Dropout$，模型也同样具有很好的泛化效果

### 局限性
* 如果 $Batch Size$ 太小 ，则 $BN$ 效果明显下降
* 对于像素级图片生成任务来说，$BN$ 效果不佳
* $RNN$ 等动态网络使用 $BN$ 效果不佳且使用起来不方便
* 训练时和推理时统计量不一致
* 导致训练时间的增加
* 对于在线学习不好

## 分裂归一化(divisive normalization)

## 组归一化(Group Normalization)

### 简介
$GN$ 是一种新的深度学习归一化方式，可以替代 $BN$。众所周知，$BN$ 是深度学习中常使用的归一化方法，在提升训练以及收敛速度上发挥了重大的作用，是深度学习上里程碑式的工作，但是其仍然存在一些问题，而新提出的 $GN$ 解决了 $BN$ 归一化对 $batch \ size$ 依赖的影响。

### BN 缺陷
$BN $是以 $batch$ 的维度做归一化，此归一化方式依赖于 $batch$，过小的 $batch \ size$ 会导致其性能下降，一般来说每 $GPU$ 上 $batch$ 设为 $32$ 最合适，但是对于一些其他深度学习任务 $batch \ size$ 往往只有 $1-2$，比如目标检测，图像分割，视频分类。较大的 $batch \ size$ 显存吃不消，较小的 $batch \ size$ 性能表现很弱，如下图所示：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Normalization/GroupNormalization/batch-size-error.jpg?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">ImageNet classification error vs. batch sizes</div>
</center>

此外，$BN$ 是在 $batch$ 这个维度上 $Normalization$，但是这个维度并不是固定不变的，比如训练和测试时一般不一样，一般都是训练的时候在训练集上通过滑动平均预先计算好 $mean$ 和 $variance$ 参数，在测试的时候，不在计算这些值，而是直接调用这些预计算好的来用。但是，当训练数据和测试数据分布有差别是时，训练机上预计算好的数据并不能代表测试数据，这就导致在训练，验证，测试这三个阶段存在矛盾。

### GN 工作机制
$GN$ 本质上仍是归一化，但是它灵活的避开了 $BN$ 的问题，同时又不同于 $Layer Norm$，$Instance Norm$，四者的工作方式从下图可窥一斑：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Normalization/GroupNormalization/normalization-methods.jpg?raw=true"
    width="720" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Normalization methods</div>
</center>

上图形象的表示了四种 $norm$ 的工作方式：
* $BN$ 在 $batch$ 的维度上 $norm$，归一化维度为 $[N，H，W]$，对 $batch$ 中对应的 $channel$ 归一化；
* $LN$ 避开了 $batch$ 维度，归一化的维度为 $[C，H，W]$；
* $IN$ 归一化的维度为 $[H，W]$；
* $GN$ 介于 $LN$ 和 $IN$ 之间，其首先将 $channel$ 分为许多组，对每一组做归一化，及先将 $feature$ 的维度由 $[N, C, H, W]$ $reshape$ 为$[N, G，C//G , H, W]$，归一化的维度为 $[C//G , H, W]$；

事实上，$GN$ 的极端情况就是 $LN$ 和 $IN$，分别对应 $G$ 等于 $C$ 和 $G$ 等于 $1$，作者在论文中给出 $G$ 设为 $32$ 较好。

```python
def GroupNorm(x, gamma, beta, G, eps=1e−5):
    # x: input features with shape [N,C,H,W]
    # gamma, beta: scale and offset, with shape [1,C,1,1]
    # G: number of groups for GN
    N, C, H, W = x.shape
    x = tf.reshape(x, [N, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep dims=True)
    x = (x − mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [N, C, H, W])
    return x ∗ gamma + beta
```

其中 $beta$ 和 $gama$ 参数是 $norm$ 中可训练参数，表示平移和缩放因子。

### 优势
$GN$ 的归一化方式避开了 $batch size$ 对模型的影响，$GN$ 同样可以解决 $Internal \ Covariate \ Shift$ 的问题，并取得较好的效果。

### 参考
[Group Normalization](https://arxiv.org/pdf/1803.08494.pdf)

[全面解读Group Normalization-（吴育昕-何恺明 ）](https://zhuanlan.zhihu.com/p/35005794)


## FRN (Filter Response Normalization)

### 简介
谷歌的提出的FRN层不仅消除了模型训练过程中对$batch$的依赖，而且当$batch \ size$较大时性能优于$BN$。

### 工作机制
$FRN$ 层包括归一化层 $FRN（Filter Response Normalization）$ 和激活层 $TLU（Thresholded Linear Unit）$，如下图所示：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Normalization/FRN/fpn.jpg?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">FRN</div>
</center>

其中 $FRN$ 的操作是 $(H, W)$ 维度上的，即对每个样例的每个 $channel$ 单独进行归一化，这里 $x$ 就是一个 $N$ 维度（$H \times W$）的向量，所以 $FRN$ 没有 $BN$ 层对 $batch$ 依赖的问题。$BN$ 层采用归一化方法是减去均值然后除以标准差，而 $FRN$ 却不同，这里没有减去均值操作，公式中的 $v^2$ 是 $x$ 的二次范数的平均值。这种归一化方式类似$BN$可以用来消除中间操作（卷积和非线性激活）带来的尺度问题，有助于模型训练。

此外，$\epsilon$ 是一个很小的正常量，以防止除 $0$。$FRN$ 是在 $H,W$ 两个维度上归一化，一般情况下网络的特征图大小 $N=H \times W$ 较大，但是有时候可能会出现 $1 \times 1$ 的情况，比如 $InceptionV3$ 和 $VGG$ 网络，此时 $\epsilon $ 就比较关键，图 $4$ 给出了当 $N=1$ 时不同下归一化的结果。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Normalization/FRN/epsilon-effect.jpg?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Effect of epsilon</div>
</center>

当 $\epsilon$ 值较小时，归一化相当于一个符号函数，这时候梯度几乎为 $0$，严重影响模型训练；当值较大时，曲线变得更圆滑，此时的梯度利于模型学习。对于这种情况，论文建议采用一个可学习的 $\epsilon$。对于不含有 $1 \times 1$特征的模型，论文中采用的是一个常量值 $1e-6$。值得说明的是 $IN$ 也是在 $H,W$ 维度上进行归一化，但是会减去均值，对于 $N=1$ 的情况归一化的结果是 $0$，但FRN可以避免这个问题。

归一化之后同样需要进行缩放和平移变换，这里的 $\gamma$ 和 $\beta$ 也是可学习的参数（参数大小为 $C$）：

$$
y=\gamma \widehat{x}+\beta 
$$

$FRN$ 缺少去均值的操作，这可能使得归一化的结果任意地偏移 $0$，如果 $FRN$ 之后是 $ReLU$ 激活层，可能产生很多 $0$ 值，这对于模型训练和性能是不利的。为了解决这个问题，$FRN$ 之后采用的阈值化的 $ReLU$，即 $TLU$：

$$
z=\max \left( y,\tau \right) =\text{Re}LU\left( y-\tau \right) +\tau 
$$

这里的 $\tau $是一个可学习的参数。论文中发现 $FRN$ 之后采用 $TLU$ 对于提升性能是至关重要的。

### 实现代码
在 $TensorFlow$ 中的实现代码如下所示：

```python
def FRNLayer(x, tau, beta, gamma, eps=1e-6):
    # x: Input tensor of shape [BxHxWxC].
    # alpha, beta, gamma: Variables of shape [1, 1, 1, C].
    # eps: A scalar constant or learnable variable.
    # Compute the mean norm of activations per channel.
    nu2 = tf.reduce _ mean(tf.square(x), axis=[1, 2],
                keepdims=True)
    # Perform FRN.
    x = x * tf.rsqrt(nu2 + tf.abs(eps))
    # Return after applying the Offset-ReLU non-linearity.
    return tf.maximum(gamma * x + beta, tau)
```

### 效果
$FRN$ 层的效果也是极好的，下图给出了 $FRN$ 与 $BN$ 和 $GN$ 的效果对比：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/Normalization/FRN/compare.jpg?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">compare</div>
</center>

可以看到 $FRN$ 是不受 $batch \ size$ 的影响，而且效果是超越 $BN$ 的。论文中还有更多的对比试验证明 $FRN$ 的优越性。

### 参考
[Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks](https://arxiv.org/pdf/1911.09737.pdf)

[超越BN和GN！谷歌提出新的归一化层：FRN](https://mp.weixin.qq.com/s/9EjTX-Al28HLV0k1FZPvIg)

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


<iframe src="https://rawcdn.githack.com/TinyJay/donate-page/51aaf216f048b8e6d5ce01443a32be930b91869d/simple/index.html" style="overflow-x:hidden;overflow-y:hidden; border:0xp none #fff; min-height:240px; width:100%;"  frameborder="0" scrolling="no"></iframe>