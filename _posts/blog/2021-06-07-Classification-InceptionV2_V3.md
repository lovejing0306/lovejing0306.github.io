---
layout: post
title: Inception-v2-v3
categories: [Classification]
description: Inception-v2-v3
keywords: Classification
---


分类模型 Inception-v2-v3
---


## inception-v2
### 背景
&emsp;&emsp;$GoogLeNet$ 凭借其优秀的表现，得到了很多研究人员的学习和使用，因此 $GoogLeNet$ 团队又对其进行了进一步地发掘改进，产生了升级版本的 $GoogLeNet$。

&emsp;&emsp;$GoogLeNet$ 设计的初衷就是要又准又快，而如果只是单纯的堆叠网络虽然可以提高准确率，但是会导致计算效率有明显的下降，所以如何在不增加过多计算量的同时提高网络的表达能力就成为了一个问题。$Inception \  v2$ 版本的解决方案就是修改 $Inception$ 的内部计算逻辑，提出了比较特殊的“卷积”计算结构。

### 网络结构

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-V2-V3/inception7.jpg?raw=true"
    width="320" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception7</div>
</center>

说明：
上表中的 $Figure \ 5$ 是指没有进化的 $Inception$，$Figure \ 6$ 是指小卷积版的 $Inception$（用 $3 \times 3$ 卷积核代替 $5 \times 5$ 卷积核），$Figure \ 7$ 是指非对称版的 $Inception$（用 $1 \times n$、$n \times 1$ 卷积核代替 $n \times n$ 卷积核）

### 网络技巧
#### Smaller Convolutions(小卷积)
&emsp;&emsp;大尺寸的卷积核可以带来更大的感受野，但也意味着会产生更多的参数，比如 $5 \times 5$ 卷积核的参数有 $25$ 个，$3 \times 3$ 卷积核的参数有 $9$ 个，前者是后者的 $25/9=2.78$ 倍。因此，$GoogLeNet$ 团队借鉴 $VGGNet$ 用 $2$ 个连续的 $3 \times 3$ 卷积层组成的小网络来代替单个的 $5 \times 5$ 卷积层，即在保持感受野范围的同时又减少了参数量，如下图：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-V2-V3/inception1.jpg?raw=true"
    width="256" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception1</div>
</center>

修正后的 $Inception \ module$ 如下图所示：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-V2-V3/inception2.jpg?raw=true"
    width="256" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception2</div>
</center>

#### Asymmetric Convoluitons(非对称卷积)
使用 $3 \times 3$ 的已经很小了，那么更小的 $2 \times 2$ 呢？

$2 \times 2$ 虽然能使得参数进一步降低，但是不如另一种方式更加有效，那就是 $Asymmetric$ 方式，即使用 $1 \times 3$ 和 $3 \times 1$ 两种来代替 $3 \times 3$，如下图所示：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-V2-V3/inception3.jpg?raw=true"
    width="196" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception3</div>
</center>

使用 $2$ 个 $2 \times 2$ 的话能节省11%的计算量，而使用这种方式则可以节省 33%。修正后的 $Inception \ module$ 如下图所示：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-V2-V3/inception4.jpg?raw=true"
    width="256" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception4</div>
</center>

> 注意：实践证明，这种模式的 $Inception$ 在前几层使用并不会导致好的效果，在中度大小的特征图（$feature \ map$）上使用效果才会更好（特征图大小建议在 $12$ 到 $20$ 之间）。

#### Auxiliary Classifiers(辅助分类器)
&emsp;&emsp;在 $GoogLeNet$ 中，在底层使用了多余的分类器，直觉上可以认为这样做可以使底层能够在梯度下降中学的比较充分，但在实践中发现两条：
* 多余的分类器在训练开始的时候并不能起到作用，在训练快结束的时候，使用它可以有所提升；
* 最底层多余的分类器去掉以后也不会有损失；
* 以为多余的分类器起到的是梯度传播下去的重要作用，但通过实验认为实际上起到的是 $regularizer$ 的作用，因为在多余的分类器前添加 $dropout$ 或者 $batch \ normalization$ 后效果更佳。

#### Grid Size Reduction(降低特征图大小)
&emsp;&emsp;一般情况下，如果想让图像缩小，可以有如下两种方式：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-V2-V3/inception5.jpg?raw=true"
    width="320" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception5</div>
</center>

&emsp;&emsp;先池化再作 $Inception$ 卷积，或者先作 $Inception$ 卷积再作池化。但是方法一（左图）先作 $pooling$（池化）会导致特征表示遇到瓶颈（特征缺失），方法二（右图）是正常的缩小，但计算量很大。为了同时保持特征表示且降低计算量，将网络结构改为下图，使用两个并行化的模块来降低计算量（卷积、池化并行执行，再进行合并）：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-V2-V3/inception6.jpg?raw=true"
    width="400" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception6</div>
</center>

#### Label Smoothing
？？？

#### Batch Normalization
&emsp;&emsp;$BN$ 是一个非常有效的归一化方法，可以让大型卷积网络的训练速度加快很多倍，同时收敛后的分类准确率也得到大幅提高。$BN$ 在用于神经网络某层时，会对每一个 $mini-batch$ 数据的内部进行标准化（$normalization$）处理，使输出规范化到 $ N(0,1) $ 的正态分布，减少了 $Internal Covariate Shift$（内部神经元分布的改变）。

&emsp;&emsp;$BN$ 的论文指出，传统的深度神经网络在训练时，每一层输入的分布都在变化，导致训练变得困难，只能使用一个很小的学习速率解决这个问题。而对每一层使用 $BN$ 之后，就可以有效地解决这个问题，学习速率可以增大很多倍，达到之前的准确率所需要的迭代次数只有 $1/14$，训练时间大大缩短。而达到之前的准确率后，可以继续训练，并最终取得远超于 $ Inception V1 $ 模型的性能—— $top-5$ 错误率 4.8%，已经优于人眼水平。因为 $BN$ 某种意义上还起到了正则化的作用，所以可以减少或者取消 $Dropout$ 和 $LRN$，简化网络结构。

## inception-v3
Inception-v2 BN-auxiliary