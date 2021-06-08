---
layout: post
title: Inception-v4
categories: [Classification]
description: Inception-v4
keywords: Classification
---


分类模型 Inception-v4
---



## 背景
&emsp;&emsp;$Google$ 于 $2014$ 年和 $2015$ 年相继提出了 $Going \ Deeper \ with \ Convolutions(Inception-v1)$ 和 $Rethinking \ the \ Inception \ Architecture \ for \ Computer \ Vision(Inception-v2-v3)$，在 $ILSVRC \ 2012 \ classification \ challenge$ 验证集上取得了 $state \ of \ the \ art$ 的表现。

&emsp;&emsp;但在同月，$Microsoft$ 的 $Kaiming \ He$ 提出了 $Deep \ Residual \ Learning \ for \ Image \ Recognition(ResNet)$，使用残差连接的方式堆叠了 $152$ 层卷积网络，达到了与 $Inception-v3$ 相近的表现，并在 $ILSVRC \ 2015 \ classification \ task$ 中位居第一。使用残差结构可以加速训练，有助于防止梯度消失和梯度爆炸，层数越多效果越好(在没有发生过拟合的情况下)，$ResNet$ 认为：想得到深度卷积网络必须使用残差结构。

&emsp;&emsp;$Google$ 见到这个结论后，于 $2016$ 年发表 $Inception-v4$, $Inception-ResNet \ and \ the \ Impact \ of \ Residual \ Connections \ on \ Learning$，提出了 $Inception-v4$ 结构，证明了即使不引入残差结构也能达到和引入残差结构相似的结果，即 $Inception-v4$和$Inception-ResNet \  v2$，这两个模型都能取得 $state \ of \ the \ art$ 的表现。

## 网络结构
### 主干

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-V4/inception1.jpg?raw=true"
    width="256" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception1</div>
</center>

&emsp;&emsp;$Stem$ 是 $Inception-v4$ 结构的主干，起到基本特征提取的作用。$Inception \ module$ 适用于 $CNN$ 的中间层，处理高维特征。若直接将 $inception$ 模块用于低维特征的提取，模型的性能会降低。

&emsp;&emsp;$Inception-v4$ 使用了 $4$ 个 $Inception-A$ 模块、$7$ 个 $Inception-B$ 模块、$3$ 个 $Inception-C$ 模块，起到高级特征提取的作用。并且各个 $Inception$ 模块的输入输出维度是相同的，$Inception-A$、$Inception-B$、$Inception-C$分 别处理输入维度为 $35 \times 35$、$17 \times 17$、$8 \times  8$ 的 $feature \  map$，这种设计是懒人式的，即直接告知哪个 $module$ 对哪种 $size$ 的 $feature \ map$ 是最合适的，你只需要根据 $size$ 选择对应的 $module$ 即可。

&emsp;&emsp;$Inception-v4$ 使用了两种 $Reduction$ 模块，$Reduction-A$ 和 $Reduction-B$，作用是在避免瓶颈的情况下减小 $feature \ map$ 的 $size$，并增加 $feature \ map$ 的 $depth$。

&emsp;&emsp;$Inception-v4$ 使用了 $Network \ in \ Network$ 中的 $average-pooling$ 方法避免 $full-connect$ 产生大量的网络参数，使网络参数量减少了 $8 \times 8 \times 1536 \times 1536=144M$。最后一层使用 $softmax$ 得到各个类别的类后验概率，并加入 $Dropout$ 正则化防止过拟合。

### Stem (299x299x3 → 35x35x384)

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-V4/inception2.png?raw=true"
    width="256" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception2</div>
</center>

### Inception-A (35x35x384 → 35x35x384)

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-V4/inception3.png?raw=true"
    width="400" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception3</div>
</center>

### Reduction-A (35x35x384 → 17x17x1024)

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-V4/inception4.png?raw=true"
    width="400" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception4</div>
</center>

### Inception-B (17x17x1024 → 17x17x1024)

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-V4/inception5.png?raw=true"
    width="400" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception5</div>
</center>

### Reduction-B (17x17x1024 → 8x8x1536)

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-V4/inception6.png?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception6</div>
</center>

### Inception-C (8x8x1536 → 8x8x1536)

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-V4/inception7.png?raw=true"
    width="400" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception7</div>
</center>

# Inception-ResNet-v1
## 网络结构
### 主干
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-Resnet-V1/inception-resnet-v1-1.jpg?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception-resnet-v1-1</div>
</center>

### Stem

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-Resnet-V1/inception-resnet-v1-2.jpg?raw=true"
    width="240" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception-resnet-v1-2</div>
</center>

### Inception-resnet-A

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-Resnet-V1/inception-resnet-v1-3.jpg?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception-resnet-v1-3</div>
</center>

### Reduction-A

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-Resnet-V1/inception-resnet-v1-4.jpg?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception-resnet-v1-4</div>
</center>

### Inception-resnet-B

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-Resnet-V1/inception-resnet-v1-5.jpg?raw=true"
    width="256" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception-resnet-v1-5</div>
</center>

### Reduction-B
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-Resnet-V1/inception-resnet-v1-6.jpg?raw=true"
    width="400" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception-resnet-v1-6</div>
</center>

### Inception-resnet-C

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-Resnet-V1/inception-resnet-v1-7.jpg?raw=true"
    width="256" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception-resnet-v1-7</div>
</center>

# Inception-ResNet-v2
## 网络结构
### 主干
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-Resnet-V2/inception-resnet-v2-1.jpg?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception-resnet-v2-1</div>
</center>

### Stem
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-Resnet-V2/inception-resnet-v2-2.png?raw=true"
    width="256" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception-resnet-v2-2</div>
</center>

### Inception-resnet-A
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-Resnet-V2/inception-resnet-v2-3.jpg?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception-resnet-v2-3</div>
</center>

### Reduction-A
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-Resnet-V2/inception-resnet-v2-4.jpg?raw=true"
    width="320" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception-resnet-v2-4</div>
</center>

### Inception-resnet-B
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-Resnet-V2/inception-resnet-v2-5.jpg?raw=true"
    width="256" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception-resnet-v2-5</div>
</center>

### Reduction-B
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-Resnet-V2/inception-resnet-v2-6.jpg?raw=true"
    width="400" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception-resnet-v2-6</div>
</center>

### Inception-resnet-C
<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Model/Inception/Inception-Resnet-V2/inception-resnet-v2-7.jpg?raw=true"
    width="256" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">inception-resnet-v2-7</div>
</center>
