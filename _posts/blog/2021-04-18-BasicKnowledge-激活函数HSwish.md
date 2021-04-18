---
layout: post
title: 激活函数HSwish
categories: [BasicKnowledge]
description: 激活函数HSwish
keywords: BasicKnowledge
---


深度学习基础知识点激活函数HSwish
---

$HSwish$ 又称为 $Hard \ Swish$ 是对 $Swish$ 激活函数的改进，由谷歌团队在 $MobileNetV3$ 中提出，目的在于减少计算量。

## Hard Swish
$Hard \ Swish$ 激活函数的原始形式：

$$
f\left( x \right) =x\frac{\text{Re}LU6\left( x+3 \right)}{6}
$$

作者基于 $ReLU6$ 对 $Swish$ 改进的原因，作者认为几乎所有的软件和硬件框架上都可以使用 $ReLU6$ 的优化实现。


$Hard \ Swish$ 和 $Swish$ 激活函数对比如下：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/DeepLearning/Skill/ActiveFunction/hswish.jpg?raw=true"
    width="560" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Hard Swish</div>
</center>

## 优点
与 $Swish$ 相比 $Hard \ Swish$ 减少了计算量，具有和 $Swish$ 同样的性质。

## 缺点
与  $ReLU6$ 相比 $Hard \ Swish$ 的计算量仍然较大。

## 总结
$Hard \ Swish$ 可以看作 $Swish$ 激活函数的低精度版本，$Hard \ Swish$ 通过用线性类型的 $ReLU6$ 函数取代指数类型的 $Sigmoid$ 函数，减少了计算量。此外，在 $MobileNetV3$ 中作者认为 $Hard \ Swish$ 能够将通道数量减少到 $16$ 个的同时保持与使用 $ReLU6$ 或 $Swish$ 的 $32$ 个通道相同的精度。

## 参考
[Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)


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