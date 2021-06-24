---
layout: post
title: 广义线性模型
categories: [MachineLearning]
description: 广义线性模型
keywords: MachineLearning
---


广义线性模型
---


## 广义线性模型的函数定义
&emsp;&emsp;考虑单调可微函数 $g\left( \cdot  \right) $，令$g\left( y \right) ={ w }^{ T }x+b$，这样得到的模型称作广义线性模型(generalized linear model)。其中函数 $g\left( \cdot  \right) $ 称作联系函数 (link function) 。

&emsp;&emsp;对数线性回归是广义线性模型在 $g\left( \cdot  \right) =\ln { \left( \cdot  \right)  } $ 时的特例。即：$\ln { \left( y \right) ={ w }^{ T }x+b } $。对数线性回归实际上试图让 ${ e }^{ { w }^{ T }x+b }$ 逼近 $y$，在形式上是线性回归，但是实质上是非线性的。

## 广义线性模型的概率定义
&emsp;&emsp;给定 $x$ 和 $w$ 之后，如果 $y$ 的条件概率分布 $p\left( y|x;w \right) $ 服从指数分布族，则该模型称作广义线性模型。指数分布族的形式为：

$$
p\left( y|\eta  \right) =b\left( y \right) \times { e }^{ \left( \eta T\left( y \right) -a\left( \eta  \right)  \right)  }
$$

其中，$\eta$ 是 $x$ 的线性函数：$\eta ={ w }^{ T }x$； $b(y)$ 和 $T(y)$ 为 $y$ 的函数；$a\left( \eta  \right) $ 为 $\eta$ 的函数。

## 常见分布的广义线性模型

### 高斯分布

$$
p\left( y \right) =\frac { 1 }{ \sqrt { 2\pi  } \sigma  } exp\left( -\frac { { \left( y-\mu  \right)  }^{ 2 } }{ 2{ \sigma  }^{ 2 } }  \right) =\frac { 1 }{ \sqrt { 2\pi  } \sigma  } exp\left( -\frac { { y }^{ 2 } }{ 2{ \sigma  }^{ 2 } }  \right) exp\left( \frac { \mu  }{ { \sigma  }^{ 2 } } \times y-\frac { { \mu  }^{ 2 } }{ { 2\sigma  }^{ 2 } }  \right) 
$$

令：

$$
\begin{aligned} 
b\left( y \right) &=\frac { 1 }{ \sqrt { 2\pi  } \sigma  } exp\left( -\frac { { y }^{ 2 } }{ 2{ \sigma  }^{ 2 } }  \right)  \\ T\left( y \right) &=y \\ \eta &=\frac { \mu  }{ { \sigma  }^{ 2 } }  \\ a\left( \eta  \right) &=\frac { { \mu  }^{ 2 } }{ { 2\sigma  }^{ 2 } } 
\end{aligned}
$$

则满足广义线性模型。
    

### 伯努利分布
&emsp;&emsp;伯努利分布（二项分布，y 为 0 或者 1，取 1的概率为 $\phi $）：

$$
p\left( y;\phi  \right) ={ \phi  }^{ y }{ \left( 1-\phi  \right)  }^{ 1-y }=exp\left( y\ln { \frac { \phi  }{ 1-\phi  }  } +\ln { \left( 1-\phi  \right)  }  \right) 
$$

令：

$$
\begin{aligned} 
b\left( y \right) &=1\\ \eta &=\ln { \frac { \phi  }{ 1-\phi  }  } \\ T\left( y \right) &=y\\ a\left( \eta  \right) &=-\ln { \left( 1-\phi  \right)  } 
\end{aligned}
$$

则满足广义线性模型。

&emsp;&emsp;根据 $\eta ={ w }^{ T }x$ 有 $\eta ={ w }^{ T }x=\ln { \frac { \phi  }{ 1-\phi  }  } $，从而得到：

$$
\phi =\frac { 1 }{ 1+{ e }^{ { w }^{ T }x } } 
$$

因此，$logistic$ 回归属于伯努利分布的广义形式。
    

### 多元伯努利分布
&emsp;&emsp;假设有 $K$ 个分类，样本标记 $\tilde { y } \in \left\{ 1,2,\cdots ,K \right\} $。每种分类对应的概率为 ${ \phi  }  _  { 1 },{ \phi  }  _  { 2 },\cdots ,{ \phi  }  _  { K }$。则根据全概率公式有:

$$
\begin{aligned} 
\sum _{ i=1 }^{ K }{ { \phi  }_{ i } } &=1\\ { \phi  }_{ K }&=1-\sum _{ i=1 }^{ K-1 }{ { \phi  }_{ i } } 
\end{aligned}
$$

定义 $T(y)$ 为一个 $K-1$ 维的列向量：

$$
T\left( 1 \right) =\left[ \begin{matrix} 1 \\ 0 \\ 0 \\ \vdots  \\ 0 \end{matrix} \right] ,T\left( 2 \right) =\left[ \begin{matrix} 0 \\ 1 \\ 0 \\ \vdots  \\ 0 \end{matrix} \right] ,\cdots ,T\left( K-1 \right) =\left[ \begin{matrix} 0 \\ 0 \\ 0 \\ \vdots  \\ 1 \end{matrix} \right] ,T\left( K \right) =\left[ \begin{matrix} 0 \\ 0 \\ 0 \\ \vdots  \\ 0 \end{matrix} \right] 
$$

定义示性函数 :$I(y=i)$ 表示属于第 $i$ 分；$I(y\neq i)$ 表示不属于第 $i$ 分。则有：

$$
T(y)_ i=I(y=i)
$$

构建概率密度函数为：

$$
\begin{aligned} 
p\left( y;\phi  \right) &={ \phi  }_ { 1 }^{ I\left( y=1 \right)  }\times { \phi  }_ { 2 }^{ I\left( y=2 \right)  }\times \cdots \times { \phi  }_ { K }^{ I\left( y=K \right)  }\\ &={ \phi  }_{ 1 }^{ I\left( y=1 \right)  }\times { \phi  }_ { 2 }^{ I\left( y=2 \right)  }\times \cdots \times { \phi  }_{ K }^{ 1-\sum _{ i=1 }^{ K-1 }{ I\left( y=i \right)  }  }\\ &={ \phi  }_{ 1 }^{ { T\left( y \right)  }_{ 1 } }\times { \phi  }_ { 2 }^{ { T\left( y \right)  }_{ 2 } }\times \cdots \times { \phi  }_{ K }^{ 1-\sum _{ i=1 }^{ K-1 }{ { T\left( y \right)  }_{ i } }  }\\ &=exp\left( { T\left( y \right)  }_{ 1 }\times \ln { { \phi  }_{ 1 } } +{ T\left( y \right)  }_{ 2 }\times \ln { { \phi  }_{ 2 } } +\cdots +\left( 1-\sum _{ i=1 }^{ K-1 }{ { T\left( y \right)  }_{ i } }  \right) \times \ln { { \phi  }_{ K } }  \right) \\ &=exp\left( { T\left( y \right)  }_{ 1 }\times \ln { \frac { { \phi  }_{ 1 } }{ { \phi  }_{ K } }  } +{ T\left( y \right)  }_{ 2 }\times \ln { \frac { { \phi  }_{ 2 } }{ { \phi  }_ { K } }  } +\cdots +{ T\left( y \right)  }_{ K-1 }\times \ln { \frac { { \phi  }_{ K-1 } }{ { \phi  }_{ K } }  } +\ln { { \phi  }_{ K } }  \right) 
\end{aligned}
$$

令:
$$
\eta ={ \left( \ln { \frac { { \phi  }_{ 1 } }{ { \phi  }_{ K } }  } ,\ln { \frac { { \phi  }_{ 2 } }{ { \phi  }_{ K } }  } ,\cdots ,\ln { \frac { { \phi  }_{ K-1 } }{ { \phi  }_{ K } }  }  \right)  }^{ T }
$$

则有：

$$
p\left( y;\phi  \right) =exp\left( \eta \cdot T\left( y \right) +\ln { { \phi  }_ { K } }  \right) 
$$

令 $b(y)=1$, $a(\eta)=-\ln\phi  _  K$，则满足广义线性模型。
        
根据：

$$
{ \eta  }_{ i }=\ln { \frac { { \phi  }_{ i } }{ { \phi  }_{ K } }  } \rightarrow { \phi  }_{ i }={ \phi  }_{ K }{ e }^{ { \eta  }_{ i } }
$$

$$
1=\sum _{ i=1 }^{ K }{ { \phi  }_{ i } } ={ \phi  }_{ K }\left( 1+\sum _{ i=1 }^{ K-1 }{ { e }^{ { \eta  }_{ i } } }  \right) \rightarrow { \phi  }_{ K }=\frac { 1 }{ 1+\sum _{ i=1 }^{ K-1 }{ { e }^{ { \eta  }_{ i } } }  } 
$$

于是得到：

$$
{ \phi  }_{ i }=\begin{cases} \frac { { e }^{ { \eta  }_{ i } } }{ 1+\sum _{ j=1 }^{ K-1 }{ { e }^{ { \eta  }_{ j } } }  } ,i=1,2,\cdots ,K-1 \\ \frac { 1 }{ 1+\sum _{ j=1 }^{ K-1 }{ { e }^{ { \eta  }_{ j } } }  } ,i=K \end{cases}
$$
