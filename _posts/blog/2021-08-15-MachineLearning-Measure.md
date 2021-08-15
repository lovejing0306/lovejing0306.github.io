---
layout: post
title: 性能度量
categories: [MachineLearning]
description: 性能度量
keywords: MachineLearning
---


性能度量
---


&emsp;&emsp;给定训练集 $\mathbb D={(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)}$ ，测试集合 $\mathbb T={(x  _  1^{\prime},\tilde y  _  1^{\prime}),(x  _  2^{\prime},\tilde y  _  2^{\prime}),\cdots,(x^{\prime}  _  {N^{\prime}},\tilde y^{\prime}  _  {N^{\prime}})}$ 。对于样本 $x$ ，假设其真实标记为 $\tilde y$ ，模型预测输出为 $\hat y$ 。

## 分类问题性能度量

### 准确率、错误率
&emsp;&emsp;测试准确率：测试数据集上的准确率：

$$
r_{test}=\frac{1}{N^{\prime}} \sum_{i=1}^{N^{\prime}}I(\tilde y_i^{\prime} =\hat y_i^{\prime})
$$

其中 $I$ 为示性函数。准确率衡量的是有多少比例的样本被正确判别。

&emsp;&emsp;测试错误率：测试数据集上的错误率：

$$
e_{test}=\frac{1}{N^{\prime}} \sum_{i=1}^{N^{\prime}}I(\tilde y_i ^{\prime}\ne \hat y_i^{\prime})
$$

错误率衡量的是有多少比例的样本被判别错误，它也是损失函数为 $0-1$ 损失时的测试误差。

### 查准率、查全率
&emsp;&emsp;对于二分类问题，通常将关注的类作为正类，其他类作为负类。令：
1. $TP$ ：分类器将正类预测为正类的数量(True Positive) ，即：真正类的数量。
2. $FN$ ：分类器将正类预测为负类的数量(False Negative) ，即：假负类的数量。
3. $FP$ ：分类器将负类预测为正类的数量(False Positive)，即：假正类的数量。
4. $TN$ ：分类器将负类预测为负类的数量(True Negative) ，即：真负类的数量。

分类结果的混淆矩阵(confusion matrix)定义为：
|   |预测：正类|预测：反类|
| :---: |:---:  |:---: |
| 真实：正类  | $TP$ | $FN$  |
| 真实：反类  | $FP$ | $TN$  |

查准率(precision)：

$$
P=\frac{TP}{TP+FP}
$$

它刻画了所有预测为正类的结果中，真正正类的比例。

查全率(recall)： 

$$
R=\frac{TP}{TP+FN}
$$

它刻画了真正正类中，被分类器找出来的比例。

&emsp;&emsp;不同的问题中，有的侧重查准率，有的侧重查全率。
1. 对于推荐系统，更侧重于查准率。即推荐的结果中，用户真正感兴趣的比例。因为给用户展示的窗口有限，必须尽可能的给用户展示他真实感兴趣的结果。
2. 对于医学诊断系统，更侧重与查全率。即疾病被发现的比例。因为疾病如果被漏诊，则很可能导致病情恶化。

&emsp;&emsp;查准率和查全率是一对矛盾的度量。一般来说查准率高时查全率往往偏低，而查全率高时查准率往往偏低。
1. 如果希望将所有的正例都找出来（查全率高），最简单的就是将所有的样本都视为正类，此时有 $FN=0$ 。此时查准率就偏低（准确性降低）。
2. 如果希望查准率高，则可以只挑选有把握的正例。最简单的就是挑选最有把握的那一个样本。此时有 $FP=0$ 。此时查全率就偏低（只挑出了一个正例）。

### P-R 曲线
&emsp;&emsp;对二类分类问题，可以根据分类器的预测结果对样本进行排序：排在最前面的是分类器认为“最可能”是正类的样本，排在最后面的是分类器认为“最不可能”是正类的样本。

&emsp;&emsp;假设排序后的样本集合为 $(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)$ ，预测为正类的概率依次为 $ (p  _  1,p  _  2,\cdots,p  _  N)$ 。在第 $i$ 轮，将 $p  _  i$ 作为分类阈值来。即：

$$
\hat y_j=\begin{cases} 1 ,&\text{if }\;p_j\ge p_i\\ 0,&\text{else} \end{cases},\quad j=1,2,\cdots,N
$$

此时计算得到的查准率记做 $P  _  i$ ，查全率记做 $R  _  i$ 。

&emsp;&emsp;以查准率为纵轴、查全率为横轴作图，就得到查准率-查全率曲线，简称 $P-R$ 曲线。该曲线由点 $ {(R  _  1,P  _  1),(R  _  2,P  _  2),\cdots,(R  _  N,P  _  N)}$ 组成。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/evaluation/P_R.png?raw=true"
    width="480" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> PR </div>
</center>

&emsp;&emsp; $P-R$ 曲线从左上角 $(0,1)$  到右下角 $(1,0)$ 。开始时第一个样本（最可能为正例的）预测为正例，其它样本都预测为负类。此时：
1. 查准率很高，几乎为 1。
2. 查全率很低，几乎为 0，大量的正例没有找到。

结束时所有的样本都预测为正类。此时：
1. 查全率很高，正例全部找到了，查全率为 1。
2. 查准率很低，大量的负类被预测为正类。

&emsp;&emsp; $P-R$ 曲线直观显示出分类器在样本总体上的查全率、查准率。因此可以在同一个测试集上通过 $P-R$  曲线来比较两个分类器的预测能力：
1. 如果分类器 $B$ 的 $P-R$ 曲线被分类器 $A$ 的曲线完全包住，则可断言： $A$ 的性能好于 $B$  。
2. 如果分类器 $A$ 的 $P-R$ 曲线与分类器 $B$ 的曲线发生了交叉，则难以一般性的断言两者的优劣，只能在具体的查准率和查全率下进行比较。
   1. 此时一个合理的判定依据是比较 $P-R$ 曲线下面积大小，但这个值通常不容易计算。
   2. 可以考察平衡点。平衡点 (Break-Even Point:BEP) 是 $P-R$ 曲线上查准率等于查全率的点，可以判定：平衡点较远的 $P-R$ 曲线较好。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/evaluation/P _ R _ AB.jpeg?raw=true"
    width="560" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> P_R_AB </div>
</center>

### ROC曲线
&emsp;&emsp;定义真正例率(True Positive Rate) 为：

$$
TPR=\frac{TP}{TP+FN}
$$

它刻画了真正的正类中，被分类器找出来的比例。它也就等于正类的查全率。

&emsp;&emsp;定义假正例率(False Positive Rate) 为：

$$
FPR=\frac{FP}{TN+FP} 
$$

它刻画了模型将真实的负样本被预测为正类的概率。它就等于 1 减去负类的查全率。

&emsp;&emsp;对二类分类问题，可以根据分类器的预测结果对样本进行排序：排在最前面的是分类器认为“最可能”是正类的样本，排在最后面的是分类器认为“最不可能”是正类的样本。假设排序后的样本集合为 $(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)$ ，预测为正类的概率依次为 $ (p  _  1,p  _  2,\cdots,p  _  N)$ 。
在第 $i$ 轮，将 $p  _  i$ 作为分类阈值来。即：

$$
 \hat y_j=\begin{cases} 1 ,&\text{if }\;p_j\ge p_i\\ 0,&\text{else} \end{cases},\quad j=1,2,\cdots,N
$$

此时计算得到的真正例率记做 $TPR  _  i$ ，假正例率记做 $FPR  _  i$ 。

&emsp;&emsp;以真正例率为纵轴、假正例率为横轴作图，就得到 $ROC$ 曲线。该曲线由点 ${(TPR  _  1,FPR  _  1),(TPR  _  2,FPR  _  2),\cdots,(RPR  _  N,FPR  _  N)}$ 组成。

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/evaluation/ROC.png?raw=true"
    width="360" height=""
    >
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> ROC </div>
</center>

&emsp;&emsp; $ROC$ 曲线从左下角 $(0,0)$  到右上角 $(1,1)$ 。开始时第一个样本（最可能为正例的）预测为正例，其它样本都预测为负类。此时：
1. 真正例率很低，几乎为 $0$ ，因为大量的正例未预测到。
2. 假正例率很低，几乎为 $0$ ，因为此时预测为正类的样本很少，所以几乎没有错认的正例。

结束时所有的样本都预测为正类。此时：
1. 真正例率很高，几乎为 $1$ ，因为所有样本都预测为正类。
2. 假正例率很高，几乎为 $1$ ，因为所有的负样本都被错认为正类。

&emsp;&emsp;在 $ROC$ 曲线中，对角线对应于随机猜想模型。点 $(0,1)$ 对应于理想模型：没有预测错误， $FPR$ 恒等于 $0$ ， $TPR$ 恒等于 $1$ 。通常 $ROC$ 曲线越靠近点 $(0,1)$ 越好。

&emsp;&emsp;可以在同一个测试集上通过 $ROC$ 曲线来比较两个分类器预测能力：
1. 如果分类器 $A$ 的 $ROC$ 曲线被分类器 $B$ 的曲线完全包住，则可断言： $B$ 的性能好于 $A$  。
2. 如果分类器 $A$ 的 $ROC$ 曲线与分类器 $B$ 的曲线发生了交叉，则难以一般性的断言两者的优劣。此时一个合理的判定依据是比较 $ROC$ 曲线下的面积，这个面积称作 $AUC$ (Area Under ROC Curve)，其中 $AUC$ 的值越大，模型的性能越优。

> AUC的特性： $AUC$ 对样本类别是否均衡并不敏感，这也是不均衡样本通常用 $AUC$ 评价分类器性能的一个原因
   
> AUC的含义：从所有正样本中随机选取一个样本，从所有负样本中随机选取一个样本，然后使用训练好的分类器对两个随机选取的样本进行预测，把正样本预测为正类的概率为 $P  _  1$ ，把负样本预测为正类的概率为 $P  _  0$ ， ${ P }  _  { 1 }>{ P }  _  { 0 }$ 的概率就等于 $AUC$ 。
        

&emsp;&emsp; $P-R$ 曲线和 $ROC$ 曲线刻画的都是阈值的选择对于分类度量指标的影响。通常一个分类器对样本预测的结果是一个概率结果，比如正类概率  $0.7$ 。但是样本是不是正类还需要与阈值比较。这个阈值会影响了分类器的分类结果，比如：是阈值 $0.5$ 还是阈值 $0.9$ 。
1. 如果更重视查准率，则将阈值提升，比如为 $0.9$ 。
2. 如果更看重查全率，则将阈值下降，比如为 $0.5$ 。

&emsp;&emsp; $P-R$ 曲线和 $ROC$ 曲线上的每一个点都对应了一个阈值的选择，该点就是在该阈值下的(查准率，查全率)或(真正例率，假正例率)。沿着横轴的方向对应着阈值的下降。

### F1 值
&emsp;&emsp; $F1$ 度量，其定义为:

$$
{ F }_ { 1 }=\frac { 2\times P\times R }{ P+R } 
$$

其中， $F1$ 的值越大越好。此外，可以发现， $F1$ 是一个准确率和召回率的调和平均数，其调和平均数更关心较小的值。因此，如果 $P$ 和 $R$  中一个值太小会对 $F1$ 产生更大的影响，但是这样的判断都是以准确率和召回率同等重要为基础。

&emsp;&emsp;对于很多其它问题，我们更关心其中一个指标，如在逃犯信息检索系统中，更希望尽可能少的漏掉逃犯，此时查全率更重要一些，这就需要用到  ${ F }  _  { \beta  }$ 度量,其能够表达出对查准率或查全率的不同偏好，定义为

$$
{ F }_ { \beta  }=\frac { \left( 1+{ \beta  }^{ 2 } \right) \times P\times R }{ \left( { \beta  }^{ 2 }\times P \right) +R } 
$$

其中，  $\beta >0$  度量了查全率对查准率的相对重要性。
1.  $\beta =1$ 时退化为标准的 $F1$ ；
2.  $\beta >1$ 时查全率有更大的影响（此时 $R$ 更小，更小的值会对 $F1$ 产生更大的影响）；
3.  $\beta <1$ 时查准率有更大影响（此时 $P$ 更小，更小的值会对 $F1$ 产生更大的影响）。

### 代价矩阵
&emsp;&emsp;实际应用过程中，不同类型的错误所造成的后果可能有所不同。如：将健康人诊断为患者，与将患者诊断为健康人，其代价就不同。为权衡不同类型错误所造成的不同损失，可以为错误赋予非均等代价(unequal cost)。

&emsp;&emsp;对于二类分类问题，可以设定一个“代价矩阵”(cost matrix)，其中 $cost  _  {ij}$ 表示将第 $i$ 类样本预测为第 $j$ 类样本的代价。通常  $cost  _  {ii}=0$ 表示预测正确时的代价为 $0$ 。
|   |预测：第0类|预测：第1类|
| :---: |:---:  |:---: |
| 真实：第0类| $0$ | $cost  _  {01}$  |
| 真实：第1类| $cost  _  {10}$ | $0$ |

> 前面讨论的性能度量都隐式的假设均等代价，即 $cost  _  {01}=cost  _  {10}$ 

&emsp;&emsp;在非均等代价下，希望找到的不再是简单地最小化错误率的模型，而是希望找到最小化总体代价(total cost)的模型。非均等代价下， $ROC$ 曲线不能直接反映出分类器的期望总体代价，此时需要使用代价曲线(cost curve)。代价曲线的横轴是正例概率代价：

$$
P_{+cost}=\frac{p\times cost_{01}}{p\times cost_{01}+(1-p)\times cost_{10}}
$$

其中 $p$ 为正例（第 $0$ 类）的概率，代价曲线的纵轴为：

$$
cost_{norm}=\frac{FNR\times p \times cost_{01}+FPR\times(1-p)\times cost_{10}}{p\times cost_{01}+(1-p)\times cost_{10}}
$$

其中：
1. $FPR$ 为假正例率 $FPR=\frac{FP}{TN+FP}$ ，它刻画了模型将真实的负样本预测为正类的概率。
2. $FNR$ 为假负例率 $FNR=1-TPR=\frac{FN}{TP+FN}$ ，它刻画了模型将真实的正样本预测为负类的概率。

### 宏查准率/宏查全率、微查准率/微查全率
&emsp;&emsp;有时候可能得到了多个二分类混淆矩阵。如：在多个数据集上进行训练/测试。此时希望在多个二分类混淆矩阵上综合考察查准率和查全率。

&emsp;&emsp;假设有 $m$  个二分类混淆矩阵，有两种方法来综合考察：
1. 宏查准率、宏查全率：先在各个混淆矩阵上分别计算查准率和查全率，记作 $(P  _  1,R  _  1),(P  _  2,R  _  2),\cdots,(P  _  m,R  _  m)$ ；然后计算平均值。这样得到的是宏查准率(macro-P)，宏查全率(macro-R)，宏 $F1$ (macro-F1)：
$$
\begin{aligned} macro-P&=\frac { 1 }{ m } \sum _ { i=1 }^{ m }{ { P }_ { i } }  \\ macro-R&=\frac { 1 }{ m } \sum _ { i=1 }^{ m }{ { R }_ { i } }  \\ macro-{ F }_ { 1 }&=\frac { 2\times macro-P\times macro-R }{ macro-P+macro-R }  \end{aligned}
$$

2. 微查准率、微查全率：先将个混淆矩阵对应元素进行平均，得到 $TP,FP,TN,FN$ 的平均值，记作 $\overline {TP},\overline{FP},\overline{TN},\overline{FN}$ ；再基于这些平均值计算微查准率(micro-P)，微查全率(micro-R)，微 $F1$ (micro-F1)：

$$
\begin{aligned} micro-P&=\frac { \bar { TP }  }{ \bar { TP } +\bar { FP }  }  \\ micro-R&=\frac { \bar { TP }  }{ \bar { TP } +\bar { FN }  }  \\ micro-{ F }_ { 1 }&=\frac { 2\times micro-P\times micro-R }{ micro-P+micro-R }  \end{aligned}
$$

## 回归问题性能度量
平均绝对误差( $mean \ absolute \ error:MAE$ )：

$$
MAE=\frac{1}{N^{\prime}} \sum_{i=1}^{N^{\prime}}|\tilde y_i^{\prime}-\hat y_i^{\prime}| 
$$

均方误差( $mean \ square \ error:MSE$ )：

$$
MSE=\frac{1}{N^{\prime}} \sum_{i=1}^{N^{\prime}}(\tilde y_i^{\prime}-\hat y_i^{\prime})^2 
$$

均方根误差( $root \ mean \ squared \ error:RMSE$ )：

$$
RMSE=\sqrt{\frac{1}{N^{\prime}} \sum_{i=1}^{N^{\prime}}(\tilde y_i^{\prime}-\hat y_i^{\prime})^2}
$$

均方根对数误差( $root \ mean \ squared \ logarithmic \ error:RMSLE$ )：

$$
RMLSE=\sqrt{\frac{1}{N^{\prime}} \sum_{i=1}^{N^{\prime}}[\log(\tilde y_i^{\prime})-\log(\hat y_i^{\prime})]^2}
$$

为使得 $log$ 有意义，也可以使用：

$$
RMSLE=\sqrt{\frac{1}{N^{\prime}} \sum_{i=1}^{N^{\prime}}[\log(\tilde y_i^{\prime}+1)-\log(\hat y_i^{\prime}+1)]^2} 
$$

优势：
1. 当真实值的分布范围比较广时（如：年收入可以从 $0$ 到非常大的数），如果使用 $MAE$ 、 $MSE$ 、 $RMSE$ 等误差，这将使得模型更关注于那些真实标签值较大的样本。而  $RMSLE$ 关注的是预测误差的比例，使得真实标签值较小的样本也同等重要。
2. 当数据中存在标签较大的异常值时，  $RMSLE$ 能够降低这些异常值的影响。

