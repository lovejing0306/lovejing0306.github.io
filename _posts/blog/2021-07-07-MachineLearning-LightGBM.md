---
layout: post
title: LightGBM
categories: [MachineLearning]
description: LightGBM
keywords: MachineLearning
---


LightGBM
---


$GBT$ 的缺点：在构建子决策树时为了获取分裂点，需要在所有特征上扫描所有的样本，从而获得最大的信息增益。
1. 当样本的数量很大，或者样本的特征很多时，效率非常低。
2. 同时 $GBT$ 也无法使用类似 $mini batch$ 方式进行训练。

$xgboost$ 缺点：
1. 每轮迭代都需要遍历整个数据集多次。
   1. 如果把整个训练集装载进内存，则限制了训练数据的大小。
   2. 如果不把整个训练集装载进内存，则反复读写训练数据会消耗非常大的 $IO$ 时间。
2. 空间消耗大。预排序（$pre-sorted$）需要保存数据的 $feature$ 值，还需要保存 $feature$ 排序的结果（如排序后的索引，为了后续的快速计算分割点）。因此需要消耗训练数据两倍的内存。
3. 时间消耗大。为了获取分裂点，需要在所有特征上扫描所有的样本，从而获得最大的信息增益，时间消耗大。
4. 对 $cache$ 优化不友好，造成 $cache miss$ 。
   1. 预排序后，$feature$ 对于梯度的访问是一种随机访问，并且不同 $feature$ 访问的顺序不同，无法对 $cache$ 进行优化。
   2. 在每一层的树生长时，需要随机访问一个行索引到叶子索引的数组，并且不同 $feature$ 访问的顺序也不同。

$LightGBM$ 的优点：
1. 更快的训练效率：在达到同样的准确率的情况下，可以达到比 $GBT$ 约 $20$ 倍的训练速度。
2. 低内存使用。
3. 更高的准确率。
4. 支持并行化学习。
5. 可处理大规模数据。


<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/LightGBM/good.jpg?raw=true"
    width="360" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> good </div>
</center>
    

## 原理
$LightGBM$ 的思想：若减少训练样本的数量，或者减少样本的训练特征数量，则可以大幅度提高训练速度。
    
$LightGBM$ 提出了两个策略：
1. $Gradient-based One-Side Sampling(GOSS)$： 基于梯度的采样。该方法用于减少训练样本的数量。
2. $Exclusive Feature Bundling(EFB)$： 基于互斥特征的特征捆绑。该方法用于减少样本的特征。

## GOSS

### 算法
减少样本的数量的难点在于：不知道哪些样本应该被保留，哪些样本被丢弃。
1. 传统方法：采用随机丢弃的策略。
2. $GOSS$ 方法：保留梯度较大的样本，梯度较小的样本则随机丢弃。

&emsp;&emsp;在 $AdaBoost$ 中每个样本都有一个权重，该权重指示了样本在接下来的训练过程中的重要性。在 $GBDT$ 中并没有这样的权重。如果能知道每个样本的重要性（即：权重），那么可以保留比较重要的样本，丢弃不那么重要的样本。由于 $GBDT$ 中，负的梯度作为当前的残差，接下来的训练就是拟合这个残差。因此 $GOSS$ 采用样本的梯度作为样本的权重：
1. 如果权重较小，则说明样本的梯度较小，说明该样本已经得到了很好的训练。因此对于权重较小的样本，则可以随机丢弃。
2. 如果权重较大，则说明样本的梯度较大，说明该样本未能充分训练。因此对于权重较大的样本，则需要保留。

&emsp;&emsp;$GOSS$ 丢弃了部分样本，因此它改变了训练样本的分布。这会影响到模型的预测准确性。为了解决这个问题，$GOSS$ 对小梯度的样本进行了修正：对每个保留下来的、小梯度的样本，其梯度乘以系数 $\frac{1-a}{b}$ （即放大一个倍数）。其中（假设样本总数为 $N$）：
1. $a$ 是一个 $0.0-1.0$ 之间的数，表示大梯度采样比。其意义为：保留梯度的绝对值在 $top$ $a\times N$ 的样本作为重要的样本。
2. $b$ 是一个 $0.0-1.0$ 之间的数，表示小梯度采样比。其意义为：从不重要的样本中随机保留 $b\times N$ 的样本。
3. $1-a$ 是一个 $0.0-1.0$ 之间的数，表示不重要的样本的比例。
4. $\frac{1-a}{b}$ 刻画了：从不重要的样本中，随机保留的样本的比例的倒数。
        
$GOSS$ 算法：
1. 输入：
   1. 训练集 $\mathbb D$，其样本数量为 $N$
   2. 大梯度采样比 $a $
   3. 小梯度采样比 $b$
   4. 当前的模型 $f(x)=\sum  _  {i=1}^{m-1}h  _  i(x)$
2. 输出：下一个子树 $h  _  m(x)$
3. 算法：
   1. 计算：
      1. 修正因子 $\text{factor}=\frac{1-a}{b}$
      2. 重要的样本数为 $\text{topN}=a\times N$
      3. 随机丢弃的样本数为 $\text{randN}=b \times N$
      4. 每个样本的损失函数的梯度 $\mathbf{\vec g}=(g  _  1,g  _  2,\cdots,g  _  N)$
   2. 根据梯度的绝对值大小，将样本按照从大到小排列。
      1. 取其中取 $\text{topN}$ 的样本作为重要性样本。
      2. 在 $\text{topN}$ 之外的样本中，随机选取 $\text{randN}$ 的样本作为保留样本，剩下的样本被丢弃。
   3. 构建新的训练集：重要性样本 + 随机保留的样本，其中：
      1. $\text{topN}$ 个重要性样本，每个样本的权重都为 $1$ 。
      2. $\text{randN}$ 个随机保留的样本，每个样本的权重都为 $\text{fractor}$。
   4. 根据新的训练集及其权重，训练决策树模型 $h  _  m(x)$ 来拟合残差（即：负梯度 $-\mathbf{\vec g}$）。返回训练好的 $h  _  m(x)$。

由于需要在所有的样本上计算梯度，因此丢弃样本的比例 ～ 加速比并不是线性的关系。
    

### 理论
&emsp;&emsp;在 $GBDT$ 生成新的子决策树 $h  _  m(x)$ 时，对于当前结点 $O$，考虑是否对它进行分裂。假设结点 $O$ 包含的样本集合为 $\mathbb O$， 样本维数为 $n$。对于第 $j$ 维，假设其拆分点为 $x  _  {i,j}=d$。

&emsp;&emsp;对于分类问题，其拆分增益为信息增益。它刻画的是划分之后混乱程度的降低，也就是纯净程度的提升：

$$
Gain_{j\mid O}(d)=p(O)H(y\mid O)- p(Left) H(y\mid Left)-p(Right)H(y\mid Right)
$$

其中：
1. $p(O)$ 表示样本属于结点 $O$ 的概率， $H(y\mid O)$ 为结点 $O$ 上的样本标记的条件熵。
2. $Left=\{x: x^j\le d\}$ 表示左子结点的样本集合；$Right=\{x: x^j\gt d\}$ 表示右子结点的样本集合。
3. $ p(Left) $ 表示样本属于结点 $O$ 的左子结点概率，$H(y\mid Left)$ 为左子结点的样本标记的条件熵。
4. $p(Right) $ 表示样本属于结点 $O$ 的右子结点概率，$H(y\mid Right)$ 为右子结点的样本标记的条件熵。

对于结点 $O$ 的任意拆分点，由于 $p(O)H(Y\mid O)$ 都相同，所以：

$$
\max_{j,d}(Gain_{j\mid O}(d)) \rightarrow \min_{j,d}p(Left) H(y\mid Left)+p(Right)H(y\mid Right)
$$

&emsp;&emsp;对于回归问题，其拆分增益为方差增益 ($variance gain:VG$)。它刻画的是划分之后方差的下降；也就是纯净程度的提升：

$$
Gain_{j\mid O}(d)=p(O)Var(y\mid O)-p(Left)Var(y\mid Left)-p(Right)Var(y\mid Right)
$$

其中：
1. $Var(y\mid O)$ 表示属于结点$O$的样本的标记的方差。
2. $Left=\{x: x^j\le d\}$ 表示左子结点的样本集合；$Right=\{x: x^j\gt d\}$ 表示右子结点的样本集合。
3. $Var(y\mid Left)$ 表示属于结点 $O$ 的左子结点的样本的标记的方差。
4. $Var(y\mid Right)$ 表示属于结点 $O$ 的右子结点的样本的标记的方差。

对于结点 $O$ 的任意拆分点，由于 $P(O)H(Y\mid O)$ 都相同，所以：

$$
\max_{j,d}(Gain_{j\mid O}(d)) \rightarrow \min_{j,d}p(Left) Var(y\mid Left)+p(Right)Var(y\mid Right)
$$


&emsp;&emsp;对于样本 $x  _  i$，设其标记为 $g  _  i$（它是残差，也是负梯度）。对于结点 $O$ 中的样本，设其样本数量为 $n  _  O$，样本的标记均值为 $\bar g=\frac{\sum  _  {i:x  _  i\in \mathbb O}g  _  i}{n  _  O}$，其方差为：

$$
Var(y\mid O)=\frac{\sum_{i:x_i\in \mathbb O}(g_i-\bar g)^2}{n_O} =\frac{\sum_{i:x_i\in \mathbb O}g_i^2-\frac{\left(\sum_{i:x_i\in \mathbb O}g_i\right)^2}{n_O}}{n_O}
$$

设总样本数量为 $N$， 则 $p(O)=\frac {n  _  O}{N}$，则有：

$$
 p(O)Var(y\mid O)=\frac{\sum_{i:x_i\in \mathbb O}g_i^2-\frac{\left(\sum_{i:x_i\in \mathbb O}g_i\right)^2}{n_O}}{N}
$$
    
&emsp;&emsp;现在考虑回归问题。对于拆分维度 $j$ 和拆分点 $d$， 令左子结点的样本下标为 $\mathbb L$，样本数量为 $n  _  {l\mid O}$ 右子结点的样本下标为 $\mathbb R$， 样本数量为 $n  _  {l\mid O}$。则方差增益：

$$
p(O)Var(y\mid O)-p(Left)Var(y\mid Left)-p(Right)Var(y\mid Right)= \frac{\sum_{i:x_i\in \mathbb O}g_i^2-\frac{\left(\sum_{i:x_i\in \mathbb O}g_i\right)^2}{n_O}}{N}-\frac{\sum_{i:x_i\in \mathbb L}g_i^2-\frac{\left(\sum_{i:x_i\in \mathbb L}g_i\right)^2}{n_{l\mid O}(d)}}{N}-\frac{\sum_{i:x_i\in \mathbb R}g_i^2-\frac{\left(\sum_{i:x_i\in \mathbb R}g_i\right)^2}{n_{r\mid O}(d)}}{N}
$$

考虑到 $\mathbb O=\mathbb L \bigcup \mathbb R$，因此有：

$$
\sum_{i:x_i\in \mathbb O}g_i^2=\sum_{i:x_i\in \mathbb L}g_i^2+\sum_{i:x_i\in \mathbb R}g_i^2
$$

因此则方差增益：

$$
p(O)Var(y\mid O)-p(Left)Var(y\mid Left)-p(Right)Var(y\mid Right) \\= \frac 1N\left[\frac{\left(\sum_{i:x_i\in \mathbb L}g_i\right)^2}{n_{l\mid O}(d)}+\frac{\left(\sum_{i:x_i\in \mathbb R}g_i\right)^2}{n_{r\mid O}(d)}-\frac{\left(\sum_{i:x_i\in \mathbb O}g_i\right)^2}{n_O}\right]
$$

考虑到总样本大小 $N$ 是个恒定值，因此可以去掉 $\frac 1N$。考虑到 $\frac{\left(\sum  _  {i:x  _  i\in \mathbb O}g  _  i\right)^2}{n  _  O}$ 并不随着结点 $O$ 的不同划分而变化因此定义：对于拆分维度 $j$ 和拆分点 $d$，方差增益为：

$$
V_{j\mid O}(d)=\left[\frac{\left(\sum_{i:x_i\in \mathbb L}g_i\right)^2}{n_{l\mid O}(d)}+\frac{\left(\sum_{i:x_i\in \mathbb R}g_i\right)^2}{n_{r\mid O}(d)}\right]
$$
    
&emsp;&emsp;考虑在 $GOSS$ 中，在划分结点 $O$ 的过程中，可能会随机丢弃一部分样本，从而 $O$ 的样本总数下降。因此重新定义方差增益：

$$
V_{j\mid O}(d)=\frac{1}{n_O}\left[\frac{\left(\sum_{i:x_i\in \mathbb L}g_i\right)^2}{n_{l\mid O}(d)}+\frac{\left(\sum_{i:x_i\in \mathbb R}g_i\right)^2}{n_{r\mid O}(d)}\right]
$$

在 $GOSS$ 中：
1. 首先根据样本的梯度的绝对值大小降序排列。
2. 然后选取其中的 $top a$ 的样本作为重要样本，设其集合为 $\mathbb A$。则剩下的样本集合 $\mathbb A^c$ 保留了 $1-a$ 比例的样本。在剩下的样本集合 $\mathbb A^c$ 中，随机选取总样本的 $b$ 比例的样本保留，设其集合为 $\mathbb B$。
3. 最后将样本 $\mathbb A\bigcup \mathbb B$ 划分到子结点中。

重新定义方差增益为：

$$
\tilde V_{j\mid O}(d)=\frac{1}{\tilde n_O}\left[\frac{\left(\sum_{i:x_i\in \mathbb A_l}g_i+\frac{1-a}{b}\sum_{i:x_i\in \mathbb B_l}g_i\right)^2}{\tilde n_{l\mid O}(d)}+\frac{\left(\sum_{i:x_i\in \mathbb A_r}g_i+\frac{1-a}{b}\sum_{i:x_i\in \mathbb B_r}g_i\right)^2}{\tilde n_{r\mid O}(d)}\right]
$$

其中：
1. $\tilde n  _  O$ 表示所有保留的样本的数量。$\tilde n  _  {l\mid O}(d),\tilde n  _  {r\mid O}(d)$ 分别表示左子结点、右子结点保留的样本的数量。
2. $\mathbb A  _  l,\mathbb A  _  r$ 分别表示左子结点、右子结点的被保留的重要样本的集合。
3. $\mathbb B  _  l,\mathbb B  _  r$ 分别表示左子结点、右子结点的被保留的不重要样本的集合。
4. $\frac{1-a}{b}$ 用于补偿由于对 $\mathbb A^c $ 的采样带来的梯度之和的偏离。

由于 $\mathbb B$ 的大小可能远远小于 $\mathbb A^c$，因此估计 $\tilde V  _  {j\mid O}(d)$ 需要的计算量可能远远小于估计 $V  _  {j\mid O}(d)$。

&emsp;&emsp;定义近似误差为：

$$
\varepsilon (d)=|\tilde V_{j\mid O}(d)-V_{j\mid O}(d)|
$$

定义标准的梯度均值：

$$
\bar g_l(d)=\frac{ \sum_{i:x_i\in \mathbb L}g_i }{n_{l\mid O}(d)} ,\quad \bar g_r(d)= \frac{ \sum_{i:x_i\in \mathbb R}g_i }{n_{r\mid O}(d)}
$$

则可以证明：至少以概率 $1-\delta$ 满足：

$$
\varepsilon(d) \le C_{a,b}^2 \ln \frac 1\delta \times\max \left\{\frac{1}{n_{l\mid O}^j(d)},\frac{1}{n_{r\mid O}^j(d)}\right\}+2D\times C_{a,b}\sqrt{\frac{\ln 1/\delta}{n}}  
$$

其中：
1.  $C_{a,b}=\frac{1-a}{\sqrt b}\max  _  {i:x  _  i}\in \mathbb A^{c} \|g  _  i \|$，刻画的是剩余样本集合 $\mathbb A^c$ 中最大梯度的加权值。
2. $D=\max(\bar g  _  l(d),\bar g  _  r(d))$， 刻画的是未采取 $GOSS$ 时，划分的左子结点的梯度均值、右子结点的梯度均值中，较大的那个。

结论：
1. 当划分比较均衡（即：$n  _  {l\mid O}(d)\ge O(\sqrt n), \quad n  _  {r\mid O}(d)\ge O(\sqrt n)$） 时，近似误差由不等式的第二项决定。此时，随着样本数量的增长，使用 $GOSS$  和原始的算法的误差逼近于 $0$。
2. 当 $a=0$ 时，$GOSS$ 退化为随机采样。

&emsp;&emsp;$GOSS$ 的采样增加了基学习器的多样性，有助于提升集成模型的泛化能力。
    

## EFB
&emsp;&emsp;减少样本特征的传统方法是：使用特征筛选。该方式通常是通过 $PCA$ 来实现的，其中使用了一个关键的假设：不同的特征可能包含了重复的信息。这个假设很有可能在实践中无法满足。

&emsp;&emsp;$LightBGM$ 的思路是：很多特征都是互斥的，即：这些特征不会同时取得非零的值。如果能将这些互斥的特征捆绑打包成一个特征，那么可以将特征数量大幅度降低。

现在有两个问题：
1. 如何找到互斥的特征？
2. 如何将互斥的特征捆绑成一个特征？

### 互斥特征发现
&emsp;&emsp;定义打包特征集为这样的特征的集合：集合中的特征两两互斥。给定数据集 $\mathbb D = \{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\}$，其中样本 $x  _  i = (x  _  {i,1},x  _  {i,2},\cdots,x  _  {i,n})^T$。
如果对每个 $i=1,2,\cdots,N$，都不会出现 $x  _  {i,j}\ne 0\; \text{and} \;x  _  {i,k}\ne0$，则特征 $j$ 和特征 $k $ 互斥。

&emsp;&emsp;可以证明：将每个特征划分到每个打包特征集中使得打包特征集的数量最小，这个问题是 $NP$ 难的。为了解决这个问题，$LightGBM$ 采用了一个贪心算法来求解一个近似的最优解。

&emsp;&emsp;将每个特征视为图中的一个顶点。
遍历每个样本 $x  _  i\in \mathbb D$， 如果特征 $j,k$ 之间不互斥（即 $x  _  {i,j}\ne0\; \text{and}\; x  _  {i,k}\ne0$ ），则：
1. 如果顶点 $j,k$ 之间不存在边，则在顶点 $j,k$ 之间连接一条边，权重为 $1$。
2. 如果顶点 $j,k$ 之间存在边，则顶点 $j,k$ 之间的边的权重加 $1$。
最终，如果一组顶点之间都不存在边，则它们是相互互斥的，则可以放入到同一个打包特征集中。

&emsp;&emsp;事实上有些特征之间并不是完全互斥的，而是存在非常少量的冲突。即：存在少量的样本，在这些样本上，这些特征之间同时取得非零的值。如果允许这种少量的冲突，则可以将更多的特征放入打包特征集中，这样就可以减少更多的特征。

&emsp;&emsp;理论上可以证明：如果随机污染小部分的样本的特征的值，则对于训练 $ accuracy $ 的影响是：最多影响 $ O([(1-\gamma)N]^{-2/3})$。其中 $\gamma$ 为污染样本的比例，$ N $ 为样本数量 。

互斥特征发现算法：
1. 输入：
   1. 数据集 $\mathbb D = \{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\}$，其中样本 $ x  _  i = (x  _  {i,1},x  _  {i,2},\cdots,x  _  {i,n})^T$。
   2. 冲突阈值 $K$。
2. 输出：打包特征集的集合 $\mathbb B$
3. 算法：
   1. 构建图 $\mathcal G$：
      1. 每个特征作为一个顶点。
      2. 遍历每个样本 $x  _  i\in \mathbb D$：
         1. 遍历所有的特征对 $(j,k)$，如果特征 $j,k$ 之间不互斥 （即 $x  _  {i,j}\ne0\; \text{and}\; x  _  {i,k}\ne0$）则：
            1. 如果顶点 $j,k$ 之间不存在边，则在顶点 $j,k$ 之间连接一条边，权重为 $1$。
            2. 如果顶点 $j,k$ 之间存在边，则顶点 $j,k$ 之间的边的权重加 $1$。
   2. 对每个顶点，根据 $degree$ （与顶点相连的边的数量）来降序排列。
   3. 初始化：$\mathbb B=\phi$
   4. 根据顶点的排序遍历顶点，设当前顶点为 $j$：
      1. 遍历打包特征集 $B\in \mathbb B$，计算顶点 $j$ 与打包特征集 $B$ 的冲突值 $cnt$。如果 $cnt \le K$， 则说明顶点 $j$ 与打包特征集 $B$ 不冲突。此时将顶点 $j$ 添加到打包特征集 $B$ 中，退出循环并考虑下一个顶点。
         > 顶点 $j$ 与 $bundle$ 特征集 $B$ 的冲突值有两种计算方法：
         >  1. 计算最大冲突值：即最大的边的权重：$cnt=\max  _  {k\in B} \text{weight}(j,k)$
         >  2. 计算所有的冲突值：即所有的边的权重：$cnt=\sum  _  {k\in B} \text{weight}(j,k)$
                
      2. 如果顶点 $j$ 未加入到任何一个打包特征集中 ，则：创建一个新的打包特征集加入到 $\mathbb B$ 中，并将顶点 $j$ 添加到这个新的打包特征集中。
                
   5. 返回 $\mathbb B$

互斥特征发现算法的算法复杂度为：$O(N\times n^2)$，其中 $N$ 为样本总数，$n$ 为样本维数。
1. 复杂度主要集中在构建图 $\mathcal G$。
2. 该算法只需要在训练之前执行一次。
3. 当特征数量较小时，该算法的复杂度还可以接受。当特征数量非常庞大时，该算法的复杂度太高。
4. 优化的互斥特征发现算法不再构建图 $\mathcal G$ ，而是仅仅计算每个特征的非零值。

优化的互斥特征发现算法：
1. 输入：
   1. 数据集 $\mathbb D = \{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\}$，其中样本 $ x  _  i = (x  _  {i,1},x  _  {i,2},\cdots,x  _  {i,n})^T$。
   2. 冲突阈值 $K$。
2. 输出：打包特征集的集合 $\mathbb B$
3. 算法：
   1. 初始化：所有特征的非零值数量组成的数组 $\text{NZ}=(0,0,\cdots,0)$
   2. 计算每个特征的非零值 (复杂度 $O(N\times n))$：遍历所有的特征 $j$、遍历所有所有的样本 $x  _  i \in \mathbb D$，获取特征 $j$ 的非零值 $\text{NZ}  _  j$。
   3. 根据 $\text{NZ}$ 对顶点降序排列。
   4. 初始化：$\mathbb B=\phi$
   5. 根据顶点的排序遍历顶点，设当前顶点为 $j$：
      1. 遍历打包特征集 $B\in \mathbb B$，计算顶点 $j$ 与打包特征集 $B$ 的冲突值 $cnt$。如果 $cnt \le K$， 则说明顶点 $j$ 与打包特征集 $B$ 不冲突。此时将顶点 $j$ 添加到 打包特征集 $B$ 中，退出循环并考虑下一个顶点。
         > 顶点 $j$ 与 $bundle$ 特征集 $B$ 的冲突值有两种计算方法：
         > 1. 计算最大冲突值：即最大的非零值：$cnt=\text{NZ}  _  j+\max  _  {k\in B}\text{NZ}  _  k$
         > 2. 计算所有的冲突值：即所有的非零值：$cnt=\text{NZ}  _  j+\sum  _  {k\in B} \text{NZ}  _  k$
         > 
         > 这里简单的将两个特征的非零值之和认为是它们的冲突值。它是实际的冲突值的上界。
                
      2. 如果顶点 $j$ 未加入到任何一个打包特征集中 ，则：创建一个新的打包特征集加入到 $\mathbb B$ 中，并将顶点 $j$ 添加到这个新的打包特征集中。
   6. 返回 $\mathbb B$
            

### 互斥特征打包
&emsp;&emsp;互斥特征打包的思想：可以从打包的特征中分离出原始的特征。假设特征 $a$ 的取值范围为 $[0,10)$， 特征 $b$ 的取值范围为 $[0,20)$ 。如果 $a,b$ 是互斥特征，那么打包的时候：对于特征 $b$ 的值，给它一个偏移量，比如 $20$。最终打包特征的取值范围为：$[0,40)$。
1. 如果打包特征的取值在 $[0,10)$， 说明该值来自于特征 $a$；
2. 如果打包特征的取值在 $[20,40)$，说明该值来自于特征 $b$；

&emsp;&emsp;基于 $histogram$ 的算法需要考虑分桶，但是原理也是类似：将 $[0,x]$ 之间的桶分给特征 $a$， 将 $[x+offset,y]$ 之间的桶分给特征 $b$。 其中 $offset > 0$ 。

互斥特征打包算法：
1. 输入：
   1. 数据集 $\mathbb D = \{(x  _  1,\tilde y  _  1),(x  _  2,\tilde y  _  2),\cdots,(x  _  N,\tilde y  _  N)\}$，其中样本 $x  _  i = (x  _  {i,1},x  _  {i,2},\cdots,x  _  {i,n})^T$。
   2. 待打包的特征集合 $B$。
2. 输出：打包之后的分桶
3. 算法：
   1. 令 $\text{totalBin}$ 记录总的分桶数量，$\text{binRanges}$ 记录不同的特征的边界。初始化：$\text{totalBin}=0,\text{binRanges}=\phi$。
   2. 计算特征边界：遍历所有的特征 $j \in B$：
      1. 获取特征 $j$ 的分桶数量 $\text{num}(j)$，增加到 $\text{totalBin}:\text{totalBin}+=\text{num}(j)$
      2. 获取特征 $j$ 的分桶边界：$\text{binRanges}.append(\text{totalBin})$
   3. 创建新特征，它有 $\text{totalBin}$ 个桶。
   4. 计算分桶点：遍历每个样本 $x  _  i\in \mathbb D$：
      1. 计算每个特征 $j\in B$：
         1. 如果 $x  _  {i,j}\ne 0$，则：如果 $x  _  {i,j}$ 在特征 $j$ 的第 $k$ 个分桶中， 那么在打包后的特征中，它位于桶 $\text{binRanges}[j]+k$ 中。
         2. 如果 $x  _  {i,j}=0$，则不考虑。

&emsp;&emsp;互斥特征打包算法的算法复杂度为 $O(N\times n)$，其中 $N$ 为样本总数 $n$ 为样本维数。也可以首先扫描所有的样本，然后建立一张扫描表，该表中存放所有样本所有特征的非零值。这样互斥特征打包算法在每个特征上仅仅需要扫描非零的样本即可。这样每个特征的扫描时间从 $O(N)$ 降低为 $O(N  _  {nz})$， 其中 $N  _  {nz}$ 为该特征上非零的样本数。该方法的缺陷是：消耗更多的内存，因为需要在整个训练期间保留这样的一张表。
    

## 优化
$LightGBM$ 优化思路：
1. 单个机器在不牺牲速度的情况下，尽可能多地用上更多的数据。
2. 多机并行时通信的代价尽可能地低，并且在计算上可以做到线性加速。

$LightGBM$ 的优化：
1. 基于 $histogram$ 的决策树算法。
2. 带深度限制的 $leaf-wise$ 的叶子生长策略。
3. 直方图做差加速。
4. 直接支持类别（$categorical$） 特征。
5. 并行优化。

### histogram 算法
基本思想：先把连续的浮点特征值离散化成 $k$ 个整数，同时构造一个宽度为 $k$ 的直方图。在遍历数据时：
1. 根据离散化后的值作为索引在直方图中累积统计量。
2. 当遍历一次数据后，直方图累积了需要的统计量。
3. 然后根据直方图的离散值，遍历寻找最优的分割点。

优点：节省空间。假设有 $N$ 个样本，每个样本有 $n$ 个特征，每个特征的值都是 $32$ 位浮点数。
1. 对于每一列特征，都需要一个额外的排好序的索引（$32$ 位的存储空间）。则 $pre-sorted$ 算法需要消耗 $ 2\times N\times n\times4$ 字节内存。
2. 如果基于 $histogram$ 算法，仅需要存储 $feature bin value$（离散化后的数值），不需要原始的 $feature value$，也不用排序。而 $bin value$ 用 $unit8  _  t$ 即可，因此 $ histogram $ 算法消耗 $N\times n\times 1$ 字节内存，是预排序算法的 $\frac 18$。

缺点：不能找到很精确的分割点，训练误差没有 $pre-sorted$ 好。但从实验结果来看，$ histogram $ 算法在测试集的误差和 $pre-sorted$ 算法差异并不是很大，甚至有时候效果更好。实际上可能决策树对于分割点的精确程度并不太敏感，而且较“粗”的分割点也自带正则化的效果。
    
采用 $ histogram $ 算法之后，寻找拆分点的算法复杂度为：
1. 构建 $ histogram $ ：$O(N\times n)$。
2. 寻找拆分点：$O(N\times k)$，其中 $k$ 为分桶的数量。

与其他算法相比：
1. $scikit-learn GBDT$、$Gbm In R$ 使用的是基于 $pre-sorted$ 的算法。
2. $pGBRT$ 使用的是基于 $histogram$ 的算法。
3. $xgboost$ 既提供了基于 $pre-sorted$ 的算法，又提供了基于 $histogram$ 的算法。
4. $lightgbm$ 使用的是基于 $histogram$ 的算法。

### leaf-wise 生长策略
大部分梯度提升树算法采用 $ level-wise $ 的叶子生长策略：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/LightGBM/leaf_wise.jpg?raw=true"
    width="560" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Level Wise </div>
</center>


而 $ lightgbm $ 采用 $ leaf-wise $ 的叶子生长策略：

<center>
    <img 
    src="https://github.com/lovejing0306/Images/blob/master/MachineLearning/models/LightGBM/level_wise.jpg?raw=true"
    width="560" height="" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Leaf Wise </div>
</center>

    
$level-wise$ ：
1. 优点：过一遍数据可以同时分裂同一层的叶子，容易进行多线程优化，也好控制模型复杂度，不容易过拟合。
2. 缺点：实际上 $level-wise$ 是一种低效算法 。它不加区分的对待同一层的叶子，带来了很多没必要的开销：实际上很多叶子的分裂增益较低，没必要进行搜索和分裂。

$leaf-wise$ 是一种更为高效的策略。每次从当前所有叶子中，找到分裂增益最大的一个叶子来分裂。
1. 优点：同 $level-wise$ 相比，在分裂次数相同的情况下，$leaf-wise$ 可以降低更多的误差，得到更好的精度。
2. 缺点：可能会长出比较深的决策树，产生过拟合。

因此 $ lightgbm $ 在 $leaf-wise$ 之上增加了一个最大深度限制，在保证高效率的同时防止过拟合。
        

### 直方图做差加速
通常构造直方图，需要遍历该叶子上的所有数据。但是事实上一个叶子的直方图可以由它的父亲结点的直方图与它兄弟的直方图做差得到。

$LightGBM$ 在构造一个叶子的直方图后，可以用非常微小的代价得到它兄弟叶子的直方图，在速度上可以提升一倍。
    

### 直接支持 categorical 特征
通常对 #categorical$ 特征进行 $one-hot$ 编码，但是这个做法在决策树学习中并不好：对于取值集合较多的 $categorical feature$，学习到的树模型会非常不平衡；树的深度需要很深才能达到较高的准确率。

$LightGBM$ 直接支持 $categorical$ 特征。
    

### 并行优化

#### 特征并行
传统的特征并行算法主要体现在决策树中的最优拆分过程中的并行化处理：
1. 沿特征维度垂直划分数据集，使得不同机器具有不同的特征集合。
2. 在本地数据集中寻找最佳划分点：(划分特征，划分阈值)。
3. 将所有机器上的最佳划分点整合，得到全局的最佳划分点。
4. 利用全局最佳划分点对数据集进行划分，完成本次最优拆分过程。

$LightGBM$ 在特征并行上进行了优化，流程如下：
1. 每个机器都有全部样本的全部特征集合。
2. 每个机器在本地数据集中寻找最佳划分点：(划分特征，划分阈值)。但是不同的机器在不同的特征集上运行。
3. 将所有机器上的最佳划分点整合，得到全局的最佳划分点。
4. 利用全局最佳划分点对数据集进行划分，完成本次最优拆分过程。
        
$LightGBM$ 不再沿特征维度垂直划分数据集，而是每个机器都有全部样本的全部特征集合。这样就节省了数据划分的通信开销。
1. 传统的特征并行算法需要在每次最优拆分中，对数据划分并分配到每台机器上。
2. $LightGBM$ 特征并行算法只需要在程序开始时，将全量样本拷贝到每个机器上。

二者交换的数据相差不大，但是后者花费的时间更少。
    
$LightGBM$ 的特征并行算法在数据量很大时，仍然存在计算上的局限。因此建议在数据量很大时采用数据并行。
    

#### 数据并行
传统的数据并行算法主要体现在决策树的学习过程中的并行化处理：
1. 水平划分数据集，使得不同机器具有不同的样本集合。
2. 以本地数据集构建本地直方图
3. 将本地直方图整合为全局直方图
4. 在全局直方图中寻找最佳划分点。

$LightGBM$ 在数据并行上进行了优化，流程如下：
1. $LightGBM$ 使用 $ Reduce scatter $ 的方式对不同机器上的不同特征进行整合。每个机器从本地整合直方图中寻找最佳划分点，并同步到全局最佳划分点中。
2. $LightGBM$ 通过直方图做差分加速。