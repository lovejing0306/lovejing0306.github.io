---
layout: post
title: HBase 调优
categories: [HBase]
description: HBase 调优
keywords: HBase
---

HBase 调优

---

#### 前言

HBase 配置参数极其繁多，参数配置可能会影响到 HBase 性能问题，因此得好好总结下。

HBase 调优是个技术活。得结合多年生产经验加测试环境下性能测试得出。

1. JVM垃圾回收优化

2. 本地 memstore 分配缓存优化

3. Region 拆分优化

4. Region 合并优化

5. Region 预先加载优化

6. 负载均衡优化

7. 启用压缩，推荐snappy

8. 进行预分区，从而避免自动 split，提高 HBase 响应速度

9. 避免出现 region 热点现象，启动按照 table 级别进行 balance

#### Compaction 参数

Compaction 的主要目的

- 将多个HFile 合并为较大HFile，从而提高查询性能
- 减少HFile 数量，减少小文件对 HDFS 影响
- 提高 Region 初始化速度

**hbase.hstore.compaction.min**

当某个列族下的 HFile 文件数量超过这个值，则会触发 minor compaction 操作 默认是3，比较小，建议设置 10-15   
设置过小会导致合并文件太频繁，特别是频繁 bulkload 或者数据量比较大的情况下 设置过大又会导致一个列族下面的 HFile 数量比较多，影响查询效率

**hbase.hstore.compaction.max**

一次最多可以合并多少个HFile，默认为 10 限制某个列族下面选择最多可选择多少个文件来进行合并   
注意需要满足条件`hbase.hstore.compaction.max` > `hbase.hstore.compaction.min`

**hbase.hstore.compaction.max.size**

默认 Long 最大值，minor_compact 时 HFile 大小超过这个值则不会被选中合并   
用来限制防止过大的 HFile 被选中合并，减少写放大以及提高合并速度

**hbase.hstore.compaction.min.size**

默认 memstore 大小，minor_compact 时 HFile 小于这个值，则一定会被选中   
可用来优化尽量多的选择合并小的文件

**hbase.regionserver.thread.compaction.small**

默认1，每个RS的 minor compaction线程数，其实不是很准确，这个线程主要是看参与合并的 HFile 数据量 有可能 minor compaction 数据量较大会使用 compaction.large 提高线程可提高 HFile 合并效率

**hbase.regionserver.thread.compaction.large**

默认1，每个RS的 major compaction线程数，其实不是很准确，这个线程主要是看参与合并的 HFile 数据量 有可能 minor compaction 数据量较大会使用 compaction.large 提高线程可提高 HFile 合并效率

**hbase.hregion.majorcompaction**

默认604800000, 单位是毫秒，即7天。major compaction 间隔

设置为0即关闭hbase major compaction，改为业务低谷手动执行。[HBase 自动大合并脚本](https://lihuimintu.github.io/2019/06/09/HBase-timing-major-compaction/)

**hbase.hregion.majorcompaction.jetter**

默认值为 0.5 对参数hbase.hregion.majoucompaction 规定的值起到浮动的作用   
防止region server 在同一时间进行major compaction

---
参考链接
* [HBase调优 HBase Compaction参数调优](https://mp.weixin.qq.com/s/uMDoSnsbcqznCvSQCW5_yA)





