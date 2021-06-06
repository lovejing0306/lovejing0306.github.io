---
layout: post
title: git clone 一半失败
categories: [Git]
description: git clone 一半失败
keywords: Git
---

解决 git clone 到一半时提示失败的问题
---

## 解决方案 1
有时候使用

```
git clone https://github.com/xxx.git
```
从github上拉取项目时会提示下载失败。

可以尝试把`https://`换成 `git://`

```
git clone git://github.com/xxx.git
```

## 参考
[git clone失败的一种解决办法](https://blog.csdn.net/gufeiyunshi/article/details/51492078)
