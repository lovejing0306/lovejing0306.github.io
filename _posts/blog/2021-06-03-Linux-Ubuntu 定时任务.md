---
layout: post
title: Cron 定时任务
categories: [Linux]
description: Cron 定时任务
keywords: Linux
---


Cron 定时任务
---


## 安装

```
apt-get install cron
```

## 查看状态
```
sudo service cron start     #启动服务
sudo service cron stop      #关闭服务
sudo service cron restart   #重启服务
sudo service cron reload    #重新载入配置
sudo service cron status    #查看服务状态
```

## 查看当前用户定时任务

```
crontab -l
```

## 添加定时任务

```
crontab -e
# 每天早上6点 
0 6 * * * echo "Good morning." >> /tmp/test.txt

service cron restart # 重启服务使新添加的任务生效
```

## 参考

[Linux之crontab定时任务](https://www.jianshu.com/p/838db0269fd0)

[ubuntu定时任务](https://juejin.cn/post/6844904006666436615)

[ubuntu centos Crontab 定时任务](https://blog.csdn.net/HybridTheory_/article/details/104901806)

[Ubuntu 使用 Cron 实现计划任务](https://zhuanlan.zhihu.com/p/350671948)