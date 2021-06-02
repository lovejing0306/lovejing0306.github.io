---
layout: page
title: Archives
description: TimeLine
keywords: 归档, Archive
comments: false
menu: 归档
permalink: /archives/
---

<!--
<h2>Archives</h2>
-->

<style>
.times {display:block;margin:15px 0;}/*首先，我们要创建一个容器class*/
.times ul {margin-left:25px;border-left:2px solid #ddd;list-style-type:none;}/*利用ul标签的特性，设置外边框左移25px，左边边框是2px粗的实心线，颜色一般要浅一点*/
.times ul li {width:100%;margin-left:-12px;line-height:20px;font-weight:narmal;}/*一般情况，通过li标签控制圆点回到时间线上，然后控制要出现的文字大小和是否粗体*/
.times ul li b {width:8px;height:8px;background:#fff;border:2px solid #555;margin:5px;border-radius:6px;-webkit-border-radius:6px;-moz-border-radius:6px;overflow:hidden;display:inline-block;float:left;}/*利用处理加粗以外没有其它特别属性b标签做时间轴的圆点。*/
.times ul li span {padding-left:7px;font-size:12px;line-height:20px;color:#555;}/*设置span标签的属性，让它来做时间显示，加一点边距，使时间显示离时间线远一点*/
.times ul li:hover b {border:2px solid #ff6600;}/*注意这一行，前面的li标签后面加了一个:hover伪属性，意思是鼠标移上来，激活后面的属性，这样可以设置鼠标移动到整个时间范围的时候，时间点和时间显示会变色*/
.times ul li:hover span {color:#ff6600;}/*同上*/
.times ul li p {padding-left:15px;font-size:14px;line-height:25px;}/*这里利用段落标签p做文字介绍*/
</style>

<div id="archives" class="times">
{% for post in site.posts %}
  {% capture this_year %}{{ post.date | date: "%Y" }}{% endcapture %}
  {% capture pre_year %}{{ post.previous.date | date: "%Y" }}{% endcapture %}
  {% if forloop.first %}
    {% assign last_day = "" %}
    {% assign last_month = "" %}
  <h3>{{this_year}}</h3>
  <ul>
  {% endif %}
    <li>
        <b></b>
        <span>{{ post.date | date:"%m-%d" }}</span>
        <span><a href="{{ post.url | relative_url }}">{{ post.title }}</a></span>
    </li>
  {% if forloop.last %}
  </ul>
  {% elsif this_year != pre_year %}
  </ul>

  <h3>{{pre_year}}</h3>
  <ul>
    {% assign last_day = "" %}
    {% assign last_month = "" %}
  {% endif %}
{% endfor %}
</div>
