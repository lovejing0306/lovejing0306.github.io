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
    .timeline-small {
        max-width: 350px;
        max-height: 630px;
        overflow: hidden;
        margin: 30px auto 0;
        box-shadow: 0 0 40px #a0a0a0;
        font-family: 'Open Sans', sans-serif;
    }
    .timeline-small-body ul {
        padding: 1em 0 0 2em;
        margin: 0;
        list-style: none;
        position: relative;
    }
    .timeline-small-body ul::before {
        content: ' ';
        height: 100%;
        width: 5px;
        background-color: #d9d9d9;
        position: absolute;
        top: 0;
        left: 2.4em;
        z-index: -1;
    }
    .timeline-small-body li div {
        display: inline-block;
        margin: 1em 0;
        vertical-align: top;
    }
    .timeline-small-body .bullet {
        width: 1rem;
        height: 1rem;
        box-sizing: border-box;
        border-radius: 50%;
        background: #fff;
        z-index: 1;
        margin-right: 1rem;
        margin-top: 7%;
    }
    .timeline-small-body .bullet.pink {
        background-color: hotpink;
        border: 3px solid #F93B69;
    }
    .timeline-small-body .bullet.green {
        background-color: lightseagreen;
        border: 3px solid #B0E8E2;
    }
    .timeline-small-body .bullet.blue {
        background-color: aquamarine;
        border: 3px solid cadetblue;
    }
    .timeline-small-body .bullet.orange {
        background-color: salmon;
        border: 3px solid #EB8B6E;
    }
    .timeline-small-body .date {
        width: 23%;
        font-size: 0.75em;
        padding-top: 0.40rem;
        padding-right: 2rem;
    }
    .timeline-small-body .desc {
        width: 50%;
    }
    .timeline-small-body h3 {
        font-size: 0.9em;
        font-weight: 400;
        margin: 0;
    }
    .timeline-small-body h4 {
        margin: 0;
        font-size: 0.7em;
        font-weight: 400;
        color: #808080;
    }
    .lead {
        font-size: 1.5rem;
        position: relative;
        left: 8px;
    
        /* archives */
        --timeline-node-bg: rgb(150, 152, 156);
        --timeline-color: rgb(63, 65, 68);
        --timeline-year-dot-color: var(--timeline-color);

        &::after { /* Year dot */
          content: "";
          display: block;
          position: relative;
          -webkit-border-radius: 50%;
          -moz-border-radius: 50%;
          border-radius: 50%;
          width: 12px;
          height: 12px;
          top: -26px;
          left: 63px;
          border: 3px solid;
          background-color: var(--timeline-year-dot-color);
          border-color: var(--timeline-node-bg);
          box-shadow: 0 0 2px 0 #c2c6cc;
          z-index: 1;
        }
    }
</style>

<div id="archives" class="timeline-small-body">
{% for post in site.posts %}
  {% capture this_year %}{{ post.date | date: "%Y" }}{% endcapture %}
  {% capture pre_year %}{{ post.previous.date | date: "%Y" }}{% endcapture %}
  {% if forloop.first %}
    {% assign last_day = "" %}
    {% assign last_month = "" %}

  <span class="lead">{{this_year}}</span>

  <ul>
  {% endif %}
    <li>
        <div class="bullet pink"></div>
        <div class="date">XXXX年XX月XX日</div>
        <!--
 {{ post.date | date:"%m-%d" }}
        <span><a href="{{ post.url | relative_url }}">{{ post.title }}</a></span>
        -->
    </li>
  {% if forloop.last %}
  </ul>
  {% elsif this_year != pre_year %}
  </ul>

  <span class="lead">{{pre_year}}</span>
  <ul>
    {% assign last_day = "" %}
    {% assign last_month = "" %}
  {% endif %}
{% endfor %}
</div>


<div class="timeline-small">
    <div class="timeline-small-body">
        <ul>
            <li>
                <div class="bullet pink"></div>
                <div class="date">XXXX年XX月XX日</div>
                <div class="desc">
                    <h3>内容段落1</h3>
                    <h4>内容段落2内容段落2内容段落2内容段落2</h4>
                </div>
            </li>
            <li>
                <div class="bullet orange"></div>
                <div class="date">XXXX年XX月XX日</div>
                <div class="desc">
                    <h3>内容段落1</h3>
                    <h4>内容段落2内容段落2内容段落2内容段落2</h4>
                </div>
            </li>
            <li>
                <div class="bullet blue"></div>
                <div class="date">XXXX年XX月XX日</div>
                <div class="desc">
                    <h3>内容段落1</h3>
                    <h4>内容段落2内容段落2内容段落2内容段落2</h4>
                </div>
            </li>
            <li>
                <div class="bullet green"></div>
                <div class="date">XXXX年XX月XX日</div>
                <div class="desc">
                    <h3>内容段落1</h3>
                    <h4>内容段落2内容段落2内容段落2内容段落2</h4>
                </div>
            </li>
        </ul>
    </div>
</div>

