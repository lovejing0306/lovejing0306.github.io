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

<!--
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
</style>
-->

<!--
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
-->

<style>
.use-motion {
  .post { opacity: 0; }
}

.page-archive {

  .archive-page-counter {
    position: relative;
    top: 3px;
    left: 20px;

    @include mobile() {
      top: 5px;
    }
  }

  .posts-collapse {

    .archive-move-on {
      position: absolute;
      top: 11px;
      left: 0;
      margin-left: -6px;
      width: 10px;
      height: 10px;
      opacity: 0.5;
      background: $black-light;
      border: 1px solid white;

      @include circle();
    }
  }
}

.collection-title {
    position: relative;
    margin: 60px 0;

    h1, h2 { margin-left: 20px; }

    small { color: $grey; margin-left: 5px; }

    &::before {
      content: " ";
      position: absolute;
      left: 0;
      top: 50%;
      margin-left: -4px;
      margin-top: -4px;
      width: 8px;
      height: 8px;
      background: $grey;
      @include circle();
    }
  }
</style>

<section id="posts" class="posts-collapse" hidden>
  <span class="archive-move-on"></span>

  <span class="archive-page-counter">
    {% assign posts_length = site.posts.size %}
    {% if posts_length > 210 %} {% assign cheers = 'excellent' %}
      {% elsif posts_length > 130 %} {% assign cheers = 'great' %}
      {% elsif posts_length > 80 %} {% assign cheers = 'good' %}
      {% elsif posts_length > 50 %} {% assign cheers = 'nice' %}
      {% elsif posts_length > 30 %} {% assign cheers = 'ok' %}
    {% else %}
      {% assign cheers = 'um' %}
    {% endif %}
    {{ __.cheers[cheers] }}!
    {% if site.posts.size == 0 %}
      {{ __.counter.archive_posts.zero }}
    {% elsif site.posts.size == 1 %}
      {{ __.counter.archive_posts.one }}
    {% else %}
      {{ __.counter.archive_posts.other | replace: '%d', site.posts.size }}
    {% endif %}
    {{ __.keep_on }}
  </span>

  {% assign paginate = site.archive.paginate | default: site.paginate  %}
  {% assign paginate_path = site.archive.paginate_path | default: site.paginate_path  %}

  {% for post in site.posts %}
    {% if paginate > 0 %}
      {% assign page_num = forloop.index0 | divided_by: paginate | plus: 1 %}
      {% if page_num == 1 %}
        {% assign route = '/' %}
      {% else %}
        {% assign route = paginate_path | replace: ':num', page_num %}
      {% endif %}
      {% assign index0_modulo_paginate = forloop.index0 | modulo: paginate %}
    {% endif %}

    {% comment %} Show year {% endcomment %}
    {% assign post_year = post.date | date: '%Y' %}
    {% if post_year != year or index0_modulo_paginate == 0 and index0_modulo_paginate %}
      {% assign year = post_year %}
      <div class="collection-title" {% if route %}route="{{ route }}"{% endif %}>
        <h2 class="archive-year motion-element" id="archive-year-{{ year }}">{{ year }}</h2>
      </div>
    {% endif %}
    {% comment %} endshow {% endcomment %}

    {% include _macro/post-collapse.html %}

  {% endfor %}

</section>