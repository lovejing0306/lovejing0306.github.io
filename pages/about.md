---
layout: page
title: About
description: LittleBei's Personal Blog
keywords: LittleBei, Blog
comments: true
menu: 关于
permalink: /about/
<!-- subtitle:   <h3>Download My CV</h3>
            <a role="button" class="btn btn-primary hvr-grow-shadow" href="/assets/files/CV_Wendy_e.pdf" target="_blanks">
                <span class="flag-icon flag-icon-gb"></span> English
            </a>
            <a role="button" class="btn btn-primary hvr-grow-shadow" href="/assets/files/CV_Wendy_e.pdf" target="_blanks">
                <span class="flag-icon flag-icon-cn"></span> 中文
            </a> -->
                            
---

个人本职工作是一名计算机视觉算法工程师。

通过记录博客的方式记录自己成长。记录自己的改变。

记录知识，坚信努力改变人生。

## Contact

{% for website in site.data.social %}
* {{ website.sitename }}：[@{{ website.name }}]({{ website.url }})
{% endfor %}

## Skill Keywords

{% for category in site.data.skills %}
### {{ category.name }}
<div class="btn-inline">
{% for keyword in category.keywords %}
<button class="btn btn-outline" type="button">{{ keyword }}</button>
{% endfor %}
</div>
{% endfor %}
