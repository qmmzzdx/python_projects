from operator import itemgetter
import requests
import plotly.express as px


# 返回一个列表包含Hacker News上当前排名靠前的文章的ID
url = 'https://hacker-news.firebaseio.com/v0/topstories.json'
r = requests.get(url)
print(f"Status code: {r.status_code}")

# 处理有关每篇文章的信息
submission_ids = r.json()

submission_dicts = []
for submission_id in submission_ids[:10]:
    # 对于每篇文章都执行一个API调用
    url = f"https://hacker-news.firebaseio.com/v0/item/{submission_id}.json"
    r = requests.get(url)
    print(f"id: {submission_id}\tstatus: {r.status_code}")
    response_dict = r.json()

    try:
        # 对于每篇文章都创建一个字典
        submission_dict = {
            'title': response_dict['title'],
            'hn_link': f"https://news.ycombinator.com/item?id={submission_id}",
            'comments': response_dict['descendants'],
        }
    except KeyError:
        print(f"KeyError: {submission_id}")
    else:
        submission_dicts.append(submission_dict)

# 根据评论数对字典列表submission_dicts排序
# 使用operator模块中的函数itemgetter(), 从这个列表的每个字典中提取与键comments对应的值
submission_dicts = sorted(submission_dicts, key=itemgetter('comments'),
                          reverse=True)

# 显示每篇文章的信息
for submission_dict in submission_dicts:
    print(f"\nTitle: {submission_dict['title']}")
    print(f"Discussion link: {submission_dict['hn_link']}")
    print(f"Comments: {submission_dict['comments']}")

# 创建包含文章链接, 评论数目和标题的列表
article_links, comment_counts, hover_texts = [], [], []
for submission_dict in submission_dicts:
    # 将较长的文章标题缩短
    title = submission_dict['title'][:30]
    discussion_link = submission_dict['hn_link']
    article_link = f'<a href="{discussion_link}"">{title}</a>'
    comment_count = submission_dict['comments']

    article_links.append(article_link)
    comment_counts.append(comment_count)
    # 悬停时显示完整的文章标题
    hover_texts.append(submission_dict['title'])

# 创建图表
title = "Most active discussions on Hacker News"
labels = {'x': 'Article', 'y': 'Comment count'}
fig = px.bar(x=article_links, y=comment_counts, title=title, labels=labels,
             hover_name=hover_texts)

fig.update_layout(title_font_size=28, xaxis_title_font_size=20,
                  yaxis_title_font_size=20)

fig.update_traces(marker_color='SteelBlue', marker_opacity=0.6)

fig.show()
