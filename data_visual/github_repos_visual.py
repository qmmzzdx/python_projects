import urllib3
import requests
import plotly.express as px


urllib3.disable_warnings()
# 执行API调用并查看响应
url = "https://api.github.com/search/repositories"
url += "?q=language:python+sort:stars+stars:>100000"

# 最新的GitHub API版本为第3版, 通过指定headers显式地要求使用这个版本的API并返回JSON格式的结果
headers = {"Accept": "application/vnd.github.v3+json"}
r = requests.get(url, headers=headers, verify=False)
print(f"Status code: {r.status_code}")

# 将响应转换为字典
response_dict = r.json()

# 查看响应字典中的键
print(response_dict.keys())

# 查看符合要求的项目总数以及GitHub是否有足够的时间处理完这个查询
print(f"Total repositories: {response_dict['total_count']}")
print(f"Complete results: {not response_dict['incomplete_results']}")

# 探索有关仓库的信息
repo_dicts = response_dict['items']
print(f"Repositories returned: {len(repo_dicts)}")
print("\nSelected information about each repository:")
for repo_dict in repo_dicts:
    print(f"Name: {repo_dict['name']}")
    print(f"Owner: {repo_dict['owner']['login']}")
    print(f"Stars: {repo_dict['stargazers_count']}")
    print(f"Repository: {repo_dict['html_url']}")
    print(f"Created: {repo_dict['created_at']}")
    print(f"Updated: {repo_dict['updated_at']}")
    print(f"Description: {repo_dict['description']}\n")

repo_links, stars, hover_texts = [], [], []
for repo_dict in repo_dicts:
    # 将仓库名转换为链接
    repo_name = repo_dict['name']
    repo_url = repo_dict['html_url']
    repo_link = f"<a href='{repo_url}'>{repo_name}</a>"
    repo_links.append(repo_link)

    stars.append(repo_dict['stargazers_count'])

    # 创建悬停文本(鼠标静置后生成的信息)
    owner = repo_dict['owner']['login']
    description = repo_dict['description']

    # 添加HTML换行符
    hover_text = f"{owner}<br />{description}"
    hover_texts.append(hover_text)

# 创建一个条形图
title = "Most-Starred Python Projects on GitHub"
labels = {'x': 'Repository', 'y': 'Stars'}
fig = px.bar(x=repo_links, y=stars, title=title, labels=labels,
             hover_name=hover_texts)

# 添加图形的标题并给每条坐标轴添加标题
fig.update_layout(title_font_size=28, xaxis_title_font_size=20,
                  yaxis_title_font_size=20)

# 将条形改为更深的蓝色并且是半透明的, 将每个标记的不透明度都设置成了0.6
fig.update_traces(marker_color='SteelBlue', marker_opacity=0.6)

fig.show()
