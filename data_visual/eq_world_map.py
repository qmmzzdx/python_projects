from pathlib import Path
import json

import plotly.express as px
import pandas as pd

# 将数据作为字符串读取并转换为Python对象
path = Path('eq_data/eq_data_30_day_m1.geojson')
try:
    contents = path.read_text()
except:
    contents = path.read_text(encoding='utf-8')

# json.loads()将这个文件的字符串表示转换为Python对象
all_eq_data = json.loads(contents)

# 以更易于阅读的方式存储这些数据, indent指定数据结构中嵌套元素的缩进量
readable_geojson_path = Path('eq_data/readable_eq_data.geojson')
readable_contents = json.dumps(all_eq_data, indent=4)
readable_geojson_path.write_text(readable_contents)

# 获取数据集中的所有地震列表
all_eq_dicts = all_eq_data['features']

# 提取震级, 位置, 经度和纬度信息
mags, titles, lons, lats = [], [], [], []
for eq_dict in all_eq_dicts:
    mags.append(eq_dict['properties']['mag'])
    titles.append(eq_dict['properties']['title'])
    lons.append(eq_dict['geometry']['coordinates'][0])
    lats.append(eq_dict['geometry']['coordinates'][1])

# 通过zip函数将列表打包成元组形式的数据, 将其作为DataFrame对象的行数据, 并指定列名
data = pd.DataFrame(
    data=zip(lons, lats, titles, mags), columns=['经度', '纬度', '位置', '震级']
)

# 设置散点图大小, 颜色, 鼠标悬停信息
# 设置x轴为经度范围是-200到200, y轴为纬度范围是-90到90
# 设置标题为全球地震散点图
# 设置散点图显示的宽度和高度均为800像素
# size参数来指定散点图中每个标记的尺寸, 此处使用震级作为散点图尺寸把图中的点显示为不同的大小
# size_max=10将最大显示尺寸缩小到10像素
# color参数来指定散点图中每个标记的颜色, 此处使用震级作为散点图颜色并使用连续的颜色映射
# hover_name参数来指定散点图中每个标记的鼠标悬停信息, 此处使用地震位置作为散点图鼠标悬停信息
fig = px.scatter(
    data,
    x='经度',
    y='纬度',
    range_x=[-200, 200],
    range_y=[-90, 90],
    width=800,
    height=800,
    title='全球地震散点图',
    size='震级',
    size_max=10,
    color='震级',
    hover_name='位置',
)

# 将散点图保存为HTML文件并显示
fig.write_html('eq_data/global_earthquakes.html')
fig.show()
