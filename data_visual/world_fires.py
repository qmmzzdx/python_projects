from pathlib import Path
import csv

import plotly.express as px
import pandas as pd

path = Path("eq_data/world_fires_7_day.csv")
try:
    lines = path.read_text().splitlines()
except:
    lines = path.read_text(encoding="utf8").splitlines()
reader = csv.reader(lines)
header_row = next(reader)

# 提取经纬度和火灾强度
lats, lons, brights = [], [], []
for row in reader:
    try:
        lat = float(row[0])
        lon = float(row[1])
        bright = float(row[2])
    except ValueError:
        # 对于无效行, 显示原始的日期信息
        print(f"Invalid data for {row[5]}")
    else:
        lats.append(lat)
        lons.append(lon)
        brights.append(bright)

# 绘制林火活动
data = pd.DataFrame(data=zip(lons, lats, brights),
                    columns=["经度", "纬度", "火灾强度"])

fig = px.scatter(
    data,
    x="经度",
    y="纬度",
    range_x=[-200, 200],
    range_y=[-90, 90],
    width=800,
    height=800,
    title="全球林火活动图",
    color="火灾强度",
    hover_name="火灾强度",
)
fig.write_html("eq_data/world_fires.html")
fig.show()
