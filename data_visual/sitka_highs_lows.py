from pathlib import Path
import csv
from datetime import datetime

import matplotlib.pyplot as plt

# 创建一个Path对象指向文件夹weather_data中的天气数据文件
# splitlines()方法链式调用来获取一个包含文件中各行的列表
path = Path('weather_data/sitka_weather_2021_simple.csv')
lines = path.read_text().splitlines()

# 创建一个reader对象解析文件的各行
# next()返回文件中的下一行, 首次调用得到的是文件的第一行, 其中包含文件头
reader = csv.reader(lines)
header_row = next(reader)

# 提取日期, 最高温度和最低温度
dates, highs, lows = [], [], []
for row in reader:
    current_date = datetime.strptime(row[2], '%Y-%m-%d')
    high, low = int(row[4]), int(row[5])
    dates.append(current_date)
    highs.append(high)
    lows.append(low)

# 根据数据绘图
# 实参alpha指定颜色的透明度, 0表示完全透明, 1(默认设置)表示完全不透明
plt.style.use('Solarize_Light2')
fig, ax = plt.subplots()
ax.plot(dates, highs, color='red', alpha=0.5)
ax.plot(dates, lows, color='blue', alpha=0.5)
# facecolor指定填充区域的颜色, 在low和high之间填充颜色
ax.fill_between(dates, highs, lows, facecolor='blue', alpha=0.1)

# 设置绘图的格式
ax.set_title("Daily High and Low Temperatures, 2021", fontsize=18)
# 绘制倾斜的日期标签以免它们彼此重叠
fig.autofmt_xdate()
ax.set_ylabel("Temperature (F)", fontsize=16)
ax.tick_params(labelsize=16)

plt.show()
