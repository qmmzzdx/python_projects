import plotly.express as px

from random import randint


class Die:
    """表示一个骰子的类"""

    def __init__(self, num_sides=6):
        """骰子默认为6面的"""
        self.num_sides = num_sides

    def roll(self):
        """"返回一个介于1和骰子面数之间的随机值"""
        return randint(1, self.num_sides)


die_1 = Die()
die_2 = Die(10)

# 掷多次骰子并将结果存储在一个列表中
results = [die_1.roll() + die_2.roll() for roll_num in range(50_000)]

# 分析结果
max_result = die_1.num_sides + die_2.num_sides
poss_results = range(2, max_result + 1)
frequencies = [results.count(value) for value in poss_results]

# 使用函数px.bar()创建一个直方图并传递一些参数
title = "Results of Rolling a D6 and a D10 50,000 Times"
labels = {'x': 'Result', 'y': 'Frequency of Result'}
fig = px.bar(x=poss_results, y=frequencies, title=title, labels=labels)

# 进一步定制图形, 给每个条形都加上标签, xaxis_dtick指定x轴上刻度标记的间距, 设置为1
fig.update_layout(xaxis_dtick=1)

# 让Plotly将生成的直方图渲染为HTML文件, 并在一个新的浏览器选项卡中打开这个文件
fig.show()

# 将图形保存为HTML文件
fig.write_html('dice_visual.html')
