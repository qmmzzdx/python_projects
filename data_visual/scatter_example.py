import matplotlib.pyplot as plt


# 设置x轴的绘图参数
x_values = range(1, 1001)

# 设置y轴的绘图参数
y_values = [x**2 for x in x_values]

# 使用matplotlib中的一些样式风格, 包含默认的背景色、网格线、线条粗细、字体、字号等设置
plt.style.use('Solarize_Light2')

# 在一个图形(figure)中绘制一个或多个绘图(plot)
# fig表示由生成的一系列绘图构成的整个图形, 变量ax表示图形中的绘图
fig, ax = plt.subplots()

# 使用scatter来绘制散点图, c类似于参数color
# 将参数c设置成了一个y坐标值列表, 并使用参数cmap告诉pyplot使用哪个颜色映射
# 将y坐标值较小的点显示为浅蓝色, 将y坐标值较大的点显示为深蓝色, 参数s设置绘图时使用的点的尺寸
ax.scatter(x_values, y_values, c=y_values, cmap=plt.cm.Blues, s=10)

# 设置图题并给坐标轴加上标签, fontsize用于指定图中各种文字的大小
ax.set_title("Square Numbers", fontsize=24)
ax.set_xlabel("Value", fontsize=14)
ax.set_ylabel("Square of Value", fontsize=14)

# 设置刻度标记的样式, 将两条轴上的刻度标记的字号都设置为14
ax.tick_params(labelsize=14)

# 设置每个坐标轴的取值范围
ax.axis([0, 1100, 0, 1_100_000])

# 在刻度标记表示的数足够大时, matplotlib将默认使用科学记数法
# 使用ticklabel_format()方法覆盖默认的刻度标记样式
ax.ticklabel_format(style='plain')

# 显示图表
plt.show()
