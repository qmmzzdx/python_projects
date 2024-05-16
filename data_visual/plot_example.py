import matplotlib.pyplot as plt


# 设置x轴的绘图参数
x_values = [1, 2, 3, 4, 5]

# 设置y轴的绘图参数
y_values = [1, 4, 9, 16, 25]

# 使用matplotlib中的一些样式风格, 包含默认的背景色、网格线、线条粗细、字体、字号等设置
plt.style.use('Solarize_Light2')

# 在一个图形(figure)中绘制一个或多个绘图(plot)
# fig表示由生成的一系列绘图构成的整个图形, 变量ax表示图形中的绘图
fig, ax = plt.subplots()

# 根据给定的数据以浅显易懂的方式绘制折线图, linewidth决定了plot()绘制的线条的粗细
ax.plot(x_values, y_values, linewidth=3)

# 设置图题并给坐标轴加上标签, fontsize用于指定图中各种文字的大小
ax.set_title("Square Numbers", fontsize=24)
ax.set_xlabel("Value", fontsize=14)
ax.set_ylabel("Square of Value", fontsize=14)

# 设置刻度标记的样式, 将两条轴上的刻度标记的字号都设置为14
ax.tick_params(labelsize=14)

# 显示图表
plt.show()
