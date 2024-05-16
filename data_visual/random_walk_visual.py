import matplotlib.pyplot as plt
from random import choice


class RandomWalk:
    """一个生成随机游走数据的类"""

    def __init__(self, num_points=5000):
        """初始化随机游走的属性"""
        self.num_points = num_points

        # 所有随机游走都始于(0, 0)
        self.x_values = [0]
        self.y_values = [0]

    def fill_walk(self):
        """计算随机游走包含的所有点"""
        # 不断游走, 直到列表达到指定的长度
        while len(self.x_values) < self.num_points:
            # 决定前进的方向以及沿这个方向前进的距离
            x_step = self.get_step()
            y_step = self.get_step()

            # 拒绝原地踏步
            if x_step or y_step:
                # 添加下一个点的x坐标值和y坐标值
                self.x_values.append(self.x_values[-1] + x_step)
                self.y_values.append(self.y_values[-1] + y_step)

    def get_step(self):
        """决定随机游走方向和距离"""
        direction = choice([1, -1])
        distance = choice([0, 1, 2, 3, 4])
        return direction * distance


# 只要程序处于活动状态, 就一直模拟随机游走
while True:
    # 创建一个RandomWalk实例
    rw = RandomWalk(50_000)
    rw.fill_walk()

    # 将所有的点都绘制出来
    # figsize指定生成的图形尺寸, 向matplotlib指出绘图窗口的尺寸, 单位为英寸
    # dpi调整屏幕的分辨率, 每英寸128像素
    plt.style.use('classic')
    fig, ax = plt.subplots(figsize=(6, 4), dpi=128)
    point_numbers = range(rw.num_points)

    # 实参edgecolors='none'来删除每个点的轮廓
    ax.scatter(rw.x_values, rw.y_values, c=point_numbers, cmap=plt.cm.Blues,
               edgecolors='none', s=1)
    # 默认情况下Matplotlib独立地缩放每个轴, 而这将水平或垂直拉伸绘图
    # 使用set_aspect()指定两条轴上刻度的间距必须相等
    ax.set_aspect('equal')

    # 突出起点和终点
    ax.scatter(0, 0, c='green', edgecolors='none', s=100)
    ax.scatter(rw.x_values[-1], rw.y_values[-1], c='red', edgecolors='none',
               s=100)

    # 隐藏坐标轴
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.show()

    keep_running = input("Make another walk? (y/n): ")
    if keep_running == 'n':
        break
