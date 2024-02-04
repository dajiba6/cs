import matplotlib.pyplot as plt
import numpy as np

# 从文件读取数据
with open("output.txt", "r") as file:
    lines = file.readlines()

# 将字符串数据转换为NumPy数组
data_array = np.array([list(map(float, line.strip().split())) for line in lines])

# 分类数据
class_0 = data_array[data_array[:, 0] == 0]
class_1 = data_array[data_array[:, 0] == 1]


# Rosenbrock函数
def rosenbrock(x, y):
    a = 1
    b = 100
    return (x - 1) ** 2 + b * (x**2 - y) ** 2


# 生成坐标网格
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

# 绘制等高线图
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 10), cmap="viridis")


# 绘制散点图
plt.plot(class_0[:, 1], class_0[:, 2], label="Class 0")
plt.plot(class_1[:, 1], class_1[:, 2], label="Class 1")
# 在1,1坐标处绘制一个大红点
plt.scatter(1, 1, color="red", s=80, marker="o", label="end Point")
plt.scatter(
    class_0[0, 1], class_0[0, 2], color="green", s=80, marker="o", label="start Point"
)
# 添加标题和标签
plt.title("Scatter Plot of Class 0 and Class 1")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# 显示图例
plt.legend()

# 显示图表
plt.show()
