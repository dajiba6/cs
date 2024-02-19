import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取数据
data = np.loadtxt(
    "/home/cyn/cs/SL/numerical_optimization/hw2/QuasiNewton/src/output_file.txt"
)  # 替换为你的数据文件名

# 提取x和y坐标
x_data = data[:, 0]
y_data = data[:, 1]


# Rosenbrock函数
def rosenbrock(x, y):
    a = 1
    b = 100
    return (a - x) ** 2 + b * (y - x**2) ** 2  # 修正了Rosenbrock函数的实现


# 生成坐标网格
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

# 绘制等高线图
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap="viridis", alpha=0.6)
plt.plot(x_data, y_data, label="QuasiNewton", color="blue", linewidth=2)
plt.scatter(1, 1, color="red", s=80, marker="o", label="end Point")
plt.scatter(x_data[0], y_data[0], color="green", s=80, marker="o", label="start Point")

# 设置图形标题和坐标轴标签
plt.title("Contour Plot of Rosenbrock Function")
plt.xlabel("x")
plt.ylabel("y")

# 添加图例
plt.legend()

# 显示图形
plt.show()
