import matplotlib.pyplot as plt
import numpy as np


def read_file(file_path):
    data = np.loadtxt(file_path)
    return data[:, 0], data[:, 1]


def plot_scatter(file_a, file_b):
    x_a, y_a = read_file(file_a)
    x_b, y_b = read_file(file_b)

    plt.plot(x_a, y_a, label="File A", marker="o", linestyle="-", linewidth=1)
    # plt.plot(x_b, y_b, label="File B", marker="x", linestyle="-", linewidth=1)

    # 在点 (1, 1) 处标记一个红点
    plt.scatter(1, 1, color="red", marker="o", s=200, label="Point (1, 1)")
    # plt.scatter(x_b[0], y_b[0], color="green", marker="o", s=100, label="Point (1, 2)")

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Scatter Plot of File A and File B")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    file_a_path = "./output1.txt"  # 替换为文件a的实际路径
    file_b_path = "./output2.txt"  # 替换为文件b的实际路径

    plot_scatter(file_a_path, file_b_path)
