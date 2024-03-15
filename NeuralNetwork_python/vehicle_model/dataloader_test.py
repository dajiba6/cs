import pandas as pd

# 假设 df 是你的 DataFrame
# 这里只是一个示例，你需要将其替换为你的实际 DataFrame
data = {
    "feature1": [1, 2, 3, 4, 5],
    "feature2": [10, 20, 30, 40, 50],
    "feature3": [100, 200, 300, 400, 500],
}
df = pd.DataFrame(data)

# 获取每个特征的最大值和最小值
max_values = df.max()
min_values = df.min()

print("最大值:")
print(max_values)
print("\n最小值:")
print(min_values)
