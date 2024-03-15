import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd


# =================模型定义======================
class VehicleDynamicsLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VehicleDynamicsLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# =====================数据准备======================
seq_length = 20
batch_size = 1
input_size = 5
output_size = 2

# 从 CSV 文件读取数据
data = pd.read_csv("RecordToCSV_2024-03-14_15-28-51.csv")

# 提取输入特征和输出标签
X_train = data[
    [
        "throttle_percentage",
        "brake_percentage",
        "steering_percentage",
        "speed_mps",
        "acceleration_current_point",
        "acceleration_next_point",
        "angular_velocity_vrf",
    ]
]
y_train = data[["acceleration_next_point", "angular_velocity_vrf"]]

num_samples = len(data) // seq_length
X_samples = []
y_samples = []

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

# 数据集创建
train_dataset = TensorDataset(
    torch.stack(
        [
            X_train_tensor[i : i + seq_length]
            for i in range(0, len(data) - seq_length + 1, seq_length)
        ]
    ),
    torch.stack(
        [
            y_train_tensor[i : i + seq_length]
            for i in range(0, len(data) - seq_length + 1, seq_length)
        ]
    ),
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 实例化模型、损失函数和优化器
hidden_size = 8
model = VehicleDynamicsLSTM(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===================训练模型====================
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}"
    )

# 评估模型
# 在实际任务中，你可以准备测试集，并使用测试集来评估模型的性能
# 这里省略了测试过程，假设训练集已经足够代表总体数据，直接使用训练集进行评估

print("训练完成!")
