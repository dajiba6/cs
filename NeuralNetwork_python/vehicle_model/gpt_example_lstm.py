import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


# 创建数据集类
class MyDataset(Dataset):
    def __init__(self, csv_file, sequence_length):
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.features = self.data.iloc[:, :2].values
        self.labels = self.data.iloc[:, 2].values
        self.scaler = MinMaxScaler()  # 创建一个最小-最大标准化器
        self.features = self.scaler.fit_transform(self.features)  # 对特征进行标准化

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        idx_end = idx + self.sequence_length
        features = self.features[idx:idx_end]
        labels = self.labels[idx_end]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(
            labels, dtype=torch.float32
        )


# 创建 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# 训练模型
def train_model(train_loader, model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels.unsqueeze(1))  # 计算损失
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader)}")


if __name__ == "__main__":
    # 设置一些超参数
    csv_file = "data.csv"
    sequence_length = 20
    input_size = 2  # 特征维度
    hidden_size = 64  # LSTM隐藏层单元数
    num_layers = 2  # LSTM层数
    output_size = 1  # 输出维度
    batch_size = 64
    learning_rate = 0.001
    epochs = 20

    # 创建数据集实例和数据加载器
    dataset = MyDataset(csv_file, sequence_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 训练模型
    train_model(train_loader, model, criterion, optimizer, epochs)
