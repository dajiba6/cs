import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# 数据集
class RecordDataset(Dataset):
    def __init__(self, csv_file, sequence_length):
        self.data = pd.read_csv(csv_file)
        self.data_samples = self.data[
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
        self.sequence_length = sequence_length

        self.samples = []
        for i in range(0, len(self.data), 1):
            if i + sequence_length <= len(self.data):
                self.samples.append(self.data_samples.iloc[i : i + sequence_length])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features = torch.tensor(
            self.samples[idx].iloc[:, :4].values, dtype=torch.float32
        )
        target = torch.tensor(
            self.samples[idx].iloc[-1, -2:].values, dtype=torch.float32
        )
        return features, target


# 网络
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
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


# 训练
def train(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for features, target in train_loader:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader)}")


# 测试
def test(model, test_loader, criterion):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for features, target in test_loader:
            features, target = features.to(device), target.to(device)
            outputs = model(features)
            loss = criterion(outputs, target)
            total_loss += loss.item()
        print(f"Test Loss: {total_loss / len(test_loader)}")


if __name__ == "__main__":

    input_size = 4
    hidden_size = 8
    num_layers = 2
    output_size = 2
    batch_size = 32
    learning_rate = 0.001
    epochs = 10
    sequence_length = 20
    csv_file = "record.csv"

    # 数据集
    dataset = RecordDataset(csv_file, sequence_length)
    train_size = int(0.8 * len(dataset))  # 训练集占比 80%
    val_size = len(dataset) - train_size  # 验证集占比 20%
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 模型
    model = LSTM(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 训练
    train(model, train_loader, criterion, optimizer, epochs)

    # 测试
    test_loader = DataLoader(val_set, batch_size=batch_size)
    test(model, test_loader, criterion)
