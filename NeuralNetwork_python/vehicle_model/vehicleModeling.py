import numpy as np
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

print(torch.version.cuda)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def print_aligned_title(title):
    total_length = 80
    title_length = len(title)
    left_padding_length = (total_length - title_length) // 2
    right_padding_length = total_length - title_length - left_padding_length
    aligned_title = (
        left_padding_length * "=" + " " + title + " " + right_padding_length * "="
    )
    logging.info(aligned_title)


# 滑动窗口滤波
def moving_average_filter(data, window_size):
    filled_data = data.rolling(window=window_size, min_periods=1).mean()
    return filled_data


# 数据集
class RecordDataset(Dataset):
    def __init__(self, csv_file, sequence_length):
        self.data = pd.read_csv(csv_file)
        self.data_samples = self.data[
            [
                "throttle_percentage",
                "brake_percentage",
                "speed_mps",
                "steering_percentage",
                "acceleration_current_point",
                "acceleration_next_point",
                "angular_velocity_vrf",
            ]
        ]
        self.data_ori = self.data_samples.copy()
        # 滑动窗口平滑滤波
        columns_to_smooth = [
            "throttle_percentage",
            "brake_percentage",
            "speed_mps",
            "steering_percentage",
            "acceleration_current_point",
            "acceleration_next_point",
            "angular_velocity_vrf",
        ]
        for column in columns_to_smooth:
            self.data_samples.loc[:, column] = moving_average_filter(
                self.data_samples[column], window_size=5
            )

        num_features = len(columns_to_smooth)
        fig, axs = plt.subplots(num_features, 1, figsize=(12, 6 * num_features))
        for i, feature in enumerate(columns_to_smooth):
            # 原始特征
            axs[i].plot(
                self.data_ori.index,
                self.data_ori[feature],
                label=f"Original {feature}",
                zorder=1,
            )
            axs[i].set_title(f"{feature} Before and After Smoothing")  # 设置子图标题
            axs[i].set_xlabel("Index")
            axs[i].set_ylabel("Feature Value")
            axs[i].legend()  # 添加图例

            # 平滑后的特征
            axs[i].plot(
                self.data_samples.index,
                self.data_samples[f"{feature}"],
                label=f"Smoothed {feature}",
                zorder=2,
            )

        # 调整布局
        plt.tight_layout()
        # 显示图像
        plt.show()

        # TODO: standardlized residual 去除 outliers
        self.sequence_length = sequence_length
        self.throttle_count = 0
        self.brake_count = 0
        big_group = []
        index_tokeep = []

        normalization_list = [[] for _ in range(len(self.data_samples.columns))]
        # 清理数据
        for i in range(len(self.data_samples)):
            steering_condition = (
                abs(self.data_samples.at[i, "steering_percentage"]) <= 1
            )
            cmd_condition1 = (
                self.data_samples.at[i, "throttle_percentage"] >= 1
                or self.data_samples.at[i, "brake_percentage"] >= 1
            )
            cmd_condition2 = not (
                self.data_samples.at[i, "throttle_percentage"] > 0
                and self.data_samples.at[i, "brake_percentage"] > 0
            )
            speed_condition = self.data_samples.at[i, "speed_mps"] > 0

            if (
                steering_condition
                and cmd_condition1
                and cmd_condition2
                and speed_condition
            ):
                index_tokeep.append(i)

            else:
                if len(index_tokeep) >= self.sequence_length:
                    # 存储需要归一化的数据
                    for idx in index_tokeep:
                        row_data = self.data_samples.iloc[idx]
                        for col_idx, col_name in enumerate(self.data_samples.columns):
                            normalization_list[col_idx].append(row_data[col_name])

                    small_group_data = [
                        self.data_samples.iloc[idx] for idx in index_tokeep
                    ]
                    small_group_df = pd.DataFrame(small_group_data)
                    big_group.append(small_group_df)
                index_tokeep = []

        # 最后一个数据
        if len(index_tokeep) >= self.sequence_length:
            for idx in index_tokeep:
                row_data = self.data_samples.iloc[idx]
                for col_idx, col_name in enumerate(self.data_samples.columns):
                    normalization_list[col_idx].append(row_data[col_name])
            small_group_data = [self.data_samples.iloc[idx] for idx in index_tokeep]
            small_group_df = pd.DataFrame(small_group_data)
            big_group.append(small_group_df)

        # 归一化
        normalization_list = np.array(normalization_list)
        max_list = np.max(normalization_list, axis=1)
        min_list = np.min(normalization_list, axis=1)
        max_difference = max_list - min_list
        for group in big_group:
            for data_idx, row_data in group.iterrows():
                if row_data["throttle_percentage"] > 0:
                    self.throttle_count += 1
                else:
                    self.brake_count += 1
                group.loc[data_idx] = (row_data - min_list) / (max_list - min_list)

        self.samples = []
        for group in big_group:
            for i in range(len(group)):
                if i + sequence_length <= len(group):
                    self.samples.append(group.iloc[i : i + sequence_length])

        print_aligned_title("Data processing")
        print(f"Total original samples: {len(self.data_samples)}")
        print(f"Total filtered samples: {len(self.samples)}")
        print(f"throttle left:{self.throttle_count}, brake left:{self.brake_count}\n")

    def __len__(self):
        return len(self.samples)

    # 选择用于训练的列
    def __getitem__(self, idx):
        features = torch.tensor(
            self.samples[idx].iloc[:, :3].values, dtype=torch.float32
        )
        target = torch.tensor(
            self.samples[idx].iloc[-1, -3:-2].values, dtype=torch.float32
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

    input_size = 3
    hidden_size = 8
    num_layers = 5
    output_size = 1
    batch_size = 32
    learning_rate = 0.002
    epochs = 100
    sequence_length = 25
    csv_file = "/home/cyn/cs/NeuralNetwork_python/vehicle_model/record.csv"
    save_path = "VehicleModel.pth"
    print_aligned_title("Config")
    print(
        f"input size:{input_size}, ouput size:{output_size}, hidden size:{hidden_size}, num layers:{num_layers}"
    )
    print(
        f"batch size:{batch_size}, learning rate:{learning_rate}, epochs:{epochs}, sequence length: {sequence_length}\n"
    )

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

    torch.save(model.state_dict(), save_path)
    print("\nModel saved successfully.")
