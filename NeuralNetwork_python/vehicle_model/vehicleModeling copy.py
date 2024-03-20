import numpy as np
import time
import threading
import os
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.signal import firwin
from torch.utils.data import Dataset, DataLoader, random_split

print(torch.version.cuda)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def interpolate_outliers(data_samples, threshold=3):
    outlier_indices = []  # 存储异常值的行号列表
    for feature in data_samples:
        feature_mean = data_samples[feature].mean()
        feature_std = data_samples[feature].std()

        standardized_residuals = (data_samples[feature] - feature_mean) / feature_std
        outliers_mask = abs(standardized_residuals) > threshold
        outliers_indices = data_samples[outliers_mask].index

        # 将当前特征的异常值索引添加到异常值行号列表中
        outlier_indices.extend(outliers_indices)

    return outlier_indices


def print_aligned_title(title):
    total_length = 80
    title_length = len(title)
    left_padding_length = (total_length - title_length) // 2
    right_padding_length = total_length - title_length - left_padding_length
    aligned_title = (
        left_padding_length * "=" + " " + title + " " + right_padding_length * "="
    )
    logging.info(aligned_title)


def moving_average_filter(data, window_size):
    """
    滑动窗口滤波
    """
    filled_data = data.rolling(window=window_size, min_periods=1).mean()
    return filled_data


def fir_filter(data, num_taps, cutoff_freq, window="hamming"):
    """
    有限冲激响应(FIR)滤波器
    参数:
        num_taps (int): 滤波器阶数（滤波器长度）。
        cutoff_freq (float): 滤波器的截止频率。
        window (str, optional): 要使用的窗口函数类型。默认为 'hamming'。
    """
    filter_coeffs = firwin(num_taps, cutoff=cutoff_freq, window=window)
    filtered_data = np.convolve(data, filter_coeffs, mode="same")
    return filtered_data


# 数据集
class RecordDataset(Dataset):
    def __init__(self, csv_file, sequence_length):
        self.mode = 1  # 1 throttle; 2 brake
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

        # 滤波
        for column in self.data_samples.columns:
            self.data_samples.loc[:, column] = moving_average_filter(
                self.data_samples[column], window_size=3
            )
        # for column in self.self.data_samples.columns:
        #     self.data_samples.loc[:, column] = fir_filter(
        #         self.data_samples[column], 3, 1e-20
        #     )
        self.data_samples_no_extraction = self.data_samples.copy()

        # TODO: standardlized residual 去除 outliers
        self.sequence_length = sequence_length
        self.throttle_count = 0
        self.brake_count = 0
        big_group = []
        index_tokeep = []
        normalization_list = [[] for _ in range(len(self.data_samples.columns))]

        # 筛选数据
        for i in range(len(self.data_samples)):

            steering_condition = (
                abs(self.data_samples.at[i, "steering_percentage"]) <= 1
            )
            cmd_condition = (
                self.data_samples.at[i, "throttle_percentage"] >= 1
                or self.data_samples.at[i, "brake_percentage"] >= 1
            )
            cmd_throttle = self.data_samples.at[i, "brake_percentage"] == 0
            cmd_brake = self.data_samples.at[i, "throttle_percentage"] == 0
            speed_condition = self.data_samples.at[i, "speed_mps"] > 0
            # 训练油门模型
            if (
                self.mode == 1
                and steering_condition
                and cmd_condition
                and cmd_throttle
                and speed_condition
            ):
                index_tokeep.append(i)
            # 训练刹车模型
            elif (
                self.mode == 2
                and steering_condition
                and cmd_condition
                and cmd_brake
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
                for col_idx, col_name in enumerate(
                    self.data_samples.columns
                ):  # 返回列序号及列名字
                    normalization_list[col_idx].append(row_data[col_name])
            small_group_data = [self.data_samples.iloc[idx] for idx in index_tokeep]
            small_group_df = pd.DataFrame(small_group_data)
            big_group.append(small_group_df)

        # 归一化
        normalization_list = np.array(normalization_list)
        self.max_list = np.max(normalization_list, axis=1)
        self.min_list = np.min(normalization_list, axis=1)
        for group in big_group:
            for data_idx, row_data in group.iterrows():
                if row_data["throttle_percentage"] > 0:
                    self.throttle_count += 1
                else:
                    self.brake_count += 1
                group.loc[data_idx] = (row_data - self.min_list) / (
                    self.max_list - self.min_list
                )

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
        if self.mode == 1:
            features = torch.tensor(
                self.samples[idx].iloc[:, [0, 2]].values, dtype=torch.float32
            )
        else:
            features = torch.tensor(
                self.samples[idx].iloc[:, [1, 2]].values, dtype=torch.float32
            )
        target = torch.tensor(self.samples[idx].iloc[-1, 5], dtype=torch.float32)
        return features, target


class GenerateDataset(Dataset):
    """
    生成标定表的数据集
    参数：
      max_list:[0]cmd,[1]speed,[2]acc
      min_list:[0]cmd,[1]speed,[2]acc
    """

    def __init__(self, sequence_length, max_list, min_list):
        self.samples = []
        cmd = np.arange(0, 81, 5)
        speed = np.arange(0, 10.1, 0.2)
        self.cmd, self.speed = np.meshgrid(cmd, speed)
        self.input_data_ori = np.column_stack((self.cmd.ravel(), self.speed.ravel()))
        input_length = len(self.input_data_ori)
        # 归一化
        max_cmd = max_list[0]
        min_cmd = min_list[0]
        max_speed = max_list[1]
        min_speed = min_list[1]
        normalize_data = self.input_data_ori.copy()
        normalize_data[:, 0] = (normalize_data[:, 0] - min_cmd) / (max_cmd - min_cmd)
        normalize_data[:, 1] = (normalize_data[:, 1] - min_speed) / (
            max_speed - min_speed
        )
        input_data = np.repeat(normalize_data, repeats=sequence_length, axis=0)
        df = pd.DataFrame(input_data, columns=["cmd", "speed"])
        for i in range(input_length):
            self.samples.append(df.iloc[i : i + sequence_length])

    def __len__(self):
        return len(self.samples)

    # 选择用于训练的列
    def __getitem__(self, idx):
        features = torch.tensor(
            self.samples[idx].iloc[:, [0, 1]].values, dtype=torch.float32
        )
        target = torch.tensor(self.samples[idx].iloc[-1, -1], dtype=torch.float32)
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
    final_loss = 100
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
        final_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {final_loss}")
    return final_loss


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


# 数据可视化
def visualize_data(data):
    for i, feature in enumerate(data.data_samples.columns):
        fig, axs = plt.subplots(figsize=(12, 6))
        # 原始特征
        axs.plot(
            data.data_ori.index,
            data.data_ori[feature],
            label=f"Original {feature}",
            zorder=1,
        )
        axs.set_title(f"{feature} Before and After Smoothing")  # 设置子图标题
        axs.set_xlabel("Index")
        axs.set_ylabel("Feature Value")
        axs.legend()  # 添加图例

        # 平滑后的特征
        axs.plot(
            data.data_samples_no_extraction.index,
            data.data_samples_no_extraction[f"{feature}"],
            label=f"Smoothed {feature}",
            zorder=2,
        )
    plt.show()


def generate_calibration_table(
    model, sequence_length, batch_size, device, max_list, min_list
):
    min_list = [min_list[0], min_list[2], min_list[4]]
    max_list = [max_list[0], max_list[2], max_list[4]]
    generate_set = GenerateDataset(sequence_length, max_list, min_list)
    loader = DataLoader(generate_set, batch_size=batch_size, shuffle=False)
    model.eval()
    calibration_table = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.cpu().numpy()
            calibration_table.append(outputs)

    calibration_table = np.vstack(calibration_table)
    # 反归一化
    min_value = min_list[2]
    max_value = max_list[2]
    calibration_table = calibration_table * (max_value - min_value) + min_value
    calibration_table = pd.DataFrame(
        {
            "cmd": generate_set.input_data_ori[:, 0],
            "speed": generate_set.input_data_ori[:, 1],
            "acc": calibration_table[:, 0],
        }
    )
    return calibration_table


if __name__ == "__main__":

    input_size = 2
    hidden_size = 8
    num_layers = 2
    output_size = 1
    batch_size = 20
    learning_rate = 0.002
    epochs = 100
    sequence_length = 25
    csv_file = "/home/cyn/cs/NeuralNetwork_python/vehicle_model/record.csv"

    print_aligned_title("Config")
    print(
        f"input size:{input_size}, ouput size:{output_size}, hidden size:{hidden_size}, num layers:{num_layers}"
    )
    print(
        f"batch size:{batch_size}, learning rate:{learning_rate}, epochs:{epochs}, sequence length: {sequence_length}\n"
    )

    # 数据集
    dataset = RecordDataset(csv_file, sequence_length)
    # train_size = int(0.8 * len(dataset))  # 训练集占比 80%
    # val_size = len(dataset) - train_size  # 验证集占比 20%
    # train_set, val_set = random_split(dataset, [train_size, val_size])
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型
    model = LSTM(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_aligned_title("Using device: " + str(device))
    model.to(device)

    # visualize_data(dataset)

    # # 训练
    final_loss = train(model, train_loader, criterion, optimizer, epochs)

    # # 测试
    # test_loader = DataLoader(val_set, batch_size=batch_size)
    # test(model, test_loader, criterion)

    # 保存
    current_time = time.strftime("%Y%m%d%H%M", time.localtime())
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)  # 确保文件夹存在
    if dataset.mode == 1:
        save_name = f"{current_time}_throttle_loss{final_loss}.pth"
    else:
        save_name = f"{current_time}_brake_loss{final_loss}.pth"
    save_path = os.path.join(save_dir, save_name)
    torch.save(model.state_dict(), save_path)
    print("\nModel saved successfully.")

    # # 生成标定表
    # model_path = "/home/cyn/cs/NeuralNetwork_python/vehicle_model/models/202403191535_throttle_loss0.0013729687514815936.pth"
    # model.load_state_dict(torch.load(model_path))
    calibration_table = generate_calibration_table(
        model, sequence_length, batch_size, device, dataset.min_list, dataset.max_list
    )
    calibration_table.to_csv("calibration_table.csv", index=False)
    print("Calibration table generated and saved.")
