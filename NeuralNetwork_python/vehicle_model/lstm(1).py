# 0. 解析record包为训练可用的数据格式CSV(时间序列) 完成 里面数据格式为float

# 1 CSV(时间序列)内数据的筛选 去除异常值 均值滤波 生成训练集和验证集 归一化

# 2. 引入头文件需要每步检查数据打印验证

# 3 数据是否需要归一化 个人认为需要

# 4. 确定是什么问题，优化还是分类等(预测)；进而确定loss函数RSME
# In order to evaluate the performances of dynamic models, the
# RMSE (rooted-mean-squared-error) between the predicted
# states (acceleration, angular speed, speed, heading, position)
# and the ground-truth states given by localization module will
# be calculated

# 5. 确定输入3 输出维度1，具体变量，hidden层数及维度8，时间序列长度20，batch_size等
# input layer of the learning-based dynamic model has five dimensions [u t , u b , u st , v i , a i ], where u t , u b , u st represent throttle, brake and steering control command respectively. v i is ego vehicle’s speed and a i is acceleration at current sample point.

# The output layer has two dimensions a_i+1 , theta_dot_i+1 , where a_i+1 is the acceleration and theta_dot_i+1 is the heading angle change rate for next sample point.

# 6. 确定采用的激活函数，明确使用原因; 训练集和测试集的科学划分 先用LSTM默认，再尝试ReLU

# 7. 训练后结果评价指标
# In order to evaluate the performances of dynamic models, the
# RMSE (rooted-mean-squared-error) between the predicted
# states (acceleration, angular speed, speed, heading, position)
# and the ground-truth states given by localization module will
# be calculated.

# 对于数据集大小为300,000的情况，你可以尝试使用中等大小的批量进行训练。以下是一些建议：

# 尝试较小的批量大小：你可以从较小的批量大小开始，比如16或32。这样可以确保模型的训练过程比较稳定，并且可以更快地观察到训练的进展情况。

# 观察训练表现：根据训练的情况，观察模型的表现以及损失函数的变化。如果模型在训练过程中表现良好，并且没有出现过拟合或欠拟合的情况，那么可以考虑尝试增大批量大小。

# 增大批量大小：如果模型和训练过程可以支持，你可以尝试增大批量大小，比如64、128或更大。增大批量大小有助于利用计算资源，并且通常可以加快训练速度。

# 观察内存使用情况：在选择较大的批量大小时，要确保你的计算资源（特别是内存）能够支持。如果批量太大导致内存溢出或者训练速度明显下降，那么可能需要降低批量大小。

# 进行实验和调优：根据观察到的训练结果，进行实验和调优，找到最适合你的数据集和模型的批量大小。

# 总的来说，选择合适的批量大小需要进行实验和调整，并且取决于你的具体情况和需求。最终的选择应该是能够在给定的计算资源下实现良好性能的批量大小。


#!/usr/bin/env python

##############################################################################

# 20240313 liujiaqiao  LSTM
# 训练用于形成离线标定表的神经网络

##############################################################################
"""
LSTM
"""
import torch
import torch.nn as nn
import csv
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import data_prepocess
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# 读取csv文件,并按需保存数据
def read_csv_file(file_path):
    data = []  # 用于保存读取到的数据的列表
    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # 将每行数据转换为浮点数并添加到二维列表中
            float_row = []
            for item in row:
                try:
                    float_value = float(item)
                    float_row.append(float_value)
                except ValueError:
                    # 如果转换失败，则保留原字符串
                    float_row.append(item)
            data.append(float_row)
    return data


# 调用data_preprocess函数 预处理CSV数据 + 去除outliers


# 根据读取到的CSV数据创建输入输出序列
def create_inout_sequences(input_data, target_data, squence_length):
    in_seq = []
    out_seq = []
    L = len(input_data)
    for i in range(L - squence_length):
        in_seq.append(input_data[i : i + squence_length])
        out_seq.append(target_data[i + squence_length - 1])
    return in_seq, out_seq


# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    # 预测结果将是一个形状为 (len(input_seq), output_size) 的二维张量。
    # c0,h0会自动隐式给定为全零张量 TODO 此处可以尝试不同初始值
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions


# 训练并在每个训练周期后验证 LSTM 模型


def train_and_verify_lstm_model(
    model,
    data_loader_for_train,
    data_loader_for_verify,
    epochs,
):
    print("===================================================================")
    loss_func = nn.MSELoss()  # 使用均方误差作为损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器 lr:学习率

    # 定义收敛条件
    prev_val_loss = float("inf")
    tolerance = 1e-5  # 设置一个收敛容差 TODO 这里还没有生效?

    for epoch in range(epochs):
        for seq, labels in data_loader_for_train:
            model.train()  # 设置模型为训练模式
            optimizer.zero_grad()  # 梯度清零
            output = model(seq)  # 前向传播 这里的输入数据应该是前四列
            loss = torch.sqrt(
                loss_func(output, labels)
            )  # 计算 RMSE 损失 target_data 是后1列(目标值)
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

        # 在每个 epoch 后验证模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq_verify, labels_verify in data_loader_for_verify:
                outputs = model(seq_verify)
                val_loss += loss_func(outputs, labels_verify).item()
            val_loss /= len(dataset_for_verify)
        # print(
        #     "len(verify_in_seq_tensor)",
        #     len(verify_in_seq_tensor),
        # )
        # print(
        #     "len(verify_out_seq_tensor)",
        #     len(verify_out_seq_tensor),
        # )
        # print(
        #     "len(dataset_for_verify)",
        #     len(dataset_for_verify),
        # )

        # 输出当前训练周期的验证损失
        # 如果验证误差开始增加，说明模型可能开始过拟合，可以提前停止训练
        print(
            f"val_loss [{val_loss}], prev_val_loss + tolerance: {prev_val_loss+tolerance}"
        )
        if val_loss > prev_val_loss + tolerance:
            print("Validation loss increased! Stopping training.")
            break
        else:
            print("-----------------------------------------------------")
            print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss}")

        prev_val_loss = val_loss


# TODO 把预测的和实际的数据可视化

if __name__ == "__main__":

    # 设置批次大小   和   序列长度(从一开始,因为要保证时间序列的不间断)
    batch_size = 16
    squence_length = 20
    # 1.step 调用函数读取CSV文件获取原始数据
    file_path = "/home/cyn/cs/NeuralNetwork_python/vehicle_model/record.csv"
    csv_data = read_csv_file(file_path)

    # 获取csv文件内样本数量 这里要排除表头,在下面处理
    csv_data_length = len(csv_data)
    print("csv_data_length", csv_data_length)

    data_set = torch.tensor(csv_data[1:], dtype=torch.float32)
    print("data_set.shape", data_set.shape)

    ########################  复用接口处 for throttle  #############################
    # 2.step 数据预处理 TODO 下面throttle和brake复用接口,用哪个打开哪个
    # (
    #     diff_values_for_train,
    #     breakpoint_index_for_train,
    #     data_set_for_train_new,
    # ) = data_prepocess.data_prepocess_for_throttle_train(data_set, squence_length)
    # # 3.step 去掉outliers breakpoint_index存储diff,即每diff_ith个数据需要断开,重新划分
    # breakpoint_index, outliers_filt_data = (
    #     data_prepocess.outliers_filt_for_throttle_train(
    #         data_set_for_train_new,
    #         breakpoint_index_for_train,
    #         squence_length,
    #     )
    # )
    ########################  复用接口处 for brake  #############################
    (
        diff_values_for_train,
        breakpoint_index_for_train,
        data_set_for_train_new,
    ) = data_prepocess.data_prepocess_for_brake_train(data_set, squence_length)
    breakpoint_index, outliers_filt_data = data_prepocess.outliers_filt_for_brake_train(
        data_set_for_train_new,
        breakpoint_index_for_train,
        squence_length,
    )

    # 4.step 均值滤波减小噪声影响 TODO

    # 5.step 划分数据集
    # 训练集:验证集 == 8:2
    # 把breakpoint_index先转换,然后近似8：2附近取截断，后续生成输入输出序列时需要把breakpoint_index也传进函数
    outliers_filt_data_length = len(outliers_filt_data)
    print("outliers_filt_data_length", outliers_filt_data_length)
    # print("outliers_filt_data.shape", outliers_filt_data_length.shape)
    train_data_length = int(0.8 * outliers_filt_data_length)  # 向下取整

    # ------------------TODO 在这里开始利用时间断点信息------------------------#
    # print("breakpoint_index", breakpoint_index)  # 此时是list
    breakpoint_index = torch.tensor(breakpoint_index)
    # print("breakpoint_index.shape", breakpoint_index.shape)

    breakpoint_index_for_length_calculate = torch.cumsum(breakpoint_index, dim=0).to(
        torch.int64
    )
    # print(
    #     "breakpoint_index_for_length_calculate", breakpoint_index_for_length_calculate
    # )
    #  224 *20 = 4460 TODO 一会看产生的样本数

    # 计算差值的绝对值
    absolute_diff = torch.abs(breakpoint_index_for_length_calculate - train_data_length)
    # 找到最小值和对应的索引
    _, min_index = torch.min(absolute_diff, dim=0)
    # 获取最接近的数
    closest_number = breakpoint_index_for_length_calculate[min_index]
    # 输出结果
    # print("closest_number:", closest_number.item())
    # print("min_index:", min_index.item())  # 索引223 第224个是分界点
    # print(
    #     "train_data_length", train_data_length
    # )  # TODO 每个不同的数据包这里需要重新计算,本例为28655,需要找出breakpoint_index_for_length_calculate内最接近这个数的数字作为训练集和验证集的分界点

    # train_data_length最终值设定为closest_number
    train_data_length = closest_number
    # print("train_data_length:", train_data_length)
    verify_data_length = outliers_filt_data_length - train_data_length
    # print("verify_data_length:", verify_data_length)

    # 训练集数据
    # 重要:此处分两种情况:一种训练throttle,另一种训练brake;上面已经复用接口了,因此这里代码得以简化,但是注意去除了outliers的输出张量维度为n x 4

    train_data_set = outliers_filt_data[:-verify_data_length]
    print("train_data_set.shape", train_data_set.shape)

    # -------保存训练集中的acc的最大最小数值，以便后续反归一化,这里也得分别对throttle和brake两种情况记录-------# TODO 验证epochs数量对数值有无影响,原则上应该没有
    # 训练数据集

    # train_throttle = train_data_set[:, 0]  # 油门数据集
    train_brake = train_data_set[:, 0]  # 刹车数据集
    train_steer = train_data_set[:, 1]  # 转角数据集
    train_speed = train_data_set[:, 2]  # 速度数据集
    train_acceleration = train_data_set[:, 3]  # 加速度数据集
    # 计算最大值和最小值
    # max_throttle = max(train_throttle)
    # min_throttle = min(train_throttle)
    max_brake = max(train_brake)
    min_brake = min(train_brake)
    max_steer = max(train_steer)
    min_steer = min(train_steer)
    max_speed = max(train_speed)
    min_speed = min(train_speed)
    max_acc = max(train_acceleration)
    min_acc = min(train_acceleration)

    # 保存最大值和最小值到文件
    with open("acceleration_stats.txt", "w") as f:
        # f.write(f"max_throttle: {max_throttle}\n")
        # f.write(f"min_throttle: {min_throttle}\n")
        f.write(f"max_brake: {max_brake}\n")
        f.write(f"min_brake: {min_brake}\n")
        f.write(f"max_steer: {max_steer}\n")
        f.write(f"min_steer: {min_steer}\n")
        f.write(f"max_speed: {max_speed}\n")
        f.write(f"min_speed: {min_speed}\n")
        f.write(f"max_acc: {max_acc}\n")
        f.write(f"min_acc: {min_acc}\n")

    # 验证集数据
    #  throttle/brake steer speed acc_next  TODO 利用acc_current,因为只平移一个时间点也不对，要平移也是系统机械延迟时间，百度用的车这个数值为200ms
    verify_data_set = outliers_filt_data[-verify_data_length:]
    print("verify_data_set.shape", verify_data_set.shape)

    # 6.step 数据归一化 TODO 这里归一化的方法怎么选择更合理
    # MinMaxScaler函数原型 axis=0是每列中最大值,即沿着数组第一个维度
    # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # X_scaled = X_std * (max - min) + min = X_std(在本情况:(0,1))
    # 需要注意的是，MinMaxScaler对异常值非常敏感，因为异常值会影响最小值和最大值的计算，从而影响缩放效果。在处理包含异常值的数据时，可能需要考虑使用其他的归一化方法，如RobustScaler或StandardScaler。
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data_set_scaled = scaler.fit_transform(train_data_set)
    # 根据对之前部分trainData进行fit的整体指标，对剩余的数据（testData）使用同样的均值、方差、最大最小值等指标进行转换transform(testData)，从而保证train、test处理方式相同
    verify_data_set_scaled = scaler.transform(verify_data_set)

    # 打印测试归一化效果
    # for row in train_data_set_scaled[4400:5400, :]:
    #     print(row)

    # 7.step 用归一化的数据创建输入输出序列
    train_input_data = train_data_set_scaled[:, :3]
    train_target_data = train_data_set_scaled[:, 3]
    verify_input_data = verify_data_set_scaled[:, :3]
    verify_target_data = verify_data_set_scaled[:, 3]
    print("train_input_data.shape", train_input_data.shape)
    print("train_target_data.shape", train_target_data.shape)

    # TODO 这里需要利用时间戳断点信息 循环制作 train_input_data train_target_data verify_input_data verify_target_data 然后把输出的不同的train_in_seq和train_out_seq堆叠在一起
    # 使用 breakpoint_index_for_length_calculate 中的两两元素作为索引获取局部行,此处需要额外代码处理第0行到breakpoint_index_for_length_calculate[0]行
    train_input_data_local_first = train_input_data[
        : breakpoint_index_for_length_calculate[0]
    ]
    train_target_data_local_first = train_target_data[
        : breakpoint_index_for_length_calculate[0]
    ]
    train_in_local_first_seq, train_out_local_first_seq = create_inout_sequences(
        train_input_data_local_first, train_target_data_local_first, squence_length
    )
    # print("type(train_in_local_first_seq)", type(train_in_local_first_seq))
    # print(
    #     "train_in_local_first_seq.shape",
    #     torch.tensor(np.array(train_in_local_first_seq)).shape,
    # )
    # print(
    #     "train_out_local_first_seq.shape", torch.tensor(train_out_local_first_seq).shape
    # )
    train_input_data_local = []
    train_target_data_local = []
    train_input_data_local.extend(train_in_local_first_seq)
    train_target_data_local.extend(train_out_local_first_seq)
    # print(
    #     "train_input_data_local.shape",
    #     torch.tensor(np.array(train_input_data_local)).shape,
    # )
    for i in range(min_index.item()):  # min_index.item()==223
        start_index = breakpoint_index_for_length_calculate[i]
        end_index = breakpoint_index_for_length_calculate[i + 1]
        train_in_seq, train_out_seq = create_inout_sequences(
            train_input_data[start_index:end_index],
            train_target_data[start_index:end_index],
            squence_length,
        )
        # 在第一个维度上合并列表,后续再转化成张量
        train_input_data_local.extend(train_in_seq)
        train_target_data_local.extend(train_out_seq)

    train_in_seq = np.array(train_input_data_local)  # list to nd.array
    train_out_seq = np.array(train_target_data_local)
    train_in_seq_tensor = torch.tensor(train_in_seq, dtype=torch.float32)
    train_out_seq_tensor = torch.tensor(train_out_seq, dtype=torch.float32)

    # verify
    verify_input_data_local_first = verify_input_data[
        breakpoint_index_for_length_calculate[
            min_index
        ] : breakpoint_index_for_length_calculate[min_index + 1]
    ]
    verify_target_data_local_first = verify_target_data[
        breakpoint_index_for_length_calculate[
            min_index
        ] : breakpoint_index_for_length_calculate[min_index + 1]
    ]
    verify_in_local_first_seq, verify_out_local_first_seq = create_inout_sequences(
        verify_input_data_local_first, verify_target_data_local_first, squence_length
    )
    verify_input_data_local = []
    verify_target_data_local = []
    verify_input_data_local.extend(verify_in_local_first_seq)
    verify_target_data_local.extend(verify_out_local_first_seq)
    for i in range(
        min_index.item(), len(breakpoint_index_for_length_calculate) - 1
    ):  # min_index.item()==223
        start_index = (
            breakpoint_index_for_length_calculate[i]
            - breakpoint_index_for_length_calculate[min_index.item()]
        )
        end_index = (
            breakpoint_index_for_length_calculate[i + 1]
            - breakpoint_index_for_length_calculate[min_index.item()]
        )
        verify_in_seq, verify_out_seq = create_inout_sequences(
            verify_input_data[start_index:end_index],
            verify_target_data[start_index:end_index],
            squence_length,
        )
        # 在第一个维度上合并列表,后续再转化成张量
        verify_input_data_local.extend(verify_in_seq)
        verify_target_data_local.extend(verify_out_seq)

    verify_in_seq = np.array(verify_input_data_local)
    verify_out_seq = np.array(verify_target_data_local)
    verify_in_seq_tensor = torch.tensor(verify_in_seq, dtype=torch.float32)
    verify_out_seq_tensor = torch.tensor(verify_out_seq, dtype=torch.float32)

    # for test
    # print(type(train_in_seq_tensor))
    # print(type(train_out_seq_tensor))
    print("train_in_seq_tensor.shape", train_in_seq_tensor.shape)
    print("train_out_seq_tensor.shape before", train_out_seq_tensor.shape)
    train_out_seq_tensor = torch.unsqueeze(train_out_seq_tensor, dim=1)
    print("train_out_seq_tensor.shape after", train_out_seq_tensor.shape)

    print("verify_in_seq_tensor.shape", verify_in_seq_tensor.shape)
    print("verify_out_seq_tensor.shape before", verify_out_seq_tensor.shape)
    verify_out_seq_tensor = torch.unsqueeze(verify_out_seq_tensor, dim=1)
    print("verify_out_seq_tensor.shape after", verify_out_seq_tensor.shape)
    # print(len(train_in_seq[1]))
    # for row in train_in_seq[1]:
    #     print(row)

    # 验证训练用的数据均为14324条 正好少于训练集数据14344条 20条(序列长度)
    # print(len(train_out_seq))
    # print(len(train_in_seq))

    # print(type(train_out_seq))
    # print(type(train_in_seq[1][0]))
    # print(train_in_seq[1][0])

    # 将数据集和标签组织成 TensorDataset 对象 TODO 时间断点会影响这里
    dataset_for_train = TensorDataset(train_in_seq_tensor, train_out_seq_tensor)
    # 创建 DataLoader 对象，并设置批次大小
    data_loader_for_train = DataLoader(
        dataset_for_train, batch_size=batch_size, shuffle=False
    )

    dataset_for_verify = TensorDataset(verify_in_seq_tensor, verify_out_seq_tensor)
    data_loader_for_verify = DataLoader(
        dataset_for_verify, batch_size=batch_size, shuffle=False
    )

    # 8.step 创建 LSTM 模型 batch_size=16
    model = LSTMModel(input_size=3, hidden_size=8, output_size=1)

    # 9.step 训练 LSTM 模型
    train_and_verify_lstm_model(
        model,
        data_loader_for_train,
        data_loader_for_verify,
        epochs=300,
    )

    # 10.step 保存 LSTM 模型
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"/home/liujiaqiao/anaconda3/LSTM/lstm_model_{current_time}.pt"
    torch.save(model.state_dict(), save_path)
