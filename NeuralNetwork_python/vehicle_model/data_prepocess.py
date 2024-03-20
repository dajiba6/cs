#!/usr/bin/env python

##############################################################################

# 20240315 liujiaqiao
# 预处理数据,包含数据筛选、outliers去除、TODO mean filter

##############################################################################
import csv
import matplotlib.pyplot as plt
import torch
import numpy as np


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


# 分别对数据进行如下处理
# 从头开始读数据  throttle0 brake1 steer2 speed3 acc_curr4 acc_next5 angular_speed_next6
# 需要删除的数据 throttle小于等于0|| brake小于等于0 || abs|steer|>1 ||  throttle*acc_next<0 || brake*acc_next>=0 || speed<0
# 只要满足这些条件之一，就记录之前的数据部分到一个临时列表；判断临时列表长度，如果小于seq length，则列表清空；反之每seq length(递增1)生成一条训练样本放入训练集，然后临时列表清空，直到读取完所有的数据
def data_prepocess_for_throttle_train(data_set, squence_length):
    # temp_list = []
    threshold = 1
    data_set_for_throttle_train = data_set[:, [0, 2, 3, 5]]  # 这样做前提是响应无延迟
    mask_throttle = torch.logical_and(data_set[:, 0] > 0, data_set[:, 0] <= 100)
    mask_steer = torch.logical_and(
        data_set[:, 2] >= -threshold, data_set[:, 2] <= threshold
    )
    mask_speed = data_set[:, 3] > 0
    mask_throttle_acc_next = (
        data_set[:, 0] * data_set[:, 5] > 0
    )  # 暂时不考虑这个 这里应该需要考虑机械系统的延迟 百度用的车是200ms

    # test 不考虑时间间断看剩余数据长度大约31000左右
    mask_total = np.vstack(
        (mask_throttle, mask_steer, mask_speed, mask_throttle_acc_next)
    ).T
    result_pairs = []  # 存储结果对
    diff_values = []  # 存储差值
    # 初始化起始索引
    start_index = 0
    # 遍历每一行
    while start_index < len(mask_total):
        # 初始化终止索引
        end_index = start_index
        # 检查当前行是否全部为True
        if all(mask_total[start_index]):
            # 继续检查下一行
            for j in range(start_index + 1, len(mask_total)):
                # 如果下一行不全为True，计算差值
                if not all(mask_total[j]):
                    end_index = j - 1
                    diff = end_index - start_index + 1
                    # 检查差值是否大于等于squence_length
                    if diff >= squence_length:
                        result_pairs.append((start_index, end_index))
                        diff_values.append(diff)
                    break

            # 更新起始索引
            start_index = end_index + 1
        else:
            # 如果当前行不全为True，则直接跳到下一行
            start_index += 1

    # print("Result Pairs:", result_pairs)
    # print("Difference Values:", diff_values)
    # print("len(mask_total)", len(mask_total))
    # print("mask_total.shape", mask_total.shape)
    # total = sum(diff_values)  # 可用数据量测试
    # print("sum(diff_values):", total)

    # 生成可用数据二维张量,保存时间戳断点的向量不要忘记在diff_values里面
    # 创建一个空列表，用于存储局部二维张量
    local_tensors = []
    # 根据 result_pairs 中的每个元组，获取局部二维张量并保存到 local_tensors 中
    for start_index, end_index in result_pairs:
        local_tensor = data_set_for_throttle_train[
            start_index : end_index + 1, :
        ]  # +1才能把end_index这行包括在内
        local_tensors.append(local_tensor)

    # 将 local_tensors 转换为一个新的二维张量
    data_set_for_throttle_train_new = torch.tensor(np.vstack(local_tensors))
    diff_values = torch.tensor(diff_values)
    breakpoint_index = torch.zeros(len(diff_values))
    for i in range(len(diff_values)):
        # 当前位置的元素值等于当前索引及之前索引位置的元素之和
        breakpoint_index[i] = torch.sum(diff_values[: i + 1])
    print(
        "data_set_for_throttle_train_new.shape", data_set_for_throttle_train_new.shape
    )
    # print("breakpoint_index_for_throttle_train.shape", breakpoint_index.shape)
    # print("breakpoint_index_for_throttle_train", breakpoint_index)
    return diff_values, breakpoint_index, data_set_for_throttle_train_new


def data_prepocess_for_brake_train(data_set, squence_length):
    # temp_list = []
    threshold = 1
    data_set_for_brake_train = data_set[:, [1, 2, 3, 5]]
    # print("data_set_for_brake_train.shape", data_set_for_brake_train.shape)
    mask_brake = torch.logical_and(data_set[:, 1] > 0, data_set[:, 1] <= 100)
    mask_steer = torch.logical_and(
        data_set[:, 2] >= -threshold, data_set[:, 2] <= threshold
    )
    mask_speed = data_set[:, 3] > 0
    mask_brake_acc_next = data_set[:, 1] * data_set[:, 5] < 0  # 暂时不考虑这个

    # test 不考虑时间间断看剩余数据长度大约31000左右
    mask_total = np.vstack((mask_brake, mask_steer, mask_speed, mask_brake_acc_next)).T
    result_pairs = []  # 存储结果对
    diff_values = []  # 存储差值
    # 初始化起始索引
    start_index = 0
    # 遍历每一行
    while start_index < len(mask_total):
        # 初始化终止索引
        end_index = start_index
        # 检查当前行是否全部为True
        if all(mask_total[start_index]):
            # 继续检查下一行
            for j in range(start_index + 1, len(mask_total)):
                # 如果下一行不全为True，计算差值
                if not all(mask_total[j]):
                    end_index = j - 1
                    diff = end_index - start_index + 1
                    # 检查差值是否大于等于squence_length
                    if diff >= squence_length:
                        result_pairs.append((start_index, end_index))
                        diff_values.append(diff)
                    break

            # 更新起始索引
            start_index = end_index + 1
        else:
            # 如果当前行不全为True，则直接跳到下一行
            start_index += 1

    # 生成可用数据二维张量,保存时间戳断点的向量不要忘记在diff_values里面
    # 创建一个空列表，用于存储局部二维张量
    local_tensors = []
    # 根据 result_pairs 中的每个元组，获取局部二维张量并保存到 local_tensors 中
    for start_index, end_index in result_pairs:
        local_tensor = data_set_for_brake_train[
            start_index : end_index + 1, :
        ]  # +1才能把end_index这行包括在内
        local_tensors.append(local_tensor)

    # 将 local_tensors 转换为一个新的二维张量
    data_set_for_brake_train_new = torch.tensor(np.vstack(local_tensors))
    diff_values = torch.tensor(diff_values)
    breakpoint_index = torch.zeros(len(diff_values))
    for i in range(len(diff_values)):
        # 当前位置的元素值等于当前索引及之前索引位置的元素之和
        breakpoint_index[i] = torch.sum(diff_values[: i + 1])
    print("data_set_for_brake_train_new.shape", data_set_for_brake_train_new.shape)
    # print("breakpoint_index_for_brake_train.shape", breakpoint_index.shape)
    # print("breakpoint_index_for_brake_train", breakpoint_index)
    return diff_values, breakpoint_index, data_set_for_brake_train_new


# TODO 继续上面的步骤分别处理outliers和进行均值滤波处理
# https://zhuanlan.zhihu.com/p/342801954


def outliers_filt_for_throttle_train(
    data_set_for_throttle_train_new,
    breakpoint_index_for_throttle_train,
    squence_length,
    threshold=3,
):
    # 计算每个特征（列）的均值和标准差
    mean = torch.mean(data_set_for_throttle_train_new, dim=0)
    std = torch.std(data_set_for_throttle_train_new, dim=0)

    # 计算 Z 分数
    z_scores = (data_set_for_throttle_train_new - mean) / std

    # 判断是否超过阈值
    is_outlier = torch.abs(z_scores) > threshold
    # print("is_outlier.shape", is_outlier.shape)

    # 找出包含异常值的整行索引
    rows_with_outliers = torch.any(is_outlier, dim=1)

    # 使用 torch.nonzero() 找到非零元素的索引
    outlier_indices = torch.nonzero(rows_with_outliers).squeeze()
    # print("outlier_indices.shape", outlier_indices.shape)
    # print("outlier_indices", outlier_indices)

    # 把两个时间戳断点一维张量合并成一个总时间戳断点一维张量
    # 创建一个新的一维张量，用于存储结果
    breakpoint_index_result_tensor = torch.sort(
        torch.cat((breakpoint_index_for_throttle_train, outlier_indices))
    ).values

    # 返回加上异常值行索引作为断点后的时间戳断点一维张量，数据集暂时保持不变
    # print("breakpoint_index_result_tensor", breakpoint_index_result_tensor)
    # print("breakpoint_index_result_tensor.shape", breakpoint_index_result_tensor.shape)
    # if 191.0 in breakpoint_index_result_tensor:
    #     print("Element", 191.0, "is in the tensor.")

    # 初始化保存索引区间的列表
    index_intervals = []
    breakpoint_index_for_throttle = []
    # 遍历一维张量的元素，检查相邻元素之间的差值
    for i in range(len(breakpoint_index_result_tensor) - 1):
        diff = (
            breakpoint_index_result_tensor[i + 1]
            - breakpoint_index_result_tensor[i]
            - 1
        )
        if diff > squence_length:  # !!!由于以下代码squence_length不要设置太小
            # 如果差值大于 squence_length，则将当前索引和下一个索引加入到列表中
            index_intervals.append(
                (
                    breakpoint_index_result_tensor[i] + 1,
                    breakpoint_index_result_tensor[i + 1] - 1,
                )
            )
            # 利用diff对时间戳断点记录,即data_set_without_outliers_for_throttle的行每diff行为一组时间不间断数据
            breakpoint_index_for_throttle.append(diff)

    # print("len(index_intervals)", len(index_intervals))
    # print("index_intervals", index_intervals)
    data_set_without_outliers_for_throttle = []
    # 遍历索引区间列表
    for start_index, end_index in index_intervals:
        # 将 start_index 和 end_index 转换为整数标量张量
        start_index = start_index.to(torch.int64)
        end_index = end_index.to(torch.int64)
        # print("start_index,end_index", start_index, end_index)
        # 使用切片操作从 data_set 中选择相应的行，并添加到结果列表中
        data_set_without_outliers_for_throttle.extend(
            data_set_for_throttle_train_new[start_index : end_index + 1]
        )
    # 将结果列表转换为一个新的二维张量
    # print(
    #     "len(data_set_without_outliers_for_throttle)",
    #     len(data_set_without_outliers_for_throttle),
    # )
    data_set_without_outliers_for_throttle = torch.stack(
        data_set_without_outliers_for_throttle
    )
    # print(
    #     "data_set_without_outliers_for_throttle.shape",
    #     data_set_without_outliers_for_throttle.shape,
    # )

    return breakpoint_index_for_throttle, data_set_without_outliers_for_throttle


def outliers_filt_for_brake_train(
    data_set_for_brake_train_new,
    breakpoint_index_for_brake_train,
    squence_length,
    threshold=3,
):
    # 计算每个特征（列）的均值和标准差
    mean = torch.mean(data_set_for_brake_train_new, dim=0)
    std = torch.std(data_set_for_brake_train_new, dim=0)

    # 计算 Z 分数
    z_scores = (data_set_for_brake_train_new - mean) / std

    # 判断是否超过阈值
    is_outlier = torch.abs(z_scores) > threshold
    # print("is_outlier.shape", is_outlier.shape)

    # 找出包含异常值的整行索引
    rows_with_outliers = torch.any(is_outlier, dim=1)

    # 使用 torch.nonzero() 找到非零元素的索引
    outlier_indices = torch.nonzero(rows_with_outliers).squeeze()
    # print("outlier_indices.shape", outlier_indices.shape)
    # print("outlier_indices", outlier_indices)

    # TODO 把两个时间戳断点一维张量合并成一个总时间戳断点一维张量
    # 创建一个新的一维张量，用于存储结果
    breakpoint_index_result_tensor = torch.sort(
        torch.cat((breakpoint_index_for_brake_train, outlier_indices))
    ).values

    # 返回加上异常值行索引作为断点后的时间戳断点一维张量，数据集暂时保持不变
    # print("breakpoint_index_result_tensor", breakpoint_index_result_tensor)
    # print("breakpoint_index_result_tensor.shape", breakpoint_index_result_tensor.shape)

    # 初始化保存索引区间的列表
    index_intervals = []
    breakpoint_index_for_brake = []
    # 遍历一维张量的元素，检查相邻元素之间的差值
    for i in range(len(breakpoint_index_result_tensor) - 1):
        diff = (
            breakpoint_index_result_tensor[i + 1]
            - breakpoint_index_result_tensor[i]
            - 1
        )
        if diff > squence_length:
            # 如果差值大于 squence_length，则将当前索引和下一个索引加入到列表中
            index_intervals.append(
                (
                    breakpoint_index_result_tensor[i] + 1,
                    breakpoint_index_result_tensor[i + 1] - 1,
                )
            )
            breakpoint_index_for_brake.append(diff)

    data_set_without_outliers_for_brake = []
    # 遍历索引区间列表
    for start_index, end_index in index_intervals:
        # 将 start_index 和 end_index 转换为整数标量张量
        start_index = start_index.to(torch.int64)
        end_index = end_index.to(torch.int64)
        # 使用切片操作从 data_set 中选择相应的行，并添加到结果列表中
        data_set_without_outliers_for_brake.extend(
            data_set_for_brake_train_new[start_index : end_index + 1]
        )
    # 将结果列表转换为一个新的二维张量
    data_set_without_outliers_for_brake = torch.stack(
        data_set_without_outliers_for_brake
    )
    return breakpoint_index_for_brake, data_set_without_outliers_for_brake


def mean_filtering__for_throttle_train():
    pass


def mean_filtering__for_brake_train():
    pass


if __name__ == "__main__":
    # 示例文件路径
    file_path = (
        "/home/liujiaqiao/apollo8.0_dtv/data/RecordToCSV_2024-03-15_16-10-27.csv"
    )
    csv_data = read_csv_file(file_path)
    # 读取CSV文件到二维列表中并去除表头
    data_set = torch.tensor(csv_data[1:], dtype=torch.float32)
    print("len(data_set)", len(data_set))

    # 筛选数据
    (
        diff_values_for_throttle_train,
        breakpoint_index_for_throttle_train,
        data_set_for_throttle_train_new,
    ) = data_prepocess_for_throttle_train(data_set, squence_length=20)
    # print(type(data_set_for_throttle_train_new)) #<class 'numpy.ndarray'>
    # print(diff_values_for_throttle_train)
    (
        diff_values_for_brake_train,
        breakpoint_index_for_brake_train,
        data_set_for_brake_train_new,
    ) = data_prepocess_for_brake_train(data_set, squence_length=20)
    # print(diff_values_for_brake_train)
    (
        breakpoint_index_for_throttle,
        data_set_without_outliers_for_throttle,
    ) = outliers_filt_for_throttle_train(
        data_set_for_throttle_train_new,
        breakpoint_index_for_throttle_train,
        squence_length=20,
    )
    breakpoint_index_for_brake, data_set_without_outliers_for_brake = (
        outliers_filt_for_brake_train(
            data_set_for_brake_train_new,
            breakpoint_index_for_brake_train,
            squence_length=20,
        )
    )
    # print(
    #     "len(breakpoint_index_for_brake)",
    #     len(breakpoint_index_for_brake),
    # )
    # print(
    #     "(breakpoint_index_for_brake)",
    #     breakpoint_index_for_brake,
    # )
    # print(
    #     "data_set_without_outliers_for_brake.shape",
    #     data_set_without_outliers_for_brake.shape,
    # )
