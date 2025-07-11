import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
import os
import pickle
from sklearn.preprocessing import StandardScaler

def load_features(feat_path, static_feat_path, num_nodes, num_dynamic_features, dtype=np.float32):
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    # Reshape the data to [time_len, num_nodes, num_features]
    time_len = feat.shape[0]
    feat = feat.reshape(time_len, num_nodes, num_dynamic_features)

    # 静态特征CSV形状为[num_nodes, 2]
    static_feat_df = pd.read_csv(static_feat_path)
    static_feat = np.array(static_feat_df, dtype=dtype)
    static_feat = static_feat.reshape(1, num_nodes, -1)  # [1, num_nodes, 2]
    # 扩展到[time_len, num_nodes, 2]
    static_feat_expanded = np.tile(static_feat, (time_len, 1, 1))
    # 拼接动态和静态特征
    combined_feat = np.concatenate([feat, static_feat_expanded], axis=-1)
    return combined_feat


def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj

def load_targets(target_path, dtype=np.float32):
    target_df = pd.read_csv(target_path)
    # (time_len, num_nodes)
    targets = np.array(target_df, dtype=dtype)
    return targets


def generate_dataset(
        data, targets, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        # 分离动态和静态部分
        num_dynamic = data.shape[-1] - 2  # 最后两列是静态特征
        data_dynamic = data[..., :num_dynamic]
        data_static = data[..., num_dynamic:]

        # 归一化动态特征
        data_min = np.min(data_dynamic, axis=0)
        data_max = np.max(data_dynamic, axis=0)
        data_dynamic = (data_dynamic - data_min) / (data_max - data_min)

        # 归一化静态特征（按全局值）
        static_min = np.min(data_static, axis=(0, 1))  # 全局最小
        static_max = np.max(data_static, axis=(0, 1))  # 全局最大
        data_static = (data_static - static_min) / (static_max - static_min)

        # 合并并保存参数
        data = np.concatenate([data_dynamic, data_static], axis=-1)


        # with open('data_min_max.pkl', 'wb') as f:
        #     pickle.dump({
        #         'dynamic_min': data_min,
        #         'dynamic_max': data_max,
        #         'static_min': static_min,
        #         'static_max': static_max
        #     }, f)

        # 归一化目标数据
        target_min = np.min(targets, axis=0)
        target_max = np.max(targets, axis=0)
        targets = (targets - target_min) / (target_max - target_min)
        with open('target_min_max.pkl', 'wb') as f:
            pickle.dump({'target_min': target_min, 'target_max': target_max}, f)

    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    train_targets = targets[:train_size]
    test_data = data[train_size:time_len]
    test_targets = targets[train_size:time_len]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = np.array(train_data[i:i + seq_len])
        y = np.array(train_targets[i + seq_len:i + seq_len + pre_len])
        trainX.append(a)
        trainY.append(y)
    for i in range(len(test_data) - seq_len - pre_len):
        a = np.array(test_data[i:i + seq_len])
        y = np.array(test_targets[i + seq_len:i + seq_len + pre_len])
        testX.append(a)
        testY.append(y)

    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)

    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)

    return trainX, trainY, testX, testY


def generate_torch_datasets(
    data, targets,seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data,
        targets,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )

    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, test_dataset