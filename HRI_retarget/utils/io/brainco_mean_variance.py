# compute mean and variance of the joint_vecs for beat dataset
# also split the dataset into train, val, and test sets


import numpy as np
import sys
import os
from os.path import join as pjoin
import random


# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def body_mean_variance(data_dir, save_dir):
    file_list = os.listdir(data_dir)
    data_list = []

    for file in file_list:
        data = np.load(pjoin(data_dir, file))
        if np.isnan(data).any():
            print(file)
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)

    Std[:246] = Std[:246].mean() / 1.0
    Std[246:263] = Std[246:263].mean() / 1.0
    Std[263:467] = Std[263:467].mean() / 1.0
    Std[467:] = Std[467:].mean() / 1.0

    assert Std.shape[-1] == 491
    np.save(pjoin(save_dir, 'Mean.npy'), Mean)
    np.save(pjoin(save_dir, 'Std.npy'), Std)

    return Mean, Std




if __name__ == '__main__':
    data_dir = '/root/workspace/HRI_MLLM/data/1025_60_60fps_kimi/new_joint_vecs'
    save_dir = '/root/workspace/HRI_MLLM/data/1025_60_60fps_kimi'
    mean, std = body_mean_variance(data_dir, save_dir)
    train_ratio = 0.8 
    val_ratio = 0.1
    test_ratio = 0.1
    file_list = os.listdir(data_dir)
    random.shuffle(file_list) 

    with open(pjoin(save_dir, 'train.txt'), 'w') as f:
        for file in file_list[:int(len(file_list) * train_ratio)]:
            f.write(file[:-4] + '\n')
    with open(pjoin(save_dir, 'val.txt'), 'w') as f:
        for file in file_list[int(len(file_list) * train_ratio):int(len(file_list) * (train_ratio + val_ratio))]:
            f.write(file[:-4] + '\n')
    with open(pjoin(save_dir, 'test.txt'), 'w') as f:
        for file in file_list[int(len(file_list) * (train_ratio + val_ratio)):]:
            f.write(file[:-4] + '\n')
