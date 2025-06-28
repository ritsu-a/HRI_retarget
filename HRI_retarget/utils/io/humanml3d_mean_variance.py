# compute mean and variance of the joint_vecs for G1ML3D dataset
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
def mean_variance(data_dir, save_dir, joints_num, links_num):
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
    Std[0:4] = Std[0:4].mean() / 1.0
    Std[4:6] = Std[4:6].mean() / 1.0
    Std[6:7] = Std[6:7].mean() / 1.0
    Std[7: 7+(links_num - 1) * 3] = Std[7: 7+(links_num - 1) * 3].mean() / 1.0
    Std[7+(links_num - 1) * 3: 7+(links_num - 1) * 3 + joints_num] = Std[7+(links_num - 1) * 3: 7+(links_num - 1) * 3 + joints_num].mean() / 1.0
    Std[7+(links_num - 1) * 3 + joints_num: 7+(links_num - 1) * 6 + joints_num + 3] = Std[7+(links_num - 1) * 3 + joints_num: 7+(links_num - 1) * 6 + joints_num].mean() / 1.0
    Std[7+(links_num - 1) * 6 + joints_num: ] = Std[7+(links_num - 1) * 6 + joints_num: ].mean() / 1.0

    assert 7 + (links_num - 1) * 6 + joints_num + 4 == Std.shape[-1]

    np.save(pjoin(save_dir, 'Mean.npy'), Mean)
    np.save(pjoin(save_dir, 'Std.npy'), Std)

    return Mean, Std




if __name__ == '__main__':
    data_dir = '/root/pengyang/codebase/HRI_MLLM/data/G1ML3D_v1/new_joint_vecs'
    save_dir = '/root/pengyang/codebase/HRI_MLLM/data/G1ML3D_v1'
    mean, std = mean_variance(data_dir, save_dir, 29, 41)
    train_ratio = 0.8 
    val_ratio = 0.1
    test_ratio = 0.1
    file_list = os.listdir(data_dir)
    random.shuffle(file_list) 

    with open(pjoin(save_dir, 'train.txt'), 'w') as f:
        for file in file_list[:int(len(file_list) * train_ratio)]:
            f.write(file + '\n')
    with open(pjoin(save_dir, 'val.txt'), 'w') as f:
        for file in file_list[int(len(file_list) * train_ratio):int(len(file_list) * (train_ratio + val_ratio))]:
            f.write(file + '\n')
    with open(pjoin(save_dir, 'test.txt'), 'w') as f:
        for file in file_list[int(len(file_list) * (train_ratio + val_ratio)):]:
            f.write(file + '\n')
