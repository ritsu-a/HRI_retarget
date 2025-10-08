### converting 53 dof g1 to vec representation 
### for g1_inspirehand, dof=53, num_links=74

### for dof:
### 0-11: lower body 
### 12-14: waist
### 15-21: left arm
### 22-33: left hand 
### 34-40: right arm
### 41-52: right hand

### for G1_links:
### 0-13: lower body 
### 14-22: torso and head links
### 23-31: left arm and left hand links
### 32-40: right arm and right hand links


### for G1_inspirehand_links
### 0-13 lower body
### 14-22 torso and head links
### 23-30 left arm
### 31-47 left hand
### 48-55 right arm
### 56-72 right hand
### 73 imu_in_pelvis
### 
### 

### body features
### shape (num_frames - 1, 54*3+17=179)
###     link_local_position (9+9+9=27) * 3
###     link_local_velocity 27 * 3  
###     dof_angle 3+7+7=17


### full body features
### shape (num_frame - 1, 395)
###     body_link_local_position (9+8+8=25) * 3
###     hand_link_local_position (17+17=34) * 3
###     body_link_local_velocity 25 * 3  
###     hand_link_local_velocity 34 * 3  
###     body_dof_angle 3+7+7=17
###     hand_dof_angle 12 + 12 = 24




### reference: https://github.com/EricGuo5513/HumanML3D/blob/main/motion_representation.ipynb

### note that  "global_rotation" and "global_translation" may be changed for normalization

### usage: python src/utils/io/inspirehand_representation.py <pickle file>
### example: python src/utils/io/inspirehand_representation.py data/motion/g1/HumanML3D/000000.pickle


import sys
import os
from HRI_mllm.external.HRI_retarget.HRI_retarget.config.joint_mapping import G1_LINKS, G1_INSPIREHANDS_LINKS
from HRI_retarget.model.g1_29 import G1_29_Motion_Model
from HRI_retarget.model.g1_inspirehands import G1_Inspirehands_Motion_Model
from HRI_retarget.utils.motion_lib.quaternion import qbetween_np, qinv, qinv_np, qmul_np, qrot, qrot_np
from HRI_retarget.utils.torch_utils.diff_quat import quat_to_matrix, vec6d_to_matrix, vec6d_to_quat, quat_to_vec6d

import torch
from tqdm import tqdm
import pickle

import ipdb

import numpy as np


def data_pkl_to_vec(data_dict):
    """
    Convert the data_dict to vec representation
    :param data_dict: dict, contains the data
    :return: vec, the vec representation
    """

    angles = np.unwrap(data_dict["angles"])
    num_frames = angles.shape[0]

    source_fps = data_dict["fps"]
    target_fps = 50

    ### resample if fps is not equal



    if source_fps != target_fps:
        print(f"source fps {source_fps} is not equal to target fps {target_fps}, resampling...")
        from scipy import interpolate
        ### resample 
        num_target_frames = int(num_frames * target_fps / source_fps)

        interpolated_data = np.zeros((num_target_frames, angles.shape[1]))
        t_original = np.linspace(0, num_frames - 1, num_frames) 
        t_target = np.linspace(0, num_frames - 1, num_target_frames)     

        # 4. 对每个自由度（每一列）进行插值
        for dof_idx in range(angles.shape[1]):
            # 获取这个DOF的原始80个数据点
            y_original = angles[:, dof_idx]
            
            # 创建线性插值函数
            f = interpolate.interp1d(t_original, y_original, kind='cubic', fill_value='extrapolate')
            
            # 在目标时间轴上计算插值后的值
            y_interpolated = f(t_target)
            
            # 将结果存入数组
            interpolated_data[:, dof_idx] = y_interpolated
        angles = interpolated_data
        num_frames = angles.shape[0]

    match data_dict["robot_name"]:
        case "g1_inspirehands":
            model = G1_Inspirehands_Motion_Model(num_frames)
            robot_link = G1_INSPIREHANDS_LINKS
        case _:
            print("wrong robot name in kinematic vis")
            quit()

    ### set unused dofs to zero
    angles = torch.from_numpy(angles).float()
    angles[:, :12] *= 0
    
    model.set_angles(angles)
  




    link_to_root_dict = model.forward_kinematics()
    link_to_root_pos = link_to_root_dict[:, :, :3, 3]
    local_positions = link_to_root_pos.detach().cpu().numpy()



    

    '''Get Joint Rotation Representation'''
    # (seq_len, dof) dof for skeleton joints
    rot_data = angles.detach().cpu().numpy()


    '''Get Joint Velocity Representation'''
    # (seq_len-1, (link-1)*3)
    local_vel = local_positions[1:] - local_positions[:-1]


    body_vec = np.concatenate([
        local_positions[:-1, 14:31, :].reshape(num_frames-1, -1),  # body_link
        local_positions[:-1, 48:56, :].reshape(num_frames-1, -1),  # body_link
        local_positions[:-1, 31:48, :].reshape(num_frames-1, -1),  # left_hand_link
        local_positions[:-1, 56:73, :].reshape(num_frames-1, -1),  # right_hand_link

        local_vel[:, 14:31, :].reshape(num_frames-1, -1),  # body_vel
        local_vel[:, 48:56, :].reshape(num_frames-1, -1),  # body_vel
        local_vel[:, 31:48, :].reshape(num_frames-1, -1),  # left_hand_vel
        local_vel[:, 56:73, :].reshape(num_frames-1, -1),  # right_hand_vel

        rot_data[:-1, 12:22], # waist and left arm
        rot_data[:-1, 34:41], # right arm
        rot_data[:-1, 22:34], # left hand
        rot_data[:-1, 41:53], # right hand
    ], axis=-1)


    ### todo: hand vec

    import ipdb;ipdb.set_trace()

    return body_vec



def vec_to_data_pkl(body_vec, fps=50, reference_motion_pth=None, robot_name="g1_inspirehands", scale=np.ones(3)):
    """
    Convert the vec representation to data_dict
    :param vec: vec, the vec representation
    :return: data_dict, the data_dict
    """
    
    ### TODO:
    assert body_vec.ndim == 2
    assert body_vec.shape[1] == 395
    assert isinstance(body_vec, np.ndarray), f"body_vec 应该是 NumPy 数组，但实际类型是 {type(body_vec)}"

    num_frames = body_vec.shape[0]
    global_positions = np.zeros((num_frames, 3))
    global_positions[:, 2] += 0.8 # height

    global_rotations = torch.from_numpy(np.broadcast_to(np.eye(3), (num_frames, 3, 3)).copy())[..., :3, :2].reshape(-1, 3, 2).numpy()




    dof_angles = np.zeros((num_frames, 53))
    dof_angles[:, 12:22] = body_vec[:, 354:364]  # waist and left arm
    dof_angles[:, 34:41] = body_vec[:, 364:371]  # right arm
    dof_angles[:, 22:34] = body_vec[:, 371:383] # left hand 
    dof_angles[:, 41:53] = body_vec[:, 383:395] # right handx

    data_dict = {
        "fps": fps,
        "reference_motion_pth": reference_motion_pth,
        "robot_name": robot_name,
        "angles": dof_angles,
        "global_rotation": global_rotations,
        "global_translation": global_positions,
        "scale": scale,
    }



    ### rot_data = data_dict["angles"]
    return data_dict


if __name__ == "__main__":

    # # if len(sys.argv) != 2:
    # #     print('Call the function with the Pickle file')
    # #     quit()
    
    # # filename = sys.argv[1]
    # filename = "/home/pengyang/codebase/HRI_retarget/HumanML3D/000093.pickle"
    # with open(filename, "rb") as file:
    #     data_dict = pickle.load(file)

    # vec = data_pkl_to_vec(data_dict)

    # data = vec_to_data_pkl(vec)
    # print(data)


    # with open("/home/pengyang/codebase/HRI_retarget/HumanML3D/test.pickle", "wb") as file:
    #     pickle.dump(data, file)



    filename = "/root/pengyang/codebase/HRI_MLLM/data/BEAT_v1/1/1_wayne_0_1_1.pickle"
    with open(filename, "rb") as file:
        data_dict = pickle.load(file)

    import ipdb;ipdb.set_trace()
    vec = data_pkl_to_vec(data_dict)




