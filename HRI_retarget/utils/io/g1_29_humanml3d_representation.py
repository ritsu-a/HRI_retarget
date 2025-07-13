### converting 29 dof g1 to humanml3d vec representation 
### for g1_29, dof=29, num_links=41
### shape (num_frames - 1, 280)
###     root_rot_quat 4(wxyz)
###     root_linear_velocity 2
###     root_y 1
###     link_local_position (41 - 1) * 3
###     dof_angle 29
###     link_local_velocity (41 - 1) * 3  
###     foot_contact 4

### reference: https://github.com/EricGuo5513/HumanML3D/blob/main/motion_representation.ipynb

### note that  "global_rotation" and "global_translation" may be changed for normalization

### usage: python src/utils/io/g1_29_humanml3d_representation.py <pickle file>
### example: python src/utils/io/g1_29_humanml3d_representation.py data/motion/g1/HumanML3D/000000.pickle


import sys
import os
from HRI_retarget.model.g1_29 import G1_29_Motion_Model
from HRI_retarget.config.joint_mapping import G1_LINKS
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
    num_frames = data_dict["angles"].shape[0]

    match data_dict["robot_name"]:
        case "g1_29":
            model = G1_29_Motion_Model(num_frames)
            robot_link = G1_LINKS
        case _:
            print("wrong robot name in kinematic vis")
            quit()

    model.set_angles(torch.tensor(data_dict["angles"]))
  

    # # # data_dict["global_rotation"] = (vec6d_to_matrix(torch.tensor(data_dict["global_rotation"]).to("cuda:0")) @ rot)[:, :, :2].detach().cpu().numpy()
    # # # data_dict["global_translation"] = (torch.tensor(data_dict["global_translation"]).to("cuda:0")[:, :, 0] @ rot).detach().cpu().numpy()[:, :, None]
    # # # ipdb.set_trace()


    link_to_root_dict = model.forward_kinematics()
    link_to_root_pos = link_to_root_dict[:, :, :3, 3]
    local_positions = link_to_root_pos.detach().cpu().numpy()

    torch_data_dict = {}
    for key in data_dict:
        if isinstance(data_dict[key], np.ndarray):
            torch_data_dict[key] = torch.tensor(data_dict[key])
        else:
            torch_data_dict[key] = data_dict[key]
    model.set_global_matrix(torch_data_dict)
    link_to_root_dict = model.forward_kinematics()
    link_to_root_pos = link_to_root_dict[:, :, :3, 3]
    positions = link_to_root_pos.detach().cpu().numpy()

    
    
    

    ### data normalization from humanml3d motion_representation.ipynb
    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[2]  # Now Z is height
    positions[:, :, 2] -= floor_height

    local_floor_height = local_positions.min(axis=0).min(axis=0)[2]
    local_positions[:, :, 2] -= local_floor_height

    '''XY at origin'''  # Now we work in XY plane (Z is up)
    root_pos_init = positions[0]

    root_pose_init_xy = root_pos_init[0] * np.array([1, 1, 0])
    positions = positions - root_pose_init_xy

    local_root_pos_init = local_positions[0]
    local_root_pose_init_xy = local_root_pos_init[0] * np.array([1, 1, 0])
    local_positions = local_positions - local_root_pose_init_xy

    '''All initially face X+'''
    r_hip, l_hip, sdr_r, sdr_l = [G1_LINKS.index(link) for link in ["right_hip_pitch_link", "left_hip_pitch_link", "right_shoulder_pitch_link", "left_shoulder_pitch_link"]]
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), now pointing to X+
    forward_init = np.cross(np.array([[0, 0, 1]]), across, axis=-1)  # Z-up cross product
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[1, 0, 0]])  # Target is now X+
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init, positions)

    # local
    local_across1 = local_root_pos_init[r_hip] - local_root_pos_init[l_hip]
    local_across2 = local_root_pos_init[sdr_r] - local_root_pos_init[sdr_l]
    local_across = local_across1 + local_across2
    local_across = local_across / np.sqrt((local_across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around z-axis
    local_forward_init = np.cross(np.array([[0, 0, 1]]), local_across, axis=-1)
    # forward (3,)
    local_forward_init = local_forward_init / np.sqrt((local_forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    local_target = np.array([[1, 0, 0]])
    local_root_quat_init = qbetween_np(local_forward_init, local_target)
    local_root_quat_init = np.ones(local_positions.shape[:-1] + (4,)) * local_root_quat_init


    local_positions = qrot_np(local_root_quat_init, local_positions)
    
    
    '''Root height'''
    root_z = data_dict["global_translation"][:, 2:3, 0]  # Now using Z for height

    '''Root rotation and linear velocity'''
    r_rot = vec6d_to_quat(data_dict["global_rotation"]).numpy()
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
 
    velocity = data_dict["global_translation"][1:, :, 0] - data_dict["global_translation"][:-1, :, 0]
    l_velocity = velocity[:, [0, 1]]  # Now using XY velocity

    root_data = np.concatenate([r_velocity, l_velocity, root_z[:-1]], axis=-1)







    '''New ground truth positions'''

    fid_l, fid_r =  [G1_LINKS.index(link) for link in ["left_ankle_pitch_link", "left_ankle_roll_link"]], [G1_LINKS.index(link) for link in ["right_ankle_pitch_link", "right_ankle_roll_link"]]
    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, thres=0.02)

    

    '''Get Joint Rotation Representation'''
    # (seq_len, dof) dof for skeleton joints
    rot_data = data_dict["angles"]

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (link-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, (link-1)*3)
    local_vel = local_positions[1:] - local_positions[:-1]
    local_vel = local_vel[:, 1:].reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(data.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)


    return data


def recover_root_rot_pos(data):
    """
    Recover root rotation and position from encoded data in X+ forward, Z-up coordinate system
    
    Args:
        data: [..., n_features] tensor where:
            data[..., 0] = rotation velocity (around Z-axis)
            data[..., 1:3] = XY linear velocity
            data[..., 3] = Z height
    
    Returns:
        r_rot_quat: [..., 4] quaternion rotation (w,x,y,z)
        r_pos: [..., 3] 3D position
    """


    r_velocity = data[:, 0:4]  
    l_velocity = data[:, 4:6]  # XY速度
    root_z = data[:, 6:7]       # 高度
    
    # 还原translation
    restored_translation = np.zeros((len(data)+1, 3))
    restored_translation[0, :] = [0, 0, root_z[0,0]]
    restored_translation[1:, 0:2] = np.cumsum(l_velocity, axis=0) + restored_translation[0, 0:2]
    restored_translation[1:, 2] = root_z[:, 0]
    
    # 还原rotation
    restored_rotation = np.zeros((len(data)+1, 4))
    restored_rotation[0] = np.array([0, 0, 0, 1])  # 初始四元数 (w, x, y, z)
    
    for i in range(1, len(restored_rotation)):
        delta_q = r_velocity[i-1]
        restored_rotation[i] = qmul_np(delta_q, restored_rotation[i-1])
    
    return restored_translation[1:], restored_rotation[1:]

def vec_to_data_pkl(vec, fps=20, reference_motion_pth=None, robot_name="g1_29", scale=np.ones(3)):
    """
    Convert the vec representation to data_dict
    :param vec: vec, the vec representation
    :return: data_dict, the data_dict
    """
    global_positions, global_rotations_quat = recover_root_rot_pos(vec)

    global_positions = global_positions.reshape(-1, 3, 1)
    global_rotations = quat_to_matrix(torch.from_numpy(global_rotations_quat))[..., :3, :2].reshape(-1, 3, 2).numpy()

    device = vec.device


    joints_num = 29 
    links_num = 41
    batch_size = vec.shape[0]
    num_frames = vec.shape[1]

    dof_angles = vec[..., 7 + (links_num - 1) * 3: 7 + (links_num - 1) * 3 + joints_num].reshape(-1, joints_num)


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

    # if len(sys.argv) != 2:
    #     print('Call the function with the Pickle file')
    #     quit()
    
    # filename = sys.argv[1]
    filename = "/home/pengyang/codebase/HRI_retarget/HumanML3D/000093.pickle"
    with open(filename, "rb") as file:
        data_dict = pickle.load(file)

    vec = data_pkl_to_vec(data_dict)

    data = vec_to_data_pkl(vec)
    print(data)


    with open("/home/pengyang/codebase/HRI_retarget/HumanML3D/test.pickle", "wb") as file:
        pickle.dump(data, file)


    




