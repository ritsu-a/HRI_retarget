### usage:
### python beat_g1_inspirehands.py {path_to_npy_file}
### python src/retarget/hand_retarget.py data/motion/human/BEAT_ZIP/beat_english_v0.2.1/1/1_wayne_0_1_1.bvh
# 2025.05.03
# retarget body motion + hand motion

import sys
import os
from HRI_retarget import ROOT,SRC_ROOT,DATA_ROOT
sys.path.append(SRC_ROOT)
sys.path.append(ROOT)

import torch
from tqdm import tqdm
import pickle

import numpy as np

from utils.vis.bvh_vis import Get_bvh_joint_local_coord, Get_bvh_joint_local_coord_parallel, calc_relative_transform
from utils.vis.kinematic_vis import vis_kinematic_result
from src.model.g1_inspirehands import G1_Inspirehands_Motion_Model
from config.joint_mapping import BBDB_LINKS, BBDB_G1_INSPIREHANDS_CORRESPONDENCE, \
    BBDB_LEFT_HAND_LINK, BBDB_RIGHT_HAND_LINK
from utils.vis.bvh_vis import Rx, Ry, Rz

import matplotlib.pyplot as plt
from dex_retargeting.retargeting_config import RetargetingConfig
from pathlib import Path
import yaml
import pandas as pd


### magic numbers
### transition from sg to galbot
# rot = torch.tensor([
#     [0, 0, 1],
#     [1, 0, 0],
#     [0, 1, 0],
# ], dtype=torch.float)
# rot = Rz(np.pi/2,in_radians=True) @ Rx(-np.pi/2,in_radians=True)
# rot = torch.from_numpy(rot.T).type(torch.float)
rot = torch.eye(3)
# left_hand_to_inspire = Rx(np.pi/2,in_radians=True)
# right_hand_to_inspire = Rz(np.pi,in_radians=True) @ Rx(np.pi/2,in_radians=True)
left_hand_to_inspire = np.array([[1,0,0],[0,0,1],[0,-1,0]])
right_hand_to_inspire = np.array([[-1,0,0],[0,0,1],[0,1,0]])


config_file_path = os.path.join(DATA_ROOT, "resources/robots/g1_inspirehands/inspire_hand.yml")
RetargetingConfig.set_default_urdf_dir(os.path.join(DATA_ROOT,"resources/robots/g1_inspirehands"))
with Path(config_file_path).open('r') as f:
    cfg = yaml.safe_load(f)
left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
left_retargeting = left_retargeting_config.build()
right_retargeting = right_retargeting_config.build()

# print("left joint names: ", left_retargeting.joint_names)
# print("right_joint_names: ", right_retargeting.joint_names)

def fingerpos_clip(qpos):
    limits = np.array([
        [-0.1, 1.3], ## L_thumb_proximal_yaw_joint
        [-0.1, 0.6], ##L_thumb_proximal_pitch_joint
        [0, 0.8], ##L_thumb_intermediate_joint
        [0, 1.2], ##L_thumb_distal_joint
        [0, 1.7], ##L_index_proximal_joint
        [0, 1.7], ##L_index_intermediate_joint
        [0, 1.7], ##L_middle_proximal_joint
        [0, 1.7], ##L_middle_intermediate_joint
        [0, 1.7], ##L_ring_proximal_joint
        [0, 1.7], ##L_ring_intermediate_joint
        [0, 1.7], ##L_pinky_proximal_joint
        [0, 1.7], ##L_pinky_intermediate_joint
    ])
    qpos[qpos < limits[:,0]] = limits[:,0][qpos < limits[:,0]]
    qpos[qpos > limits[:,1]] = limits[:,1][qpos > limits[:,1]]
    return qpos


if __name__ == "__main__":

    # if len(sys.argv) != 2:
    #     print('Call the function with the BVH file')
    #     quit()

    # filename = sys.argv[1]
    filename = os.path.join(DATA_ROOT, "motion/human/misc/suisei_vivideba_motion_.bvh")
    # bvh_joint_local_coord = Get_bvh_joint_local_coord_parallel(filename, link_list=BBDB_LINKS)
    bvh_joint_local_coord, bvh_joint_local_rot = Get_bvh_joint_local_coord_parallel(filename, link_list = BBDB_LINKS)

    num_frames = len(bvh_joint_local_coord)
    print("Num of frames: ", num_frames)
    
    model = G1_Inspirehands_Motion_Model(batch_size=num_frames, joint_correspondence=BBDB_G1_INSPIREHANDS_CORRESPONDENCE)



    # print(bvh_joint_local_coord.shape)

    model.set_gt_joint_positions(bvh_joint_local_coord @ rot.T)
    print("Links of robot: ", model.chain.get_link_names())

    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    model.train()

    history_losses = []
    
    pbar = tqdm(range(200))
    for epoch in pbar:
        
        ### normalize
        with torch.no_grad():
            model.normalize()
    
        joint_local_velocity_loss, joint_local_accel_loss = model.joint_local_velocity_loss()
        joint_global_position_loss = model.retarget_joint_loss()
        dof_limit_loss = model.dof_limit_loss()
        # collision_loss = model.collision_loss()
        # init_angle_loss = model.init_angle_loss()
        # elbow_loss = model.elbow_loss()


        loss_dict = {
            "joint_global_position_loss": [1.0, joint_global_position_loss],
            "joint_local_velocity_loss": [1.0, joint_local_velocity_loss],
            "joint_local_accel_loss": [0.0, joint_local_accel_loss],
            "dof_limit_loss": [1.0, dof_limit_loss],
            # "collision_loss": [1.0, collision_loss],
        }

        loss = 0
        log_str = "#" * 50 + "\n"
        for loss_name in loss_dict.keys():
            loss += loss_dict[loss_name][0] * loss_dict[loss_name][1]
            log_str += f"{loss_name}: {loss_dict[loss_name][0] * loss_dict[loss_name][1].item()}" + "\n"
        # pbar.set_description(log_str)  
        # print("dof_limit_loss", dof_limit_loss.item())
        # print("collision_loss", collision_loss.item())

        pbar.set_description(f"loss:, {loss.item()}")
        history_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        

    with torch.no_grad():
        
        pred_joint_angles = model.joint_angles.detach().cpu().numpy()
        global_rotation = model.global_rot.detach().cpu().numpy()
        global_translation = model.global_trans.detach().cpu().numpy()
        scale = model.scale.detach().cpu().numpy()
        
    # 2025.04.28 HIT-xiaowangzi 
    # Second step : refine the hand position using dex-retargeting
    left_rel_pos = torch.zeros([num_frames,5,3])
    right_rel_pos = torch.zeros([num_frames, 5,3])
    left_root_id = BBDB_LEFT_HAND_LINK["base_link"]
    left_tip_id_list = BBDB_LEFT_HAND_LINK["tip_link"]
    right_root_id = BBDB_RIGHT_HAND_LINK["base_link"]
    right_tip_id_list = BBDB_RIGHT_HAND_LINK["tip_link"]
    left_rot_batch = torch.from_numpy(left_hand_to_inspire).view(1,3,3).repeat(num_frames,1,1).type(torch.float)
    right_rot_batch = torch.from_numpy(right_hand_to_inspire).view(1,3,3).repeat(num_frames,1,1).type(torch.float)
        
    for i,tip_id in enumerate(left_tip_id_list):
        pos, rot = calc_relative_transform(bvh_joint_local_coord, bvh_joint_local_rot, left_root_id, tip_id)
        # pos[:,0,:] = -pos[:,0,:]
        left_rel_pos[:,i,:] = torch.bmm(left_rot_batch,pos).squeeze(2)
    for i,tip_id in enumerate(right_tip_id_list):
        pos, rot = calc_relative_transform(bvh_joint_local_coord, bvh_joint_local_rot, right_root_id, tip_id)
        # pos[:,0,:] = -pos[:,0,:]
        right_rel_pos[:,i,:] = torch.bmm(right_rot_batch,pos).squeeze(2)
    
    left_qpos_list = np.zeros((num_frames,12))
    right_qpos_list = np.zeros((num_frames,12))
    left_last_qpos = np.zeros((12,1))
    right_last_qpos = np.zeros((12,1))
    for i in range(num_frames):
        left_ref = left_rel_pos[i,:,:].numpy()
        right_ref = right_rel_pos[i,:,:].numpy()
        left_qpos = left_retargeting.retarget(left_ref)
        right_qpos = right_retargeting.retarget(right_ref)
        # left_qpos_list[i,:] = left_qpos[[8,9,10,11,0,1,2,3,6,7,4,5]]
        # right_qpos_list[i,:] = right_qpos
        pred_joint_angles[i,22:34] = fingerpos_clip(left_qpos[[1,2,3,4,5,6,7,8,9,10,11,12]])
        pred_joint_angles[i,41:53] = fingerpos_clip(right_qpos[[1,2,3,4,5,6,7,8,9,10,11,12]])
    # print("left_qpos_list: ", left_qpos_list)
    # print("left_qpos_list shape: ", left_qpos_list.shape)      
    # print("right_qpos_list: ", right_qpos_list) 
    # print("right_qpos_list shape: ", right_qpos_list.shape) 
    
    left_df = pd.DataFrame(left_qpos_list)
    right_df = pd.DataFrame(right_qpos_list)
    left_ref_df = pd.DataFrame(left_rel_pos.reshape(-1,15))
    right_ref_df = pd.DataFrame(right_rel_pos.reshape(-1,15))
    # left_df.columns = ['L_thumb_proximal_yaw_joint', 'L_thumb_proximal_pitch_joint',
    #                    'L_thumb_intermediate_joint', 'L_thumb_distal_joint',
    #                    'L_index_proximal_joint', 'L_index_intermediate_joint',
    #                    'L_middle_proximal_joint', 'L_middle_intermediate_joint',
    #                    'L_ring_proximal_joint', 'L_ring_intermediate_joint',
    #                    'L_pinky_proximal_joint', 'L_pinky_intermediate_joint',
    #                    ]
    # right_df.columns = ['R_thumb_proximal_yaw_joint', 'R_thumb_proximal_pitch_joint',
    #                     'R_thumb_intermediate_joint', 'R_thumb_distal_joint',
    #                     'R_index_proximal_joint', 'R_index_intermediate_joint',
    #                     'R_middle_proximal_joint', 'R_middle_intermediate_joint',
    #                     'R_ring_proximal_joint', 'R_ring_intermediate_joint',
    #                     'R_pinky_proximal_joint', 'R_pinky_intermediate_joint',
    #                     ]
    # left_ref_df.columns = ["thumb_x","thumb_y","thumb_z",
    #                        "index_x","index_y","index_z",
    #                        "middle_x","middle_y","middle_z",
    #                        "ring_x","ring_y","ring_z",
    #                        "pinky_x","pinky_y","pinky_z"]
    # right_ref_df.columns = ["thumb_x","thumb_y","thumb_z",
    #                         "index_x","index_y","index_z",
    #                         "middle_x","middle_y","middle_z",
    #                         "ring_x","ring_y","ring_z",
    #                         "pinky_x","pinky_y","pinky_z"]
    # left_df.to_csv(os.path.join(ROOT,"log","left_finger_joints.csv"))
    # right_df.to_csv(os.path.join(ROOT,"log","right_finger_joints.csv"))
    # left_ref_df.to_csv(os.path.join(ROOT,"log","left_fingertip_pos.csv"))
    # right_ref_df.to_csv(os.path.join(ROOT,"log","right_fingertip_pos.csv"))
    

    data_dict = {
        "fps": 120,
        "reference_motion_pth": filename,
        "robot_name": "g1_inspirehands",
        "angles": pred_joint_angles,
        "global_rotation": global_rotation,
        "global_translation": global_translation,
        "scale": scale,
    }

    with open(os.path.join(DATA_ROOT,"motion/g1/BBDB", filename.split("/")[-1][:-4] + ".pickle"), "wb") as file:
        pickle.dump(data_dict, file)
        
    joints_df = pd.DataFrame(pred_joint_angles)
    joints_df.to_csv(os.path.join(ROOT,"log","g1_inspirehands_joints.csv"))

    

    # ### visualize results.

    # ### draw loss curve
    # plt.plot(history_losses[len(history_losses) // 10:], label='Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Curve')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    # ### vis motion
    # ### press esc to quit plt visualization

    # vis_kinematic_result(os.path.join(DATA_ROOT,"motion/g1/BBDB", filename.split("/")[-1][:-4] + ".pickle"), dataset="BBDB", robot="g1_inspirehands", correspondence=BBDB_G1_INSPIREHANDS_CORRESPONDENCE)
