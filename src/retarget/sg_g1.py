### usage:
### python sg_galbot.py {path_to_bvh_file}
### python sg_galbot.py /home/pengyang/data/human/SG/output.bvh
### todo: 
###   tune magic numbers

import sys
import os
sys.path.append("/home/pengyang/codebase/H1_RL/src")

import torch
from tqdm import tqdm
import pickle

from utils.vis.bvh_vis import Get_bvh_joint_local_coord
from utils.vis.kinematic_vis import vis_kinematic_result
from src.model.g1_15 import G1_15_Motion_Model
from config.joint_mapping import SG_LINKS, SG_G1_CORRESPONDENCE


### magic numbers
### transition from sg to galbot
rot = torch.tensor([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
], dtype=torch.float)
# # pos = torch.tensor([0, 0, 0.8], dtype=torch.float)
# scale = 1.5

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print('Call the function with the BVH file')
        quit()

    filename = sys.argv[1]
    bvh_joint_local_coord = Get_bvh_joint_local_coord(filename, link_list=SG_LINKS)

   
    num_frames = len(bvh_joint_local_coord)
    print("Num of frames: ", num_frames)
    
    model = G1_15_Motion_Model(batch_size=num_frames, joint_correspondence=SG_G1_CORRESPONDENCE)



    print(bvh_joint_local_coord.shape)

    # model.set_gt_joint_positions((bvh_joint_local_coord) @ rot.T @ scale + pos)
    model.set_gt_joint_positions(bvh_joint_local_coord @ rot.T)
    print("Links of robot: ", model.chain.get_link_names())

    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    model.train()

    pbar = tqdm(range(2000))
    for epoch in pbar:
        
        joint_local_velocity_loss, joint_local_accel_loss = model.joint_local_velocity_loss()
        joint_global_position_loss = model.retarget_joint_loss()
        dof_limit_loss = model.dof_limit_loss()
        # init_angle_loss = model.init_angle_loss()
        # elbow_loss = model.elbow_loss()


        loss_dict = {
            "joint_global_position_loss": [1.0, joint_global_position_loss],
            "joint_local_velocity_loss": [1.0, joint_local_velocity_loss],
            "joint_local_accel_loss": [0.0, joint_local_accel_loss],
            "dof_limit_loss": [1.0, dof_limit_loss],
        }

        loss = 0
        log_str = "#" * 50 + "\n"
        for loss_name in loss_dict.keys():
            loss += loss_dict[loss_name][0] * loss_dict[loss_name][1]
            log_str += f"{loss_name}: {loss_dict[loss_name][0] * loss_dict[loss_name][1].item()}" + "\n"
        pbar.set_description(log_str)  
        print("dof_limit_loss", dof_limit_loss.item())


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred_joint_angles = model.joint_angles.detach().cpu().numpy()
        global_rotation = model.global_rot.detach().cpu().numpy()
        global_translation = model.global_trans.detach().cpu().numpy()
        scale = model.scale.detach().cpu().numpy()

    data_dict = {
        "angles": pred_joint_angles,
        "global_rotation": global_rotation,
        "global_translation": global_translation,
        "scale": scale,
    }

    with open(os.path.join("/home/pengyang/data/motion/g1/SG", filename.split("/")[-1][:-4] + ".pickle"), "wb") as file:
        pickle.dump(data_dict, file)
    
    vis_kinematic_result(os.path.join("/home/pengyang/data/motion/g1/SG", filename.split("/")[-1][:-4] + ".pickle"), dataset="SG", robot="g1", correspondence=SG_G1_CORRESPONDENCE)
