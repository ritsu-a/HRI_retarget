### usage: visualize galbot and bvh at the same time
### specific to /data/SeG_dataset results
### python kinematic_vis.py data/SeG_dataset/galbot_motion/ARMS_SELF_EMBRACE-1.pickle

import torch
import time
import pickle 
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pytorch_kinematics as pk
from HRI_retarget import ROOT,SRC_ROOT,DATA_ROOT
sys.path.append(SRC_ROOT)
# sys.path.append("/home/pengyang/codebase/H1_RL/src")

from config.joint_mapping import GALBOT_CHARLIE_LINKS, G1_LINKS,SG_LINKS, SEG_LINKS, SMPL_LINKS, SG_GALBOT_CHARLIE_CORRESPONDENCE, SG_G1_CORRESPONDENCE, SMPL_G1_CORRESPONDENCE
from utils.vis.bvh_vis import Draw_bvh_frame, ProcessBVH, Get_bvh_joint_local_coord
from model.galbot_charlie import Galbot_Charlie_Motion_Model
from model.g1_15 import G1_15_Motion_Model


def Draw_bvh_urdf(bvh_link_pos, bvh_skeleton_data, urdf_link_pos, urdf_chain, reference_link=SG_LINKS, robot_link=G1_LINKS, correspondence=SG_G1_CORRESPONDENCE):

    bvh_joints = bvh_skeleton_data[0]
    bvh_joints_hierarchy = bvh_skeleton_data[2]
    fig = plt.figure(figsize=(8, 6), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
   

    """
    Number of frames skipped is controlled with this variable below. If you want all frames, set to 1.
    """
    frame_skips = 1

    figure_limit = 2 #used to set figure axis limits

    
    idx = 0
    while True:
        idx += 1
        i = idx % len(bvh_link_pos) 
        
        if isinstance(bvh_link_pos[i], torch.Tensor):
            bvh_pos = bvh_link_pos[i].detach().cpu().numpy()
        else:
            bvh_pos = bvh_link_pos[i]
        #calculate the limits of the figure. Usually the last joint in the dictionary is one of the feet.
        if figure_limit == None:
            lim_min = np.abs(np.min(bvh_pos[-1]))
            lim_max = np.abs(np.max(bvh_pos[-1]))
            lim = lim_min if lim_min > lim_max else lim_max
            figure_limit = lim
        
        
        ### visualizeing bvh
        for joint in bvh_joints:
        
            if joint == bvh_joints[0]: continue #skip root joint
            parent_joint = bvh_joints_hierarchy[joint][0]
            plt.plot(xs = [bvh_pos[reference_link.index(parent_joint)][0], bvh_pos[reference_link.index(joint)][0]],
                     zs = [bvh_pos[reference_link.index(parent_joint)][1], bvh_pos[reference_link.index(joint)][1]],
                     ys = [bvh_pos[reference_link.index(parent_joint)][2], bvh_pos[reference_link.index(joint)][2]], c = 'blue', lw = 2.5)

        ### visualizing urdf 
        frames_to_draw = [robot_link[0]]
        urdf_pos = urdf_link_pos[i].detach().cpu().numpy()

        while frames_to_draw != []:
            link = frames_to_draw[0]
            frames_to_draw.pop(0)
            for child_frame in urdf_chain.find_frame(link).children:
                child_link = child_frame.name 
                frames_to_draw.append(child_link)
                plt.plot(xs = [urdf_pos[robot_link.index(link)][1] + 0.5, urdf_pos[robot_link.index(child_link)][1] + 0.5],
                         zs = [urdf_pos[robot_link.index(link)][2], urdf_pos[robot_link.index(child_link)][2]],
                         ys = [urdf_pos[robot_link.index(link)][0], urdf_pos[robot_link.index(child_link)][0]],c = 'red', lw = 2.5)


        ### visualizing corespondence
        for ii, jj, v in correspondence:
            plt.plot(xs = [urdf_pos[jj][1] + 0.5, bvh_pos[ii][0]],
                    zs = [urdf_pos[jj][2], bvh_pos[ii][1]],
                    ys = [urdf_pos[jj][0], bvh_pos[ii][2]],c = 'green', lw = 2.5)

            #uncomment here if you want to see the world coords. If nothing appears on screen, change the axis limits below!
            # plt.plot(xs = [world_pos[parent_joint][0], world_pos[joint][0]],
            #          zs = [world_pos[parent_joint][1], world_pos[joint][1]],
            #          ys = [world_pos[parent_joint][2], world_pos[joint][2]], c = 'red', lw = 2.5)

        #Depending on the file, the axis limits might be too small or too big. Change accordingly.
        ax.set_axis_off()
        ax.set_xlim(-0.6*figure_limit, 0.6*figure_limit)
        ax.set_ylim(-0.6*figure_limit, 0.6*figure_limit)
        ax.set_zlim(-0.2*figure_limit, 1.*figure_limit)
        plt.title('frame: ' + str(i))
        plt.pause(0.01)
        ax.cla()

    pass



def vis_kinematic_result(filename, dataset="SG", robot="g1", correspondence=SG_G1_CORRESPONDENCE):
    ### loading estimated joint angles
    with open(filename, "rb") as file:
        data_dict = pickle.load(file)
        joints_angle = data_dict["angles"]

    
    ### loading dataset
    if dataset == "SG":
        reference_link = SG_LINKS 
    elif dataset == "SeG":
        reference_link = SEG_LINKS
    elif dataset == "MDM":
        reference_link = SMPL_LINKS

    ### loading bvh data
    if dataset in ["SG", "SeG"]:
        bvh_path = os.path.join(DATA_ROOT,f"/motion/human/{dataset}", filename.split("/")[-1][:-7] + ".bvh")
        skeleton_data = ProcessBVH(bvh_path)
        bvh_joint_local_coord = Get_bvh_joint_local_coord(bvh_path, link_list=reference_link)
        num_frames = len(bvh_joint_local_coord)
        print("Num of frames: ", num_frames)

    elif dataset in ["MDM"]:
        ### creating pseudo skeleton for smpl-like joints
        npy_path = os.path.join(DATA_ROOT,f"/motion/human/{dataset}", filename.split("/")[-1][:-7] + ".npy")
        joint_data = np.load(npy_path)
        skeleton_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
        skeleton = {}
        for chain in skeleton_chain:
            for idx, link_idx in enumerate(chain):
                if idx == 0:
                    continue
                skeleton[SMPL_LINKS[chain[idx]]] = [SMPL_LINKS[chain[idx-1]]]
        skeleton_data = [SMPL_LINKS, None, skeleton]
        num_frames = len(joint_data)
        bvh_joint_local_coord = joint_data


    ### loading galbot model 
    if "galbot" in robot:
        model = Galbot_Charlie_Motion_Model(num_frames)
        robot_link = GALBOT_CHARLIE_LINKS
    elif "g1" in robot:
        model = G1_15_Motion_Model(num_frames)
        robot_link = G1_LINKS
    else:
        print("wrong robot name in kinematic vis")

    model.set_angles(torch.tensor(joints_angle))
    model.set_global_matrix(data_dict)

    link_to_root_dict = model.forward_kinematics()
    link_to_root_pos = link_to_root_dict[:, :, :3, 3]


    Draw_bvh_urdf(bvh_joint_local_coord, skeleton_data, link_to_root_pos, model.chain, reference_link=reference_link, robot_link=robot_link, correspondence=correspondence)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Call the function with the motion file')
        filename = os.path.join(DATA_ROOT,"motion/g1/SG/output.pickle")

    else:
        filename = sys.argv[1]
    vis_kinematic_result(filename)






