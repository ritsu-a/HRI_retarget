### usage: visualize galbot and bvh at the same time
### specific to /data/SeG_dataset results
### python kinematic_vis.py /home/pengyang/codebase/H1_RL/data/SeG_dataset/galbot_motion/ARMS_SELF_EMBRACE-1.pickle
### todo: 
###   sort magic numbers in aligning bvh and galbot

import torch
import time
import pickle 
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pytorch_kinematics as pk
sys.path.append("/home/pengyang/codebase/H1_RL/src")

from config.joint_mapping import GALBOT_CHARLIE_LINKS, SEG_LINKS
from utils.bvh_vis import Draw_bvh_frame, ProcessBVH, Get_bvh_joint_local_coord
from model.galbot_charlie import Galbot_Charlie_Motion_Model


def Draw_bvh_urdf(bvh_link_pos, bvh_skeleton_data, urdf_link_pos, urdf_chain):

    bvh_joints = bvh_skeleton_data[0]
    bvh_joints_hierarchy = bvh_skeleton_data[2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
   

    """
    Number of frames skipped is controlled with this variable below. If you want all frames, set to 1.
    """
    frame_skips = 1

    figure_limit = 1 #used to set figure axis limits

    
    idx = 0
    while True:
        idx += 1
        i = idx % len(bvh_link_pos) 
        
        bvh_pos = bvh_link_pos[i].detach().cpu().numpy()
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
            plt.plot(xs = [bvh_pos[SEG_LINKS.index(parent_joint)][0], bvh_pos[SEG_LINKS.index(joint)][0]],
                     zs = [bvh_pos[SEG_LINKS.index(parent_joint)][1], bvh_pos[SEG_LINKS.index(joint)][1]],
                     ys = [bvh_pos[SEG_LINKS.index(parent_joint)][2], bvh_pos[SEG_LINKS.index(joint)][2]], c = 'blue', lw = 2.5)


        ### visualizing urdf 
        frames_to_draw = [GALBOT_CHARLIE_LINKS[0]]
        urdf_pos = urdf_link_pos[i].detach().cpu().numpy()
        while frames_to_draw != []:
            link = frames_to_draw[0]
            frames_to_draw.pop(0)
            for child_frame in urdf_chain.find_frame(link).children:
                child_link = child_frame.name 
                frames_to_draw.append(child_link)
                plt.plot(xs = [urdf_pos[GALBOT_CHARLIE_LINKS.index(link)][1], urdf_pos[GALBOT_CHARLIE_LINKS.index(child_link)][1]],
                         zs = [urdf_pos[GALBOT_CHARLIE_LINKS.index(link)][2] - 0.8, urdf_pos[GALBOT_CHARLIE_LINKS.index(child_link)][2] - 0.8],
                         ys = [urdf_pos[GALBOT_CHARLIE_LINKS.index(link)][0], urdf_pos[GALBOT_CHARLIE_LINKS.index(child_link)][0]],c = 'red', lw = 2.5)




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


def vis_kinematic_result(filename):
    ### loading estimated joint angles
    with open(filename, "rb") as file:
        data_dict = pickle.load(file)
        joints_angle = data_dict["angles"]
        

    ### loading bvh data
    bvh_path = os.path.join(os.path.abspath(os.path.join(filename, os.pardir, os.pardir, "bvh")), filename.split("/")[-1][:-7] + ".bvh")
    skeleton_data = ProcessBVH(bvh_path)
    bvh_joint_local_coord = Get_bvh_joint_local_coord(bvh_path)
    num_frames = len(bvh_joint_local_coord)
    print("Num of frames: ", num_frames)

    ### loading galbot model 
    model = Galbot_Charlie_Motion_Model(num_frames)
    model.load_urdf_as_chain("/home/pengyang/codebase/H1_RL/data/urdf/galbot_one_charlie_10/galbot_one_charlie_retarget.urdf")

    model.set_angles(torch.tensor(joints_angle))
    model.set_global_matrix(data_dict)

    link_to_root_dict = model.forward_kinematics()
    link_to_root_pos = link_to_root_dict[:, :, :3, 3]


    Draw_bvh_urdf(bvh_joint_local_coord, skeleton_data, link_to_root_pos, model.chain)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Call the function with the motion file')
        quit()
    filename = sys.argv[1]
    vis_kinematic_result(filename)






