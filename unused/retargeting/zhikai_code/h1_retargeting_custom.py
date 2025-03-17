"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Franka Cube Pick
----------------
Use Jacobian matrix and inverse kinematics control of Franka robot to pick up a box.
Damped Least Squares method from: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
"""

# conda activate isaac
# python unitree_h1_retargeting_lsk.py
# import keyboard
import sys
import select
import math
import numpy as np
import torch
import time
import os
from tqdm import tqdm
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles, axis_angle_to_matrix

def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats


def control_ik(dpose):
    global damping, j_eef, num_envs
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)
    return u


def control_osc(dpose):
    global kp, kd, kp_null, kd_null, default_dof_pos_tensor, mm, j_eef, num_envs, dof_pos, dof_vel, hand_vel
    mm_inv = torch.inverse(mm)
    m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    m_eef = torch.inverse(m_eef_inv)
    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (
        kp * dpose - kd * hand_vel.unsqueeze(-1))

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    j_eef_inv = m_eef @ j_eef @ mm_inv
    u_null = kd_null * -dof_vel + kp_null * (
        (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi)
    u_null = u_null[:, :7]
    u_null = mm @ u_null
    u += (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
    return u.squeeze(-1)


# set random seed
np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments

# Add custom arguments
custom_parameters = [
    {"name": "--controller", "type": str, "default": "ik",
     "help": "Controller to use for Franka. Options are {ik, osc}"},
    {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"},
    {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
    {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
]
args = gymutil.parse_arguments(
    description="Franka Jacobian Inverse Kinematics (IK) + Operational Space Control (OSC) Example",
    custom_parameters=custom_parameters,
)

# Grab controller
controller = args.controller
assert controller in {"ik", "osc"}, f"Invalid controller specified -- options are (ik, osc). Got: {controller}"

# set torch device
# ipdb.set_trace()
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

# configure sim
sim_params = gymapi.SimParams()
sim_fps = 30
sim_params.dt = dt = 1.0 / sim_fps
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

# args:
# Namespace(compute_device_id=0, 
#           controller='ik', 
#           flex=False, 
#           graphics_device_id=0, 
#           num_envs=256, 
#           num_threads=0, 
#           physics_engine=SimType.SIM_PHYSX, 
#           physx=False, 
#           pipeline='gpu', 
#           show_axis=False, 
#           sim_device='cuda:0', 
#           sim_device_type='cuda', 
#           slices=0, 
#           speed_scale=1.0, 
#           subscenes=0, 
#           use_gpu=True, 
#           use_gpu_pipeline=True)

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer

# ipdb.set_trace()
show = True
save = True
save_fps = 10
if show:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("Failed to create viewer")


asset_root = "./assets"

# load h1 asset
h1_asset_file = "h1_description/urdf/h1_with_hand.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = True
h1_asset = gym.load_asset(sim, asset_root, h1_asset_file, asset_options)
h1_dof_names = gym.get_asset_dof_names(h1_asset)
# ['left_hip_yaw_joint', 
#  'left_hip_roll_joint', 
#  'left_hip_pitch_joint', 
#  'left_knee_joint', 
#  'left_ankle_joint',
#  'right_hip_yaw_joint', 
#  'right_hip_roll_joint', 
#  'right_hip_pitch_joint', 
#  'right_knee_joint', 
#  'right_ankle_joint', 
#  'torso_joint', 
#  'left_shoulder_pitch_joint', 
#  'left_shoulder_roll_joint', 
#  'left_shoulder_yaw_joint', 
#  'left_elbow_joint', 
#  'left_hand_joint', 
#  'L_index_proximal_joint', 
#  'L_index_intermediate_joint', 
#  'L_middle_proximal_joint', 
#  'L_middle_intermediate_joint', 
#  'L_pinky_proximal_joint', 
#  'L_pinky_intermediate_joint', 
#  'L_ring_proximal_joint', 
#  'L_ring_intermediate_joint', 
#  'L_thumb_proximal_yaw_joint', 
#  'L_thumb_proximal_pitch_joint', 
#  'L_thumb_intermediate_joint', 
#  'L_thumb_distal_joint', 
#  'right_shoulder_pitch_joint', 
#  'right_shoulder_roll_joint', 
#  'right_shoulder_yaw_joint', 
#  'right_elbow_joint', 
#  'right_hand_joint', 
#  'R_index_proximal_joint', 
#  'R_index_intermediate_joint',
#  'R_middle_proximal_joint', 
#  'R_middle_intermediate_joint', 
#  'R_pinky_proximal_joint', 
#  'R_pinky_intermediate_joint', 
#  'R_ring_proximal_joint', 
#  'R_ring_intermediate_joint', 
#  'R_thumb_proximal_yaw_joint', 
#  'R_thumb_proximal_pitch_joint', 
#  'R_thumb_intermediate_joint', 
#  'R_thumb_distal_joint']

h1_dof_props = gym.get_asset_dof_properties(h1_asset)
h1_num_dofs = gym.get_asset_dof_count(h1_asset)
h1_dof_states = np.zeros(h1_num_dofs, dtype=gymapi.DofState.dtype)
h1_dof_types = [gym.get_asset_dof_type(h1_asset, i) for i in range(h1_num_dofs)]
h1_dof_positions = h1_dof_states['pos']
h1_lower_limits = h1_dof_props["lower"]
h1_upper_limits = h1_dof_props["upper"]
h1_ranges = h1_upper_limits - h1_lower_limits
h1_mids = 0.3 * (h1_upper_limits + h1_lower_limits)
h1_stiffnesses = h1_dof_props['stiffness']
h1_dampings = h1_dof_props['damping']
h1_armatures = h1_dof_props['armature']
h1_has_limits = h1_dof_props['hasLimits']
h1_dof_props['hasLimits'] = np.array([True]*h1_num_dofs)

defaults = np.zeros(h1_num_dofs)
speeds = np.zeros(h1_num_dofs)
for i in range(h1_num_dofs):
    if h1_has_limits[i]:
        if h1_dof_types[i] == gymapi.DOF_ROTATION:
            h1_lower_limits[i] = clamp(h1_lower_limits[i], -math.pi, math.pi)
            h1_upper_limits[i] = clamp(h1_upper_limits[i], -math.pi, math.pi)
        # make sure our default position is in range
        if h1_lower_limits[i] > 0.0:
            defaults[i] = h1_lower_limits[i]
        elif h1_upper_limits[i] < 0.0:
            defaults[i] = h1_upper_limits[i]
    else:
        # set reasonable animation limits for unlimited joints
        if h1_dof_types[i] == gymapi.DOF_ROTATION:
            # unlimited revolute joint
            h1_lower_limits[i] = -math.pi
            h1_upper_limits[i] = math.pi
        elif h1_dof_types[i] == gymapi.DOF_TRANSLATION:
            # unlimited prismatic joint
            h1_lower_limits[i] = -1.0
            h1_upper_limits[i] = 1.0
    # set DOF position to default
    h1_dof_positions[i] = defaults[i]
    # set speed depending on DOF type and range of motion
    # ipdb.set_trace()
    if h1_dof_types[i] == gymapi.DOF_ROTATION:
        speeds[i] = args.speed_scale * clamp(2 * (h1_upper_limits[i] - h1_lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi)
    else:
        speeds[i] = args.speed_scale * clamp(2 * (h1_upper_limits[i] - h1_lower_limits[i]), 0.1, 7.0)

# load human motion data
motion_name = "walk.npy"
pos_motion_root = "./human_motion_data/controlvae_human_joint_pos_data"
pos_motion_path = os.path.join(pos_motion_root, motion_name)
pos_motion_data = np.load(pos_motion_path)
ori_motion_root = "./human_motion_data/controlvae_human_joint_ori_data"
ori_motion_path = os.path.join(ori_motion_root, motion_name)
ori_motion_data = np.load(ori_motion_path)
print('!!!!!!!!!!!!!!!!!!##############')
pos_motion_data -= [pos_motion_data[0,0,0], 0.0, pos_motion_data[0,0,2]] # move motion data to origin
pos_motion_data_x = pos_motion_data[:, :, :1]
pos_motion_data_y = pos_motion_data[:, :, 1:2]
pos_motion_data_z = pos_motion_data[:, :, 2:]

ori_motion_data_x = ori_motion_data[:, :, :1]
ori_motion_data_y = ori_motion_data[:, :, 1:2]
ori_motion_data_z = ori_motion_data[:, :, 2:3]
ori_motion_data_w = ori_motion_data[:, :, 3:]

motion_data = np.concatenate([pos_motion_data_x, pos_motion_data_y, -pos_motion_data_z,  -ori_motion_data_x, -ori_motion_data_y, ori_motion_data_z,  ori_motion_data_w], axis = -1)
human_joint_names = ['pelvis', 'pelvis_lowerback', 'lowerback_torso', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_toes', 'right_toes', 'neck', 'left_torso_clavicle', 'right_torso_clavicle', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
h1_dof_names_without_hand = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_joint', 'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_joint', 'torso_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint']
# joint list for retargeting
true_h1_joint_names = ['pelvis',"torso_joint", "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_joint", "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_hand_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_hand_joint"]
true_h1_joint_parents = [-1, 0, 0, 2, 3, 4, 5, 0, 7, 8, 9, 10, 1, 12, 13, 14, 15, 1, 17, 18, 19, 20]
# correspondence joint dict between h1 and human
correspondence_dict = {"left_ankle_joint": "left_ankle", "right_ankle_joint" : "right_ankle", "left_knee_joint": "left_knee", "right_knee_joint": "right_knee", "left_elbow_joint": "left_elbow", "right_elbow_joint": "right_elbow", "left_hand_joint": "left_wrist", "right_hand_joint": "right_wrist"}
# configure env grid
num_envs = 1
num_per_row = 1
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)


envs = []
actor_handles = []
joint_handles = {}

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

initial_pose_ori = [-0.5,0.5,0.5,0.5]
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 1.08, 0.0)
    pose.r = gymapi.Quat(*initial_pose_ori)
    
    actor_handle = gym.create_actor(env, h1_asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, h1_dof_states, gymapi.STATE_ALL)

    # body_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_ALL)
    
    # body_names = gym.get_actor_joint_names(env, actor_handle)
    # print(body_states)
    # print(body_names)
    # print(len(body_states))

# position the camera
if show:
    cam_pos = gymapi.Vec3(3, 2.0, 0)
    cam_target = gymapi.Vec3(-3, 0, 0)
    gym.viewer_camera_look_at(viewer, envs[0], cam_pos, cam_target)



# !!!!!!!!!!!!!!!!!!!!!!!!
gym.prepare_sim(sim)
# !!!!!!!!!!!!!!!!!!!!!!!!
print('!!!!!!!!!!!^^^^^^^^^^^^^^^^*********')
# print(h1_dof_states.shape)
def set_root_pose(gym, sim, env, actor_handle, tar_pos, tar_ori):
    gym.simulate(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    # actor_count = gym.get_sim_actor_count(sim)

    actor_root_state = gym.acquire_actor_root_state_tensor(sim)
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)

    root_states = gymtorch.wrap_tensor(actor_root_state)
    # print()
    dof_states = gymtorch.wrap_tensor(dof_state_tensor)
    dof_states[:,0:2] = torch.tensor(0,device=dof_states.device)

    one_indice = gym.get_actor_index(env, actor_handle, gymapi.DOMAIN_SIM)
    root_indices = []
    root_indices.append(one_indice)
    root_positions = root_states[:, 0:3]
    root_orientations = root_states[:, 3:7]

    root_pose_pos = tar_pos

    root_pose_ori = R.from_rotvec(tar_ori).as_quat()

    root_positions[one_indice]= torch.tensor(root_pose_pos)
    root_orientations[one_indice] = torch.tensor(root_pose_ori)

    root_reset_actors_indices = torch.tensor(root_indices).to(dtype=torch.int32)
    gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root_states), gymtorch.unwrap_tensor(root_reset_actors_indices), len(root_reset_actors_indices))
    gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(dof_states), gymtorch.unwrap_tensor(root_reset_actors_indices), len(root_reset_actors_indices))
    ####################################################################################################################################################

def getpose(gym, env, actor_handle, true_h1_joint_names):
    # get h1 joint pos, ori
    
    ori = [1,0,0]
    joint_pose = gym.get_actor_joint_transforms(env, actor_handle)
    joint_names = gym.get_actor_joint_names(env, actor_handle)
    h1_joint_ori_list = []
    h1_joint_pos_list = []
    for name in true_h1_joint_names:
        if name == "pelvis":
            h1_joint_ori_list.append(np.array([[0,0,0]]))
            h1_joint_pos_list.append(np.array([[0,0,0]]))
            continue
        cur_joint_pose = joint_pose[joint_names.index(name)]
        h1_joint_pos_list.append(np.array([[cur_joint_pose[0][0], cur_joint_pose[0][1], cur_joint_pose[0][2]]]))
        axis = np.array([R([cur_joint_pose[1][0],cur_joint_pose[1][1],cur_joint_pose[1][2],cur_joint_pose[1][3]]).as_matrix() @ ori])

        h1_joint_ori_list.append(axis)
    
    return np.concatenate(h1_joint_pos_list, axis=0), np.concatenate(h1_joint_ori_list, axis=0)

epoch = 10
alpha = 0.1
cnt = 0
motion_cnt = 0
motion_data_t = torch.tensor(motion_data,device='cuda:0')
true_h1_joint_rotation = np.zeros(len(true_h1_joint_names))
joint_rotation_t = [torch.tensor(data, device='cuda:0',requires_grad=True) for data in true_h1_joint_rotation]

h1_motion = []
for _ in tqdm(range(motion_data.shape[0])):
    # set motion root ori and pos to h1 base, set all dofs 0 
    set_root_pose(gym, sim, envs[0], actor_handles[0], motion_data[motion_cnt][0][:3]+[0,0.2,0], (R(motion_data[motion_cnt][0][3:])*R(initial_pose_ori)).as_rotvec())

    true_h1_joint_orientation = np.zeros([len(true_h1_joint_names), 3])
    true_h1_joint_position, true_h1_joint_axis = getpose(gym, envs[0], actor_handles[0], true_h1_joint_names)
    true_h1_joint_offset = [true_h1_joint_position[j] - true_h1_joint_position[true_h1_joint_parents[j]] for j in range(len(true_h1_joint_position))]
    true_h1_joint_offset[0] = np.array([0.0, 0.0, 0.0])

    joint_axis_t = [torch.tensor(data, device='cuda:0') for data in true_h1_joint_axis]
    joint_offset_t = [torch.tensor(data, device='cuda:0') for data in true_h1_joint_offset]
    joint_position_t = [torch.tensor(data, device='cuda:0') for data in true_h1_joint_position]
    joint_orientation_t = [torch.tensor(R.from_euler('XYZ', data).as_matrix(), device='cuda:0') for data in true_h1_joint_orientation]
    
    if  show and gym.query_viewer_has_closed(viewer):
        break
    # forward kinematics
    for j in range(epoch):
        if  show and gym.query_viewer_has_closed(viewer):
            break
        for k in range(1, len(true_h1_joint_names)):
            joint_orientation_t[k] = joint_orientation_t[true_h1_joint_parents[k]]  @ axis_angle_to_matrix(joint_rotation_t[k]*joint_axis_t[k])
            joint_position_t[k] = joint_position_t[true_h1_joint_parents[k]] + joint_orientation_t[true_h1_joint_parents[k]] @ joint_offset_t[k].double()
        optimize_target = 0
        for h1_joint in correspondence_dict:
            optimize_target += torch.norm(joint_position_t[true_h1_joint_names.index(h1_joint)] - motion_data_t[motion_cnt][human_joint_names.index(correspondence_dict[h1_joint])][:3])
        cnt = 1
        # TODO joint constraint
        # TODO adam better?
        # TODO add true root joint
        # TODO multiframe
        optimize_target.backward()

        for num, name in enumerate(h1_dof_names):
            if name not in true_h1_joint_names:
                continue
            id = true_h1_joint_names.index(name)
            if joint_rotation_t[id].grad is not None:
                if name =='torso_joint':
                    continue
                step = alpha * joint_rotation_t[id].grad
                joint_rotation_t[id] = torch.tensor(clamp((joint_rotation_t[id] - step).detach().cpu().numpy(), h1_lower_limits[num], h1_upper_limits[num]),device='cuda:0')
                # print(clamp(joint_rotation_t[id] - step, h1_lower_limits[num], h1_upper_limits[num]).type)
                # joint_rotation_t[id] = joint_rotation_t[id] + 0.1
                joint_rotation_t[id] = joint_rotation_t[id].clone().detach().requires_grad_(True)
                h1_dof_positions[num] = joint_rotation_t[id]
    
    greenballgeo = gymutil.WireframeSphereGeometry(radius=0.1,color = (0, 1, 0))
    redballgeo = gymutil.WireframeSphereGeometry(radius=0.1)
    blueballgeo = gymutil.WireframeSphereGeometry(radius=0.1,color=(0, 0, 1))
    # vis human joint pos
    for id, name in enumerate(human_joint_names):
        if name in correspondence_dict.values():
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(motion_data[motion_cnt][id][0], motion_data[motion_cnt][id][1], motion_data[motion_cnt][id][2])
            gymutil.draw_lines(redballgeo, gym, viewer, envs[0], pose)
            # bbox = np.array([motion_data[motion_cnt][id][:3]-box_size,motion_data[motion_cnt][id][:3]+box_size])
            # boxgeo = gymutil.WireframeBBoxGeometry(bbox)
            # pose = gymapi.Transform()
            # gymutil.draw_lines(boxgeo, gym, viewer, envs[0], pose)
    # vis humanoid joint pos
    for id, name in enumerate(true_h1_joint_names):
        if name in correspondence_dict.keys():
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(joint_position_t[id][0], joint_position_t[id][1], joint_position_t[id][2])
            if name =='left_elbow_joint':
                gymutil.draw_lines(greenballgeo, gym, viewer, envs[0], pose)
            else:
                gymutil.draw_lines(blueballgeo, gym, viewer, envs[0], pose)
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    # ipdb.set_trace()
    gym.set_actor_dof_states(envs[0], actor_handles[0], h1_dof_states, gymapi.STATE_POS)
    if show:
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.clear_lines(viewer)
    gym.sync_frame_time(sim)
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    if save and motion_cnt%2 and motion_cnt<50:
        dof_pos = []
        for name in h1_dof_names_without_hand:
            idx = h1_dof_names.index(name)
            cur_dof_pos = h1_dof_positions[idx]
            dof_pos.append(cur_dof_pos)
        dof_pos = np.stack(dof_pos)
        h1_motion.append(dof_pos)

        # frame_pos = [data.detach().cpu().numpy() for data in joint_position_t]
        # frame_pos = np.stack(frame_pos,axis=0)
        # frame_pos[0] = motion_data[motion_cnt][0][:3]+[0,0.2,0]
        # frame_orientation = [R.from_matrix(data.detach().cpu().numpy()).as_quat() for data in joint_orientation_t]
        # frame_orientation = np.stack(frame_orientation,axis=0)
        # frame_orientation[0] = (R(motion_data[motion_cnt][0][3:])*R([-0.707107, 0.0, 0.0, 0.707107])).as_quat()
        # frame = np.concatenate([frame_pos,frame_orientation],axis=1)
        # unitree_motion.append(frame)

    motion_cnt += 1
    print(motion_cnt)
    sys.stdin.fileno()
    if select.select([sys.stdin], [], [], 0)[0]:
        break


h1_motion = np.stack(h1_motion)
print(h1_motion.shape)
pre_name = "h1_"
if save:
    np.save("../../h1_motion_data/"+pre_name+motion_name,h1_motion)
print("Done")
if show:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

