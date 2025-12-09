### converting 53 dof g1 to vec representation 
### for g1_brainco, dof=53, num_links=72

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


### for G1_brainco_links
### 0-13 lower body
### 14-22 torso and head links
### 23-30 left arm
### 31-46 left hand
### 47-54 right arm
### 55-70 right hand
### 71 imu_in_pelvis
### 
### 


### for split G1_brainco_links
### body links: 0-40
### left hand links: 41-57
### right hand links: 58-74


### body features
### shape (num_frame - 1, 263)
###     body_link_local_position 41 * 3
###     body_link_local_velocity 41 * 3  
###     body_dof_angle 3+7+7=17

### hand features
### shape (num_frame - 1, 228)
###     hand_link_local_position 34 * 3
###     hand_link_local_velocity 34 * 3  
###     hand_dof_angle 12 + 12 = 24






### reference: https://github.com/EricGuo5513/HumanML3D/blob/main/motion_representation.ipynb

### note that  "global_rotation" and "global_translation" may be changed for normalization

### usage: python src/utils/io/inspirehand_representation.py <pickle file>
### example: python src/utils/io/inspirehand_representation.py data/motion/g1/HumanML3D/000000.pickle


import sys
import os
from HRI_mllm.external.HRI_retarget.HRI_retarget.config.joint_mapping import G1_LINKS, G1_BRAINCO_LINKS, G1_BRAINCO_LEFT_HAND_LINKS, G1_BRAINCO_RIGHT_HAND_LINKS
from HRI_retarget.model.g1_brainco import G1_Brainco_Motion_Model
from HRI_retarget.utils.motion_lib.quaternion import qbetween_np, qinv, qinv_np, qmul_np, qrot, qrot_np
from HRI_retarget.utils.torch_utils.diff_quat import quat_to_matrix, vec6d_to_matrix, vec6d_to_quat, quat_to_vec6d

import torch
from tqdm import tqdm
import pickle

import ipdb

import numpy as np

from HRI_mllm import DATA_ROOT
hparams_mean = np.load(os.path.join(f"{DATA_ROOT}/BEAT_v2_kimi", "Mean.npy"))
hparams_std = np.load(os.path.join(f"{DATA_ROOT}/BEAT_v2_kimi", "Std.npy"))
target_fps = 25

def data_pkl_to_vec(data_dict):
    """
    Convert the data_dict to vec representation
    :param data_dict: dict, contains the data
    :return: vec, the vec representation
    """

    angles = np.unwrap(data_dict["angles"])
    num_frames = angles.shape[0]

    source_fps = data_dict["fps"]


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
        case "g1_brainco":
            model = G1_Brainco_Motion_Model(num_frames)
            robot_link = G1_BRAINCO_LINKS
        case _:
            print("wrong robot name in kinematic vis")
            quit()

    ### set unused dofs to zero
    angles = torch.from_numpy(angles).float()
    angles[:, :12] *= 0
    
    model.set_angles(angles)
  




    link_to_root_dict = model.forward_kinematics_split()
    link_to_root_pos = link_to_root_dict[:, :, :3, 3]
    local_positions = link_to_root_pos.detach().cpu().numpy()


    

    '''Get Joint Rotation Representation'''
    # (seq_len, dof) dof for skeleton joints
    rot_data = angles.detach().cpu().numpy()


    '''Get Joint Velocity Representation'''
    # (seq_len-1, (link-1)*3)
    local_vel = local_positions[1:] - local_positions[:-1]


    body_vec = np.concatenate([
        local_positions[:-1, :41, :].reshape(num_frames-1, -1),  # body_link

        local_vel[:, :41, :].reshape(num_frames-1, -1),  # body_vel

        rot_data[:-1, 12:22], # waist and left arm
        rot_data[:-1, 34:41], # right arm
    ], axis=-1)


    hand_vec = np.concatenate([
      
        local_positions[:-1, 41:58, :].reshape(num_frames-1, -1),  # left_hand_link
        local_positions[:-1, 58:75, :].reshape(num_frames-1, -1),  # right_hand_link

        local_vel[:, 41:58, :].reshape(num_frames-1, -1),  # left_hand_vel
        local_vel[:, 58:75, :].reshape(num_frames-1, -1),  # right_hand_vel

        rot_data[:-1, 22:34], # left hand
        rot_data[:-1, 41:53], # right hand
    ], axis=-1)                                     

    ### todo: hand vec


    return np.concatenate([body_vec, hand_vec], axis=-1)  # (num_frame - 1, 263 + 228 = 491)



def vec_to_data_pkl(body_vec, fps=target_fps, reference_motion_pth=None, robot_name="g1_brainco", scale=np.ones(3), use_ik=True, ik_iterations=10, ik_lr=0.01, position_weight=1.0, angle_weight=0.1):
    """
    Convert the vec representation to data_dict using IK optimization
    :param body_vec: vec, the vec representation (num_frames, 491)
    :param fps: target fps
    :param reference_motion_pth: path to reference motion
    :param robot_name: robot name
    :param scale: scale factor
    :param use_ik: whether to use IK optimization (default: True)
    :param ik_iterations: number of IK optimization iterations (default: 10)
    :param ik_lr: learning rate for IK optimization (default: 0.01)
    :param position_weight: weight for position loss (default: 1.0)
    :param angle_weight: weight for angle regularization (default: 0.1)
    :return: data_dict, the data_dict
    """
    
    assert body_vec.ndim == 2
    assert body_vec.shape[1] == 491
    assert isinstance(body_vec, np.ndarray), f"body_vec 应该是 NumPy 数组，但实际类型是 {type(body_vec)}"

    num_frames = body_vec.shape[0]
    global_positions = np.zeros((num_frames, 3))
    global_positions[:, 2] += 0.8 # height

    global_rotations = torch.from_numpy(np.broadcast_to(np.eye(3), (num_frames, 3, 3)).copy())[..., :3, :2].reshape(-1, 3, 2).numpy()

    # Extract position information from body_vec
    # body_vec structure:
    # 0-122: body_link_local_position (41 * 3 = 123)
    # 123-245: body_link_local_velocity (41 * 3 = 123)
    # 246-255: waist and left arm dof angles (10)
    # 256-262: right arm dof angles (7)
    # 263-466: hand positions and velocities (204)
    # 467-478: left hand dof angles (12)
    # 479-490: right hand dof angles (12)
    
    # Extract local positions (relative to root)
    # Note: body_vec has shape (num_frames-1, 491) because positions are computed from velocities
    # We need to handle the last frame by duplicating or interpolating
    actual_frames = body_vec.shape[0]  # This is num_frames - 1
    
    # Extract positions for available frames
    body_local_positions = body_vec[:, :123].reshape(actual_frames, 41, 3)  # body links
    left_hand_local_positions = body_vec[:, 263:314].reshape(actual_frames, 17, 3)  # left hand links (17 * 3 = 51)
    right_hand_local_positions = body_vec[:, 314:365].reshape(actual_frames, 17, 3)  # right hand links (17 * 3 = 51)
    
    # Extract initial dof angles from body_vec (also num_frames-1)
    initial_dof_angles = np.zeros((actual_frames, 53))
    initial_dof_angles[:, 12:22] = body_vec[:, 246:256]  # waist and left arm
    initial_dof_angles[:, 34:41] = body_vec[:, 256:263]  # right arm
    initial_dof_angles[:, 22:34] = body_vec[:, 467:479] # left hand 
    initial_dof_angles[:, 41:53] = body_vec[:, 479:491] # right hand
    
    # If we need num_frames frames, pad the last frame by duplicating
    if num_frames > actual_frames:
        # Duplicate the last frame
        body_local_positions = np.concatenate([body_local_positions, body_local_positions[-1:]], axis=0)
        left_hand_local_positions = np.concatenate([left_hand_local_positions, left_hand_local_positions[-1:]], axis=0)
        right_hand_local_positions = np.concatenate([right_hand_local_positions, right_hand_local_positions[-1:]], axis=0)
        initial_dof_angles = np.concatenate([initial_dof_angles, initial_dof_angles[-1:]], axis=0)

    if use_ik:
        # Use IK optimization to refine joint angles based on position constraints
        dof_angles = _optimize_ik_with_positions(
            initial_dof_angles=initial_dof_angles,
            body_local_positions=body_local_positions,
            left_hand_local_positions=left_hand_local_positions,
            right_hand_local_positions=right_hand_local_positions,
            robot_name=robot_name,
            num_iterations=ik_iterations,
            lr=ik_lr,
            position_weight=position_weight,
            angle_weight=angle_weight
        )
    else:
        # Use direct mapping (original method)
        dof_angles = initial_dof_angles

    data_dict = {
        "fps": fps,
        "reference_motion_pth": reference_motion_pth,
        "robot_name": robot_name,
        "angles": dof_angles,
        "global_rotation": global_rotations,
        "global_translation": global_positions,
        "scale": scale,
    }

    return data_dict


def _optimize_ik_with_positions(initial_dof_angles, body_local_positions, left_hand_local_positions, 
                                right_hand_local_positions, robot_name="g1_brainco", num_iterations=10, 
                                lr=0.01, position_weight=1.0, angle_weight=0.1):
    """
    Optimize joint angles using position constraints from body_vec
    
    :param initial_dof_angles: initial joint angles (num_frames, 53)
    :param body_local_positions: target body link positions (num_frames, 41, 3)
    :param left_hand_local_positions: target left hand link positions (num_frames, 17, 3)
    :param right_hand_local_positions: target right hand link positions (num_frames, 17, 3)
    :param robot_name: robot name
    :param num_iterations: number of optimization iterations
    :param lr: learning rate
    :param position_weight: weight for position loss
    :param angle_weight: weight for angle regularization
    :return: optimized joint angles (num_frames, 53)
    """
    import torch.optim as optim
    
    num_frames = initial_dof_angles.shape[0]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Convert to torch tensors
    initial_angles_torch = torch.from_numpy(initial_dof_angles).float().to(device)
    body_target_positions = torch.from_numpy(body_local_positions).float().to(device)
    left_hand_target_positions = torch.from_numpy(left_hand_local_positions).float().to(device)
    right_hand_target_positions = torch.from_numpy(right_hand_local_positions).float().to(device)
    
    # Create motion model
    match robot_name:
        case "g1_brainco":
            model = G1_Brainco_Motion_Model(batch_size=num_frames, device=device)
        case _:
            print("wrong robot name in IK optimization")
            return initial_dof_angles
    
    # Set model to training mode to enable gradients
    model.train()
    
    # Make angles learnable (don't use set_angles as it creates a new Parameter)
    optimized_angles = torch.nn.Parameter(initial_angles_torch.clone(), requires_grad=True)
    optimizer = optim.Adam([optimized_angles], lr=lr)
    
    # Get link indices for position matching
    # Body links: 0-40 (41 links)
    # Left hand links: 41-57 (17 links)
    # Right hand links: 58-74 (17 links)
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Directly use optimized_angles for forward kinematics
        # We need to manually call forward_kinematics_split with the angles
        # Split angles for body, left hand, and right hand chains
        body_angles = torch.cat([optimized_angles[:, :22], optimized_angles[:, 34:41]], dim=1)
        left_hand_angles = torch.cat([optimized_angles[:, 22:25], optimized_angles[:, 26:34]], dim=1)
        right_hand_angles = torch.cat([optimized_angles[:, 41:44], optimized_angles[:, 45:53]], dim=1)
        
        # Forward kinematics to get current positions
        body_dict = model.body_chain.forward_kinematics(body_angles)
        left_hand_dict = model.left_hand_chain.forward_kinematics(left_hand_angles)
        right_hand_dict = model.right_hand_chain.forward_kinematics(right_hand_angles)
        
        # Combine results
        link_to_root_dict = []
        for link_name in G1_LINKS:
            link_to_root_dict.append(body_dict[link_name].get_matrix())
        for link_name in G1_BRAINCO_LEFT_HAND_LINKS:
            link_to_root_dict.append(left_hand_dict[link_name].get_matrix())
        for link_name in G1_BRAINCO_RIGHT_HAND_LINKS:
            link_to_root_dict.append(right_hand_dict[link_name].get_matrix())
        
        link_to_root_dict = torch.stack(link_to_root_dict, dim=1)  # (num_frames, 75, 4, 4)
        current_positions = link_to_root_dict[:, :, :3, 3]  # (num_frames, 75, 3)
        
        # Split into body, left hand, and right hand positions
        current_body_positions = current_positions[:, :41, :]  # (num_frames, 41, 3)
        current_left_hand_positions = current_positions[:, 41:58, :]  # (num_frames, 17, 3)
        current_right_hand_positions = current_positions[:, 58:75, :]  # (num_frames, 17, 3)
        
        # Position loss: MSE between target and current positions
        body_pos_loss = torch.mean((current_body_positions - body_target_positions) ** 2)
        left_hand_pos_loss = torch.mean((current_left_hand_positions - left_hand_target_positions) ** 2)
        right_hand_pos_loss = torch.mean((current_right_hand_positions - right_hand_target_positions) ** 2)
        
        # Use equal weights for all parts (hand positions are already in the same scale)
        # The original 10x weight was too large and caused instability
        position_loss = body_pos_loss + left_hand_pos_loss + right_hand_pos_loss
        
        # Angle regularization: keep angles close to initial prediction
        angle_reg_loss = torch.mean((optimized_angles - initial_angles_torch) ** 2)
        
        # Total loss
        total_loss = position_weight * position_loss + angle_weight * angle_reg_loss
        
        # Print loss every 10 iterations or on first/last iteration
        if iteration % 10 == 0 or iteration == 0 or iteration == num_iterations - 1:
            print(f"Iter {iteration}: pos_loss={position_loss.item():.6f}, "
                  f"body={body_pos_loss.item():.6f}, "
                  f"left_hand={left_hand_pos_loss.item():.6f}, "
                  f"right_hand={right_hand_pos_loss.item():.6f}, "
                  f"angle_reg={angle_reg_loss.item():.6f}, "
                  f"total={total_loss.item():.6f}")
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping to prevent instability
        torch.nn.utils.clip_grad_norm_([optimized_angles], max_norm=1.0)
        
        optimizer.step()
        
        # Clamp angles to valid ranges (in-place operation, preserves gradients)
        for i in range(num_frames):
            for j in range(53):
                min_angle = model.dof_max_limits[0, j, 0].item()
                max_angle = model.dof_max_limits[0, j, 1].item()
                optimized_angles.data[i, j] = torch.clamp(optimized_angles.data[i, j], min_angle, max_angle)
    
    # Set model back to eval mode
    model.eval()
    
    # Return optimized angles as numpy array
    return optimized_angles.detach().cpu().numpy()


def vec_to_joints(body_vec, robot_name="g1_brainco"):
    assert body_vec.dim() == 3, "Input data must be a 3D tensor (batch_size, frame, num_features)"
    assert isinstance(body_vec, torch.Tensor), f"body_vec 应该是 torch.tensor，但实际类型是 {type(body_vec)}"

    device = body_vec.device
    batch_size = body_vec.shape[0]
    num_frames = body_vec.shape[1]

    total_frames = batch_size * num_frames


    body_vec_expanded = body_vec.view(total_frames, -1)
    dof_angles = torch.zeros((total_frames, 53))


    dof_angles[:, 12:22] = body_vec_expanded[:, 246:256]  # waist and left arm
    dof_angles[:, 34:41] = body_vec_expanded[:, 256:263]  # right arm
    dof_angles[:, 22:34] = body_vec_expanded[:, 467:479] # left hand 
    dof_angles[:, 41:53] = body_vec_expanded[:, 479:491] # right handx

   

    match robot_name:
        case "g1_brainco":
            model = G1_Brainco_Motion_Model(total_frames)
            robot_link = G1_BRAINCO_LINKS
        case _:
            print("wrong robot name in kinematic vis")
            quit()

    
    model.set_angles(dof_angles)
  


    link_to_root_dict = model.forward_kinematics()
    link_to_root_pos = link_to_root_dict[:, :, :3, 3] 
    positions = link_to_root_pos.view(batch_size, num_frames, -1, 3)

    return positions

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




