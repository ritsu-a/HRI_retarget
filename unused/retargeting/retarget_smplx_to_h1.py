import os
import torch
from SAMP_utils.load_data import load_smplx_motion
from h1_kinematics import H1_Motion_Model, load_urdf, get_h1_link_names, forward_kinematics, save_predicted_h1_motion


def get_joint_correspondence():
    """
    [h1 joint index, corresponding SMPLX joint index]
    """
    joint_correspondence = [
        [0, 0],
        [1, 1],
        [2, 1],
        # [3, 1],
        [4, 4],
        [5, 7],
        [6, 2],
        [7, 2],
        # [8, 2],
        [9, 5],
        [10, 8],
        [11, 0],
        [12, 16],
        # [13, 16],
        [14, 18],
        [15, 20],
        [16, 17],
        # [17, 17],
        [18, 19],
        [19, 21],
    ]
    return joint_correspondence


def retarget_smplx_to_h1(chain, link_names, smplx_motion, device="cuda:0"):
    """
    chain: h1 chain
    link_names: h1 link names
    smplx_motion: {"joint_positions": (N, 127, 3)}, all items are numpy

    return: {"joint_angles": (N, 19), "global_rotation": (N, 3), "global_orientation": (N, 3)}
    """

    gt_joint_positions = torch.from_numpy(smplx_motion["joint_positions"]).to(device)
    cx, cy = gt_joint_positions[0, 0, 0], gt_joint_positions[0, 0, 1]
    gt_joint_positions[:, :, 0] = gt_joint_positions[:, :, 0] - cx
    gt_joint_positions[:, :, 1] = gt_joint_positions[:, :, 1] - cy
    N_frame = gt_joint_positions.shape[0]

    ########################## start optimization #################################
    h1_motion_model = H1_Motion_Model(batch_size=N_frame, device=device)

    optimizer = torch.optim.Adam(h1_motion_model.parameters(), lr=2e-2)
    h1_motion_model.train()

    joint_corrs = get_joint_correspondence()

    for epoch in range(3000):
        h1_motion = h1_motion_model()

        pred_link_to_world_dict = forward_kinematics(chain, link_names, h1_motion["joint_angles"], global_rotation=h1_motion["global_rotations"], global_translation=h1_motion["global_translations"], device=device)

        joint_global_position_loss = 0
        for joint_corr in joint_corrs:
            joint_global_position_loss += ((pred_link_to_world_dict[link_names[joint_corr[0]]][:, :3, 3] - gt_joint_positions[:, joint_corr[1]])**2).sum(dim=-1).mean()
        
        pred_joint_angles = h1_motion["joint_angles"]
        pred_joint_velocities = pred_joint_angles[1:] - pred_joint_angles[:-1]
        pred_joint_accelerations = pred_joint_velocities[1:] - pred_joint_velocities[:-1]
        pred_root_linear_velocities = pred_link_to_world_dict["pelvis"][1:, :3, 3] - pred_link_to_world_dict["pelvis"][:-1, :3, 3]
        pred_root_linear_acceleration = pred_root_linear_velocities[1:] - pred_root_linear_velocities[:-1]
        joint_local_velocity_loss = pred_joint_velocities.abs().sum(dim=-1).mean()
        joint_local_acceleration_loss = pred_joint_accelerations.abs().sum(dim=-1).mean()
        root_global_linear_acceleration_loss = (pred_root_linear_acceleration**2).sum(dim=-1).mean()

        # TODO: add joint rotation loss

        # TODO: add contact loss

        loss = 1.0 * joint_global_position_loss + 1.0 * joint_local_acceleration_loss + 0.0 * root_global_linear_acceleration_loss

        if epoch % 100 == 0:
            print(epoch, loss.item(), joint_global_position_loss.item(), joint_local_acceleration_loss.item(), root_global_linear_acceleration_loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    ########################## finish optimization ################################

    h1_motion = h1_motion_model()
    return h1_motion


if __name__ == "__main__":

    device = "cuda:0"

    chain = load_urdf("/home/liuyun/Humanoid_IL_Benchmark/retargeting/assets/h1_description/urdf/h1.urdf", device=device)
    link_names = get_h1_link_names()

    # load gt data
    # smplx_file_path = "/media/liuyun/TOSHIBA EXT/SAMP/pkl/chair_mo001_stageII.pkl"
    # smplx_file_path = "/media/liuyun/TOSHIBA EXT/SAMP/pkl/armchair007_stageII.pkl"
    # smplx_file_path = "/media/liuyun/TOSHIBA EXT/SAMP/pkl/sofa012_stageII.pkl"
    smplx_file_path = "/media/liuyun/TOSHIBA EXT/SAMP/pkl/run_circular_stageII.pkl"
    smplx_model_path = "./SAMP_utils/models"
    smplx_motion = load_smplx_motion(smplx_file_path, smplx_model_path, start_frame=0, end_frame=-1, sampling_rate=6)
    # assert False

    h1_motion = retarget_smplx_to_h1(chain, link_names, smplx_motion, device=device)
    save_predicted_h1_motion(h1_motion, "./pred.npz")
