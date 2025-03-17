import os
import numpy as np
import torch
from torch import nn
import pytorch_kinematics as pk
from torch_utils import batch_rodrigues


class H1_Motion_Model(nn.Module):
    def __init__(self, batch_size=1, device="cuda:0"):
        super(H1_Motion_Model, self).__init__()

        self.batch_size = batch_size
        self.device = device

        # N * (3 + 3 + 19) DoF's solution space
        self.global_rotations = nn.Parameter(torch.zeros(batch_size, 3).to(device), requires_grad=True)  # (N, 3)  # NOTE: 需要接近最优解的初始化
        self.global_translations = nn.Parameter(torch.zeros(batch_size, 3).to(device), requires_grad=True)  # (N, 3)
        self.joint_angles = nn.Parameter(torch.zeros(batch_size, 19).to(device), requires_grad=True)  # (N, 19)
    
    def forward(self):
        return {
            "global_rotations": self.global_rotations,
            "global_translations": self.global_translations,
            "joint_angles": self.joint_angles,    
        }


def get_h1_link_names():
    link_names = [
        'pelvis',
        'left_hip_yaw_link',
        'left_hip_roll_link',
        'left_hip_pitch_link',
        'left_knee_link',
        'left_ankle_link',
        'right_hip_yaw_link',
        'right_hip_roll_link',
        'right_hip_pitch_link',
        'right_knee_link',
        'right_ankle_link',
        'torso_link',
        'left_shoulder_pitch_link',
        'left_shoulder_roll_link',
        'left_shoulder_yaw_link',
        'left_elbow_link',
        'right_shoulder_pitch_link',
        'right_shoulder_roll_link',
        'right_shoulder_yaw_link',
        'right_elbow_link',
        'imu_link',
        'logo_link',
        'd435_left_imager_link',
        'd435_rgb_module_link',
        'mid360_link',
    ]
    return link_names


def load_urdf(urdf_path, device="cuda:0"):
    # 19个revolute的关节, 5个fix的关节, 共25个link (root link是pelvis)
    chain = pk.build_chain_from_urdf(open(urdf_path, "rb").read())
    # chain = pk.build_serial_chain_from_urdf(open(urdf_path, "rb").read(), "left_ankle_link")
    chain = chain.to(dtype=torch.float32, device=device)
    return chain


def forward_kinematics(chain, link_names, joint_angle, global_rotation=None, global_translation=None, device="cuda:0"):
    """
    chain: pytorch_kinematics.chain.Chain
    link_names: list, len = 25
    joint_angle: (N, 19) 19D vector
    global_translation: (N, 3) 3D vector, root_to_world
    global_orientation: (N, 3) 3D axis-angle, root_to_world

    return: a dict contains the global poses of 25 links
    """

    N_frame = joint_angle.shape[0]

    if not global_rotation is None:
        R = batch_rodrigues(global_rotation)  # (N_frame, 3, 3)
    else:
        R = torch.eye(3).unsqueeze(0).repeat(N_frame, 1, 1).to(device)  # (N_frame, 3, 3)
    if not global_translation is None:
        t = global_translation.reshape(N_frame, 3, 1)  # (N_frame, 3, 1)
    else:
        t = torch.zeros((N_frame, 3, 1)).to(device)  # (N_frame, 3, 1)
    root_to_world = torch.cat((torch.cat((R, t), dim=-1), torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat(N_frame, 1, 1).to(device)), dim=1)  # (N_frame, 4, 4)

    link_to_root_dict = chain.forward_kinematics(joint_angle)  # link to root
    link_to_world_dict = {}
    for link_name in link_names:
        T = link_to_root_dict[link_name].get_matrix()  # link to root
        link_to_world_dict[link_name] = torch.einsum('bij,bjk->bik', root_to_world, T)
    
    return link_to_world_dict


def save_predicted_h1_motion(h1_motion, save_path):
    result = {}
    for key in h1_motion:
        result[key] = h1_motion[key].detach().cpu().numpy()
    np.savez(save_path, result)


def test():

    device = "cuda:0"

    chain = load_urdf("/home/liuyun/Humanoid_IL_Benchmark/retargeting/assets/h1_description/urdf/h1.urdf", device=device)
    link_names = get_h1_link_names()

    # load h1 motion data
    motion_data_path = "/home/liuyun/Humanoid_IL_Benchmark/humanplus_zhikai/HST/isaacgym/h1_motion_data/ACCAD_walk_10fps.npy"
    motion_data = np.load(motion_data_path)
    motion_offset = np.array([0.0000, 0.0000, -0.3490, 0.6980, -0.3490, 0.0000, 0.0000, -0.3490, 0.6980, -0.3490, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
    motion_data += motion_offset
    N_frame = motion_data.shape[0]
    global_rotation = torch.tensor([0, 0, 1.0]).unsqueeze(0).repeat(N_frame, 1).to(device=device)
    global_translation = torch.tensor([0, 0, 1.05]).unsqueeze(0).repeat(N_frame, 1).to(device=device)

    gt_link_to_world_dict = forward_kinematics(chain, link_names, motion_data, global_rotation=global_rotation, global_translation=global_translation, device=device)
    # print(gt_link_to_world_dict, gt_link_to_world_dict["pelvis"].shape)
    # assert False

    ########################## start optimization #################################
    h1_motion_model = H1_Motion_Model(batch_size=N_frame, device=device)

    optimizer = torch.optim.Adam(h1_motion_model.parameters(), lr=3e-3)
    h1_motion_model.train()

    constraint_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    for epoch in range(1000):
        h1_motion = h1_motion_model()

        pred_link_to_world_dict = forward_kinematics(chain, link_names, h1_motion["joint_angles"], global_rotation=h1_motion["global_rotations"], global_translation=h1_motion["global_translations"], device=device)

        loss = 0
        for idx in constraint_ids:
            loss += ((pred_link_to_world_dict[link_names[idx]][:, :3, 3] - gt_link_to_world_dict[link_names[idx]][:, :3, 3])**2).sum(dim=-1).mean()
        
        if epoch % 100 == 0:
            print(epoch, loss.item())
            print(h1_motion["joint_angles"][0].detach().cpu().numpy() - motion_data[0])
            print(h1_motion["global_rotations"][0].detach().cpu().numpy())
            print(h1_motion["global_translations"][0].detach().cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # save
    h1_motion = h1_motion_model()
    save_predicted_h1_motion(h1_motion, "./pred.npz")
    
    ########################## finish optimization ################################


if __name__ == "__main__":
    test()
