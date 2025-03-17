import torch

def forward_kinematics(chain, joint_angle, global_rotvec=None, global_translation=None, device="cuda:0"):
    """
    chain: pytorch_kinematics.chain.Chain
    link_names: list, len = 25
    joint_angle: (N, 19) 19D vector
    global_translation: (N, 3) 3D vector, root_to_world
    global_orientation: (N, 3) 3D axis-angle, root_to_world

    return: a dict contains the global poses of 25 links
    """

    N_frame = joint_angle.shape[0]
    R = vec6d_to_matrix(global_rotvec)  # (N_frame, 3, 3)
    t = global_translation.reshape(N_frame, 3, 1)  # (N_frame, 3, 1)
    root_to_world = torch.cat((torch.cat((R, t), dim=-1), torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat(N_frame, 1, 1).to(device)), dim=1)  # (N_frame, 4, 4)

    link_to_root_dict = chain.forward_kinematics(joint_angle)  # link to root
    link_to_world_dict = []
    for link_name in h1_rigid_body_names:
        T = link_to_root_dict[link_name].get_matrix()  # link to root
        link_to_world_dict.append(torch.einsum('bij,bjk->bik', root_to_world, T))
    link_to_world_dict = torch.stack(link_to_world_dict, dim=1) # (22, N_frame, 4, 4)
    return link_to_world_dict
