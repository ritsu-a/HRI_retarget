import math
import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
import pytorch_kinematics as pk

from utils.torch_utils.diff_quat import vec6d_to_matrix

from config.joint_mapping import G1_INSPIREHANDS_LINKS, BEAT_G1_INSPIREHANDS_CORRESPONDENCE, G1_LOWERBODY_LINKS
# from config.joint_mapping import G1_COLLISION_CAPSULE, G1_COLLISION
### TODO replace with g1_inspirehand collision
from HRI_retarget import ROOT,SRC_ROOT,DATA_ROOT

from utils.motion_lib.strechable_chain import load_urdf_as_stretchable_chain
from model.g1_base_model import G1_Base_Motion_Model

from dex_retargeting.retargeting_config import RetargetingConfig
import yaml


class G1_Inspirehands_Motion_Model(G1_Base_Motion_Model):
    def __init__(self, batch_size=1, joint_correspondence=BEAT_G1_INSPIREHANDS_CORRESPONDENCE, device="cuda:0"):
        super(G1_Inspirehands_Motion_Model, self).__init__(batch_size=batch_size, joint_correspondence=joint_correspondence, device=device)

      
        self.dof = 53

        self.init_angles = torch.zeros(self.batch_size, self.dof)

        self.dof_max_limits = torch.from_numpy(np.array([
                [-2.530, 2.879], ## left_hip_pitch
                [-0.523, 2.967], ## left_hip_roll
                [-0.087267, 2.757], ## left_hip_yaw
                [-0.087267, 2.879], ## left_knee
                [-0.87267, 0.523], ## left_ankle_pitch
                [-0.2618, 0.2618], ## left_ankle_roll
                [-2.5307, 2.8798], ## right_hip_pitch
                [-2.9671, 0.5236], ## right_hip_roll
                [-2.7576, 2.7576], ## right_hip_yaw
                [-0.087267, 2.8798], ## right_knee
                [-0.87267, 0.5236], ## right_ankle_pitch
                [-0.2618, 0.2618], ## right_ankle_roll
                [-2.618, 2.618], ## waist_yaw
                [-0.52, 0.52], ## waist_roll
                [-0.52, 0.52], ## waist_pitch
                [-3.0892, 2.6704], ## left_shoulder_pitch
                [-1.5882, 2.2515], ## left_should_roll
                [-2.618, 2.618], ## left_shoulder_yaw
                [-1.0472, 2.0944], ## left_elbow
                [-1.972222054, 1.972222054], ## left_wrist_roll
                [-1.614429558, 1.614429558], ##left_wrist_pitch
                [-1.614429558, 1.614429558], ## left_wrist_yaw
                [0, 1.7], ##L_index_proximal_joint
                [0, 1.7], ##L_index_intermediate_joint
                [0, 1.7], ##L_middle_proximal_joint
                [0, 1.7], ##L_middle_intermediate_joint
                [0, 1.7], ##L_pinky_proximal_joint
                [0, 1.7], ##L_pinky_intermediate_joint
                [0, 1.7], ##L_ring_proximal_joint
                [0, 1.7], ##L_ring_intermediate_joint
                [-0.1, 1.3], ## L_thumb_proximal_yaw_joint
                [-0.1, 0.6], ##L_thumb_proximal_pitch_joint
                [0, 0.8], ##L_thumb_intermediate_joint
                [0, 1.2], ##L_thumb_distal_joint
                [-3.0892, 2.6704], ## right_shoulder_pitch
                [-2.2515, 1.5882], ## right_should_roll
                [-2.618, 2.618], ## right_shoulder_yaw
                [-1.0472, 2.0944], ## right_elbow
                [-1.972222054, 1.972222054], ## right_wrist_roll
                [-1.614429558, 1.614429558], ## right_wrist_pitch
                [-1.614429558, 1.614429558], ## right_wrist_yaw
                [0, 1.7], ##R_index_proximal_joint
                [0, 1.7], ##R_index_intermediate_joint
                [0, 1.7], ##R_middle_proximal_joint
                [0, 1.7], ##R_middle_intermediate_joint
                [0, 1.7], ##R_pinky_proximal_joint
                [0, 1.7], ##R_pinky_intermediate_joint
                [0, 1.7], ##R_ring_proximal_joint
                [0, 1.7], ##R_ring_intermediate_joint
                [-0.1, 1.3],##R_thumb_proximal_yaw_joint
                [-0.1, 0.6], ##R_thumb_proximal_pitch_joint
                [0, 0.8], ##R_thumb_intermediate_joint
                [0, 1.2], ##R_thumb_distal_joint
        ])).repeat(self.batch_size, 1, 1).to(dtype=torch.float32, device=self.device)


        ### soft threshold
        self.dof_limits = self.dof_max_limits * 0.9
       

        ### joint scales upper and lower bound 
        self.joint_scales_min = 0.8
        self.joint_scales_max = 1.2

        ## learnable parameters 
        self.joint_angles = nn.Parameter(torch.zeros(batch_size, self.dof).to(device), requires_grad=True)  # (N, dof)
        self.joint_scales = nn.Parameter(torch.ones(self.dof).to(device), requires_grad=True)  # (dof)

        self.joint_correspondence = joint_correspondence

        self.chain = None

        self.links = G1_INSPIREHANDS_LINKS
        self.lower_body_links = G1_LOWERBODY_LINKS

        ### apply scale and transformation on robot frame
        ### full body retargeting should have different rot and trans for each frame
        self.scale = nn.Parameter(torch.ones(3).to(device), requires_grad=True)
        self.global_rot = nn.Parameter(torch.eye(3)[:, :2].repeat(self.batch_size, 1, 1).to(device), requires_grad=True)
        self.global_trans = nn.Parameter(torch.zeros(3).reshape(3, 1).repeat(self.batch_size, 1, 1).to(device), requires_grad=True)

        ### modify lowerbody scale to match robot and human shape
        self.lower_body_scale = torch.ones(3).requires_grad_(False).to(device)

        urdf_rel_path = "resources/robots/g1_inspirehands/G1_inspire_hands.urdf"
        self.chain = load_urdf_as_stretchable_chain(os.path.join(DATA_ROOT,urdf_rel_path)).to(dtype=torch.float32, device=self.device)
        
    

    def forward_kinematics(self):
        """
        chain: pytorch_kinematics.chain.Chain
        joint_angle: (N, dof) 24D vector
        global_translation: (N, 3) 3D vector, root_to_world
        global_orientation: (N, 3) 3D axis-angle, root_to_world

        return: a tensor contains the global poses of 52 links
        """
        # R = vec6d_to_matrix(self.global_rot) * self.scale.repeat(self.batch_size, 3, 1) # (N_frame, 3, 3)
        R = vec6d_to_matrix(self.global_rot)# (N_frame, 3, 3)
        t = self.global_trans # (N_frame, 3, 1)
        root_to_world = torch.cat((torch.cat((R, t), dim=-1), torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat(self.batch_size, 1, 1).to(self.device)), dim=1)  # (N_frame, 4, 4)
        
        # R_lower_body = vec6d_to_matrix(self.global_rot) * self.scale.repeat(self.batch_size, 3, 1) * self.lower_body_scale.repeat(self.batch_size, 3, 1)# (N_frame, 3, 3)
        R_lower_body = vec6d_to_matrix(self.global_rot)
        lower_body_root_to_world = torch.cat((torch.cat((R_lower_body, t), dim=-1), torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat(self.batch_size, 1, 1).to(self.device)), dim=1)  # (N_frame, 4, 4)
        

        link_to_root_dict = self.chain.forward_kinematics(self.joint_angles, self.joint_scales)  # link to root
        link_to_world_dict = []
        for link_name in self.links:
            T = link_to_root_dict[link_name].get_matrix()  # link to root
            if link_name in self.lower_body_links:
                link_to_world_dict.append(torch.einsum('bij,bjk->bik', lower_body_root_to_world, T))
            else:
                link_to_world_dict.append(torch.einsum('bij,bjk->bik', root_to_world, T))


        link_to_world_dict = torch.stack(link_to_world_dict, dim=1) # (N_frame, 52, 4, 4)
        return link_to_world_dict
    
    def set_hand_optimizer(self,config_file_path,default_urdf_dir):
        """
        set the dex-retargeting optimizer
        """
        RetargetingConfig.set_default_urdf_dir(default_urdf_dir)
        with open(config_file_path, 'r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_hand_optimizer = left_retargeting_config.build()
        self.right_hand_optimizer = right_retargeting_config.build()

        # self.left_hand_optimizer = left_retargeting
        # self.right_hand_optimizer = right_retargeting

    def set_hand_tip_positions(self,left_hand_tip_positions, right_hand_tip_positions):
        """ 
        hand_tip_positions:(N_frammes ,5 ,3)
        """
        self.left_hand_tip_positions = left_hand_tip_positions
        self.right_hand_tip_positions = right_hand_tip_positions
        
    def set_hand_rotations(self,left_hand_rotations, right_hand_rotations):
        """
        hand_rotations:(N_frammes ,3 ,3)
        the relative rotation from LeftHand to LeftForeArm; from RightHand to RightForeArm
        """
        self.left_hand_rotations = left_hand_rotations
        self.right_hand_rotations = right_hand_rotations

    def resolve_euler_angles(self,rot):
        """
        resolve the wrist_roll,wrist_pitch, wrist_yaw from the rotations
        rot: (N_frames, 3, 3)
        return : wrist_roll, wrist_pitch, wrist_yaw(N_frames,3)
        """
        r11 = rot[:,0,0]
        r12 = rot[:,0,1]
        r13 = rot[:,0,2]
        r23 = rot[:,1,2]
        r33 = rot[:,2,2]
        
        pitch = torch.atan2(r13, torch.sqrt(r11**2 + r12**2))
        yaw = torch.atan2(-r12/torch.cos(pitch),r11/torch.cos(pitch))
        roll = torch.atan2(-r23/torch.cos(pitch),r33/torch.cos(pitch))
        return torch.stack([roll,pitch,yaw],dim=1)
    
    def refine_wrist_angle(self):
        left_angles = self.resolve_euler_angles(self.left_hand_rotations)
        right_angles = self.resolve_euler_angles(self.right_hand_rotations)
        self.joint_angles[:,19:22] = left_angles # left_wrist_roll, left_wrist_pitch, left_wrist_yaw
        self.joint_angles[:,38:41] = right_angles # right_wrist_roll, right_wrist_pitch, right_wrist_yaw

    def refine_hand_angle(self):
        self.left_qpos_list = np.zeros((self.batch_size,12))
        self.right_qpos_list = np.zeros((self.batch_size,12))
        for i in range(self.batch_size):
            left_ref = self.left_hand_tip_positions[i,:,:].numpy()
            right_ref = self.right_hand_tip_positions[i,:,:].numpy()
            left_qpos = self.left_hand_optimizer.retarget(left_ref)
            right_qpos = self.right_hand_optimizer.retarget(right_ref)
            self.left_qpos_list[i,:] = self.fingerpos_clip(left_qpos[1:])
            self.right_qpos_list[i,:] = self.fingerpos_clip(right_qpos[1:])
            
    def fingerpos_clip(self,qpos):
        limits = np.array([
            [0, 1.7], ##L_index_proximal_joint
            [0, 1.7], ##L_index_intermediate_joint
            [0, 1.7], ##L_middle_proximal_joint
            [0, 1.7], ##L_middle_intermediate_joint
            [0, 1.7], ##L_pinky_proximal_joint
            [0, 1.7], ##L_pinky_intermediate_joint
            [0, 1.7], ##L_ring_proximal_joint
            [0, 1.7], ##L_ring_intermediate_joint
            [-0.1, 1.3], ## L_thumb_proximal_yaw_joint
            [-0.1, 0.6], ##L_thumb_proximal_pitch_joint
            [0, 0.8], ##L_thumb_intermediate_joint
            [0, 1.2], ##L_thumb_distal_joint
        ])
        qpos[qpos < limits[:,0]] = limits[:,0][qpos < limits[:,0]]
        qpos[qpos > limits[:,1]] = limits[:,1][qpos > limits[:,1]]
        return qpos 