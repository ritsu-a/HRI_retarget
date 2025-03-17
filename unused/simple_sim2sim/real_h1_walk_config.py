import time
from enum import Enum

import numpy as np


class H1JointID(Enum):
    right_hip_roll_joint = 0
    right_hip_pitch_joint = 1
    right_knee_joint = 2
    left_hip_roll_joint = 3
    left_hip_pitch_joint = 4
    left_knee_joint = 5
    waist_yaw_joint = 6
    left_hip_yaw_joint = 7
    right_hip_yaw_joint = 8
    none_joint = 9
    left_ankle_joint = 10
    right_ankle_joint = 11
    right_shoulder_pitch_joint = 12
    right_shoulder_roll_joint = 13
    right_shoulder_yaw_joint = 14
    right_elbow_joint = 15
    left_shoulder_pitch_joint = 16
    left_shoulder_roll_joint = 17
    left_shoulder_yaw_joint = 18
    left_elbow_joint = 19


# v3
# default_joint_angles = {
#     'left_hip_yaw_joint': 0.0,
#     'left_hip_roll_joint': 0.0,
#     'left_hip_pitch_joint': -0.45,
#     'left_knee_joint': 1.05,
#     'left_ankle_joint': -0.585,
#     'right_hip_yaw_joint': 0.0,
#     'right_hip_roll_joint': 0.0,
#     'right_hip_pitch_joint': -0.45,
#     'right_knee_joint': 1.05,
#     'right_ankle_joint': -0.585,
#     'torso_joint': 0.0,
#     'left_shoulder_pitch_joint': 0.44,
#     'left_shoulder_roll_joint': -0.01,
#     'left_shoulder_yaw_joint': 0.0,
#     'left_elbow_joint': 0.308,
#     'right_shoulder_pitch_joint': 0.44,
#     'right_shoulder_roll_joint': 0.01,
#     'right_shoulder_yaw_joint': 0.0,
#     'right_elbow_joint': 0.308
# }

DEFAULT_JOINT_POSITION = {
    # "left_hip_yaw_joint": 0.0,
    # "left_hip_roll_joint": 0.0,
    # "left_hip_pitch_joint": -0.45,
    # "left_knee_joint": 1.05,
    # "left_ankle_joint": -0.585,
    # "right_hip_yaw_joint": 0.0,
    # "right_hip_roll_joint": 0.0,
    # "right_hip_pitch_joint": -0.45,
    # "right_knee_joint": 1.05,
    # "right_ankle_joint": -0.585,
    # "torso_joint": 0.0,
    # "left_shoulder_pitch_joint": 0.0,
    # "left_shoulder_roll_joint": 0.0,
    # "left_shoulder_yaw_joint": 0.0,
    # "left_elbow_joint": 0.0,
    # "right_shoulder_pitch_joint": 0.0,
    # "right_shoulder_roll_joint": 0.0,
    # "right_shoulder_yaw_joint": 0.0,
    # "right_elbow_joint": 0.0,
    # 'left_hip_yaw_joint': 0.0,
    # 'left_hip_roll_joint': 0.0,
    # 'left_hip_pitch_joint': -0.349,
    # 'left_knee_joint': 0.698,
    # 'left_ankle_joint': -0.349,
    # 'right_hip_yaw_joint': 0.0,
    # 'right_hip_roll_joint': 0.0,
    # 'right_hip_pitch_joint': -0.349,
    # 'right_knee_joint': 0.698,
    # 'right_ankle_joint': -0.349,
    # 'torso_joint': 0.0,
    # 'left_shoulder_pitch_joint': 0.0,
    # 'left_shoulder_roll_joint': 0.0,
    # 'left_shoulder_yaw_joint': 0.0,
    # 'left_elbow_joint': 0.0,
    # 'right_shoulder_pitch_joint': 0.0,
    # 'right_shoulder_roll_joint': 0.0,
    # 'right_shoulder_yaw_joint': 0.0,
    # 'right_elbow_joint': 0.0,
            'left_hip_yaw_joint': 0.0,
            'left_hip_roll_joint': 0.0,
            'left_hip_pitch_joint': -0.45,
            'left_knee_joint': 1.05,
            'left_ankle_joint': -0.585,
            'right_hip_yaw_joint': 0.0,
            'right_hip_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.45,
            'right_knee_joint': 1.05,
            'right_ankle_joint': -0.585,
            'torso_joint': 0.0,
            'left_shoulder_pitch_joint': 0.44,
            'left_shoulder_roll_joint': -0.01,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.308,
            'right_shoulder_pitch_joint': 0.44,
            'right_shoulder_roll_joint': 0.01,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 0.308
}


class WholeBodyJointID(Enum): # no 9
    left_hip_yaw_joint = 7
    left_hip_roll_joint = 3
    left_hip_pitch_joint = 4
    left_knee_joint = 5
    left_ankle_joint = 10
    right_hip_yaw_joint = 8
    right_hip_roll_joint = 0
    right_hip_pitch_joint = 1
    right_knee_joint = 2
    right_ankle_joint = 11
    torso_joint = 6
    left_shoulder_pitch_joint = 16
    left_shoulder_roll_joint = 17
    left_shoulder_yaw_joint = 18
    left_elbow_joint = 19
    right_shoulder_pitch_joint = 12
    right_shoulder_roll_joint = 13
    right_shoulder_yaw_joint = 14
    right_elbow_joint = 15


class RealH1WalkConfig:
    class deploy:
        use_gait = True
        pre_inference = True  # inference when waiting before robot apply action

    class topics:
        state_estimator_topic_name = "/state/state_estimator"
        remote_controller_topic_name = "/command/remote_controller"
        h1_low_cmd_topic_name = "/command/low_cmd"

    class policy:
        dt = 0.02
        action_scale = 0.25 # ?

        class normalization:
            class obs_scales: # ?
                dof_pos = 1.0
                dof_vel = 0.05
                last_action = 1.0
                base_ang_vel = 0.25
                target_dof_pos = 1.0
                target_dof_vel = 0.05
                target_root_ori = 1.0
                target_root_vel = 0.05
                target_root_ang_vel = 0.05

            clip_observations = 100.0
            clip_actions = 5.0

        class commands:
            lin_vel_clip = 0.3
            ang_vel_clip = 0.3

            class ranges:
                lin_vel_x = [-0.5, 0.6]  # min max [m/s]
                lin_vel_y = [-0.5, 0.5]  # min max [m/s]
                ang_vel_yaw = [-0.51, 0.51]  # min max [rad/s]

        @staticmethod
        def get_default_sim_joint_position():
            return np.float32(list(DEFAULT_JOINT_POSITION.values()))

        @staticmethod
        def joint_sim_to_real(sim_q):
            h1_q = np.zeros(len(H1JointID), dtype=np.float32)
            for i, h1_id in enumerate(WholeBodyJointID):
                h1_q[h1_id.value] = sim_q[i]
            return h1_q

        @staticmethod
        def joint_real_to_sim(real_q):
            urdf_q = np.zeros(len(WholeBodyJointID), dtype=np.float32)
            for i, h1_id in enumerate(WholeBodyJointID):
                urdf_q[i] = real_q[h1_id.value]
            return urdf_q

    class gait:
        frequencies = 1.5
        phase_offset = 0.5
        stance_ratio = 0.7
        kappa_gait_probs = 0.07
