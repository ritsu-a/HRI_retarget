# Reference Unitree H1 SDK https://support.unitree.com/home/en/H1_developer/Joint_motor_sequence
from enum import Enum


# Message Joint Table
# Joint Index	Joint Name	Servo Mode
# 0	    right hip roll	0x0A
# 1	    right hip pitch	0x0A
# 2	    right knee	0x0A
# 3	    left hip roll	0x0A
# 4	    left hip pitch	0x0A
# 5	    left knee	0x0A
# 6	    waist yaw	0x0A
# 7	    left hip yaw	0x0A
# 8	    right hip yaw	0x0A
# 9	-
# 10    left ankle	0x01
# 11	right ankle	0x01
# 12	right shoulder pitch	0x01
# 13	right shoulder roll	0x01
# 14	right shoulder yaw	0x01
# 15	right elbow	0x01
# 16	left shoulder pitch	0x01
# 17	left shoulder roll	0x01
# 18	left shoulder yaw	0x01
# 19	left elbow	0x01

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


# URDF joint index
# right hip yaw	8	0x0A
# right hip roll	0	0x0A
# right hip pitch	1	0x0A
# right knee	2	0x0A
# right ankle	11	0x01
# left hip yaw	7	0x0A
# left hip roll	3	0x0A
# left hip pitch	4	0x0A
# left knee	5	0x0A
# left ankle	10	0x01
# waist yaw	6	0x0A
# right shoulder pitch	12	0x01
# right shoulder roll	13	0x01
# right shoulder yaw	14	0x01
# right elbow	15	0x01
# left shoulder pitch	16	0x01
# left shoulder roll	17	0x01
# left shoulder yaw	18	0x01
# left elbow	19	0x01

class URDFJointID(Enum):
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
    waist_yaw_joint = 6
    right_shoulder_pitch_joint = 12
    right_shoulder_roll_joint = 13
    right_shoulder_yaw_joint = 14
    right_elbow_joint = 15
    left_shoulder_pitch_joint = 16
    left_shoulder_roll_joint = 17
    left_shoulder_yaw_joint = 18
    left_elbow_joint = 19


_MESSAGE_JOINT_TABLE = [
    (0, H1JointID.right_hip_roll_joint, 0x0A),
    (1, H1JointID.right_hip_pitch_joint, 0x0A),
    (2, H1JointID.right_knee_joint, 0x0A),
    (3, H1JointID.left_hip_roll_joint, 0x0A),
    (4, H1JointID.left_hip_pitch_joint, 0x0A),
    (5, H1JointID.left_knee_joint, 0x0A),
    (6, H1JointID.waist_yaw_joint, 0x0A),
    (7, H1JointID.left_hip_yaw_joint, 0x0A),
    (8, H1JointID.right_hip_yaw_joint, 0x0A),
    (9, H1JointID.none_joint, 0x00),
    (10, H1JointID.left_ankle_joint, 0x01),
    (11, H1JointID.right_ankle_joint, 0x01),
    (12, H1JointID.right_shoulder_pitch_joint, 0x01),
    (13, H1JointID.right_shoulder_roll_joint, 0x01),
    (14, H1JointID.right_shoulder_yaw_joint, 0x01),
    (15, H1JointID.right_elbow_joint, 0x01),
    (16, H1JointID.left_shoulder_pitch_joint, 0x01),
    (17, H1JointID.left_shoulder_roll_joint, 0x01),
    (18, H1JointID.left_shoulder_yaw_joint, 0x01),
    (19, H1JointID.left_elbow_joint, 0x01),
]

MESSAGE_JOINT_DICT = {name: (idx, mode) for idx, name, mode in _MESSAGE_JOINT_TABLE}

RIGHT_LEG_JOINTS_ID = [
    H1JointID.right_hip_roll_joint,
    H1JointID.right_hip_pitch_joint,
    H1JointID.right_hip_yaw_joint,
    H1JointID.right_knee_joint,
    H1JointID.right_ankle_joint,
]

LEFT_LEG_JOINTS_ID = [
    H1JointID.left_hip_roll_joint,
    H1JointID.left_hip_pitch_joint,
    H1JointID.left_hip_yaw_joint,
    H1JointID.left_knee_joint,
    H1JointID.left_ankle_joint,
]

TORSO_ID = [H1JointID.waist_yaw_joint]

RIGHT_ARM_JOINTS_ID = [
    H1JointID.right_shoulder_pitch_joint,
    H1JointID.right_shoulder_roll_joint,
    H1JointID.right_shoulder_yaw_joint,
    H1JointID.right_elbow_joint,
]

LEFT_ARM_JOINTS_ID = [
    H1JointID.left_shoulder_pitch_joint,
    H1JointID.left_shoulder_roll_joint,
    H1JointID.left_shoulder_yaw_joint,
    H1JointID.left_elbow_joint,
]


def get_joint_ids_by_names(names: list, table: dict):
    joint_ids = [table[name][0] for name in names]
    return joint_ids


# Control Mode
HIGHLEVEL = 0xEE
LOWLEVEL = 0xFF
TRIGERLEVEL = 0xF0
PosStopF = 2.146e9
VelStopF = 16000.0

H1TorqueLimit = {
    H1JointID.right_hip_roll_joint: 200,
    H1JointID.right_hip_pitch_joint: 200,
    H1JointID.right_knee_joint: 300,
    H1JointID.left_hip_roll_joint: 200,
    H1JointID.left_hip_pitch_joint: 200,
    H1JointID.left_knee_joint: 300,
    H1JointID.waist_yaw_joint: 200,
    H1JointID.left_hip_yaw_joint: 200,
    H1JointID.right_hip_yaw_joint: 200,
    H1JointID.none_joint: 1,
    H1JointID.left_ankle_joint: 40,
    H1JointID.right_ankle_joint: 40,
    H1JointID.right_shoulder_pitch_joint: 40,
    H1JointID.right_shoulder_roll_joint: 40,
    H1JointID.right_shoulder_yaw_joint: 18,
    H1JointID.right_elbow_joint: 18,
    H1JointID.left_shoulder_pitch_joint: 40,
    H1JointID.left_shoulder_roll_joint: 40,
    H1JointID.left_shoulder_yaw_joint: 18,
    H1JointID.left_elbow_joint: 18,
}
