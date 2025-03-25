### dataset link names
SEG_LINKS = ['Hips', 'Chest', 'Chest2', 'Neck', 'Head',
    'LeftCollar', 'LeftShoulder', 'LeftElbow', 'LeftWrist', 
    'LeftFinger0', 'LeftFinger01', 'LeftFinger02', 'LeftFinger1', 
    'LeftFinger11', 'LeftFinger12', 'LeftFinger2', 'LeftFinger21', 
    'LeftFinger22', 'LeftFinger3', 'LeftFinger31', 'LeftFinger32', 
    'LeftFinger4', 'LeftFinger41', 'LeftFinger42', 'RightCollar', 
    'RightShoulder', 'RightElbow', 'RightWrist', 'RightFinger0', 
    'RightFinger01', 'RightFinger02', 'RightFinger1', 'RightFinger11', 
    'RightFinger12', 'RightFinger2', 'RightFinger21', 'RightFinger22', 
    'RightFinger3', 'RightFinger31', 'RightFinger32', 'RightFinger4', 
    'RightFinger41', 'RightFinger42', 'LeftHip', 'LeftKnee', 
    'LeftAnkle', 'LeftToe', 'RightHip', 'RightKnee', 'RightAnkle', 'RightToe']

SG_LINKS = ['Hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg',
    'LeftLeg', 'LeftFoot', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Neck1', 
    'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 
    'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightInHandIndex', 
    'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightInHandMiddle', 
    'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightInHandRing', 
    'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightInHandPinky', 
    'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'LeftShoulder', 
    'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandThumb1', 'LeftHandThumb2', 
    'LeftHandThumb3', 'LeftInHandIndex', 'LeftHandIndex1', 'LeftHandIndex2', 
    'LeftHandIndex3', 'LeftInHandMiddle', 'LeftHandMiddle1', 'LeftHandMiddle2', 
    'LeftHandMiddle3', 'LeftInHandRing', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 
    'LeftInHandPinky', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3']






### robot link names
GALBOT_CHARLIE_LINKS = ['mobile_base', 'base_link_x', 'base_link_y', 'base_link_z',
    'base_link', 'omni_chassis_base_link', 'omni_chassis_leg_mount_link', 'leg_base_link',
    'leg_link1', 'leg_link2', 'leg_link3', 'leg_link4', 
    'leg_end_effector_mount_link', 'torso_base_link', 'torso_head_mount_link', 
    'head_base_link', 'head_link1', 'head_link2', 'head_end_effector_mount_link', 
    'torso_right_arm_mount_link', 'right_arm_base_link', 'right_arm_link1', 
    'right_arm_link2', 'right_arm_link3', 'right_arm_link4', 'right_arm_link5', 
    'right_arm_link6', 'right_arm_link7', 'right_arm_end_effector_mount_link', 
    'right_arm_force_sensor_sim_view_frame', 'right_suction_cup_base_link', 
    'right_suction_cup_link1', 'right_suction_cup_tcp_link', 'right_flange_link', 
    'torso_left_arm_mount_link', 'left_arm_base_link', 'left_arm_link1', 
    'left_arm_link2', 'left_arm_link3', 'left_arm_link4', 'left_arm_link5', 
    'left_arm_link6', 'left_arm_link7', 'left_arm_end_effector_mount_link', 
    'left_arm_force_sensor_sim_view_frame', 'left_gripper_base_link', 'left_gripper_l1_link', 
    'left_gripper_l3_link', 'left_gripper_l2_link', 'left_gripper_r1_link', 'left_gripper_r3_link', 
    'left_gripper_r2_link', 'left_gripper_left_link', 'left_gripper_right_link', 'left_gripper_tcp_link', 
    'left_flange_link']

G1_LINKS = ['pelvis', 'pelvis_contour_link', 'left_hip_pitch_link', 'left_hip_roll_link', 
            'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 
            'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 
            'right_ankle_pitch_link', 'right_ankle_roll_link', 'waist_yaw_link', 'waist_roll_link', 
            'torso_link', 'logo_link', 'head_link', 'waist_support_link', 'imu_link', 'd435_link', 
            'mid360_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 
            'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 
            'left_rubber_hand', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 
            'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link', 'right_rubber_hand']



### mapping from dataset to robot
SEG_GALBOT_CHARLIE_CORRESPONDENCE = [
    [SEG_LINKS.index("Hips"), GALBOT_CHARLIE_LINKS.index("leg_link3"), 5],
    [SEG_LINKS.index("Neck"), GALBOT_CHARLIE_LINKS.index("head_link1"), 5],

    [SEG_LINKS.index("LeftShoulder"), GALBOT_CHARLIE_LINKS.index("left_arm_link1"), 3],
    [SEG_LINKS.index("LeftElbow"), GALBOT_CHARLIE_LINKS.index("left_arm_link3"), 2],
    [SEG_LINKS.index("LeftWrist"), GALBOT_CHARLIE_LINKS.index("left_arm_link5"), 3],

    [SEG_LINKS.index("RightShoulder"), GALBOT_CHARLIE_LINKS.index("right_arm_link1"), 3],
    [SEG_LINKS.index("RightElbow"), GALBOT_CHARLIE_LINKS.index("right_arm_link3"), 2],
    [SEG_LINKS.index("RightWrist"), GALBOT_CHARLIE_LINKS.index("right_arm_link5"), 3],
]

SG_GALBOT_CHARLIE_CORRESPONDENCE = [
    [SG_LINKS.index("Spine"), GALBOT_CHARLIE_LINKS.index("leg_link3"), 5],
    # [SG_LINKS.index("Neck"), GALBOT_CHARLIE_LINKS.index(""), 5],

    [SG_LINKS.index("LeftArm"), GALBOT_CHARLIE_LINKS.index("left_arm_link1"), 3],
    [SG_LINKS.index("LeftForeArm"), GALBOT_CHARLIE_LINKS.index("left_arm_link3"), 2],
    [SG_LINKS.index("LeftHand"), GALBOT_CHARLIE_LINKS.index("left_arm_link5"), 3],
    
    [SG_LINKS.index("RightArm"), GALBOT_CHARLIE_LINKS.index("right_arm_link1"), 3],
    [SG_LINKS.index("RightForeArm"), GALBOT_CHARLIE_LINKS.index("right_arm_link3"), 2],
    [SG_LINKS.index("RightHand"), GALBOT_CHARLIE_LINKS.index("right_arm_link5"), 3],
]

SG_G1_CORRESPONDENCE = [
    [SG_LINKS.index("Spine"), G1_LINKS.index("pelvis"), 1],
    [SG_LINKS.index("Head"), G1_LINKS.index("mid360_link"), 1],

    [SG_LINKS.index("LeftArm"), G1_LINKS.index("left_shoulder_pitch_link"), 3],
    # [SG_LINKS.index("LeftArm"), G1_LINKS.index("left_shoulder_roll_link"), 1],
    # [SG_LINKS.index("LeftArm"), G1_LINKS.index("left_shoulder_yaw_link"), 1],
    [SG_LINKS.index("LeftForeArm"), G1_LINKS.index("left_elbow_link"), 3],
    # [SG_LINKS.index("LeftHand"), G1_LINKS.index("left_wrist_roll_link"), 1],
    [SG_LINKS.index("LeftHand"), G1_LINKS.index("left_wrist_pitch_link"), 3],
    # [SG_LINKS.index("LeftHand"), G1_LINKS.index("left_wrist_yaw_link"), 1],
    [SG_LINKS.index("LeftHandMiddle3"), G1_LINKS.index("left_rubber_hand"), 3],

    

    [SG_LINKS.index("RightArm"), G1_LINKS.index("right_shoulder_pitch_link"), 3],
    # [SG_LINKS.index("RightArm"), G1_LINKS.index("right_shoulder_roll_link"), 1],
    # [SG_LINKS.index("RightArm"), G1_LINKS.index("right_shoulder_yaw_link"), 1],
    [SG_LINKS.index("RightForeArm"), G1_LINKS.index("right_elbow_link"), 3],
    # [SG_LINKS.index("RightHand"), G1_LINKS.index("right_wrist_roll_link"), 1],
    [SG_LINKS.index("RightHand"), G1_LINKS.index("right_wrist_pitch_link"), 3],
    # [SG_LINKS.index("RightHand"), G1_LINKS.index("right_wrist_yaw_link"), 1],
    [SG_LINKS.index("RightHandMiddle3"), G1_LINKS.index("right_rubber_hand"), 3],

]

