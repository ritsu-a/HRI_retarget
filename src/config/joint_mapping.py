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
