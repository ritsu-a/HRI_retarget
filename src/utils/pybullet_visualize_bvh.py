### visualize bvh via pybullet
### ALERT: buggy abandoned
### TODO: decouple bvh load and vis
### usage: 
### python pybullet_visualize_bvh.py

import pybullet as p
import time
import ipdb
from bvh_io import Bvh
import os
import numpy as np

# 禁用OpenGL的调试输出
os.environ["PYBULLET_VERBOSE"] = "0"

# 连接物理引擎
physicsClient = p.connect(p.GUI)
# p.setGravity(0, 0, -9.81)

# 加载BVH文件
with open("/home/pengyang/codebase/H1_RL/data/motion/ARMS_RAISE_TOWORDS_SKY-1.bvh") as f:
    mocap = Bvh(f.read())

# 获取骨骼层次结构
joints = mocap.get_joints()

# 创建骨架的根节点
base_pos = [0, 0, 0]
base_orient = p.getQuaternionFromEuler([0, 0, 0])
base_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
base_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 1])
base_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=base_shape, baseVisualShapeIndex=base_visual, basePosition=base_pos, baseOrientation=base_orient)


# 创建其他关节
joint_bodies = {}
for joint in joints:
    if joint.value[0] == "ROOT":
        continue  # 跳过根节点
    parent = mocap.joint_parent(joint.value[1])
    offset = mocap.joint_offset(joint.value[1])
    joint_pos = [base_pos[0] + offset[0], base_pos[1] + offset[1], base_pos[2] + offset[2]]
    joint_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
    joint_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 1])
    joint_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=joint_shape, baseVisualShapeIndex=joint_visual, basePosition=joint_pos)
    joint_bodies[joint] = joint_body

    # 创建关节
    if parent.value[0] == "ROOT":
        parent_body = base_body
    else:
        parent_body = joint_bodies[parent]
    p.createConstraint(parent_body, -1, joint_body, -1, p.JOINT_POINT2POINT, [0, 0, 0], parentFramePosition=offset, childFramePosition=[0, 0, 0])

# 获取运动数据
frames = np.array(mocap.frames).astype('float32')
num_frames = len(frames)
frame_time = mocap.frame_time

# 主循环
try:
    for frame in frames:
        for i, joint in enumerate(joints):
            if joint.value[0] == "ROOT":
                continue  # 跳过根节点
            # 获取关节的旋转数据
            rotation = frame[i * 3:(i + 1) * 3]  # 假设每个关节有3个旋转通道
            # 将旋转数据应用到PyBullet中的关节
            p.resetBasePositionAndOrientation(joint_bodies[joint], joint_pos, p.getQuaternionFromEuler(rotation))
        p.stepSimulation()
        time.sleep(frame_time)
except KeyboardInterrupt:
    pass

# 断开连接
p.disconnect()