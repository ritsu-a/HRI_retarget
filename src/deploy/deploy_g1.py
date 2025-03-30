### from the official code example of https://github.com/unitreerobotics/unitree_sdk2_python
### refer to unitree doc for network connect
### usage example:(replace enp0s31f6 with net card name)
###     python src/deploy/deploy_g1.py enp0s31f6


import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

import numpy as np
import pickle

G1_NUM_MOTOR = 29

Kp = [
    60,
    60,
    60,
    100,
    40,
    40,  # legs
    60,
    60,
    60,
    100,
    40,
    40,  # legs
    60,
    40,
    40,  # waist
    40,
    40,
    40,
    40,
    40,
    40,
    40,  # arms
    40,
    40,
    40,
    40,
    40,
    40,
    40  # arms
]

Kd = [
    1,
    1,
    1,
    2,
    1,
    1,  # legs
    1,
    1,
    1,
    2,
    1,
    1,  # legs
    1,
    1,
    1,  # waist
    1,
    1,
    1,
    1,
    1,
    1,
    1,  # arms
    1,
    1,
    1,
    1,
    1,
    1,
    1  # arms 
]


class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11
    WaistYaw = 12
    WaistRoll = 13  # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13  # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14  # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14  # NOTE: INVALID for g1 23dof/29dof with waist locked
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20  # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21  # NOTE: INVALID for g1 23dof
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28  # NOTE: INVALID for g1 23dof


class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints


class Custom:

    def __init__(self):
        self.time_ = 0.0
        self.control_dt_ = 0.1  # [2ms]
        self.duration_ = 6.0  # [3 s]
        self.counter_ = 0
        self.mode_pr_ = Mode.PR
        self.mode_machine_ = 0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.update_mode_machine_ = False
        self.crc = CRC()
        self.time_step = 0

    def Init(self, joint_angles):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        self.joint_angles = joint_angles
        self.num_frames = joint_angles.shape[0]

        status, result = self.msc.CheckMode()
        while result['name']:
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        # create publisher #
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        # create subscriber #
        self.__lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.__lowstate_subscriber.Init(self.LowStateHandler, 10)

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(interval=self.control_dt_,
                                                    target=self.__LowCmdWrite,
                                                    name="control")
        while self.update_mode_machine_ == False:
            time.sleep(1)

        if self.update_mode_machine_ == True:
            self.lowCmdWriteThreadPtr.Start()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        if self.update_mode_machine_ == False:
            self.mode_machine_ = self.low_state.mode_machine
            self.update_mode_machine_ = True

        self.counter_ += 1
        if (self.counter_ % 500 == 0):
            self.counter_ = 0
            print(self.low_state.imu_state.rpy)

    def __LowCmdWrite(self):
        self.time_ += self.control_dt_

        if self.time_ < self.duration_:
            # [Stage 1]: set robot to zero posture
            for i in range(G1_NUM_MOTOR):
                ratio = np.clip(self.time_ / self.duration_, 0.0, 1.0)
                self.low_cmd.mode_pr = Mode.PR
                self.low_cmd.mode_machine = self.mode_machine_
                self.low_cmd.motor_cmd[i].mode = 1  # 1:Enable, 0:Disable
                self.low_cmd.motor_cmd[i].tau = 0.
                self.low_cmd.motor_cmd[i].q = (1.0 - ratio) * self.low_state.motor_state[i].q
                self.low_cmd.motor_cmd[i].dq = 0.
                self.low_cmd.motor_cmd[i].kp = Kp[i]
                self.low_cmd.motor_cmd[i].kd = Kd[i]

        elif self.time_ < self.duration_ * 2:
            # [Stage 2]: set robot to init posture
            ratio = np.clip((self.time_ - self.duration) / self.duration_, 0.0, 1.0)
            self.low_cmd.motor_cmd[G1JointIndex.WaistYaw].q = self.joint_angles[0, 0] * ratio
            self.low_cmd.motor_cmd[G1JointIndex.WaistRoll].q = self.joint_angles[0, 1] * ratio
            self.low_cmd.motor_cmd[G1JointIndex.WaistPitch].q = self.joint_angles[0, 2] * ratio
            self.low_cmd.motor_cmd[G1JointIndex.LeftShoulderPitch].q = self.joint_angles[0, 3] * ratio
            self.low_cmd.motor_cmd[G1JointIndex.LeftShoulderRoll].q = self.joint_angles[0, 4] * ratio
            self.low_cmd.motor_cmd[G1JointIndex.LeftShoulderYaw].q = self.joint_angles[0, 5] * ratio
            self.low_cmd.motor_cmd[G1JointIndex.LeftElbow].q = self.joint_angles[0, 6] * ratio
            self.low_cmd.motor_cmd[G1JointIndex.LeftWristRoll].q = self.joint_angles[0, 7] * ratio
            self.low_cmd.motor_cmd[G1JointIndex.LeftWristYaw].q = self.joint_angles[0, 8] * ratio
            self.low_cmd.motor_cmd[G1JointIndex.RightShoulderPitch].q = self.joint_angles[0, 9] * ratio
            self.low_cmd.motor_cmd[G1JointIndex.RightShoulderRoll].q = self.joint_angles[0, 10] * ratio
            self.low_cmd.motor_cmd[G1JointIndex.RightShoulderYaw].q = self.joint_angles[0, 11] * ratio
            self.low_cmd.motor_cmd[G1JointIndex.RightElbow].q = self.joint_angles[0, 12] * ratio
            self.low_cmd.motor_cmd[G1JointIndex.RightWristRoll].q = self.joint_angles[0, 13] * ratio
            self.low_cmd.motor_cmd[G1JointIndex.RightWristYaw].q = self.joint_angles[0, 14] * ratio

        elif self.time_step < self.num_frames:
            # [Stage 2]: run g1 motion
            self.time_step += 1

            self.low_cmd.motor_cmd[G1JointIndex.WaistYaw].q = self.joint_angles[self.time_step, 0]
            self.low_cmd.motor_cmd[G1JointIndex.WaistRoll].q = self.joint_angles[self.time_step, 1]
            self.low_cmd.motor_cmd[G1JointIndex.WaistPitch].q = self.joint_angles[self.time_step, 2]
            self.low_cmd.motor_cmd[G1JointIndex.LeftShoulderPitch].q = self.joint_angles[self.time_step, 3]
            self.low_cmd.motor_cmd[G1JointIndex.LeftShoulderRoll].q = self.joint_angles[self.time_step, 4]
            self.low_cmd.motor_cmd[G1JointIndex.LeftShoulderYaw].q = self.joint_angles[self.time_step, 5]
            self.low_cmd.motor_cmd[G1JointIndex.LeftElbow].q = self.joint_angles[self.time_step, 6]
            self.low_cmd.motor_cmd[G1JointIndex.LeftWristRoll].q = self.joint_angles[self.time_step, 7]
            self.low_cmd.motor_cmd[G1JointIndex.LeftWristYaw].q = self.joint_angles[self.time_step, 8]
            self.low_cmd.motor_cmd[G1JointIndex.RightShoulderPitch].q = self.joint_angles[self.time_step, 9]
            self.low_cmd.motor_cmd[G1JointIndex.RightShoulderRoll].q = self.joint_angles[self.time_step, 10]
            self.low_cmd.motor_cmd[G1JointIndex.RightShoulderYaw].q = self.joint_angles[self.time_step, 11]
            self.low_cmd.motor_cmd[G1JointIndex.RightElbow].q = self.joint_angles[self.time_step, 12]
            self.low_cmd.motor_cmd[G1JointIndex.RightWristRoll].q = self.joint_angles[self.time_step, 13]
            self.low_cmd.motor_cmd[G1JointIndex.RightWristYaw].q = self.joint_angles[self.time_step, 14]

        else:
            # goto stage 1
            self.time_ = 0.0
            self.time_step = 0


        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)


def deploy_g1():
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    # input("Press Enter to continue...")

    with open("/home/pengyang/codebase/H1_RL/data/motion/g1/SG/output.pickle", "rb") as file:
        joint_angles = pickle.load(file)["angles"]
    print(joint_angles.shape)

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init(joint_angles)
    custom.Start()

    while True:
        time.sleep(1)


if __name__ == '__main__':
    deploy_g1()
