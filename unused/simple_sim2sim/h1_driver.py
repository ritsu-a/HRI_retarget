import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC

from h1_constant import (
    H1JointID,
    get_joint_ids_by_names,
    MESSAGE_JOINT_DICT,
    RIGHT_LEG_JOINTS_ID,
    LEFT_LEG_JOINTS_ID,
    TORSO_ID,
    RIGHT_ARM_JOINTS_ID,
    LEFT_ARM_JOINTS_ID,
    PosStopF,
    VelStopF,
    H1TorqueLimit,
)

NUM_JOINTS = 20
VALID_JOINT_IDS = list(set(range(NUM_JOINTS)) - {9})
LEFT_LEG_MSG_IDS = get_joint_ids_by_names(LEFT_LEG_JOINTS_ID, MESSAGE_JOINT_DICT)
RIGHT_LEG_MSG_IDS = get_joint_ids_by_names(RIGHT_LEG_JOINTS_ID, MESSAGE_JOINT_DICT)
TORSO_MSG_IDS = get_joint_ids_by_names(TORSO_ID, MESSAGE_JOINT_DICT)
LEFT_ARM_MSG_IDS = get_joint_ids_by_names(LEFT_ARM_JOINTS_ID, MESSAGE_JOINT_DICT)
RIGHT_ARM_MSG_IDS = get_joint_ids_by_names(RIGHT_ARM_JOINTS_ID, MESSAGE_JOINT_DICT)
JOINT_MODES = [m for _, m in MESSAGE_JOINT_DICT.values()]

TORQUE_LIMIT = np.float32(list(H1TorqueLimit.values()))


@dataclass
class LowState:
    tick: int
    imu_quaternion: np.ndarray
    imu_gyroscope: np.ndarray
    imu_accelerometer: np.ndarray
    imu_rpy: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    joint_acceleration: np.ndarray
    joint_torque: np.ndarray

    def as_dict(self) -> Dict:
        return {
            "tick": self.tick,
            "imu_quaternion": self.imu_quaternion.tolist(),
            "imu_gyroscope": self.imu_gyroscope.tolist(),
            "imu_accelerometer": self.imu_accelerometer.tolist(),
            "imu_rpy": self.imu_rpy.tolist(),
            "joint_position": self.joint_position.tolist(),
            "joint_velocity": self.joint_velocity.tolist(),
            "joint_acceleration": self.joint_acceleration.tolist(),
            "joint_torque": self.joint_torque.tolist(),
        }


@dataclass
class LowCmd:
    activated_joint_ids: List[H1JointID]
    joint_position: np.ndarray
    Kp: np.ndarray
    Kd: np.ndarray
    joint_velocity: np.ndarray = np.zeros(NUM_JOINTS)
    joint_torque: np.ndarray = np.zeros(NUM_JOINTS)

    def __post_init__(self):
        assert len(self.joint_position) == NUM_JOINTS
        assert len(self.joint_velocity) == NUM_JOINTS
        assert len(self.joint_torque) == NUM_JOINTS
        assert len(self.Kp) == NUM_JOINTS
        assert len(self.Kd) == NUM_JOINTS


def is_safe_to_move(
    curr_joint_position,
    curr_joint_velocity,
    target_joint_position,
    kp,
    kd,
    save_ratio=1.0,
):
    torque_limit = TORQUE_LIMIT * save_ratio
    torque = (
        kp * (target_joint_position - curr_joint_position) - kd * curr_joint_velocity
    )
    safe_mask = np.abs(torque) < torque_limit
    unsafe_id = np.where(~safe_mask)[0]
    is_safe = np.all(safe_mask)
    if not is_safe:
        raise ValueError(
            f"Unsafe Joint ID: {np.array(list(H1JointID))[unsafe_id]}\n"
            f"Target Torque: {np.rad2deg(torque[unsafe_id])} [Nm]\n"
        )
    return is_safe


def torque_clip(
    target_joint_position,
    curr_joint_position,
    curr_joint_velocity,
    kp,
    kd,
    save_ratio=0.9,
):
    torque_limit = TORQUE_LIMIT * save_ratio
    joint_pos_low = (kd * curr_joint_velocity - torque_limit) / kp + curr_joint_position
    joint_pos_high = (
        kd * curr_joint_velocity + torque_limit
    ) / kp + curr_joint_position
    clipped_target_joint_pos = np.clip(
        target_joint_position, joint_pos_low, joint_pos_high
    )
    return clipped_target_joint_pos


class H1LowLevelDriver:
    dt = 0.002
    _low_state: LowState

    def __init__(self, channel_id: int = 0, channel_network_interface: str = None):
        self.channel_id = channel_id
        self.channel_network_interface = channel_network_interface
        ChannelFactoryInitialize(channel_id, channel_network_interface)

        # initialize subscribers
        self.low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_sub.Init(self.low_state_handler, 1)

        # initialize publishers
        self.low_cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.low_cmd_pub.Init()

        # initialize low level control message
        self._crc = CRC()
        self.cmd_msg = unitree_go_msg_dds__LowCmd_()
        self.cmd_msg.head[0] = 0xFE
        self.cmd_msg.head[1] = 0xEF
        self.cmd_msg.level_flag = 0xFF
        self.cmd_msg.gpio = 0
        for msg_id in range(20):
            self.cmd_msg.motor_cmd[msg_id].mode = JOINT_MODES[msg_id]
            self.cmd_msg.motor_cmd[msg_id].q = PosStopF
            self.cmd_msg.motor_cmd[msg_id].kp = 0
            self.cmd_msg.motor_cmd[msg_id].dq = VelStopF
            self.cmd_msg.motor_cmd[msg_id].kd = 0
            self.cmd_msg.motor_cmd[msg_id].tau = 0

    def get_low_state(self) -> LowState:
        return self._low_state

    def low_state_handler(self, msg: LowState_):
        # print front right hip motor states
        tick = int(msg.tick)
        imu_quaternion = np.float32(msg.imu_state.quaternion)
        imu_gyroscope = np.float32(msg.imu_state.gyroscope)
        imu_accelerometer = np.float32(msg.imu_state.accelerometer)
        imu_rpy = np.float32(msg.imu_state.rpy)

        joint_position = np.float32([s.q for s in msg.motor_state])
        joint_velocity = np.float32([s.dq for s in msg.motor_state])
        joint_acceleration = np.float32([s.ddq for s in msg.motor_state])
        joint_torque = np.float32([s.tau_est for s in msg.motor_state])
        self._low_state = LowState(
            tick=tick,
            imu_quaternion=imu_quaternion,
            imu_gyroscope=imu_gyroscope,
            imu_accelerometer=imu_accelerometer,
            imu_rpy=imu_rpy,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            joint_acceleration=joint_acceleration,
            joint_torque=joint_torque,
        )

    def send_low_cmd(self, low_cmd: LowCmd):
        low_state = self.get_low_state()
        assert is_safe_to_move(
            low_state.joint_position,
            low_state.joint_velocity,
            low_cmd.joint_position,
            low_cmd.Kp,
            low_cmd.Kd,
        )

        for msg_id in range(NUM_JOINTS):
            if msg_id in low_cmd.activated_joint_ids:
                self.cmd_msg.motor_cmd[msg_id].q = low_cmd.joint_position[msg_id]
                self.cmd_msg.motor_cmd[msg_id].dq = low_cmd.joint_velocity[msg_id]
                self.cmd_msg.motor_cmd[msg_id].kp = low_cmd.Kp[msg_id]
                self.cmd_msg.motor_cmd[msg_id].kd = low_cmd.Kd[msg_id]
                self.cmd_msg.motor_cmd[msg_id].tau = low_cmd.joint_torque[msg_id]
            else:
                self.cmd_msg.motor_cmd[msg_id].q = PosStopF
                self.cmd_msg.motor_cmd[msg_id].dq = VelStopF
                self.cmd_msg.motor_cmd[msg_id].kp = 0
                self.cmd_msg.motor_cmd[msg_id].kd = 0
                self.cmd_msg.motor_cmd[msg_id].tau = 0

        self.cmd_msg.crc = self._crc.Crc(self.cmd_msg)
        self.low_cmd_pub.Write(self.cmd_msg)
        time.sleep(0.001)  # todo: maybe delete this


class MujocoH1LowLevelDriver(H1LowLevelDriver):
    def get_low_state(self) -> LowState:
        self._low_state.imu_quaternion = self._low_state.imu_quaternion
        return self._low_state

    def send_low_cmd(self, low_cmd: LowCmd):
        low_state = self.get_low_state()
        assert is_safe_to_move(
            low_state.joint_position,
            low_state.joint_velocity,
            low_cmd.joint_position,
            low_cmd.Kp,
            low_cmd.Kd,
        )

        for msg_id in range(NUM_JOINTS):
            if msg_id in low_cmd.activated_joint_ids:
                self.cmd_msg.motor_cmd[msg_id].q = low_cmd.joint_position[msg_id]
                self.cmd_msg.motor_cmd[msg_id].dq = low_cmd.joint_velocity[msg_id]
                self.cmd_msg.motor_cmd[msg_id].kp = low_cmd.Kp[msg_id]
                self.cmd_msg.motor_cmd[msg_id].kd = low_cmd.Kd[msg_id]
                self.cmd_msg.motor_cmd[msg_id].tau = low_cmd.joint_torque[msg_id]
            else:
                self.cmd_msg.motor_cmd[msg_id].q = PosStopF
                self.cmd_msg.motor_cmd[msg_id].dq = VelStopF
                self.cmd_msg.motor_cmd[msg_id].kp = 0
                self.cmd_msg.motor_cmd[msg_id].kd = 0
                self.cmd_msg.motor_cmd[msg_id].tau = 0

        self.cmd_msg.crc = self._crc.Crc(self.cmd_msg)
        self.low_cmd_pub.Write(self.cmd_msg)
        time.sleep(0.001)  # todo: maybe delete this


if __name__ == "__main__":
    h1_low_level_driver = H1LowLevelDriver()
    last_t = h1_low_level_driver.get_low_state().tick
    while True:
        time.sleep(0.002)
        state = h1_low_level_driver.get_low_state()
        left_arm_joint_position = state.joint_position[LEFT_ARM_MSG_IDS]
        run_time = state.tick - last_t
        last_t = state.tick
        print("Mean State Frequency: ", 1 / (run_time * 0.001))
