import time
import numpy as np
from h1_driver import (
    H1LowLevelDriver,
    H1JointID,
    LowState,
    LowCmd,
    LEFT_ARM_MSG_IDS, RIGHT_ARM_MSG_IDS,
    TORSO_MSG_IDS,
    LEFT_LEG_MSG_IDS, RIGHT_LEG_MSG_IDS,
    torque_clip
)
from h1_pd_param import H1_KP, H1_KD

WHOLE_BODY_IDS = np.hstack([
    LEFT_ARM_MSG_IDS, RIGHT_ARM_MSG_IDS,
    TORSO_MSG_IDS,
    LEFT_LEG_MSG_IDS, RIGHT_LEG_MSG_IDS
]).tolist()

# DEFAULT_JOINT_POSITION_DICT = {
#     H1JointID.right_hip_roll_joint: 0.05,
#     H1JointID.right_hip_pitch_joint: -0.4,
#     H1JointID.right_knee_joint: 0.8,
#     H1JointID.left_hip_roll_joint: 0,
#     H1JointID.left_hip_pitch_joint: -0.4,
#     H1JointID.left_knee_joint: 0.8,
#     H1JointID.waist_yaw_joint: 0,
#     H1JointID.left_hip_yaw_joint: 0,
#     H1JointID.right_hip_yaw_joint: 0,
#     H1JointID.none_joint: 0.0,
#     H1JointID.left_ankle_joint: -0.4,
#     H1JointID.right_ankle_joint: -0.4,
#     H1JointID.right_shoulder_pitch_joint: 0,
#     H1JointID.right_shoulder_roll_joint: 0,
#     H1JointID.right_shoulder_yaw_joint: 0,
#     H1JointID.right_elbow_joint: 0,
#     H1JointID.left_shoulder_pitch_joint: 0,
#     H1JointID.left_shoulder_roll_joint: 0,
#     H1JointID.left_shoulder_yaw_joint: 0,
#     H1JointID.left_elbow_joint: 0,
# }

DEFAULT_JOINT_POSITION_DICT = {
    H1JointID.right_hip_roll_joint: 0.05,
    H1JointID.right_hip_pitch_joint: -0.45,
    H1JointID.right_knee_joint: 1.0,
    H1JointID.left_hip_roll_joint: -0.05,
    H1JointID.left_hip_pitch_joint: -0.45,
    H1JointID.left_knee_joint: 1.0,
    H1JointID.waist_yaw_joint: 0.0,
    H1JointID.left_hip_yaw_joint: 0.0,
    H1JointID.right_hip_yaw_joint: 0.0,
    H1JointID.none_joint: 0.0,
    H1JointID.left_ankle_joint: -0.53,
    H1JointID.right_ankle_joint: -0.53,
    H1JointID.right_shoulder_pitch_joint: 0.35,
    H1JointID.right_shoulder_roll_joint: 0.04,
    H1JointID.right_shoulder_yaw_joint: 0.01,
    H1JointID.right_elbow_joint: 0.15,
    H1JointID.left_shoulder_pitch_joint: 0.35,
    H1JointID.left_shoulder_roll_joint: -0.04,
    H1JointID.left_shoulder_yaw_joint: -0.01,
    H1JointID.left_elbow_joint: 0.15,
}

DEFAULT_JOINT_POSITION = np.float32(list(DEFAULT_JOINT_POSITION_DICT.values()))


def interpolate(start, end, ratio):
    return start + (end - start) * np.clip(ratio, 0, 1)


class H1LowLevelAgent:
    def __init__(self,
                 channel_id: int = 0,
                 channel_network_interface: str = None,
                 init_joint_position=DEFAULT_JOINT_POSITION,
                 activate_ids=WHOLE_BODY_IDS,
                 driver_class=H1LowLevelDriver):
        self.driver = driver_class(channel_id, channel_network_interface)
        self.last_t = self.driver.get_low_state().tick
        self.activate_ids = activate_ids
        self.init_joint_position = init_joint_position
        self._is_warmed_up = False

    def get_low_state(self) -> LowState:
        return self.driver.get_low_state()

    def apply_position_control(self, target_joint_position, clip_torque=True):
        assert self._is_warmed_up, "Must warm up first."
        assert len(target_joint_position) == len(H1JointID)
        curr_state = self.driver.get_low_state()
        if clip_torque:
            target_joint_position = torque_clip(
                target_joint_position, curr_state.joint_position, curr_state.joint_velocity, H1_KP, H1_KD)
        cmd = LowCmd(self.activate_ids, target_joint_position, H1_KP, H1_KD)
        self.driver.send_low_cmd(cmd)

    def warm_up(self, duration=3.0, interpolate_method='tanh', warm_up_joint_position=None):
        if warm_up_joint_position is None:
            warm_up_joint_position = self.init_joint_position.copy()

        assert duration > 1.0, "Must warm up for at least 1 second."
        # state
        curr_state = self.driver.get_low_state()
        start_kp = H1_KP * 0.7
        start_kd = H1_KD * 0.7
        start_joint_position = curr_state.joint_position

        # plan
        num_warm_up = int(duration / self.driver.dt)
        schedule = np.linspace(0, 1, num_warm_up)
        if interpolate_method == 'tanh':
            schedule = np.clip(np.tanh(schedule * 3), 0., 1)
        elif interpolate_method == 'exp':
            schedule = np.clip(np.exp2(schedule) - 1, 0., 1)

        # open loop control
        for i in range(num_warm_up):
            scale = schedule[i]
            curr_kp = interpolate(start_kp, H1_KP, scale)
            curr_kd = interpolate(start_kd, H1_KD, scale)
            step_joint_position = interpolate(start_joint_position, warm_up_joint_position, scale)
            cmd = LowCmd(self.activate_ids, step_joint_position, curr_kp, curr_kd)
            self.driver.send_low_cmd(cmd)
        self._is_warmed_up = True

    def emergency_stop(self, duration=1.0):
        for i in range(int(duration / self.driver.dt)):
            cmd = LowCmd(WHOLE_BODY_IDS, self.init_joint_position, np.zeros(20), H1_KD)
            self.driver.send_low_cmd(cmd)
        print("Emergency stop.")


if __name__ == '__main__':
    robot_drive = H1LowLevelAgent()
    state = robot_drive.get_low_state()
    print(state)
