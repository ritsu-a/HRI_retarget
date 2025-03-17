import numpy as np
import time
from enum import Enum
from collections import deque

import onnxruntime as ort

from real_robot_utils.monitor.base import SequentialDataMonitor, load_data
from real_robot_utils.driver.dualsense.dualsense_controller_zzk import (
    DualSenseLocomotionController, Command, ContinuousState, ControlMode)
from real_robot_utils.logger.logru import LOGGER

from src.real_robot_deploy.h1_driver.h1_agent import H1LowLevelAgent, DEFAULT_JOINT_POSITION
from src.real_robot_deploy.h1_driver.h1_constant import H1JointID
from src.real_robot_deploy.runner.state_estimator import StateEstimator
from galaxy_legged_gym.utils.diff_quat import vec6d_to_quat, broadcast_quat_multiply, quat_to_vec6d, broadcast_quat_apply
from galaxy_legged_gym.utils.diff_rot import calc_heading_quat_inv, quat_rotate_inverse

from src.real_robot_deploy.runner.policy_runner import H1PolicyRunner
import joblib
import torch

# @dataclass
# class H1RefMotion(IdlStruct, typename="H1RefMotion"):
#     # bool
#     is_alive: bool

#     # continuous
#     target_dof_pos: types.array[types.float32, 19]
#     target_dof_vel: types.array[types.float32, 19]
#     target_xy_vel: types.array[types.float32, 2]
#     target_yaw_vel: types.array[types.float32, 1]
#     target_projected_gravity: types.array[types.float32, 3]
class H1FullSimConfig:
    dt = 0.02
    LEG_ONLY_URDF_NAMES = ['left_hip_yaw_joint',
                           'left_hip_roll_joint',
                           'left_hip_pitch_joint',
                           'left_knee_joint',
                           'left_ankle_joint',
                           'right_hip_yaw_joint',
                           'right_hip_roll_joint',
                           'right_hip_pitch_joint',
                           'right_knee_joint',
                           'right_ankle_joint']

    default_joint_angles = {  # = target angles [rad] when action = 0.0
        # 'left_hip_yaw_joint': 0.,
        # 'left_hip_roll_joint': 0,
        # 'left_hip_pitch_joint': -0.4,
        # 'left_knee_joint': 0.8,
        # 'left_ankle_joint': -0.4,
        # 'right_hip_yaw_joint': 0.,
        # 'right_hip_roll_joint': 0,
        # 'right_hip_pitch_joint': -0.4,
        # 'right_knee_joint': 0.8,
        # 'right_ankle_joint': -0.4,
        # 'torso_joint': 0.,
        # 'left_shoulder_pitch_joint': 0.,
        # 'left_shoulder_roll_joint': 0,
        # 'left_shoulder_yaw_joint': 0.,
        # 'left_elbow_joint': 0.,
        # 'right_shoulder_pitch_joint': 0.,
        # 'right_shoulder_roll_joint': 0.0,
        # 'right_shoulder_yaw_joint': 0.,
        # 'right_elbow_joint': 0.,
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

    default_joint_angle_list = [0, 0, -0.4, 0.8, -0.4, 0, 0, -0.4, 0.8, -0.4]

    class FullJointID(Enum):
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

        left_shoulder_pitch_joint = 16
        left_shoulder_roll_joint = 17
        left_shoulder_yaw_joint = 18
        left_elbow_joint = 19
        right_shoulder_pitch_joint = 12
        right_shoulder_roll_joint = 13
        right_shoulder_yaw_joint = 14
        right_elbow_joint = 15

    class LegJointID(Enum):
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

    class ArmJointID(Enum):
        left_shoulder_pitch_joint = 16
        left_shoulder_roll_joint = 17
        left_shoulder_yaw_joint = 18
        left_elbow_joint = 19
        right_shoulder_pitch_joint = 12
        right_shoulder_roll_joint = 13
        right_shoulder_yaw_joint = 14
        right_elbow_joint = 15

    class LeftArmJointID(Enum):
        left_shoulder_pitch_joint = 16
        left_shoulder_roll_joint = 17
        left_shoulder_yaw_joint = 18
        left_elbow_joint = 19
    
    class WaistJointID(Enum):
        waist_yaw_joint = 6

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            roll_pitch = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05

        clip_observations = 100.
        clip_actions = 100. # ？

    action_scale = 0.25

    # class commands:
    #     ang_vel_yaw_clip = 0.5
    #     lin_vel_x_clip = 0.2
    #     lin_vel_y_clip = 0.2

    #     class ranges:
    #         lin_vel_x = [-0.8, 0.6]  # min max [m/s]
    #         lin_vel_y = [-0.6, 0.6]  # min max [m/s]
    #         ang_vel_yaw = [-0.6, 0.6]  # min max [rad/s]
    #         squat_z = [0.3, 1.3]

    def get_full_default_joint_position(self):
        return np.float32(list(self.default_joint_angles.values())[:])

    def simfull_to_h1(self, leg_q):
        h1_q = np.zeros(len(H1JointID), dtype=np.float32)
        for i, h1_id in enumerate(self.FullJointID):
            h1_q[h1_id.value] = leg_q[i]
        return h1_q

    def h1_to_simfull(self, h1_q):
        urdf_q = np.zeros(len(self.FullJointID), dtype=np.float32)
        for i, h1_id in enumerate(self.FullJointID):
            urdf_q[i] = h1_q[h1_id.value]
        return urdf_q
    
    def get_leg_default_joint_position(self):
        return np.float32(list(self.default_joint_angles.values())[:10])

    def simleg_to_h1(self, leg_q):
        h1_q = np.zeros(len(H1JointID), dtype=np.float32)
        for i, h1_id in enumerate(self.LegJointID):
            h1_q[h1_id.value] = leg_q[i]
        return h1_q

    def h1_to_simleg(self, h1_q):
        urdf_q = np.zeros(len(self.LegJointID), dtype=np.float32)
        for i, h1_id in enumerate(self.LegJointID):
            urdf_q[i] = h1_q[h1_id.value]
        return urdf_q
    
    def simarm_to_h1(self, arm_q):
        h1_q = np.zeros(len(H1JointID), dtype=np.float32)
        for i, h1_id in enumerate(self.WaistJointID):
            h1_q[h1_id.value] = arm_q[10]
        for i, h1_id in enumerate(self.ArmJointID):
            h1_q[h1_id.value] = arm_q[i+11]
        for i, h1_id in enumerate(self.LegJointID):
            h1_q[h1_id.value] = self.default_joint_angle_list[i]
        return h1_q

    def h1_to_simarm(self, h1_q):
        urdf_q = np.zeros(len(self.ArmJointID), dtype=np.float32)
        for i, h1_id in enumerate(self.ArmJointID):
            urdf_q[i] = h1_q[h1_id.value]
        return urdf_q


class SquatDownPolicyRunner(H1PolicyRunner):
    def __init__(self, onnx_model_path, use_remote=True):
        # configs
        self.cfg = H1FullSimConfig()
        self.dt = self.cfg.dt

        # init modules
        self.agent = H1LowLevelAgent()
        self.policy_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
        self.state_estimator = StateEstimator(self.agent.driver)

        self.use_remote = use_remote
        if use_remote:
            self._remote = DualSenseLocomotionController(
                x_vel_command=Command("x_vel", 0.0,
                                      self.cfg.commands.ranges.lin_vel_x[0],
                                      self.cfg.commands.ranges.lin_vel_x[1],
                                      ContinuousState.LY, 0.5, ControlMode.DRIVE),
                y_vel_command=Command("y_vel", 0.0,
                                      self.cfg.commands.ranges.lin_vel_y[0],
                                      self.cfg.commands.ranges.lin_vel_y[1],
                                      ContinuousState.LX, 0.5, ControlMode.DRIVE),
                yaw_vel_command=Command("yaw_vel", 0.0,
                                        self.cfg.commands.ranges.ang_vel_yaw[0],
                                        self.cfg.commands.ranges.ang_vel_yaw[1],
                                        ContinuousState.RX, 0.5, ControlMode.DRIVE),
                height_command=Command("body_height", 0.98, 0.3, 0.3, ContinuousState.LY, 0.5, ControlMode.WORK),
                pitch_command=Command("body_pitch", 0.0, -0.5, 0.5, ContinuousState.RY, 0.5, ControlMode.WORK),
                left_hand_height_command=Command("left_hand_height", 0.08, 0.0, 0.5, ContinuousState.RY, 0.2, ControlMode.LEFT_HAND),
                right_hand_height_command=Command("right_hand_height", 0.08, 0.0, 0.5, ContinuousState.RY, 0.2, ControlMode.RIGHT_HAND),
                left_hand_x_command=Command("left_hand_x", 0.33, 0.1, 0.2, ContinuousState.LY, 0.2, ControlMode.LEFT_HAND),
                right_hand_x_command=Command("right_hand_x", 0.33, 0.1, 0.2, ContinuousState.LY, 0.2, ControlMode.RIGHT_HAND),
                left_hand_y_command=Command("left_hand_y", 0.215, 0.1, 0.4, ContinuousState.LX, 0.2, ControlMode.LEFT_HAND),
                right_hand_y_command=Command("right_hand_y", -0.215, -0.6, 0.1, ContinuousState.LX, 0.2, ControlMode.RIGHT_HAND),
            )
        else:
            Warning("Warning: Remote control is disabled.")

        # buffers
        self.default_dof_pos = self.cfg.get_full_default_joint_position()
        self._last_action = np.zeros(19, dtype=np.float32)
        self._time_buf = deque(maxlen=10)
        self.num_bodies = 22
        self.motion_id = 0
        self.frame_id = 0
        # self.fps = 30
        self.load_ref_motion(
            # motion_path
            "/home/ubuntu/Desktop/H1_RL-cc/logs/H1-RL/1101_xh_demo/nvel_dn_35_mdm.pkl"
            )

    def load_ref_motion(self, path):
        data_dict = joblib.load(path)
        target_jt = []
        target_global = []
        target_length = []
        
        if 'hand' in path or 'nvel' in path:
            self.cmd_v=0.
            self.fps = 15#15
        elif 'swing' in path:
            self.cmd_v=1.
            self.fps = 30#15
        elif 'walk' in path:
            self.cmd_v=1.
            self.fps = 20#15
        else:
            assert False

        for name, data in data_dict.items():
            # if 'hand' not in name and 'swing' not in name:
            #     continue
            # if 'hand' not in name:
            #     continue
            # if '0-ACCAD_Female1Walking_c3d_B21_s2_-_put_down_box_to_walk_stageii' not in name and '0-SOMA_soma_subject1_sit_001_stageii' not in name:
            #     continue
            one_target_jt = data['jt']
            one_target_global = data['global']
            # if one_target_jt.shape[0] < 100:
            #     continue
            target_jt.append(one_target_jt)
            target_global.append(one_target_global)
            target_length.append(one_target_jt.shape[0])
        # target_jt = np.concatenate(target_jt, dim=0)
        # target_global = torch.cat(target_global, dim=0)
        # target_length = torch.tensor(target_length, dtype=torch.long)
        # start_id = torch.zeros_like(target_length, dtype=torch.long)
        # start_id[1:] = torch.cumsum(target_length[:-1], dim=0)
        self.target_jt = np.concatenate(target_jt, axis=0)
        self.target_global = np.concatenate(target_global, axis=0)
        self.target_length = np.array(target_length)
        self.start_id = np.zeros_like(target_length)
        self.start_id[1:] = np.cumsum(target_length[:-1], axis=0)
        self.target_global_pos, self.target_global_ori, self.target_global_vel, self.target_global_ang_vel = \
            self.target_global[:,:3*self.num_bodies], self.target_global[:,3*self.num_bodies:9*self.num_bodies], self.target_global[:,9*self.num_bodies:12*self.num_bodies], self.target_global[:,12*self.num_bodies:15*self.num_bodies]
        self.update_ref_motion()
        
    def update_ref_motion(self):
        b, e = self.start_id[self.motion_id], self.start_id[self.motion_id] + self.target_length[self.motion_id]
        target_heading_quat_inv = calc_heading_quat_inv(vec6d_to_quat(
            torch.from_numpy(self.target_global_ori[b:e,:6]).reshape(-1,3,2)))

        target_root_ori_quat = broadcast_quat_multiply(target_heading_quat_inv,
            vec6d_to_quat(
                torch.from_numpy(self.target_global_ori[b:e,:6]).reshape(-1,3,2)))
        # breakpoint()
        self.target_jt_pos, self.target_jt_vel = self.target_jt[b:e,:19], self.target_jt[b:e,19:]
        # breakpoint()
        self.target_projected_gravity=quat_rotate_inverse(
            vec6d_to_quat(
                torch.from_numpy(self.target_global_ori[b:e,:6]).reshape(-1,3,2)), 
                torch.tensor([[0., 0., -1.]]).repeat((e-b, 1)))
        # self.target_root_ori=quat_to_vec6d(target_root_ori_quat).reshape(-1,6)
        # # breakpoint()
        # self.target_root_local_vel = broadcast_quat_apply(target_heading_quat_inv, 
        #                                     torch.from_numpy(self.target_global_vel[b:e,:3]))
        # self.target_root_local_ang_vel = broadcast_quat_apply(target_heading_quat_inv, 
        #                                     torch.from_numpy(self.target_global_ang_vel[b:e,:3]))
        
    def compute_ref_motion(self):
        self.frame_id += (self.fps / 50.)
        if self.frame_id >= self.target_length[self.motion_id]:
            self.frame_id = 0
            # self.motion_id += 1
            print(self.motion_id)
            self.update_ref_motion()
        ref_id = int(self.frame_id)
        # if ref_id<0:
        #     target_dof_pos=self.default_dof_pos
        #     target_dof_vel=0*target_dof_pos
        # else:
        target_dof_pos=self.target_jt_pos[ref_id]
        target_dof_vel=self.target_jt_vel[ref_id]
        
        # v = min(self.frame_id/50.,1)
        v=self.cmd_v
        commands = np.array([v,0.,0.])
        # commands[...,0] = 1.0
        return  dict(
            is_alive=True,
            target_dof_pos=target_dof_pos,
            target_dof_vel=target_dof_vel,
            target_xy_vel=commands[0:2],
            target_yaw_vel=commands[2:3],
            target_projected_gravity=self.target_projected_gravity[ref_id],
            # target_root_ori=self.target_root_ori[ref_id],
            # target_root_vel=self.target_root_local_vel[ref_id],
            # target_root_ang_vel=self.target_root_local_ang_vel[ref_id],
        )
    
    def compute_observation(self):
        # scales
        obs_scales = self.cfg.normalization.obs_scales
        clip_observations = self.cfg.normalization.clip_observations
        commands_scale = np.float32([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        # values
        commands = self.get_commands()
        print(f"commands: {commands}")
        # roll_pitch = self.state_estimator.get_rpy()[:2]
        projected_gravity = self.state_estimator.get_get_projected_gravity()
        # ang_vel = self.state_estimator.get_angular_velocity()  # todo: try without filtering
        ang_vel = self.state_estimator.get_raw_angular_velocity()
        dof_pos = self.cfg.h1_to_simfull(self.state_estimator.get_joint_position())
        dof_vel = self.cfg.h1_to_simfull(self.state_estimator.get_joint_velocity())
        # result
        ref_motion = self.compute_ref_motion()
        obs_buf = np.hstack([
            obs_scales.dof_pos * 
                (ref_motion['target_dof_pos'] - self.default_dof_pos), # 19
            obs_scales.dof_vel * 
                ref_motion['target_dof_vel'], # 19
            obs_scales.dof_pos * 
                (ref_motion['target_dof_pos'] - dof_pos), # 19
            obs_scales.dof_vel * 
                (ref_motion['target_dof_vel'] - dof_vel), # 19
            ref_motion['target_projected_gravity'],
            ref_motion['target_xy_vel'],
            ref_motion['target_yaw_vel'],
            projected_gravity,
            obs_scales.ang_vel * 
                ang_vel,  # 3
            obs_scales.dof_pos * 
                (dof_pos - self.default_dof_pos),  # 19
            obs_scales.dof_vel * 
                dof_vel,  # 19
            self._last_action  # 19
        ])

        obs_buf = np.clip(obs_buf, -clip_observations, clip_observations)
        return obs_buf

    # def get_commands(self):
    #     if self.use_remote:
    #         return np.hstack((
    #             self._remote.get_lin_vel_command(),
    #             self._remote.get_ang_vel_command(),
    #             self._remote.get_left_hand_x_command(),
    #             self._remote.get_left_hand_y_command(),
    #             self._remote.get_left_hand_height_command(),
    #             self._remote.get_right_hand_x_command(),
    #             self._remote.get_right_hand_y_command(),
    #             self._remote.get_right_hand_height_command(),
    #             self._remote.get_base_height_command(),
    #         ))
    #     else:
    #         return np.zeros(4, dtype=np.float32)
        
    def inference(self, obs):
        clip_actions = self.cfg.normalization.clip_actions
        action_scale = self.cfg.action_scale
        obs = obs.astype(np.float32).reshape(1, -1)
        action = self.policy_session.run(None, {'obs': obs})[0][0]
        self._last_action = action.copy()
        action = np.clip(action, -clip_actions, clip_actions)
        target_joint_pos = self.default_dof_pos + action_scale * action
        h1_target_joint_pos = self.cfg.simfull_to_h1(target_joint_pos)
        # print(h1_target_joint_pos)
        return h1_target_joint_pos, action
    
    def run(self, log_path=None, duration=10.0):

        if log_path is not None:
            monitor = SequentialDataMonitor(log_path)
        hardware_dt = self.agent.driver.dt
        decimation = int(self.dt / hardware_dt)

        self._remote.start()
        self.state_estimator.start()
        self.agent.warm_up()

        while not self._remote.get_state_start():
            self.agent.apply_position_control(self.agent.init_joint_position.copy())
            print("\rWaiting for start signal.", end='')
            if self._remote.get_state_terminate():
                self.agent.emergency_stop()
                self.state_estimator.stop()
                self._remote.stop()
                exit(0)

        try:
            for i in range(int(duration / self.dt)):
                start_t = time.perf_counter()
                obs = self.compute_observation()
                if self._remote.get_state_terminate() or self.frame_id >= self.target_length[self.motion_id]:
                    break
                h1_target_joint_pos, action = self.inference(obs)
                # print(h1_target_joint_pos)
                # target_joint_pos = self.default_dof_pos + 1.0 * action_scale * action
                # h1_target_joint_pos = self.cfg.simfull_to_h1(target_joint_pos)

                if log_path is not None:
                    curr_state = self.agent.get_low_state()
                    state_dict = curr_state.as_dict()
                    state_dict['obs'] = obs
                    state_dict['action'] = action
                    state_dict['h1_target_joint_pos'] = h1_target_joint_pos
                    monitor.add(state_dict, save=False)
                for d_i in range(1, decimation + 1):
                    self.agent.apply_position_control(h1_target_joint_pos)


                run_t = time.perf_counter() - start_t
                if run_t < self.dt:
                    time.sleep(self.dt-run_t)
                run_t = time.perf_counter() - start_t
                self._time_buf.append(run_t)
                print(f"Frequency: {1 / np.mean(list(self._time_buf)):.2f} [Hz]")
        except Exception as e:
            LOGGER.error(f"Error: {e}")

        if log_path is not None:
            monitor.save()
        self.agent.emergency_stop()
        self.state_estimator.stop()
        self._remote.stop()


def demo_walk(duration=270.):
    onnx_path = "logs/0722_walk_NoBlance/policy_28000.onnx"
    h1_policy_runner = H1PolicyRunner(onnx_path)
    h1_policy_runner.run(log_path="./logs/real_robot_data/0723_walk_02", duration=duration)

    # only for debug
    h1_policy_runner._remote.start()
    h1_policy_runner.state_estimator.start()
    for _ in range(10):
        obs = h1_policy_runner.compute_observation() #(39,)
        # print(obs.shape)
        action = h1_policy_runner.inference(obs)


def demo_squat(duration=270.):
    onnx_path = "/home/ubuntu/Desktop/H1_RL/logs/H1-RL/0728_full_dof_squat_down_resample_commands/exported/0728_full_dof_squat_down_resample_commands/policy_5000.onnx"
    squat_down_policy_runner = SquatDownPolicyRunner(onnx_path)
    # squat_down_policy_runner.run(log_path="./logs/real_robot_data/0728_upper_body_squat_down_no_add_noise_00", duration=duration)
    
    # only for debug
    squat_down_policy_runner._remote.start()
    squat_down_policy_runner.state_estimator.start()
    for _ in range(10):
        obs = squat_down_policy_runner.compute_observation() #(39,)
        # print(obs.shape)
        h1_target_joint_pos, action = squat_down_policy_runner.inference(obs)

def demo_touch_point(duration=800.):
    onnx_path = "/home/ubuntu/Desktop/H1_RL-cc/logs/H1-RL/1101_xh_demo/policy_30000.onnx"
    touch_point_policy_runner = SquatDownPolicyRunner(onnx_path)
    touch_point_policy_runner.run(log_path="./logs/real_robot_data/0808_full_dof_touch_point_00", duration=duration)
    
    # only for debug
    # touch_point_policy_runner._remote.start()
    # touch_point_policy_runner.state_estimator.start()
    # for _ in range(10):
    #     obs = touch_point_policy_runner.compute_observation() #(39,)
    #     # print(obs.shape)
    #     h1_target_joint_pos, action = touch_point_policy_runner.inference(obs)

if __name__ == '__main__':
    # demo_squat()
    # demo_walk()
    demo_touch_point()

# '/home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/h1-s2r/1028_lctk_walkforward_hipyaw_newmotion_dr_resume/policy_60000.onnx',
    # '/home/ubuntu/data/PHC/tk15_walk_mdm.pkl', # 0
    
# '/home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/h1-s2r/1101_lctk_swingwalk_dr/policy_90000.onnx',
    # '/home/ubuntu/data/PHC/rcp_10swingwalk_mdm.pkl' # 1 3 7.

# '/home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/h1-s2r/1102_17box/policy_22000.onnx',
    # '/home/ubuntu/data/PHC/rcp_10box_mdm.pkl'  # 2 5 6 7 8 9
# '/home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/h1-s2r/1102_17hand/policy_28000.onnx',
    # '/home/ubuntu/data/PHC/nvel_dn_35_mdm.pkl', # 3 7 33

# '/home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/h1-s2r/1103_lctk_hand_ankle_dr/policy_37000.onnx',
    # '/home/ubuntu/data/PHC/nvel_dn_35_mdm.pkl', # 3 7 33
