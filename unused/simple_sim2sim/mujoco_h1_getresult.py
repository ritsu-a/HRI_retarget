import os
import time
from dataclasses import dataclass
import numpy as np
from cyclonedds.idl import IdlStruct, types
from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer
import onnxruntime as ort

from real_h1_walk_config import RealH1WalkConfig
from h1_pd_param import H1_KP, H1_KD
from dualsense_node import LocomotionCommand
from h1_agent import WHOLE_BODY_IDS
import joblib
import torch
from diff_quat import vec6d_to_quat, broadcast_quat_multiply, quat_to_vec6d, broadcast_quat_apply
from diff_rot import calc_heading_quat_inv, quat_rotate_inverse


def quat_wxyz_to_xyzw(quat_wxyz):
    return quat_wxyz[[1, 2, 3, 0]]


@dataclass
class H1LowState(IdlStruct, typename="H1LowState"):
    # bool
    is_alive: bool

    # continuous
    base_projected_gravity: types.array[types.float32, 3]
    base_angular_velocity: types.array[types.float32, 3]
    joint_position: types.array[types.float32, 20]
    joint_velocity: types.array[types.float32, 20]
    timestamp: float

@dataclass
class H1StateResult(IdlStruct, typename="H1StateResult"):

    # continuous
    root: types.array[types.float32, 7]
    # root_quat: types.array[types.float32, 4]
    jt: types.array[types.float32, 19]
    torque: types.array[types.float32, 19]


@dataclass
class H1RefMotion(IdlStruct, typename="H1RefMotion"):
    # bool
    is_alive: bool

    # continuous
    target_dof_pos: types.array[types.float32, 19]
    target_dof_vel: types.array[types.float32, 19]
    target_xy_vel: types.array[types.float32, 2]
    target_yaw_vel: types.array[types.float32, 1]
    target_projected_gravity: types.array[types.float32, 3]
    # target_root_ori: types.array[types.float32, 6]
    # target_root_vel: types.array[types.float32, 3]
    # target_root_ang_vel: types.array[types.float32, 3]

@dataclass
class H1LowCommand(IdlStruct, typename="H1LowCommand"):
    joint_position: types.array[types.float32, 20] = (0.0,) * 20
    Kp: types.array[types.float32, 20] = (0.0,) * 20
    Kd: types.array[types.float32, 20] = (0.0,) * 20
    activated_joint_ids: types.array[types.int64, 19] = tuple(WHOLE_BODY_IDS)

    # bool
    terminate: bool = False
    warm_up: bool = False


class ElasticBand:

    def __init__(self):
        self.stiffness = 200
        self.damping = 100
        self.point = np.array([0, 0, 3])
        self.length = 0
        self.enable = True

    def Advance(self, x, dx):
        """
        Args:
          δx: desired position - current position
          dx: current velocity
        """
        δx = self.point - x
        distance = np.linalg.norm(δx)
        direction = δx / distance
        v = np.dot(dx, direction)
        f = (self.stiffness * (distance - self.length) - self.damping * v) * direction
        return f

    def MujuocoKeyCallback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable


class BasePolicy:
    def __init__(self, cfg: RealH1WalkConfig, onnx_model_path: str):
        self.cfg = cfg
        self.dt = cfg.policy.dt
        self.use_gait = cfg.deploy.use_gait
        self.policy_session = ort.InferenceSession(
            onnx_model_path, providers=["CPUExecutionProvider"]
        )

        # buffer
        self._default_sim_joint_position = np.float32(
            cfg.policy.get_default_sim_joint_position()
        )
        self._last_action = np.zeros_like(
            self._default_sim_joint_position, dtype=np.float32
        )

        # scales
        self._obs_scales = self.cfg.policy.normalization.obs_scales
        self._clip_actions = self.cfg.policy.normalization.clip_actions
        self._clip_observations = self.cfg.policy.normalization.clip_observations


    def inference(self, low_state: H1LowState) -> H1LowCommand:
        raise NotImplementedError

    def compute_observation(self, low_state: H1LowState):
        raise NotImplementedError
    
    def action_to_low_command(self, action) -> H1LowCommand:
        action_scale = self.cfg.policy.action_scale
        sim_target_joint_pos = self._default_sim_joint_position + action_scale * action
        real_target_joint_pos = self.cfg.policy.joint_sim_to_real(sim_target_joint_pos)
        return H1LowCommand(joint_position=real_target_joint_pos, Kp=H1_KP, Kd=H1_KD)

class TrackPolicy(BasePolicy):
    def __init__(self, cfg: RealH1WalkConfig, onnx_model_path: str, motion_path: str):
        super().__init__(cfg, onnx_model_path)
        self.num_bodies = 22
        self.motion_id = 0 # 1 3 7.
        self.init_id = -0
        self.frame_id = self.init_id
        self.onnx_model_path=onnx_model_path
        self.motion_path=motion_path
        self.load_ref_motion(
            motion_path
            )

    def inference(self, low_state: H1LowState) -> H1LowCommand:
        ref_motion = self.compute_ref_motion()
        obs = self.compute_observation(low_state, ref_motion)
        # action = self.policy_session(obs)
        # breakpoint()
        action = self.policy_session.run(None, {'obs': obs})[0][0]

        action = np.clip(action, -self._clip_actions, self._clip_actions)
        # print(action-self._last_action)
        self._last_action = action
        low_command = self.action_to_low_command(action)
        return low_command

    def compute_observation(self, low_state: H1LowState, ref_motion: H1RefMotion):
        # convert to numpy
        projected_gravity = np.float32(low_state.base_projected_gravity)
        # print(projected_gravity)
        ang_vel = np.float32(low_state.base_angular_velocity)
        dof_pos = self.cfg.policy.joint_real_to_sim(
            np.float32(low_state.joint_position)
        )
        dof_vel = self.cfg.policy.joint_real_to_sim(
            np.float32(low_state.joint_velocity)
        )

        obs_buf = np.hstack(
            [
                self._obs_scales.target_dof_pos * 
                    (ref_motion.target_dof_pos - self._default_sim_joint_position), # 19
                self._obs_scales.target_dof_vel * 
                    ref_motion.target_dof_vel, # 19
                self._obs_scales.target_dof_pos * 
                    (ref_motion.target_dof_pos - dof_pos), # 19
                self._obs_scales.target_dof_vel * 
                    (ref_motion.target_dof_vel - dof_vel), # 19
                ref_motion.target_projected_gravity,
                ref_motion.target_xy_vel,
                ref_motion.target_yaw_vel,
                projected_gravity,
                self._obs_scales.base_ang_vel * 
                    ang_vel,  # 3
                self._obs_scales.dof_pos * 
                    (dof_pos - self._default_sim_joint_position),  # 19
                self._obs_scales.dof_vel * 
                    dof_vel,  # 19
                self._obs_scales.last_action * 
                    self._last_action,  # 19
                # self._obs_scales.target_root_ori * ref_motion.target_root_ori, # 6
                # self._obs_scales.target_root_vel * ref_motion.target_root_vel, # 3
                # self._obs_scales.target_root_ang_vel * ref_motion.target_root_ang_vel, # 3
            ]
        )

        obs_buf = np.clip(obs_buf, -self._clip_observations, self._clip_observations)
        obs_buf = obs_buf.astype(np.float32).reshape(1, -1)
        return obs_buf
    
    def load_ref_motion(self, path):
        data_dict = joblib.load(path)
        target_jt = []
        target_global = []
        target_length = []
        self.target_name = []
        
        if 'hand' in path or '35' in path or 'box' in path:
            self.cmd_v=0.
            self.fps = 10#15
        elif 'swing' in path:
            self.cmd_v=.8
            self.fps = 30#15
        elif 'walk' in path:
            self.cmd_v=1.
            self.fps = 20#15
        else:
            assert False
        for name, data in data_dict.items():
            # if 'hand' in name:
            #     continue
            #     self.cmd_v=0.
            #     self.fps = 15#15
            # else:
            #     self.cmd_v=1.
            #     self.fps = 30#15
            if 'hand' not in name and 'swing' not in name:
                continue
            # if '0-ACCAD_Female1Walking_c3d_B21_s2_-_put_down_box_to_walk_stageii' not in name and '0-SOMA_soma_subject1_sit_001_stageii' not in name:
            #     continue
            one_target_jt = data['jt']
            one_target_global = data['global']
            # if one_target_jt.shape[0] < 100:
            #     continue
            target_jt.append(one_target_jt)
            target_global.append(one_target_global)
            target_length.append(one_target_jt.shape[0])
            self.target_name.append(name)
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
        self.motion_name = self.target_name[self.motion_id]
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
            self.frame_id = self.init_id
            self.motion_id += 1
            print(self.motion_id)
            self.update_ref_motion()
        ref_id = int(self.frame_id)
        if ref_id<0:
            target_dof_pos=self._default_sim_joint_position
            target_dof_vel=0*target_dof_pos
            # v=0.
        else:
            target_dof_pos=self.target_jt_pos[ref_id]
            target_dof_vel=self.target_jt_vel[ref_id]
        v=self.cmd_v
        
        # v = min(self.frame_id/50.,1)
        commands = np.array([v,0.,0.])
        # commands[...,0] = 1.0
        return  H1RefMotion(
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
    
class MujocoSim:
    sensor_dict = {
        "joint_position": range(0, 19),
        "joint_velocity": range(19, 38),
        "joint_torque": range(38, 57),
        "imu_quat": range(57, 61),
        "imu_gyroscope": range(61, 64),
        "imu_accelerometer": range(64, 67),
        "frame_pos": range(67, 70),
        "frame_vel": range(70, 73),
    }

    def __init__(
        self,
        cfg: RealH1WalkConfig,
        policy: BasePolicy,
        robot_scene="h1/scene.xml",
        simulate_dt=0.001,
        viewer_dt=0.02, # policy dt
        domain_id=1,
        interface="lo",
        enable_elastic_band=True,
        print_scene_information=True,
    ):
        self.cfg = cfg
        self.policy = policy

        self.robot_scene = robot_scene
        self.simulate_dt = simulate_dt
        self.viewer_dt = viewer_dt
        self.domain_id = domain_id
        self.interface = interface

        self.enable_elastic_band = enable_elastic_band
        self.print_scene_information = print_scene_information

        self.mj_model = mujoco.MjModel.from_xml_path(robot_scene)
        self.mj_data = mujoco.MjData(self.mj_model)

        if self.enable_elastic_band:
            self.elastic_band = ElasticBand()
            self.band_attached_link = self.mj_model.body("torso_link").id
            self.viewer = mujoco.viewer.launch_passive(
                self.mj_model,
                self.mj_data,
                key_callback=self.elastic_band.MujuocoKeyCallback,
            )
        else:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        self.mj_model.opt.timestep = self.simulate_dt
        num_motor_ = self.mj_model.nu

        self.result_dict = {name:{'root':[],'jt':[], 'torque':[]} for name in self.policy.target_name}

    def run(self):
        ChannelFactoryInitialize(self.domain_id, self.interface)

        self._remote_command = None
        command_sub = ChannelSubscriber(
            RealH1WalkConfig.topics.remote_controller_topic_name, LocomotionCommand
        )
        command_sub.Init(self._remote_command_handler)

        env_start_t = time.time()
        release_time = 2.0
        time.sleep(1.0)
        last_frame_id = -1
        while True:
            step_start = time.perf_counter()
            # if self._remote_command is not None:
            #     if self.enable_elastic_band and not self._remote_command.state_move:
            #         if self.elastic_band.enable:
            #             self.mj_data.xfrc_applied[self.band_attached_link, :3] = (
            #                 self.elastic_band.Advance(
            #                     self.mj_data.qpos[:3], self.mj_data.qvel[:3]
            #                 )
            #             )
            # if self.elastic_band.enable and self.policy.frame_id < -50:
            #     self.mj_data.xfrc_applied[self.band_attached_link, :3] = (
            #         self.elastic_band.Advance(
            #             self.mj_data.qpos[:3], self.mj_data.qvel[:3]
            #         )
            #     )
            # print(last_frame_id)
            if (last_frame_id +1) ==int(self.policy.frame_id):
                motion_result = self.get_result()
                self.result_dict[self.policy.motion_name]['root'].append(motion_result.root)
                self.result_dict[self.policy.motion_name]['jt'].append(motion_result.jt)
                self.result_dict[self.policy.motion_name]['torque'].append(motion_result.torque)
            if self.policy.frame_id >= self.policy.target_length[self.policy.motion_id] - 1:
                last_frame_id = -1
                root = np.stack(self.result_dict[self.policy.motion_name]['root'], axis=0)
                jt = np.stack(self.result_dict[self.policy.motion_name]['jt'], axis=0)
                torque = np.stack(self.result_dict[self.policy.motion_name]['torque'], axis=0)
                self.result_dict[self.policy.motion_name]['root'] = root
                self.result_dict[self.policy.motion_name]['jt'] = jt
                self.result_dict[self.policy.motion_name]['torque'] = torque
                if (self.policy.motion_id+1) == len(self.policy.target_name):
                    onnx_model_path = self.policy.onnx_model_path
                    motion_path = self.policy.motion_path
                    # get save path using onnx_model_path and motion_path
                    model_tag = onnx_model_path.split('/')[-2]+onnx_model_path.split('/')[-1].split('.')[0]
                    save_path = motion_path.replace('.pkl', f'_{model_tag}.pkl')
                    # breakpoint()
                    joblib.dump(self.result_dict, save_path)
            else:
                last_frame_id = int(self.policy.frame_id)

            low_cmd = self.policy.inference(self.get_low_state())
            for _ in range(int(self.viewer_dt / self.simulate_dt)):
                sim_torque = self.low_command_to_torque(low_cmd)
                mujoco.mj_step(self.mj_model, self.mj_data)
            self.viewer.sync()
            end_t = time.perf_counter()
            if (end_t - step_start) < self.viewer_dt:
                time.sleep(self.viewer_dt - (end_t - step_start))
            # else:
            #     breakpoint()

    def _remote_command_handler(self, remote_command: LocomotionCommand):
        self._remote_command = remote_command

    def get_low_state(self) -> H1LowState:
        imu_quaternion = quat_wxyz_to_xyzw(self.mj_data.sensor("imu_quat").data)
        rot_mat = R.from_quat(imu_quaternion).as_matrix()
        ang_vel = self.mj_data.sensor("imu_gyro").data.astype(np.double)
        gravity_vec = np.dot(rot_mat.T, np.array([0, 0, -1]))
        joint_position = self.cfg.policy.joint_sim_to_real(
            self.mj_data.sensordata[self.sensor_dict["joint_position"]]
        )
        joint_velocity = self.cfg.policy.joint_sim_to_real(
            self.mj_data.sensordata[self.sensor_dict["joint_velocity"]]
        )

        return H1LowState(
            is_alive=True,
            base_projected_gravity=gravity_vec,
            base_angular_velocity=ang_vel,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            timestamp=time.perf_counter(),
        )
    
    def get_result(self) -> H1StateResult:
        root_quat = quat_wxyz_to_xyzw(self.mj_data.sensor("imu_quat").data)
        # rot_mat = R.from_quat(imu_quaternion).as_matrix()
        # root_pos = self.mj_data.sensor("imu_gyro").data.astype(np.double)
        root_pos = self.mj_data.sensordata[self.sensor_dict["frame_pos"]]
        jt = self.mj_data.sensordata[self.sensor_dict["joint_position"]]
        torque = self.mj_data.sensordata[self.sensor_dict["joint_torque"]]
        root = np.concatenate([root_pos, root_quat], axis=-1)
        return H1StateResult(
            root=root,
            # root_quat=root_quat,
            jt=jt,
            torque=torque,
        )

    def low_command_to_torque(self, low_command: H1LowCommand):
        low_state = self.get_low_state()
        tar_q = np.float32(low_command.joint_position)
        tar_dq = 0.0
        curr_q = np.float32(low_state.joint_position)
        curr_dq = np.float32(low_state.joint_velocity)
        kp = np.float32(low_command.Kp)
        kd = np.float32(low_command.Kd)

        target_torque = kp * (tar_q - curr_q) + kd * (tar_dq - curr_dq)
        sim_torque = self.cfg.policy.joint_real_to_sim(target_torque)
        self.mj_data.ctrl[:] = sim_torque
        return sim_torque

    def joint_sim2real(self, sim_joint_data: np.ndarray):
        return self.cfg.policy.joint_sim_to_real(sim_joint_data)

    def joint_real2sim(self, real_joint_data: np.ndarray):
        return self.cfg.policy.joint_real_to_sim(real_joint_data)

    def convert_to_low_state(self, sim_sensor_data: np.ndarray):
        joint_position = self.joint_sim2real(
            sim_sensor_data[self.sensor_dict["joint_position"]]
        )
        joint_velocity = self.joint_sim2real(
            sim_sensor_data[self.sensor_dict["joint_velocity"]]
        )
        joint_torque = self.joint_sim2real(
            sim_sensor_data[self.sensor_dict["joint_torque"]]
        )
        imu_quat = sim_sensor_data[self.sensor_dict["imu_quat"]]
        imu_gyroscope = sim_sensor_data[self.sensor_dict["imu_gyroscope"]]
        imu_accelerometer = sim_sensor_data[self.sensor_dict["imu_accelerometer"]]
        return {
            "q": joint_position,
            "dq": joint_velocity,
            "ddq": joint_torque,
            "imu_quat": imu_quat,
            "imu_gyroscope": imu_gyroscope,
            "imu_accelerometer": imu_accelerometer,
        }

    def convert_to_high_state(self, sim_sensor_data: np.ndarray):
        frame_pos = sim_sensor_data[self.sensor_dict["frame_pos"]]
        frame_vel = sim_sensor_data[self.sensor_dict["frame_vel"]]
        return {
            "frame_pos": frame_pos,
            "frame_vel": frame_vel,
        }


if __name__ == "__main__":

    cfg = RealH1WalkConfig()
    policy = TrackPolicy(
        cfg, 
        # "/home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/h1-s2r/1025_lctk_still_dr/policy_60000.onnx",
        # '/home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/h1-s2r/1027_lctk_adof_still_dr/policy_27000.onnx', # NOTE useful
        # '/home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/h1-s2r/1029_lctk_nowalk_lean/policy_35000.onnx',
        # '/home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/h1-s2r/1029_lctk_still_dr_resume/policy_30000.onnx',
        # '/home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/h1-s2r/1101_lctk_box_dr/policy_60000.onnx',
        # '/home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/h1-s2r/1102_lctk_hand_dr/policy_30000.onnx',
        # '/home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/h1-s2r/1102_lctk_dfstill_hip_scretch_dr/policy_30000.onnx',
        # '/home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/h1-s2r/1029_lctk_swingwalk_dr_resume/policy_65000.onnx',
        # '/home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/h1-s2r/1028_lctk_walkforward_hipyaw_newmotion_dr_resume/policy_60000.onnx',
        # '/home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/h1-s2r/1101_lctk_swingwalk_dr/policy_90000.onnx',
        # '/home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/h1-s2r/1103_lctk_hand_ankle_dr/policy_37000.onnx',
        # '/home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/h1-s2r/1102_17hand/policy_28000.onnx',
        # '/home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/h1-s2r/1102_17box/policy_22000.onnx',
        '/home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/h1-s2r/1110_hand_randstill/policy_60000.onnx',
        # '/home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/h1-s2r/1110_swingwalk_randstill/policy_100000.onnx',

        # '/home/ubuntu/data/MDM/dn_10_swingwalk_mdm.pkl', # 0
        '/home/ubuntu/data/MDM/dn_35hand_mdm.pkl', # 3 7 33
        # '/home/ubuntu/data/MDM/rcphp_10box.pkl'  # 2 5 6 7 8 9
        # '/home/ubuntu/data/MDM/rcp_10swingwalk_mdm.pkl' # 1 3 7.
    )
    sim = MujocoSim(RealH1WalkConfig(), policy)
    sim.run()
