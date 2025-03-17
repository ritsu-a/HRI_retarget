# # Copyright (c) 2018-2023, NVIDIA Corporation
# # All rights reserved.
# #
# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:
# #
# # 1. Redistributions of source code must retain the above copyright notice, this
# #    list of conditions and the following disclaimer.
# #
# # 2. Redistributions in binary form must reproduce the above copyright notice,
# #    this list of conditions and the following disclaimer in the documentation
# #    and/or other materials provided with the distribution.
# #
# # 3. Neither the name of the copyright holder nor the names of its
# #    contributors may be used to endorse or promote products derived from
# #    this software without specific prior written permission.
# #
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# import glob
# import os
# import sys
# import pdb
# import os.path as osp
# os.environ["OMP_NUM_THREADS"] = "1"

# sys.path.append(os.getcwd())

# from phc.utils.config import set_np_formatting, set_seed, SIM_TIMESTEP
# from phc.utils.parse_task import parse_task
# from isaacgym import gymapi
# from isaacgym import gymutil


# from rl_games.algos_torch import players
# from rl_games.algos_torch import torch_ext
# from rl_games.common import env_configurations, experiment, vecenv
# from rl_games.common.algo_observer import AlgoObserver
# from rl_games.torch_runner import Runner

# from phc.utils.flags import flags

# import numpy as np
# import copy
# import torch
# # import wandb

# from learning import im_amp
# from learning import im_amp_players
# from learning import amp_agent
# from learning import amp_players
# from learning import amp_models
# from learning import amp_network_builder
# from learning import amp_network_mcp_builder
# from learning import amp_network_pnn_builder

# from env.tasks import humanoid_amp_task
# import hydra
# from omegaconf import DictConfig, OmegaConf
# from easydict import EasyDict

# args = None
# cfg = None
# cfg_train = None


# def parse_sim_params(cfg):
#     # initialize sim
#     sim_params = gymapi.SimParams()
#     sim_params.dt = SIM_TIMESTEP
#     sim_params.num_client_threads = cfg.sim.slices
    
#     if cfg.sim.use_flex:
#         if cfg.sim.pipeline in ["gpu"]:
#             print("WARNING: Using Flex with GPU instead of PHYSX!")
#         sim_params.use_flex.shape_collision_margin = 0.01
#         sim_params.use_flex.num_outer_iterations = 4
#         sim_params.use_flex.num_inner_iterations = 10
#     else : # use gymapi.SIM_PHYSX
#         sim_params.physx.solver_type = 1
#         sim_params.physx.num_position_iterations = 4
#         sim_params.physx.num_velocity_iterations = 1
#         sim_params.physx.num_threads = 4
#         sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]
#         sim_params.physx.num_subscenes = cfg.sim.subscenes
#         if flags.test and not flags.im_eval:
#             sim_params.physx.max_gpu_contact_pairs = 4 * 1024 * 1024
#         else:
#             sim_params.physx.max_gpu_contact_pairs = 16 * 1024 * 1024

#     sim_params.use_gpu_pipeline = cfg.sim.pipeline in ["gpu"]
#     sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]

#     # if sim options are provided in cfg, parse them and update/override above:
#     if "sim" in cfg:
#         gymutil.parse_sim_config(cfg["sim"], sim_params)

#     # Override num_threads if passed on the command line
#     if not cfg.sim.use_flex and cfg.sim.physx.num_threads > 0:
#         sim_params.physx.num_threads = cfg.sim.physx.num_threads
    
#     return sim_params

# def create_rlgpu_env(**kwargs):
#     use_horovod = cfg_train['params']['config'].get('multi_gpu', False)
#     if use_horovod:
#         import horovod.torch as hvd

#         rank = hvd.rank()
#         print("Horovod rank: ", rank)

#         cfg_train['params']['seed'] = cfg_train['params']['seed'] + rank

#         args.device = 'cuda'
#         args.device_id = rank
#         args.rl_device = 'cuda:' + str(rank)

#         cfg['rank'] = rank
#         cfg['rl_device'] = 'cuda:' + str(rank)
    
#     sim_params = parse_sim_params(cfg)
#     args = EasyDict({
#         "task": cfg.env.task, 
#         "device_id": cfg.device_id,
#         "rl_device": cfg.rl_device,
#         "physics_engine": gymapi.SIM_PHYSX if not cfg.sim.use_flex else gymapi.SIM_FLEX,
#         "headless": cfg.headless,
#         "device": cfg.device,
#     }) #### ZL: patch 
#     task, env = parse_task(args, cfg, cfg_train, sim_params)

#     print(env.num_envs)
#     print(env.num_actions)
#     print(env.num_obs)
#     print(env.num_states)

#     frames = kwargs.pop('frames', 1)
#     if frames > 1:
#         env = wrappers.FrameStack(env, frames, False)
#     return env


# class RLGPUAlgoObserver(AlgoObserver):

#     def __init__(self, use_successes=True):
#         self.use_successes = use_successes
#         return

#     def after_init(self, algo):
#         self.algo = algo
#         self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
#         self.writer = self.algo.writer
#         return

#     def process_infos(self, infos, done_indices):
#         if isinstance(infos, dict):
#             if (self.use_successes == False) and 'consecutive_successes' in infos:
#                 cons_successes = infos['consecutive_successes'].clone()
#                 self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
#             if self.use_successes and 'successes' in infos:
#                 successes = infos['successes'].clone()
#                 self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
#         return

#     def after_clear_stats(self):
#         self.mean_scores.clear()
#         return

#     def after_print_stats(self, frame, epoch_num, total_time):
#         if self.consecutive_successes.current_size > 0:
#             mean_con_successes = self.consecutive_successes.get_mean()
#             self.writer.add_scalar('successes/consecutive_successes/mean', mean_con_successes, frame)
#             self.writer.add_scalar('successes/consecutive_successes/iter', mean_con_successes, epoch_num)
#             self.writer.add_scalar('successes/consecutive_successes/time', mean_con_successes, total_time)
#         return


# class RLGPUEnv(vecenv.IVecEnv):

#     def __init__(self, config_name, num_actors, **kwargs):
#         self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
#         self.use_global_obs = (self.env.num_states > 0)

#         self.full_state = {}
#         self.full_state["obs"] = self.reset()
#         if self.use_global_obs:
#             self.full_state["states"] = self.env.get_state()
#         return

#     def step(self, action):
#         next_obs, reward, is_done, info = self.env.step(action)

#         # todo: improve, return only dictinary
#         self.full_state["obs"] = next_obs
#         if self.use_global_obs:
#             self.full_state["states"] = self.env.get_state()
#             return self.full_state, reward, is_done, info
#         else:
#             return self.full_state["obs"], reward, is_done, info

#     def reset(self, env_ids=None):
#         self.full_state["obs"] = self.env.reset(env_ids)
#         if self.use_global_obs:
#             self.full_state["states"] = self.env.get_state()
#             return self.full_state
#         else:
#             return self.full_state["obs"]

#     def get_number_of_agents(self):
#         return self.env.get_number_of_agents()

#     def get_env_info(self):
#         info = {}
#         info['action_space'] = self.env.action_space
#         info['observation_space'] = self.env.observation_space
#         info['amp_observation_space'] = self.env.amp_observation_space
        
#         info['enc_amp_observation_space'] = self.env.enc_amp_observation_space
        
#         if isinstance(self.env.task, humanoid_amp_task.HumanoidAMPTask):
#             info['task_obs_size'] = self.env.task.get_task_obs_size()
#         else:
#             info['task_obs_size'] = 0

#         if self.use_global_obs:
#             info['state_space'] = self.env.state_space
#             print(info['action_space'], info['observation_space'], info['state_space'])
#         else:
#             print(info['action_space'], info['observation_space'])

#         return info


# vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
# env_configurations.register('rlgpu', {'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs), 'vecenv_type': 'RLGPU'})


# def build_alg_runner(algo_observer):
#     runner = Runner(algo_observer)
#     runner.player_factory.register_builder('amp_discrete', lambda **kwargs: amp_players.AMPPlayerDiscrete(**kwargs))
    
#     runner.algo_factory.register_builder('amp', lambda **kwargs: amp_agent.AMPAgent(**kwargs))
#     runner.player_factory.register_builder('amp', lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs))

#     runner.model_builder.model_factory.register_builder('amp', lambda network, **kwargs: amp_models.ModelAMPContinuous(network))
#     runner.model_builder.network_factory.register_builder('amp', lambda **kwargs: amp_network_builder.AMPBuilder())
#     runner.model_builder.network_factory.register_builder('amp_mcp', lambda **kwargs: amp_network_mcp_builder.AMPMCPBuilder())
#     runner.model_builder.network_factory.register_builder('amp_pnn', lambda **kwargs: amp_network_pnn_builder.AMPPNNBuilder())
    
#     runner.algo_factory.register_builder('im_amp', lambda **kwargs: im_amp.IMAmpAgent(**kwargs))
#     runner.player_factory.register_builder('im_amp', lambda **kwargs: im_amp_players.IMAMPPlayerContinuous(**kwargs))
    
#     return runner

# @hydra.main(
#     version_base=None,
#     config_path="../phc/data/cfg",
#     config_name="config",
# )
# def main(cfg_hydra: DictConfig) -> None:
#     global cfg_train
#     global cfg
    
#     cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
    
#     set_np_formatting()

#     # cfg, cfg_train, logdir = load_cfg(args)
#     flags.debug, flags.follow, flags.fixed, flags.divide_group, flags.no_collision_check, flags.fixed_path, flags.real_path,  flags.show_traj, flags.server_mode, flags.slow, flags.real_traj, flags.im_eval, flags.no_virtual_display, flags.render_o3d = \
#         cfg.debug, cfg.follow, False, False, False, False, False, True, cfg.server_mode, False, False, cfg.im_eval, cfg.no_virtual_display, cfg.render_o3d

#     flags.test = cfg.test
#     flags.add_proj = cfg.add_proj
#     flags.has_eval = cfg.has_eval
#     flags.trigger_input = False

#     if cfg.server_mode:
#         flags.follow = cfg.follow = True
#         flags.fixed = cfg.fixed = True
#         flags.no_collision_check = True
#         flags.show_traj = True
#         cfg['env']['episode_length'] = 99999999999999

#     if cfg.real_traj:
#         cfg['env']['episode_length'] = 99999999999999
#         flags.real_traj = True
    
#     cfg.train = not cfg.test
#     project_name = cfg.get("project_name", "egoquest")
#     # if (not cfg.no_log) and (not cfg.test) and (not cfg.debug):
#     #     wandb.init(
#     #         project=project_name,
#     #         resume=not cfg.resume_str is None,
#     #         id=cfg.resume_str,
#     #         notes=cfg.get("notes", "no notes"),
#     #     )
#     #     wandb.config.update(cfg, allow_val_change=True)
#     #     wandb.run.name = cfg.exp_name
#     #     wandb.run.save()
    
#     set_seed(cfg.get("seed", -1), cfg.get("torch_deterministic", False))

#     # Create default directories for weights and statistics
#     cfg_train = cfg.learning
#     cfg_train['params']['config']['network_path'] = cfg.output_path
#     cfg_train['params']['config']['train_dir'] = cfg.output_path
#     cfg_train["params"]["config"]["num_actors"] = cfg.env.num_envs
#     cfg_train["params"]["config"]["multi_gpu"] = cfg.horovod
    
#     if cfg.epoch > 0:
#         cfg_train["params"]["load_checkpoint"] = True
#         cfg_train["params"]["load_path"] = osp.join(cfg.output_path, cfg_train["params"]["config"]['name'] + "_" + str(cfg.epoch).zfill(8) + '.pth')
#     elif cfg.epoch == -1:
#         path = osp.join(cfg.output_path, cfg_train["params"]["config"]['name'] + '.pth')
#         if osp.exists(path):
#             cfg_train["params"]["load_path"] = path
#             cfg_train["params"]["load_checkpoint"] = True
#         else:
#             print(path)
#             raise Exception("no file to resume!!!!")

    
#     os.makedirs(cfg.output_path, exist_ok=True)
    
#     algo_observer = RLGPUAlgoObserver()
#     runner = build_alg_runner(algo_observer)
#     runner.load(cfg_train)
#     runner.reset()
#     player = runner.create_player()
#     from IPython import embed;embed()
#     # runner.run(cfg)

#     return


# if __name__ == '__main__':
#     main()
# from ast import Try
# import glob
import os
import sys
# import pdb
# import os.path as osp

sys.path.append(os.getcwd())
# from enum import Enum
# from matplotlib.pyplot import flag
import numpy as np
# from typing import Dict, Optional

# from isaacgym import gymapi
# from isaacgym import gymtorch

# from phc.env.tasks.humanoid import Humanoid, dof_to_obs, remove_base_rot, dof_to_obs_smpl
# from phc.env.util import gym_util
from phc_h1.utils.motion_lib_h1 import MotionLibSMPL 
from phc_h1.utils.motion_lib_base import FixHeightMode
from easydict import EasyDict

# from isaacgym.torch_utils import *
# from phc.utils import torch_utils
# from smpl_sim.smpllib.smpl_parser import (
#     SMPL_Parser,
#     SMPLH_Parser,
#     SMPLX_Parser,
# )
# import gc
from phc_h1.utils.flags import flags
# from collections import OrderedDict
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
import torch
import torch.multiprocessing as mp
from poselib.poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from poselib.retarget_motion import project_joints
from tqdm import tqdm
import json
VISUALIZE = False
motion_file = 'data/amass/pkls/amass_isaac_train_0.pkl'

motion_lib_cfg = EasyDict({
    "motion_file": motion_file,
    "device": torch.device("cpu"),
    "fix_height": FixHeightMode.full_fix,
    "min_length": -1,
    "max_length": -1,
    "im_eval": flags.im_eval,
    "multi_thread": True ,
    "smpl_type": 'smpl',
    "randomrize_heading": True,
    "device": 'cuda:0',
    "min_length": 5, 
})
motion_lib = MotionLibSMPL(motion_lib_cfg=motion_lib_cfg)
gender_beta = torch.zeros(17)

asset_file_real = f"phc/data/assets/mjcf/smpl_{int(gender_beta[0])}_humanoid.xml"
skeleton = SkeletonTree.from_mjcf(asset_file_real)

# # generate zero rotation pose
# from poselib.poselib.core.rotation3d import *
# from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
# zero_pose = SkeletonState.zero_pose(skeleton)

# # local_rotation = zero_pose.local_rotation
# # local_rotation[skeleton.index("L_Shoulder")] = quat_mul(
# #     quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
# #     local_rotation[skeleton.index("L_Shoulder")]
# # )
# # local_rotation[skeleton.index("R_Shoulder")] = quat_mul(
# #     quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
# #     local_rotation[skeleton.index("R_Shoulder")]
# # )
# translation = zero_pose.root_translation
# translation += torch.tensor([0, 0, 0.9])

# # save and visualize T-pose
# zero_pose.to_file("data/smpl_tpose.npy")
# plot_skeleton_state(zero_pose)

mp.set_sharing_strategy('file_descriptor')

manager = mp.Manager()
queue = manager.Queue()
num_jobs = min(mp.cpu_count(), 64)
res_acc = {}  # using dictionary ensures order of the results.
jobs = motion_lib._motion_data_list
chunk = np.ceil(len(jobs) / num_jobs).astype(int)
ids = np.arange(len(jobs))

jobs = [(ids[i:i + chunk], jobs[i:i + chunk], [skeleton]*chunk, [gender_beta]*chunk,  motion_lib.mesh_parsers, motion_lib.m_cfg) for i in range(0, len(jobs), chunk)]
job_args = [jobs[i] for i in range(len(jobs))]
# for i in range(1, len(jobs)):
#     worker_args = (*job_args[i], queue, i)
#     worker = mp.Process(target=motion_lib.load_motion_with_skeleton, args=worker_args)
#     worker.start()
res_acc.update(motion_lib.load_motion_with_skeleton(*jobs[0], None, 0))


retarget_data_path = "data/retarget_smpl2h1.json"
with open(retarget_data_path) as f:
    retarget_data = json.load(f)
source_tpose = SkeletonState.from_file("data/smpl_tpose.npy")
from IPython import embed;embed()
if VISUALIZE:
    plot_skeleton_state(source_tpose)
# ['Pelvis',
#  'L_Hip',
#  'L_Knee',
#  'L_Ankle',
#  'L_Toe',
#  'R_Hip',
#  'R_Knee',
#  'R_Ankle',
#  'R_Toe',
#  'Torso',
#  'Spine',
#  'Chest',
#  'Neck',
#  'Head',
#  'L_Thorax',
#  'L_Shoulder',
#  'L_Elbow',
#  'L_Wrist',
#  'L_Hand',
#  'R_Thorax',
#  'R_Shoulder',
#  'R_Elbow',
#  'R_Wrist',
#  'R_Hand']
target_tpose = SkeletonState.from_file("data/h1_tpose.npy")
if VISUALIZE:
    plot_skeleton_state(target_tpose)
    
# ['pelvis',
#  'left_hip_yaw_link',
#  'left_hip_roll_link',
#  'left_hip_pitch_link',
#  'left_knee_link',
#  'left_ankle_link',
#  'right_hip_yaw_link',
#  'right_hip_roll_link',
#  'right_hip_pitch_link',
#  'right_knee_link',
#  'right_ankle_link',
#  'torso_link',
#  'left_shoulder_pitch_link',
#  'left_shoulder_roll_link',
#  'left_shoulder_yaw_link',
#  'left_elbow_link',
#  'right_shoulder_pitch_link',
#  'right_shoulder_roll_link',
#  'right_shoulder_yaw_link',
#  'right_elbow_link']
joint_mapping = retarget_data["joint_mapping"]
rotation_to_target_skeleton = torch.tensor(retarget_data["rotation"])
for f in tqdm(range(len(res_acc))):
    motion_file_data, source_motion = res_acc[f]
    # from IPython import embed;embed()
    
    target_motion = source_motion.retarget_to_by_tpose(
      joint_mapping=retarget_data["joint_mapping"],
      source_tpose=source_tpose,
      target_tpose=target_tpose,
      rotation_to_target_skeleton=rotation_to_target_skeleton,
      scale_to_target_skeleton=retarget_data["scale"]
    )
    # plot_skeleton_motion_interactive(source_motion)
    # plot_skeleton_motion_interactive(target_motion)

    # keep frames between [trim_frame_beg, trim_frame_end - 1]
    # frame_beg = retarget_data["trim_frame_beg"]
    # frame_end = retarget_data["trim_frame_end"]
    # if (frame_beg == -1):
    #     frame_beg = 0
        
    # if (frame_end == -1):
    #     frame_end = target_motion.local_rotation.shape[0]
        
    # local_rotation = target_motion.local_rotation
    # root_translation = target_motion.root_translation
    # local_rotation = local_rotation[frame_beg:frame_end, ...]
    # root_translation = root_translation[frame_beg:frame_end, ...]
      
    # new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation, root_translation, is_local=True)
    # target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

    # need to convert some joints from 3D to 1D (e.g. elbows and knees)
    # target_motion = project_joints(target_motion)

    # move the root so that the feet are on the ground
    local_rotation = target_motion.local_rotation
    root_translation = target_motion.root_translation
    tar_global_pos = target_motion.global_translation
    min_h = torch.min(tar_global_pos[..., 2])
    root_translation[:, 2] += -min_h
    
    # adjust the height of the root to avoid ground penetration
    root_height_offset = retarget_data["root_height_offset"]
    root_translation[:, 2] += root_height_offset
    
    new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation, root_translation, is_local=True)
    target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

    # save retargeted motion
    # target_motion.to_file(retarget_data["target_motion_path"])

    # visualize retargeted motion
    plot_skeleton_motion_interactive(target_motion)
# for i in tqdm(range(len(jobs) - 1)):
#     res = queue.get()
#     res_acc.update(res)
# from IPython import embed;embed()
    