import os
import time
import copy
from tabulate import tabulate

import isaacgym
from galaxy_legged_gym.envs import *
from galaxy_legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger, get_load_path

import numpy as np
import onnxruntime as ort
from datetime import datetime
import torch

from real_robot_utils.monitor.base import SequentialDataMonitor


def export(args, device='cuda:0'):
    log_pth = "logs/{}/".format(args.proj_name) + args.exptid
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root=log_pth, env=env, name=args.task, args=args,
                                                          train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    policy_model = ppo_runner.alg.actor_critic.actor

    path = f"model_weights/{args.exptid}"
    os.makedirs(path, exist_ok=True)

    # HERE: convert pt to onnx
    resume_path = get_load_path(log_pth, checkpoint=args.checkpoint)
    checkpoint = int(resume_path.split("_")[-1].split(".")[0])
    onnx_export_path = f"{path}/policy_{checkpoint}.onnx"
    print(f"Exporting policy to {onnx_export_path}")

    torch.onnx.export(policy_model, obs[:1], onnx_export_path,
                      verbose=False, input_names=['obs'], output_names=['action'])

    # # use onnx cpu to run the policy
    # onnx_session = ort.InferenceSession(onnx_export_path, providers=['CPUExecutionProvider'])

    # for i in range(1 * int(env.max_episode_length)):
    #     obs[:, 0] = 0.5
    #     obs[:, 1] = 0.0
    #     obs[:, 2] = 0.0
    #     actions = onnx_session.run(['action'], {'obs': obs.detach().cpu().numpy()})[0]
    #     obs, _, rews, dones, infos = env.step(torch.tensor(actions, device=env.device))


def export_benchmarking(args, device='cuda:0'):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    experiment_name = train_cfg.runner.experiment_name
    num_obs = env_cfg.env.num_observations

    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', experiment_name, 'exported', 'policies')
    os.makedirs(path, exist_ok=True)
    onnx_export_path = f"{path}/policy.onnx"
    torch_script_export_path = f"{path}/policy.pt"
    # load torch script
    # Load the TorchScript model
    jit_model = torch.jit.load(torch_script_export_path)
    jit_model.eval()
    jit_model.to(device)

    # load and play with onnx
    onnx_session = ort.InferenceSession(onnx_export_path, providers=['CUDAExecutionProvider'])

    # CPU
    cpu_session = ort.InferenceSession(onnx_export_path, providers=['CPUExecutionProvider'])

    def get_obs():
        return torch.rand((1, num_obs), device=device)

    inference_test_num = 3000

    onnx_runtime_list = []
    for _ in range(inference_test_num):
        start_t = time.perf_counter()
        actions = onnx_session.run(['action'], {'obs': get_obs().detach().cpu().numpy()})[0]
        onnx_runtime_list.append(time.perf_counter() - start_t)

    cpu_runtime_list = []
    for _ in range(inference_test_num):
        start_t = time.perf_counter()
        actions = cpu_session.run(['action'], {'obs': get_obs().detach().cpu().numpy()})[0]
        cpu_runtime_list.append(time.perf_counter() - start_t)

    torch_script_runtime_list = []
    for i in range(inference_test_num):
        start_t = time.perf_counter()
        actions = jit_model(get_obs().detach())
        torch_script_runtime_list.append(time.perf_counter() - start_t)

    info = {
        'Method': ['ONNX-GPU', 'ONNX-CPU', 'TorchScript'],
        'Frequency': [1 / np.mean(onnx_runtime_list),
                      1 / np.mean(cpu_runtime_list),
                      1 / np.mean(torch_script_runtime_list)],
    }
    print(tabulate(info, headers='keys', tablefmt='fancy_grid'))


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    args = get_args()
    export(args)

    # benchmarking
    # args = get_args()
    # export_benchmarking(args)
