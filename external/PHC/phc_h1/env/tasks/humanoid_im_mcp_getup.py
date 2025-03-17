

from typing import OrderedDict
import torch
import numpy as np
from phc_h1.utils.torch_utils import quat_to_tan_norm
import phc_h1.env.tasks.humanoid_im_getup as humanoid_im_getup
import phc_h1.env.tasks.humanoid_im_mcp as humanoid_im_mcp
from phc_h1.env.tasks.humanoid_amp import HumanoidAMP, remove_base_rot
from phc_h1.utils.motion_lib_h1 import MotionLibSMPL 

from phc_h1.utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from phc_h1.utils.flags import flags
import joblib
import gc
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from rl_games.algos_torch import torch_ext
import torch.nn as nn
from phc_h1.learning.network_loader import load_mcp_mlp, load_pnn
from collections import deque

class HumanoidImMCPGetup(humanoid_im_getup.HumanoidImGetup, humanoid_im_mcp.HumanoidImMCP):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        return

