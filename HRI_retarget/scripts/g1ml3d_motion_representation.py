import os
import subprocess
from HRI_retarget import DATA_ROOT
from tqdm import tqdm
import time 
from datetime import timedelta
from queue import Queue
from threading import Thread
import pickle
import numpy as np
from pathlib import Path

from HRI_retarget.utils.io.brainco_representation import data_pkl_to_vec

# 文件夹路径
folder_path = "/root/workspace/HRI_MLLM/data/BEAT_v2"
tgt_dir = "/root/workspace/HRI_MLLM/data/BEAT_v2_kimi/new_joint_vecs"
starting_time = time.time()


def find_pickle_files(root_dir):
    root_path = Path(root_dir)
    pickle_files = list(root_path.rglob('*.npz'))
    return [str(file) for file in pickle_files]

pickle_files = find_pickle_files(folder_path)


for source_file in tqdm(pickle_files):
    
       
    dof = np.load(source_file)['qpos'][:, 7:]
    data_dict = {"angles":dof, 
                "robot_name": "g1_brainco",
                "fps":60}

    vec = data_pkl_to_vec(data_dict)

    save_name = os.path.basename(source_file).replace(".npz", ".npy")
    np.save(os.path.join(tgt_dir, save_name), vec)
   

