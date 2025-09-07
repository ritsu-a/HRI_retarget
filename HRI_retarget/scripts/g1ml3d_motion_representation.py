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

from HRI_retarget.utils.io.inspirehand_representation import data_pkl_to_vec

# 文件夹路径
folder_path = "/root/pengyang/codebase/HRI_MLLM/data/BEAT_v1"
tgt_dir = "/root/pengyang/codebase/HRI_MLLM/data/BEAT_v1_kimi/new_joint_vecs"
starting_time = time.time()


def find_pickle_files(root_dir):
    root_path = Path(root_dir)
    pickle_files = list(root_path.rglob('*.pickle'))
    return [str(file) for file in pickle_files]

pickle_files = find_pickle_files(folder_path)


for source_file in tqdm(pickle_files):
    try:

        with open(source_file, "rb") as file:
            data_dict = pickle.load(file)

        vec = data_pkl_to_vec(data_dict)

        save_name = os.path.basename(source_file).replace(".pickle", ".npy")
        np.save(os.path.join(tgt_dir, save_name), vec)
    except Exception as e:
        print(source_file)
        print(e)


