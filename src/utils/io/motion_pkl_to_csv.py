### convert asap pkl files to unitree csv files

### usage python src/utils/io/asap_to_csv.py --pkl {pkl path} --csv {csv_path}

import sys
import os
from HRI_retarget import ROOT,SRC_ROOT,DATA_ROOT
sys.path.append(SRC_ROOT)

import joblib
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as sRot
from utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
import torch

from utils.torch_utils.diff_quat import vec6d_to_quat
from config.joint_mapping import G1_29_DOFS, G1_15_DOFS

def pkl_to_csv(input_path, output_path):
    with open(input_path, "rb") as file:
        data = joblib.load(file)
    # print(data.keys())
    csv_data = np.zeros((data["angles"].shape[0], 36))
    csv_data[:, :3] = data["global_translation"]


    csv_data[:, 3:7] = vec6d_to_quat(torch.tensor(data['global_rotation'])).numpy()

    for idx in range(15):
        csv_data[:, 7 + G1_29_DOFS.index(G1_15_DOFS[idx])] = data["angles"][:, idx]

    
    np.savetxt(output_path, csv_data, delimiter=',', fmt='%.8f')




    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', type=str, help="File name", default=os.path.join(DATA_ROOT,"motion/g1/SG/output.pickle"))
    parser.add_argument('--csv', type=str, help="csv file name", default=os.path.join(DATA_ROOT,"motion/g1/SG/output.csv"))
    args = parser.parse_args()

    pkl_to_csv(args.pkl, args.csv)
    # csv_to_pkl(args.csv, args.pkl)





