### convert asap pkl files to unitree csv files

### usage python src/utils/io/asap_to_csv.py --pkl {pkl path} --csv {csv_path}


import sys
import os
from HRI_retarget import ROOT,SRC_ROOT,DATA_ROOT
sys.path.append(SRC_ROOT)
# sys.path.append("/home/pengyang/codebase/retarget/src")


import joblib
import numpy as np
import argparse
import torch
from scipy.spatial.transform import Rotation as sRot
from utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

def asap_to_csv(input_path, output_path):
    with open(input_path, "rb") as file:
        data = joblib.load(file)
    # print(data.keys())

    key = list(data.keys())[0]
    csv_data = np.zeros((data[key]["root_trans_offset"].shape[0], 36))
    csv_data[:, :3] = data[key]["root_trans_offset"]
    csv_data[:, 3:7] = data[key]["root_rot"]
    csv_data[:, 7:26] = data[key]["dof"][:, :19]
    csv_data[:, 29:33] = data[key]["dof"][:, 19:23]

    print(data[key]["pose_aa"][0, -2, :], data[key]["root_rot"][0])

    root_rot_vec = torch.from_numpy(sRot.from_quat(data[key]["root_rot"]).as_rotvec()).float()

    cfg = {
        # "assetRoot": "/home/pengyang/codebase/retarget/data/resources/robots/g1_asap",
        "assetRoot": os.path.join(DATA_ROOT,"resources/robots/g1_asap"),
        "assetFileName": "g1_29dof_anneal_23dof_fitmotionONLY.xml",
        "extend_config": [{
                "joint_name": "left_hand_link",
                "parent_name": "left_elbow_link",
                "pos": [0.25, 0.0, 0.0],
                "rot": [1.0, 0.0, 0.0, 0.0] # w x y z
            },{
                "joint_name": "right_hand_link",
                "parent_name": "right_elbow_link",
                "pos": [0.25, 0.0, 0.0],
                "rot": [1.0, 0.0, 0.0, 0.0] # w x y z
            },{
                "joint_name": "head_link",
                "parent_name": "torso_link",
                "pos": [0.0, 0.0, 0.42],
                "rot": [1.0, 0.0, 0.0, 0.0] # w x y z
            },
        ]
    }

    mesh_parsers = Humanoid_Batch(cfg)
    print(mesh_parsers)
    print(root_rot_vec.shape)
    a = mesh_parsers.dof_axis * data[key]["dof"][:, :, None]
    print(a.shape)
    pose_aa = torch.cat([root_rot_vec[:, None, :], mesh_parsers.dof_axis * data[key]["dof"][:, :, None], torch.zeros((data[key]["root_trans_offset"].shape[0], 3, 3))], axis = 1)
    print(pose_aa.shape)
    print(data[key]["pose_aa"].shape)

    idx = 100
    print(pose_aa[idx] - data[key]["pose_aa"][idx])
    # print(data[key]["pose_aa"][0])
    # print(pose_aa - data[key]["pose_aa"].mean())
    for k in data[key].keys():
        print(k, data[key][k].shape)

def csv_to_asap(csv_path, pkl_path):
    csv_data = np.genfromtxt(csv_path, delimiter=',')[:400, :]
    exp_name = csv_path
    print(csv_data.shape)

    data = {}
    data["root_trans_offset"] = csv_data[:, :3]
    data["root_rot"] = csv_data[:, 3:7]
    data["dof"] = np.zeros((csv_data.shape[0], 23))
    data["dof"][:, :19] = csv_data[:, 7:26]
    data["dof"][:, 19:23] = csv_data[:, 29:33]
    data["fps"] = 30
    pkl_data = {
        exp_name: data
    }
    joblib.dump(pkl_data, pkl_path)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', type=str, help="File name", default=os.path.join(DATA_ROOT,"motion/g1/cr7/asap.pkl"))
    parser.add_argument('--csv', type=str, help="csv file name", default=os.path.join(DATA_ROOT,"motion/g1/LAFAN1/dance1_subject2.csv"))
    args = parser.parse_args()

    asap_to_csv(args.pkl, args.csv)





