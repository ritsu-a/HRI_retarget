from pytorch_kinematics import build_chain_from_urdf
import torch
from HRI_retarget import ROOT,SRC_ROOT,DATA_ROOT
import os

def print_chain_offsets(urdf_path):
    with open(urdf_path, 'rb') as f:
        chain = build_chain_from_urdf(f.read())

    print(f"{'Frame':<25} | {'Joint Type':<10} | {'Joint Offset'}\n" + "-" * 80)
    for i, (frame_name, idx) in enumerate(chain.frame_to_idx.items()):
        jtype = "FIXED"
        if chain.joint_indices[idx] >= 0:
            jidx = chain.joint_indices[idx].item()
            jtype = ['FIXED', 'REVOLUTE', 'PRISMATIC'][chain.joint_type_indices[idx].item()]
        joint_offset = chain.joint_offsets[idx]
        link_offset = chain.link_offsets[idx]

        print(f"Frame {i:<2} ({frame_name:<20})")
        print(f"  Joint type  : {jtype}")
        print(f"  joint_offset (parent link ➜ joint):")
        print(f"{joint_offset.cpu().numpy() if joint_offset is not None else 'None'}")
        print(f"  link_offset  (joint ➜ current link):")
        print(f"{link_offset.cpu().numpy() if link_offset is not None else 'None'}\n")
        
urdf_rel_path = "resources/robots/g1_asap/g1_29dof_anneal_15dof.urdf"
urdf_path = os.path.join(DATA_ROOT,urdf_rel_path)
        
print_chain_offsets(urdf_path)