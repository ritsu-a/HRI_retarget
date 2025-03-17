# What's in this repo

1. retarget for galbot from seg_dataset

# Basic usage:
seg_dataset bvh to galbot dof position retarget:
    python src/retarget/bvh_galbot.py /home/pengyang/codebase/H1_RL/data/SeG_dataset/bvh/ARMS_FOLD-1.bvh
after this, you should generate a file as /home/pengyang/codebase/H1_RL/data/SeG_dataset/galbot_motion/ARMS_FOLD-1.pickle


visualize kinematic results:
    python src/utils/kinematic_vis.py /home/pengyang/codebase/H1_RL/data/SeG_dataset/galbot_motion/ARMS_FOLD-1.pickle

visualize dynamic results via pybullet and position pd control:
    python src/utils/pybullet_visualize_galbot_dynamic.py /home/pengyang/codebase/H1_RL/data/SeG_dataset/galbot_motion/ARMS_FOLD-1.pickle

