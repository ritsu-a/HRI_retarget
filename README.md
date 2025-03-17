# What's in this repo

1. retarget for galbot from seg_dataset

# Basic usage:
you may search for /home/pengyang/data and replace it with your directory

seg_dataset bvh to galbot dof position retarget:
    python src/retarget/bvh_galbot.py /home/pengyang/data/motion/human/SeG_dataset/bvh/ARMS_FOLD-1.bvh
after this, you should generate a file as /home/pengyang/data/motion/galbot/SeG_dataset/ARMS_FOLD-1.pickle


visualize kinematic results:
    python src/utils/vis/kinematic_vis.py /home/pengyang/codebase/H1_RL/data_old/SeG_dataset/galbot_motion/ARMS_FOLD-1.pickle

visualize dynamic results via pybullet and position pd control:
    python src/utils/vis/pybullet_visualize_galbot_dynamic.py /home/pengyang/codebase/H1_RL/data_old/SeG_dataset/galbot_motion/ARMS_FOLD-1.pickle

