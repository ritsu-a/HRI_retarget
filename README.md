# What's in this repo

1. retarget for galbot from seg_dataset

# Basic usage:
you may search for /home/pengyang/data and replace it with your directory

semantic gesticulator bvh to galbot dof position retarget:
    python src/retarget/sg_galbot.py /home/pengyang/data/motion/human/SG/output.bvh

    python src/retarget/seg_galbot.py /home/pengyang/data/motion/human/SeG_dataset/bvh/ARMS_FOLD-1.bvh
after this, you should generate a file as /home/pengyang/data/motion/galbot/SG/ARMS_FOLD-1.pickle


visualize kinematic results:
    python src/utils/vis/kinematic_vis.py /home/pengyang/data/motion/galbot/SG/output.pickle

visualize dynamic results via pybullet and position pd control:
    python src/utils/vis/pybullet_visualize_galbot_dynamic.py   /home/pengyang/data/motion/galbot/SG/output.pickle


### for g1:
    python src/retarget/sg_g1.py /home/pengyang/data/motion/human/SG/output.bvh


    python src/utils/vis/pybullet_visualize_g1_dynamic.py /home/pengyang/data/motion/g1/SG/output.pickle



### read here if you want to reproduce the speech+motion example:
1. contact pengyang for SG data, urdf file, replace paths like /home/pengyang/data to your data directory
2. run python src/retarget/sg_galbot.py {path_to_bvh_file} to generate a pickle file containing robot motion
3. run python src/utils/vis/pybullet_visualize_galbot_dynamic.py {path_to_pickle_file} to visualize robot motion in pybullet
4. find the corresponding audio file, and make your own video

tips: you may need to change some paths for bvh and pickle files, it may take sometime to debug




