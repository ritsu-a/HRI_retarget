# What's in this repo

1. retarget for galbot from seg_dataset

### TODO LIST:
- change sample-based collision loss into analytical loss 
- change clip_angles in retarget files to normalize (follow model/g1_29.py and retarget/smpl_g1_29.py)
- change pytorch_kinematics.chain.Chain into utils.motionlib.strechable_chain.Strechable_Chain(follow model/g1_29.py)


# preparation
write current folder path into bashrc, then
    git clone git@github.com:ritsu-a/HRI_retarget.git
    cd HRI_retarget
    git checkout g1


the environment is tested with cuda12.1 with python=3.10
    pip install -r requirements.txt

contact pengyang for data
then create a soft link to HRI_retarget/data


    


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


### use gradio for control robot with mobile phone
1. sudo ufw allow 7990
4. python src/utils/vis/gradio_test.py


### real deployment pipeline:
1. retarget to generate the raw motion 
    src/retarget/sg_g1.py 
2. apply clipping and filter of the raw motion
    src/deploy/filter_motion.py #TODO change IO
3. visualize motion either via pybullet(dynamic) or rerun (kinamatic)
    src/utils/vis/rerun_kinematic.py 
    src/utils/vis/pybullet_visualize_g1_dynamic.py  #TODO change IO
4. deploy on real G1 
    src/deploy/deploy_g1.py


### 0419 retarget humanml3d 
    src/retarget/




