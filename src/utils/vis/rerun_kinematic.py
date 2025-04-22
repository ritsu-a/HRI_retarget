### https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset
### usage:
#        python src/utils/vis/rerun_kinematic.py --file_name dance1_subject2
### todo:
#       change IO to normal standard
import argparse
import numpy as np
import pinocchio as pin
import rerun as rr
import trimesh
import joblib
import os
import sys
from HRI_retarget import ROOT,SRC_ROOT,DATA_ROOT
sys.path.append(SRC_ROOT)


from utils.io.motion_pkl_to_csv import load_motion_pkl_as_csv_data


class RerunURDF():
    def __init__(self, robot_type):
        self.name = robot_type
        match robot_type:
            case 'g1_29':
                self.robot = pin.RobotWrapper.BuildFromURDF(os.path.join(DATA_ROOT,'resources/robots/g1_asap/g1_29dof.urdf'), os.path.join(DATA_ROOT,'resources/robots/g1_asap'), pin.JointModelFreeFlyer())
                self.Tpose = np.array([0,0,0.785,0,0,0,1,
                                       -0.15,0,0,0.3,-0.15,0,
                                       -0.15,0,0,0.3,-0.15,0,
                                       0,0,0,
                                       0, 1.57,0,1.57,0,0,0,
                                       0,-1.57,0,1.57,0,0,0]).astype(np.float32)
                
            case 'g1_inspirehands':
                self.robot = pin.RobotWrapper.BuildFromURDF(os.path.join(DATA_ROOT,'resources/robots/g1_inspirehands/G1_inspire_hands.urdf'), os.path.join(DATA_ROOT,'resources/robots/g1_inspirehands'), pin.JointModelFreeFlyer())
                self.Tpose = np.array([0,0,0.785,0,0,0,1,
                                       -0.15,0,0,0.3,-0.15,0,
                                       -0.15,0,0,0.3,-0.15,0,
                                       0,0,0,
                                       0, 1.57,0,1.57,0,0,0,
                                       0,0,0,0,0,0,0,0,0,0,0,0,
                                       0,-1.57,0,1.57,0,0,0,
                                       0,0,0,0,0,0,0,0,0,0,0,0,]).astype(np.float32)
           
         
            case _:
                print(robot_type)
                raise ValueError('Invalid robot type')
        
        # print all joints names
        # for i in range(self.robot.model.njoints):
        #     print(self.robot.model.names[i])
        
        self.link2mesh = self.get_link2mesh()
        self.load_visual_mesh()
        self.update()
    
    def get_link2mesh(self):
        link2mesh = {}
        for visual in self.robot.visual_model.geometryObjects:
            mesh = trimesh.load_mesh(visual.meshPath)
            name = visual.name[:-2]
            mesh.visual = trimesh.visual.ColorVisuals()
            mesh.visual.vertex_colors = visual.meshColor
            link2mesh[name] = mesh
        return link2mesh
   
    def load_visual_mesh(self):       
        self.robot.framesForwardKinematics(pin.neutral(self.robot.model))
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            mesh = self.link2mesh[frame_name]
            
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parentJoint
            parent_joint_name = self.robot.model.names[parent_joint_id]
            frame_tf = self.robot.data.oMf[frame_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]
            rr.log(f'urdf_{self.name}/{parent_joint_name}',
                   rr.Transform3D(translation=joint_tf.translation,
                                  mat3x3=joint_tf.rotation,
                                  axis_length=0.01))
            
            relative_tf = joint_tf.inverse() * frame_tf
            mesh.apply_transform(relative_tf.homogeneous)
            rr.log(f'urdf_{self.name}/{parent_joint_name}/{frame_name}',
                   rr.Mesh3D(
                       vertex_positions=mesh.vertices,
                       triangle_indices=mesh.faces,
                       vertex_normals=mesh.vertex_normals,
                       vertex_colors=mesh.visual.vertex_colors,
                       albedo_texture=None,
                       vertex_texcoords=None,
                   ),
                   static=True)
    
    def update(self, configuration = None):
        self.robot.framesForwardKinematics(self.Tpose if configuration is None else configuration)
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parentJoint
            parent_joint_name = self.robot.model.names[parent_joint_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]
            rr.log(f'urdf_{self.name}/{parent_joint_name}',
                   rr.Transform3D(translation=joint_tf.translation,
                                  mat3x3=joint_tf.rotation,
                                  axis_length=0.01))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, help="File name", default='/home/pengyang/codebase/HRI_retarget/data/motion/g1/BEAT/1_wayne_0_1_1.pickle')

    args = parser.parse_args()

    rr.init('Reviz', spawn=True)
    rr.log('', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    file_name = args.file_name
   
    with open(file_name, "rb") as file:
        data = joblib.load(file)
        robot_type = data["robot_name"]
    csv_data = load_motion_pkl_as_csv_data(args.file_name)

    rerun_urdf = RerunURDF(robot_type)
    for frame_nr in range(csv_data.shape[0]):
        rr.set_time_sequence('frame_nr', frame_nr)
        configuration = csv_data[frame_nr, :]
        rerun_urdf.update(configuration)
