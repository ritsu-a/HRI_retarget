# 2025.04.23 HIt-xiaowangzi
# (1) 加载我们自己的pickle文件，看一看有没有碰撞的问题
# (2) 将碰撞结果导出log日志，便于后面排查


import argparse
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import rerun as rr
import trimesh
import pandas as pd
from pinocchio import FrameType
from datetime import datetime

import os
import sys
from HRI_retarget import ROOT,SRC_ROOT,DATA_ROOT
sys.path.append(SRC_ROOT)

import trimesh.visual
from HRI_retarget import ROOT,SRC_ROOT,DATA_ROOT
from copy import deepcopy
from utils.io.motion_pkl_to_csv import pkl_to_csv



urdf_path = os.path.join(DATA_ROOT, 'resources/robots/g1/g1_29dof_rev_1_0.urdf')
mesh_dir = os.path.join(DATA_ROOT, 'resources/robots/g1/meshes')
srdf_path = os.path.join(DATA_ROOT, 'resources/robots/g1/g1_29dof_rev_1_0.srdf')
Tpose = np.array([0,0,0.785,0,0,0,1,
                -0.15,0,0,0.3,-0.15,0,
                -0.15,0,0,0.3,-0.15,0,
                0,0,0,
                0, 1.57,0,1.57,0,0,0,
                0,-1.57,0,1.57,0,0,0]).astype(np.float32)


class RerunURDF():
    def __init__(self, robot_type, urdf_path, mesh_dir,srdf_path, Tpose):
        self.name = robot_type
        self.robot = RobotWrapper.BuildFromURDF(urdf_path, mesh_dir, pin.JointModelFreeFlyer())

        # self.robot = RobotWrapper.BuildFromURDF(urdf_path, mesh_dir)
        self.urdf_path = urdf_path
        self.mesh_dir = mesh_dir
        self.Tpose = Tpose
        self.srdf_path = srdf_path
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_path = os.path.join(SRC_ROOT, "utils/vis/log", timestamp+".txt")
        
        self.last_collision_set= set()
        self.curr_collision_set = set()
        
        
        self.link2mesh = self.get_link2mesh()
        self.load_visual_mesh()
        self.init_collision_checking()
        q = pin.neutral(self.robot.model)
        self.update(q)
    
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
            
    def init_collision_checking(self):
        
        # Load model
        self.model = pin.buildModelFromUrdf(self.urdf_path, pin.JointModelFreeFlyer())
        
        # Load collision geometries
        self.geom_model = pin.buildGeomFromUrdf(
            self.model, self.urdf_path, pin.GeometryType.COLLISION, self.mesh_dir
        )
        
        # Add collisition pairs
        self.geom_model.addAllCollisionPairs()
        print("num collision pairs - initial:", len(self.geom_model.collisionPairs))
        
        
        pin.removeCollisionPairs(self.model, self.geom_model, self.srdf_path)
        print(
            "num collision pairs - after removing useless collision pairs:",
            len(self.geom_model.collisionPairs),
        )
        
        # Load reference configuration
        pin.loadReferenceConfigurations(self.model, self.srdf_path)
        print(f"Total joints in Pinocchio model: {self.model.njoints}")
        for i, joint in enumerate(self.model.joints):
            print(f"Joint {i}: {self.model.names[i]}")
            
        # Create data structures
        self.data = self.model.createData()
        self.geom_data = pin.GeometryData(self.geom_model)
        
        # Create a table to visualize collision pairs
        collision_link_names = set()
        for geom_obj in self.geom_model.geometryObjects:
            frame_id = geom_obj.parentFrame
            frame = self.model.frames[frame_id]
            if frame.type == pin.FrameType.BODY:
                collision_link_names.add(frame.name)

        collision_link_names = list(collision_link_names)
                     
        link_num = len(collision_link_names) 
            
        acm = pd.DataFrame(np.zeros((link_num,link_num)),index = collision_link_names, columns= collision_link_names) 
        for k in range(len(self.geom_model.collisionPairs)):
            cp = self.geom_model.collisionPairs[k]

            # 原始 mesh 名称
            geo1 = self.geom_model.geometryObjects[cp.first]
            geo2 = self.geom_model.geometryObjects[cp.second]

            # 真正的 link 名称（无 _0/_1 后缀）
            link1 = self.model.frames[geo1.parentFrame].name
            link2 = self.model.frames[geo2.parentFrame].name 
            acm[link1][link2] = 1
            acm[link2][link1] = 1
            
        acm.to_excel(os.path.join(SRC_ROOT,"utils","vis","acm.xlsx"))
        
    def update(self, configuration = None, frame_nr = None):
        
        collision_set = self.check_collsion(configuration)
        removed,added,flag = self.update_mesh(collision_set)
        self.robot.framesForwardKinematics(self.Tpose if configuration is None else configuration)
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parentJoint
            parent_joint_name = self.robot.model.names[parent_joint_id]
            frame_tf = self.robot.data.oMf[frame_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]
            rr.log(f'urdf_{self.name}/{parent_joint_name}',
                   rr.Transform3D(translation=joint_tf.translation,
                                  mat3x3=joint_tf.rotation,
                                  axis_length=0.01))
            
            if(flag): # 如果碰撞状态发生变化，更新网格
                if(frame_name in removed):
                    mesh = self.link2mesh[frame_name]
                    relative_tf = joint_tf.inverse() * frame_tf
                    mesh.apply_transform(relative_tf.homogeneous)
                    rr.log(f'urdf_{self.name}/{parent_joint_name}/{frame_name}',
                        rr.Mesh3D(
                            vertex_positions=mesh.vertices,
                            triangle_indices=mesh.faces,
                            vertex_normals=mesh.vertex_normals,
                            vertex_colors=mesh.visual.vertex_colors,
                            albedo_factor=mesh.visual.vertex_colors,
                            vertex_texcoords=None,
                        ),
                        )
                if (frame_name in added):
                    mesh = self.link2mesh[frame_name]
                    relative_tf = joint_tf.inverse() * frame_tf
                    mesh.apply_transform(relative_tf.homogeneous)
                    rr.log(f'urdf_{self.name}/{parent_joint_name}/{frame_name}',
                        rr.Mesh3D(
                            vertex_positions=mesh.vertices,
                            triangle_indices=mesh.faces,
                            vertex_normals=mesh.vertex_normals,
                            vertex_colors=mesh.visual.vertex_colors,
                            albedo_factor=[1.0,0.0,0.0,1.0],
                            vertex_texcoords=None,
                        ),
                        )
                    
                
        if len(collision_set) > 0:
            self.generate_log(self.log_path, frame_nr, collision_set)
            
    def check_collsion(self,configuration):
        collision_set = set()
        pin.computeCollisions(self.model,self.data,self.geom_model,
                              self.geom_data,configuration,False)
        # print("2")
        for k in range(len(self.geom_model.collisionPairs)):
            cr = self.geom_data.collisionResults[k]
            cp = self.geom_model.collisionPairs[k]

            # 原始 mesh 名称
            geo1 = self.geom_model.geometryObjects[cp.first]
            geo2 = self.geom_model.geometryObjects[cp.second]

            # 真正的 link 名称（无 _0/_1 后缀）
            link1 = self.model.frames[geo1.parentFrame].name
            link2 = self.model.frames[geo2.parentFrame].name
            
            if(cr.isCollision()):
                collision_set.add(link1)
                collision_set.add(link2)
        
        return collision_set
    
    def update_mesh(self,collison_set):
        if collison_set == self.last_collision_set:
            return set(),set(),False
        removed_collision_links = self.last_collision_set - collison_set
        added_collision_links = collison_set - self.last_collision_set
        self.last_collision_set = collison_set
        return removed_collision_links,added_collision_links,True
    
    def generate_log(self,log_path, frame_nr, collision_set):
        with open(log_path, 'a') as log_file:
            log_file.write(f"Frame {frame_nr} collision: ", )
            for item in collision_set:
                log_file.write(f" {item}")
            log_file.write("\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, help="File name", default='dance1_subject2')
    parser.add_argument('--robot_type', type=str, help="Robot type", default='g1')
    parser.add_argument('--dataset', type=str, help="dataset", default='LAFAN1')

    args = parser.parse_args()

    rr.init('Reviz', spawn=True)
    rr.log('', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    file_name = args.file_name
    robot_type = args.robot_type
    dataset = args.dataset
    
    # input the pickle data
    data_path = os.path.join(DATA_ROOT,"motion/g1/SG/output.pickle")
    curr_dir_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(curr_dir_path,"log","output.csv")
    data = pkl_to_csv(data_path,output_path)


    rerun_urdf = RerunURDF('g1',urdf_path,mesh_dir,srdf_path,Tpose)
    print("Total Frames: ", data.shape[0])
    for frame_nr in range(data.shape[0]):
        rr.set_time_sequence('frame_nr', frame_nr)
        configuration = data[frame_nr, :]
        rerun_urdf.update(configuration,frame_nr)