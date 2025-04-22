# 2025.04.21 HIT-xiaowangzi
# (1) Visualize the pickle data in the remote visualizer provided by unitree
# (2) Detect the collison links by pinocchio library
# (3) Highlight the collision links in red

import argparse
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import rerun as rr
import trimesh
import os

import trimesh.visual
from HRI_retarget import ROOT,SRC_ROOT,DATA_ROOT
from copy import deepcopy

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
        self.robot = RobotWrapper.BuildFromURDF(urdf_path, mesh_dir)
        # self.model = pin.buildModelFromUrdf(urdf_path)

        # self.robot = RobotWrapper.BuildFromURDF(urdf_path, mesh_dir)
        self.Tpose = Tpose
        self.srdf_path = srdf_path
        
        self.last_collision_set= set()
        self.curr_collision_set = set()
        
        # print all joints names
        # for i in range(self.robot.model.njoints):
        #     print(self.robot.model.names[i])
        self.geom_model = pin.buildGeomFromUrdf(
            self.model, urdf_path, pin.GeometryType.COLLISION, mesh_dir
        )
        self.geom_data = pin.GeometryData(self.geom_model)
        
        self.link2mesh = self.get_link2mesh()
        self.load_visual_mesh()
        self.init_collision_checking()
        q = pin.neutral(self.model)

        self.update(q)
    
    def get_link2mesh(self):
        link2mesh = {}
        for visual in self.robot.visual_model.geometryObjects:
            mesh = trimesh.load_mesh(visual.meshPath)
            name = visual.name[:-2]
            mesh.visual = trimesh.visual.ColorVisuals()
            mesh.visual.vertex_colors = visual.meshColor
            # mesh.visual.vertex_colors = [1,0,0,1]
            # mesh.visual.face_colors = [1,0,0,1]
            link2mesh[name] = mesh
        # link2mesh["torso_link"].visual.vertex_colors = [1.0,0.0,0.0,1.0]
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
        pin.loadReferenceConfigurations(self.robot.model, self.srdf_path)
        self.collision_model.addAllCollisionPairs()
        pin.removeCollisionPairs(self.robot.model, self.collision_model, self.srdf_path)
        print(f"有效碰撞对数量: {len(self.collision_model.collisionPairs)}")
        
        
        # copy all the meshes to red
        self.link2mesh_red = deepcopy(self.link2mesh)
        for key in self.link2mesh_red.keys():
            self.link2mesh_red[key].visual.vertex_colors = [1.0,0.0,0.0,1.0]
        
    def update(self, configuration = None):
        collision_set = self.check_collsion(configuration)
        self.update_mesh(collision_set)
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
            
    def check_collsion(self,configuration):
        if configuration is not None:
            print("输入配置维度：", configuration.shape)
            print("robot.model.nq:", self.robot.model.nq)
        print(configuration)
        collision_set = set()
        self.data = self.robot.model.createData()
        pin.computeCollisions(self.robot.model,self.data,self.collision_model,
                              self.collision_data,configuration,False)
        print("2")
        for k in range(len(self.collision_model.collisionPairs)):
            cr = self.collision_data.collisionResults[k]
            cp = self.collision_model.collisionPairs[k]

            # 原始 mesh 名称
            geo1 = self.collision_model.geometryObjects[cp.first]
            geo2 = self.collision_model.geometryObjects[cp.second]

            # 真正的 link 名称（无 _0/_1 后缀）
            link1 = self.robot.model.frames[geo1.parentFrame].name
            link2 = self.robot.model.frames[geo2.parentFrame].name
            
            if(cr.isCollision()):
                print(
                    "collision pair:",
                    {link1},
                    ",",
                    {link2},
                    "- collision:",
                    "Yes"
                )
                collision_set.add(link1)
                collision_set.add(link2)
        
        return collision_set
    
    def update_mesh(self,collison_set):
        if collison_set == self.last_collision_set:
            pass
        removed_collision_links = self.last_collision_set - collison_set
        added_collision_links = collison_set - self.last_collision_set
        for link in removed_collision_links:
            self.robot.visual_model.geometryObjects.meshColor = \
                self.link2mesh[link].visual.vertex_colors
        for link in added_collision_links:
            self.robot.visual_model.geometryObjects.meshColor = \
                self.link2mesh_red[link].visual.vertex_colors



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
    # csv_files = "/home/pengyang/data/motion" + '/'  + robot_type + '/' + dataset + '/' + file_name + '.csv'
    # csv_files = "/home/pengyang/codebase/retarget/output.csv"
    csv_files = DATA_ROOT+ "/motion" + '/'  + robot_type + '/' + dataset + '/' + file_name + '.csv'
    data = np.genfromtxt(csv_files, delimiter=',')

    print("0")
    rerun_urdf = RerunURDF('g1',urdf_path,mesh_dir,srdf_path,Tpose)
    print("1")
    for frame_nr in range(data.shape[0]):
        rr.set_time_sequence('frame_nr', frame_nr)
        configuration = data[frame_nr, :]
        print("configuration.shape: ",configuration.shape)
        rerun_urdf.update(configuration)