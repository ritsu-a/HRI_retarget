import os
from os.path import dirname
import numpy as np
import trimesh


def merge_mesh(meshes):
    """
    meshes: list of trimesh
    return: overall trimesh
    """

    vertices, faces = None, None
    N_v = 0
    for m in meshes:
        v = m.vertices
        f = m.faces
        if vertices is None:
            vertices = v.copy()
            faces = f.copy()
        else:
            vertices = np.concatenate((vertices, v), axis=0)
            faces = np.concatenate((faces, f + N_v), axis=0)
        N_v += v.shape[0]

    overall_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return overall_mesh


def save_mesh(trimesh_mesh, save_path):
    os.makedirs(dirname(save_path), exist_ok=True)
    mesh_txt = trimesh.exchange.obj.export_obj(trimesh_mesh, include_normals=False, include_color=False, include_texture=False, return_texture=False, write_texture=False, resolver=None, digits=8)
    with open(save_path, "w") as fp:
        fp.write(mesh_txt)
