# -*- coding: utf-8 -*-

import numpy as np
from mayavi import mlab 
from matplotlib import pyplot as plt
import trimesh
import scipy
from skimage import measure
import plotly.graph_objects as go

from funcs.utils import *
from funcs.projection_module import projection


def viz_surface(data_name, voxels, iso_v, images, h, w, Ps, is_carving=True, is_zf=False):
    print("==> Refill 3D space.")
    if is_carving:
        ux = np.unique(voxels.X)
        uy = np.unique(voxels.Y)
        uz = np.unique(voxels.Z)

        ux = np.insert(np.append(ux, ux[-1]+R), 0, ux[0]-R)
        uy = np.insert(np.append(uy, uy[-1]+R), 0, uy[0]-R)
        uz = np.insert(np.append(uz, uz[-1]+R), 0, uz[0]-R)

        X_arr, Y_arr, Z_arr = np.meshgrid(ux, uy, uz)
        V_arr = np.zeros(X_arr.shape)
        N = voxels.X.size

        for ii in range(N):
            ix = np.where(ux == voxels.X[ii])
            iy = np.where(uy == voxels.Y[ii])
            iz = np.where(uz == voxels.Z[ii])

            # note the coordinate: y,x,z. 
            V_arr[iy[0], ix[0], iz[0]] = voxels.V[ii]
    else:
        X_arr = voxels.X.reshape(voxels.V_Shape)
        Y_arr = voxels.Y.reshape(voxels.V_Shape)
        Z_arr = voxels.Z.reshape(voxels.V_Shape)
        V_arr = voxels.V.reshape(voxels.V_Shape)

    print("==> Visualize 3D smoothed surface.")
    # TODO: the triangulated mesh from "skimage" makes the visualization 
    # artificial as the camera is very close to the object.
    # a solution will be found to solve the visualization problem.
    # at this moment, a triangulated mesh is obtained using mayavi.
    # vertices_show, faces_show, _, _ = measure.marching_cubes_lewiner(V_arr, level=iso_v, spacing=(1,1,1))
    my_obj = mlab.contour3d(V_arr, contours=[iso_v])
    my_actor = my_obj.actor.actors[0]
    poly_data_object = my_actor.mapper.input
    vertices = np.array(poly_data_object.points)
    the_cells = np.reshape(poly_data_object.polys.data.to_array(),[-1,4])
    faces = the_cells[:,1:]
    mesh = smooth_surface(data_name, vertices, faces)
    show_mesh(mesh)
    show_color_mesh(mesh, images, X_arr, Y_arr, Z_arr, h, w, Ps, is_zf=is_zf)
    
    if is_zf:
        print("==> Compute 3D metrics for the zebrafish")
        R = voxels.Resolution
        vertices, faces, _, _ = measure.marching_cubes_lewiner(V_arr, level=iso_v, spacing=(R,R,R))
        mesh = smooth_surface(data_name, vertices, faces)

    return mesh.vertices, mesh.faces


def smooth_surface(data_name, vertices, facets):
    mesh = trimesh.Trimesh(vertices=vertices, faces=facets, vertex_colors=[169,169,169])
    la_matrix = trimesh.smoothing.laplacian_calculation(mesh, equal_weight=True)
    trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=5, implicit_time_integration=True, laplacian_operator=la_matrix)
    return mesh


def show_mesh(mesh):
    scene = trimesh.Scene(mesh)
    scene.show(smooth=True)
    return mesh


def show_color_mesh(mesh, images, X, Y, Z, h, w, Ps, is_zf=False):
    camera_directions = get_camera_direction(h, w, Ps)
    vertices = mesh.vertices
    num_vertices = vertices.shape[0]
    vertex_colors = np.ones((num_vertices, 4), np.uint8) * 255
    normals = mesh.vertex_normals
    for ii in range(num_vertices):
        angles = np.matmul(normals[ii,:], camera_directions) / np.linalg.norm(normals[ii,:])
        cam_idx = np.argmin(angles)
        #TODO: find a way to locate the accurate xyz location for the vertices
        x, y, z = int(vertices[ii,0]), int(vertices[ii,1]), int(vertices[ii,2])
        imx, imy = projection(Ps[cam_idx], X[x,y,z], Y[x,y,z], Z[x,y,z])
        if is_zf:
            vertex_colors[ii,:3] = images[cam_idx][np.int64(imy-1)-387, np.int64(imx-1), :]
        else:
            vertex_colors[ii,:3] = images[cam_idx][np.int64(imy-1), np.int64(imx-1), :]

    mesh.visual.vertex_colors = vertex_colors
    mesh.vertices -= np.mean(mesh.vertices)
    scene = trimesh.Scene(mesh)
    scene.show(smooth=False)
