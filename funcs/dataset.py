# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import argparse
import cv2
import math
import collections
from matplotlib import pyplot as plt

from funcs.utils import make_proj_matrix


class multiview_data_SR(object):
    def __init__(self, dataset, param_dir):
        image_list = glob.glob(os.path.join(dataset, "images/*.png"))
        image_list.sort()
        mask_list = glob.glob(os.path.join(dataset, "silhouettes/*.png"))
        mask_list.sort()
        self.image_list = image_list
        self.mask_list = mask_list
        self.calib_data = param_dir

    def read_projs(self):
        with open(self.calib_data, "rb") as fid:
            lines = fid.readlines()
        Ps = []
        for ll in lines[1:]:
            words = ll.strip().split()
            k = np.array(words[1:10], np.float).reshape(3,3)
            r = np.array(words[10:19], np.float).reshape(3,3)
            t = np.array(words[19:], np.float).reshape(3,1)
            Ps.append(make_proj_matrix(k, r, t))
        return Ps

    def load_data(self):
        if self.calib_data is not None:
            self.Ps = self.read_projs()
        self.images = read_img(self.image_list)
        self.masks = read_img(self.mask_list, is_mask=True)
        self.masks_planer = []
        for mask in self.masks:
            self.masks_planer.append(mask.reshape(-1))


class multiview_data_Dinosaur(multiview_data_SR):
    def __init__(self, dataset, param_dir):
        super(multiview_data_Dinosaur, self).__init__(dataset, param_dir)
        image_list = glob.glob(os.path.join(dataset, "images/*.ppm"))
        image_list.sort()
        mask_list = glob.glob(os.path.join(dataset, "silhouettes/*.pgm"))
        mask_list.sort()
        self.image_list = image_list
        self.mask_list = mask_list
        self.calib_data = param_dir

    def read_projs(self):
        with open(self.calib_data, "rb") as fid:
            lines = fid.readlines()
        P = []
        for ll in lines:
            words = ll.strip().split()
            p = np.array(words[1:], np.float).reshape(3,4)
            P.append(p)
        return P


class multiview_data_zf(multiview_data_Dinosaur):
    def __init__(self, dataset, param_dir=None):
        super(multiview_data_zf, self).__init__(dataset, param_dir)
        image_list = glob.glob(os.path.join(dataset, "images/*.tif"))
        image_list.sort()
        mask_list = glob.glob(os.path.join(dataset, "silhouettes/*.tif"))
        mask_list.sort()
        self.image_list = image_list
        self.mask_list = mask_list
        self.calib_data = param_dir
    

def read_img(img_list, is_mask=False):
    out = []
    for img in img_list:
        if is_mask:
            image = cv2.imread(img, 0)
            out.append(image[...].copy())
        else:
            image = cv2.imread(img)
            out.append(image[...,::-1].copy())
    return out


'''
Make a voxel class
'''
class voxels(object):
    def __init__(self, XLim, YLim, ZLim, N):
        self.XLim = XLim
        self.YLim = YLim
        self.ZLim = ZLim
        self.N = N
        self.make_voxels()
    
    def make_voxels(self):
        volume = (self.XLim[1] - self.XLim[0]) * (self.YLim[1] - self.YLim[0]) * (self.ZLim[1] - self.ZLim[0])
        self.Resolution = (volume / float(self.N)) ** (1./3.)
        x = np.arange(self.XLim[0], self.XLim[1], self.Resolution)
        y = np.arange(self.YLim[0], self.YLim[1], self.Resolution)
        z = np.arange(self.ZLim[0], self.ZLim[1], self.Resolution)
        X, Y, Z = np.meshgrid(x, y, z)
        self.V_Shape = X.shape
        self.X = X.reshape(-1)
        self.Y = Y.reshape(-1)
        self.Z = Z.reshape(-1)
        self.V = np.zeros(len(self.Z)) # initialize voxel values as 0s for space carving
    
    def _carve_voxels(self, keep_idx):
        self.X = self.X[keep_idx]
        self.Y = self.Y[keep_idx]
        self.Z = self.Z[keep_idx]
        self.V = self.V[keep_idx]
        self.N = keep_idx.size

    def _set_voxels(self, points):
        self.X = points[0,:]
        self.Y = points[1,:]
        self.Z = points[2,:]
        self.V = np.ones(self.X.size) # set voxel values as 1s for voxel accumulation
        self.N = points.shape[1]
    
    def _update_iso(self, idx):
        self.V[idx] += 1


def make_dataloader(data_name, data_dir, param_dir=None):
    if data_name == "dinoSparseRing" or data_name == "templeSparseRing":
        dataloader = multiview_data_SR(data_dir, param_dir)
    elif data_name == "dinosaur":
        dataloader = multiview_data_Dinosaur(data_dir, param_dir)
    elif data_name.startswith("zf"):
        dataloader = multiview_data_zf(data_dir, param_dir=param_dir)
    else:
        print('==> No support this data.')
        return 0

    dataloader.load_data()

    return dataloader
    

def initialize_voxels(data_name, voxel_num=100000):
    if data_name == "dinoSparseRing": # in meters
        xlim = [-0.065, 0.015]
        ylim = [-0.020, 0.07]
        zlim = [-0.06, 0.020]
    elif data_name == "templeSparseRing":
        xlim = [-0.080, 0.03]
        ylim = [0.025, 0.20]
        zlim = [-0.015, 0.065]
    elif data_name == "dinosaur":
        xlim = [-0.1307, 0.1156]
        ylim = [-0.1307, 0.1156]
        zlim = [-0.7484, -0.5021]
    elif data_name.startswith("zf"): # in micrometers
        xlim = [-2900, 2900]
        ylim = [-800, 800]
        zlim =  [-800, 800]
    initial_voxels = voxels(xlim, ylim, zlim, voxel_num)
    return initial_voxels


if __name__ == "__main__":

    # data_dir = "/Users/didi/Downloads/multiview datasets"
    # dataset = os.path.join(data_dir, "dinoSparseRing")
    # calib_data = os.path.join(dataset, "dinoSR_par.txt")

    # data = multivew_data(dataset, calib_data)
    # Ps = data.Ps
    # Ks = data.Ks
    # Rs = data.Rs
    # Ts = data.Ts
    # for indx, key in enumerate(Ts.keys()):
    #     if indx == 0:
    #         tt = Ts[key]
    #     else:
    #         tt = np.hstack((tt, Ts[key]))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(tt[0,:], tt[1,:], tt[2,:], c='r', marker='o')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.axis('equal')
    # tt = tt.T
    # for indx, key in enumerate(Ts.keys()):
    #     r = tt[indx]
    #     ax.text(r[0], r[1], r[2], key, color="blue")
    # plt.show()


    # data_dir =  "/Users/yuanhaoguo/Downloads/dataset/multiview_datasets/head_data"
    # data_obj = multivew_data_c(data_dir)

    # Ps = data_obj.Ps_dict

    # for indx, key in enumerate(Ps.keys()):
    #     P = Ps[key]
    #     K, R, T = decompose_proj(P)
    #     P_recon = make_proj_matrix(K, R, T)
    #     if indx == 0:
    #         Ts = T
    #     else:
    #         Ts = np.hstack((Ts, T))
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(Ts[0,:], Ts[1,:], Ts[2,:], c='r', marker='o')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.axis('equal')
    # Ts = Ts.T
    # for indx, key in enumerate(Ps.keys()):
    #     cords = Ts[indx]
    #     ax.text(cords[0], cords[1], cords[2], key, color="blue")

    # plt.show()


    a = voxels([1,10], [1,10], [1,5], 100000)
    b = voxels([1,10], [1,10], [1,5], 200000)

    print(a.V.size)
    print(b.V.size)

    print(a.Resolution)
    print(b.Resolution)
