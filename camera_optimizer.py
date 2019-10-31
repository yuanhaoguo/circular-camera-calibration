# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import numpy as np
import cv2
import math
import time
import random
from scipy.optimize import minimize
import mayavi.mlab as mlab
import cma

from funcs.dataset import make_dataloader, initialize_voxels
from funcs.projection_module import *
from funcs.viz import viz_surface
from funcs.loss_func import *


def main(opts):

    # create data loader
    data_name = opts.dataset
    print("==> Load data: %s." % (data_name))
    root_dir = "./dataset"
    data_dir = os.path.join(root_dir, data_name)
    param_dir = os.path.join(data_dir, data_dir.split('/')[-1]+"_par.txt")
    dataloader = make_dataloader(data_name, data_dir, param_dir=param_dir)

    # load masks and projection matrix
    masks = dataloader.masks
    masks_planer = dataloader.masks_planer
    images = dataloader.images
    N = len(images)
    h, w = masks[0].shape
    Ps = dataloader.Ps
    
    # decompse camera parameters
    print("==> Decompose camera projection matrix.")
    select_num = 10
    noise_level = 10
    args, X0, X_gt = decompose_P_add_noise(Ps, select_num, noise_level)
    
    # start optimization
    print("==> Start optimization ...")
    initial_voxels = initialize_voxels(data_name, voxel_num=500000)

    if opts.optim_mode == "voxel_residual":
        optim_func = voxels_residual
        params = (masks_planer, h, w, N, initial_voxels.X, initial_voxels.Y, initial_voxels.Z, args)
        options = {'xtol': 0.01, 'fatol': 1e-6, 'disp': True}

    elif opts.optim_mode == "prob":
        print("==> Estimate 2D probability model.")
        mu_obj, mu_bck, sigma_obj, sigma_bck = get_prob_model(images, masks, N)
        prob_maps_obj, prob_maps_bck,_,_ = get_prob_map(images, mu_obj, mu_bck, sigma_obj, sigma_bck)
        optim_func = max_prob
        params = (prob_maps_obj, prob_maps_bck, h, w, N, initial_voxels.X, initial_voxels.Y, initial_voxels.Z, args)
        options = {'xtol': 0.01, 'fatol': 1e-6, 'disp': True}

    else:
        print("==> Not support this optim mode!")
        return 0

    t0 = time.time()
    if opts.optimizer == "NM":
        # Nelder-Mead algorithm
        res = minimize(optim_func, 
                       X0, 
                       args=params,
                       method='nelder-mead',
                       options=options) 
        X = res.x # parse optimal results

    elif opts.optimizer == "CMA":
        # Evolution Stratergy
        cma_options = cma.CMAOptions()
        cma_options['tolfun'] = 1e-6
        cma_options['tolx'] = 0.001
        cma_options['bounds'] = [np.array(X_gt)-5, np.array(X_gt)+10]
        cma_options['verb_disp'] = 1
        es = cma.CMAEvolutionStrategy(X0, 1, cma_options)
        es.optimize(optim_func, args=params)
        X = es.result.xbest

    elapse = time.time() - t0
    print("==> Elapse: %d seconds." % elapse)

    print("Groundtruth: ")
    for idx in range(select_num):
        print(X_gt[idx])
    print("---------------------")

    print("Noisy values: ")
    for idx in range(select_num):
        print(X0[idx])
    print("---------------------")
    
    print("Optimized values: ")
    for idx in range(select_num):
        print(X[idx])
    print("---------------------")

    print("Error: ")
    for idx in range(select_num):
        print(abs(X_gt[idx]-X[idx]))
    print("---------------------")

    # construct projection matrix
    Ps_new = []
    for idx in range(N):
        K = args['Ks'][idx]
        if idx in range(select_num):
            R_Y = X[idx]
        else:
            R_Y = args['RYs'][idx]
        R_X = args['RXs'][idx]
        R_Z = args['RZs'][idx]
        t = args['ts'][idx]
        R = make_rotation_matrix(R_X, R_Y, R_Z, mode='degree')
        P = make_proj_matrix(K, R, t)
        Ps_new.append(P)

    # save optimized parameters
    save_result_name = os.path.join(data_dir, data_name+"_result_"+opts.optim_mode+"_"+opts.optimizer+".txt")
    with open(save_result_name, "a+") as fid:
        fid.write("runtime: %s.\n" % (str(elapse)))
        for v in X:
            fid.write(str(v) + " ")
        fid.write("\n")
        for v in X0:
            fid.write(str(v) + " ")

    # save projection matrix
    save_Ps_name = os.path.join(data_dir, data_name+"_par_"+opts.optim_mode+"_"+opts.optimizer+".txt")
    with open(save_Ps_name, "a+") as fid:
        for idx, P in enumerate(Ps_new):
            fid.write(os.path.split(dataloader.image_list[idx])[1].split(".")[0] + " ")
            P_planer = P.reshape(-1)
            for p in P_planer:
                fid.write(str(p) + " ")
            fid.write("\n")

    # 3D reconstruction
    print("==> Start reconstruction.")
    is_carving = False
    recon_voxels = initialize_voxels(data_name, voxel_num=5000000)
    carve(recon_voxels, Ps_new, masks_planer, h, w, is_carving=is_carving)

    # visualize reconstruction surface
    if is_carving:
        iso_value = 0.5
        print("==> Remaining voxels: %d." % (recon_voxels.X.size))
    else:
        thrd = 0.9
        iso_value = round(N * thrd) + 0.5
        print("==> Remaining voxels: %d." % (np.sum(recon_voxels.V >= iso_value)))
    
    viz_surface(data_name, recon_voxels, iso_value, images, h, w, Ps_new, is_carving=is_carving, is_zf=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dinosaur")
    parser.add_argument("--optimizer", default="CMA")
    parser.add_argument("--optim_mode", default="prob") # "prob" or "voxel_residual"
    opts = parser.parse_args()
    main(opts)