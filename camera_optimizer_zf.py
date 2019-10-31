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
from scipy.optimize import minimize, fmin
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
    dataloader = make_dataloader(data_name, data_dir, param_dir=None)

    # load masks and projection matrix
    masks_planer = dataloader.masks_planer
    masks = dataloader.masks
    images = dataloader.images
    N = len(images)
    
    # start optimization
    print("==> Start optimization ...")
    initial_voxels = initialize_voxels(data_name, voxel_num=100000)
    X0 = initialize_variable_zf(N)

    if opts.optim_mode == "voxel_residual":
        optim_func = voxels_residual_zf
        params = (masks_planer, N, initial_voxels.X, initial_voxels.Y, initial_voxels.Z)
        options={'fatol': 1e-6, 'disp': True, 'adaptive': True}

    elif opts.optim_mode == "prob":
        print("==> Estimate 2D probability model.")
        mu_obj, mu_bck, sigma_obj, sigma_bck = get_prob_model(images, masks, N)
        print("==> Estimate 2D confidence maps.")
        prob_maps_obj, prob_maps_bck, _, _ = get_prob_map(images, mu_obj, mu_bck, sigma_obj, sigma_bck)
        optim_func = max_prob_zf
        params = (prob_maps_obj, prob_maps_bck, N, initial_voxels.X, initial_voxels.Y, initial_voxels.Z)
        options={'fatol': 1e-6, 'xtol':0.01, 'disp': True, 'adaptive': True}
    else:
        print("==> Not support this loss function.")
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
        cma_options['tolfun'] = 0.1
        cma_options['tolx'] = 0.001
        t1 = np.array([-0.03, -0.04, -0.01])
        t2 = np.array([0.03, 0.04, 0.01])
        t3 = X0[3:] - 0.018
        t4 = X0[3:] + 0.018
        cma_options['bounds'] = [np.concatenate((t1,t3)), np.concatenate((t2,t4))]
        cma_options['verb_disp'] = 1
        es = cma.CMAEvolutionStrategy(X0, 0.01, cma_options)
        es.optimize(optim_func, args=params)
        X = es.result.xbest
    
    else:
        print("==> Not support this optimizer !!!")
        return 0 

    elapse = time.time() - t0
    print("==> Elapse: %d seconds." % elapse)

    # save optimized parameters
    save_result_name = os.path.join(data_dir, data_name+"_result_"+opts.optim_mode+"_"+opts.optimizer+".txt")
    with open(save_result_name, "a+") as fid:
        fid.write("runtime: %s.\n" % (str(elapse)))
        for v in X:
            fid.write(str(v) + " ")

    # save projection matrix
    Ps_new = construct_Ps_zf(X, N)
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
    h, w = masks[0].shape
    recon_voxels = initialize_voxels(data_name, voxel_num=5000000)
    carve(recon_voxels, Ps_new, masks_planer, h, w,  is_carving=is_carving, progress=True, is_zf=True)

    # visualize reconstruction surface
    if is_carving:
        iso_value = 0.5
        print("==> Remaining voxels: %d." % (recon_voxels.X.size))
    else:
        thrd = 0.9
        iso_value = round(N * thrd) + 0.5
        print("==> Remaining voxels: %d." % (np.sum(recon_voxels.V >= iso_value)))
    
    viz_surface(data_name, recon_voxels, iso_value, images, h, w, Ps_new, is_carving=is_carving, is_zf=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="zf_5dpf_s009") # "zf_3dpf_s001" "zf_4dpf_s008" "zf_5dpf_s009"
    parser.add_argument("--optimizer", default="CMA")         # "NM" or "CMA"
    parser.add_argument("--optim-mode", default="voxel_residual") # "prob" or "voxel_residual"
    opts = parser.parse_args()
    main(opts)