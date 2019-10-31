# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import numpy as np
import cv2
import math
import mayavi.mlab as mlab

from funcs.dataset import make_dataloader, initialize_voxels
from funcs.projection_module import carve
from funcs.viz import viz_surface
from funcs.utils import compute_shape


def main(args):
    # create data loader
    data_name = args.dataset
    print("==> Load data: %s." % (data_name))
    root_dir = "./dataset"
    data_dir = os.path.join(root_dir, data_name)
    if args.type == "groundtruth":
        param_dir = os.path.join(data_dir, data_dir.split('/')[-1]+"_par.txt")
    elif args.type == "recon":
        param_dir = os.path.join(data_dir, data_name+"_par_"+args.optim_mode+"_"+args.optimizer+".txt")
    else:
        print("Please check your argments.")
        return 0
    
    dataloader = make_dataloader(data_name, data_dir, param_dir=param_dir)

    # load masks and projection matrix
    masks_planer = dataloader.masks_planer
    images = dataloader.images
    Ps = dataloader.Ps
    
    # initialize voxels
    recon_voxels = initialize_voxels(data_name, voxel_num=5000000)

    # 3D reconstruction: space carving or visual hull
    is_carving = False
    h, w = dataloader.masks[0].shape
    is_zf = True if data_name.startswith("zf") else False
    print("==> Start reconstruction.")
    carve(recon_voxels, Ps, masks_planer, h, w, is_carving=is_carving, is_zf=is_zf)

    # visualize reconstruction surface
    if is_carving:
        iso_value = 0.5
        print("==> Remaining voxels: %d." % (recon_voxels.X.size))
    else:
        thrd = 0.9
        iso_value = len(dataloader.images) * thrd + 0.5
        print("==> Remaining voxels: %d." % (np.sum(recon_voxels.V >= iso_value)))
    
    vertex, facet = viz_surface(data_name, recon_voxels, iso_value, images, h, w, Ps, is_carving=is_carving, is_zf=is_zf)
    
    if is_zf:
        volume, surface_area = compute_shape(recon_voxels, vertex, facet, len(masks_planer))
        print("==> Volume: %.3f um^3, Surface Area: %.2f um^2." % (volume/(1e+6), surface_area/(1e+4)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="zf_3dpf_s001") # "zf_3dpf_s001" "zf_4dpf_s008" "zf_5dpf_s009" "dinosaur"
    parser.add_argument("--type", default="recon") # "groundtruth" or "recon"
    parser.add_argument("--optimizer", default="CMA")     # "NM" or "CMA"
    parser.add_argument("--optim-mode", default="prob")  # "voxel_residual" or "prob"
    args = parser.parse_args()
    main(args)

    