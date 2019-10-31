# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from funcs.utils import *
from funcs.dataset import *


def carve(voxels, Ps, masks_planer, h, w, is_carving=True, progress=True, is_zf=False):
    for idx in tqdm(range(len(Ps))):
        P, mask = Ps[idx], masks_planer[idx]
        x, y = projection(P, voxels.X, voxels.Y, voxels.Z)

        # keep points in the image range
        if is_zf:
            keep_idx = np.where((x >= 1) & (x <= w) & (y >= 388) & (y <= 637))[0] 
            # the original image is cut-off from the real size
            y = y - 387
        else:
            keep_idx = np.where((x >= 1) & (x <= w) & (y >= 1) & (y <= h))[0]

        x = x[keep_idx]
        y = y[keep_idx]

        # keep the points in the mask
        ind = sub2ind([h,w], (y-1).astype(np.int64), (x-1).astype(np.int64)) # index from [0,0] to [h-1, w-1]
        keep_idx = keep_idx[mask[ind] >= 1]
        
        if is_carving:
            voxels._carve_voxels(keep_idx)
        else:
            voxels._update_iso(keep_idx)

        # viz_surface(voxels)
        # fig, ax = plt.subplots()
        # for M in masks:
        #     ax.cla()
        #     ax.imshow(M)
        #     ax.plot(x, y, color='r', alpha=0.6)   
        #     plt.show()


def sub2ind(array_shape, rows, cols):
    ind = rows * array_shape[1] + cols
    return ind


def ind2sub(array_shape, ind):
    rows = ind.astype('int') / array_shape[1]
    cols = ind % array_shape[1]
    return (rows, cols)


def projection(P, X, Y, Z):
    z = P[2,0] * X + P[2,1] * Y + P[2,2] * Z + P[2,3]
    y = np.round((P[1,0] * X + P[1,1] * Y + P[1,2] * Z + P[1,3]) / z)
    x = np.round((P[0,0] * X + P[0,1] * Y + P[0,2] * Z + P[0,3]) / z)
    return x, y