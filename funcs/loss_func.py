# -*- coding: utf-8 -*-

import math
import copy
import numpy as np
from tqdm import tqdm
import random
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import numba

from funcs.utils import *
from funcs.dataset import *
from funcs.projection_module import *

eps = 2e-16

# @numba.jit
def voxels_residual(X, masks_planar, h, w, N, world_X, world_Y, world_Z, args):
    for idx in range(N):
        K = args['Ks'][idx]
        if idx in range(X.size):
            R_Y = X[idx]
        else:
            R_Y = args['RYs'][idx]
        R_X = args['RXs'][idx]
        R_Z = args['RZs'][idx]
        t = args['ts'][idx]
        R = make_rotation_matrix(R_X, R_Y, R_Z, mode='degree')
        P = make_proj_matrix(K, R, t)
        x, y = projection(P, world_X, world_Y, world_Z)

        # keep points in the image range
        keep_idx = np.where((x >= 1) & (x <= w) & (y >= 1) & (y <= h))[0]
        x = x[keep_idx]
        y = y[keep_idx]

        # keep the points in the mask
        ind = sub2ind([h,w], (y-1).astype(np.int64), (x-1).astype(np.int64))
        keep_idx = keep_idx[masks_planar[idx][ind] >= 1]
        
        world_X = world_X[keep_idx]
        world_Y = world_Y[keep_idx]
        world_Z = world_Z[keep_idx]

    out = -world_Z.size

    print("Evaluation value: %d." % (-out))

    return out


@numba.jit
def voxels_residual_zf(X, masks_planar, N, world_X, world_Y, world_Z):
    X_ratation = X[3:]
    K, R_Z_Y, t = parse_zf_params(X)

    for idx in range(N):
        # make projection matrix
        P = construct_P_zf(X_ratation[idx], R_Z_Y, t, K)
        x, y = projection(P, world_X, world_Y, world_Z)

        # keep points in the image range
        keep_idx = np.where((x >= 1) & (x <= 1024) & (y >= 388) & (y <= 637))[0]
        y = y - 387
        x = x[keep_idx]
        y = y[keep_idx]

        # keep the points in the mask
        ind = sub2ind([250,1024], (y-1).astype(np.int64), (x-1).astype(np.int64)) # index from [0,0] to [h-1, w-1]
        keep_idx = keep_idx[masks_planar[idx][ind] >= 1]
        
        world_X = world_X[keep_idx]
        world_Y = world_Y[keep_idx]
        world_Z = world_Z[keep_idx]

    out = -world_Z.size

    # print("Evaluation value: %d." % (-out))

    return out


def get_prob_model(images, masks, N):
    select_idx = random.sample(range(N), int(0.1*N))
    select_idx.sort()
    for ind, idx in enumerate(select_idx):
        I = images[idx]
        M = masks[idx]
        obj_ind_x, obj_ind_y = np.where(M >  0)
        bck_ind_x, bck_ind_y = np.where(M <= 0)
        if ind == 0:
            obj_color = I[obj_ind_x, obj_ind_y, :]
            bck_color = I[bck_ind_x, bck_ind_y, :]
        else:
            obj_color = np.vstack((obj_color, I[obj_ind_x, obj_ind_y, :]))
            bck_color = np.vstack((bck_color, I[bck_ind_x, bck_ind_y, :]))
    
    mu_obj = np.mean(obj_color.astype(np.float), axis=0)
    mu_bck = np.mean(bck_color.astype(np.float), axis=0)
    sigma_obj = np.cov(obj_color.astype(np.float), rowvar=False)
    sigma_bck = np.cov(bck_color.astype(np.float), rowvar=False)
    
    return mu_obj, mu_bck, sigma_obj, sigma_bck


def get_prob_map(images, mu_obj, mu_bck, sigma_obj, sigma_bck):
    obj_prob_planer = []
    bck_prob_planer = []
    obj_prob = []
    bck_prob = []
    for img in tqdm(images):
        obj_p = multivariate_normal.pdf(img, mu_obj, sigma_obj)
        bck_p = multivariate_normal.pdf(img, mu_bck, sigma_bck)
        obj_prob_planer.append(obj_p.reshape(-1))
        bck_prob_planer.append(bck_p.reshape(-1))
        obj_prob.append(obj_p)
        bck_prob.append(bck_p)

        # plt.subplot(1,2,1)
        # plt.imshow(obj_p)
        # plt.subplot(1,2,2)
        # plt.imshow(bck_p)
        # plt.show()

    return obj_prob_planer, bck_prob_planer, obj_prob, bck_prob


def max_prob(X, prob_map_obj, prob_map_bck, h, w, N, world_X, world_Y, world_Z, args):
    p1 = np.ones(world_X.size, dtype=np.float64)
    p2 = np.array(p1, dtype=np.float64)
    
    for idx in range(N):
        K = args['Ks'][idx]
        if idx in range(X.size):
            R_Y = X[idx]
        else:
            R_Y = args['RYs'][idx]
        R_X = args['RXs'][idx]
        R_Z = args['RZs'][idx]
        t = args['ts'][idx]
        R = make_rotation_matrix(R_X, R_Y, R_Z, mode='degree')
        P = make_proj_matrix(K, R, t)
        x, y = projection(P, world_X, world_Y, world_Z)

        # remove the points locating out of the image range
        out_idx = np.where((x < 1) | (x > w) | (y < 1) | (y > h))[0]
        x[out_idx] = 1
        y[out_idx] = 1

        # convert the confidence maps and coords into linear form
        xy_ind = sub2ind([h,w], (y-1).astype(np.int64), (x-1).astype(np.int64)) # index from [0,0] to [h-1, w-1]

        # extract confidence map
        prob_obj = prob_map_obj[idx]
        prob_bck = prob_map_bck[idx]

        # multiply the probablities of a voxel on each image 
        p1 = p1 * np.power(prob_obj[xy_ind], 1./N)
        p2 = p2 * np.power(1 - prob_bck[xy_ind], 1./N)

    p2 = 1 - p2
    prob_model = np.log(p1 + eps)-np.log(p2 + eps)
    out = -np.sum(prob_model > 0)

    print("Evaluation value: %d." % (-out))

    return out


def max_prob_zf(X, prob_map_obj, prob_map_bck, N, world_X, world_Y, world_Z):
    p1 = np.ones(world_X.size, dtype=np.float64)
    p2 = np.array(p1, dtype=np.float64)
    X_ratation = X[3:]
    K, R_Z_Y, t = parse_zf_params(X)

    for idx in range(N):
        P = construct_P_zf(X_ratation[idx], R_Z_Y, t, K)
        x, y = projection(P, world_X, world_Y, world_Z)
        
        # remove the points locating out of the image range
        out_idx = np.where((x < 1) | (x > 1024) | (y < 388) | (y > 637))[0]
        y = y - 387
        x[out_idx] = 1
        y[out_idx] = 1
        xy_ind = sub2ind([250,1024], (y-1).astype(np.int64), (x-1).astype(np.int64)) # index from [0,0] to [h-1, w-1]

        # extract confidence map
        prob_obj = prob_map_obj[idx]
        prob_bck = prob_map_bck[idx]

        # multiply the probablities of a voxel on each image 
        p1 = p1 * np.power(prob_obj[xy_ind], 1./N)
        p2 = p2 * np.power(1 - prob_bck[xy_ind], 1./N)

    p2 = 1 - p2
    prob_model = np.log(p1 + eps)-np.log(p2 + eps)
    out = -np.sum(prob_model > 0)

    # print("Evaluation value: %d." % (-out))

    return out

