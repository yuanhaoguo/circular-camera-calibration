
import math
import numpy as np


'''
Create rotation matrix from three angles
'''
def make_rotation_matrix(theta_x, theta_y, theta_z, mode='radian'):
    if mode == 'degree':
        theta_x, theta_y, theta_z = angle_2_rad(theta_x), angle_2_rad(theta_y), angle_2_rad(theta_z)
    r_x = np.array([[1, 0, 0], 
                    [0, math.cos(theta_x), -math.sin(theta_x)], 
                    [0, math.sin(theta_x), math.cos(theta_x)]], dtype=np.float)
    r_y = np.array([[math.cos(theta_y), 0, math.sin(theta_y)], 
                    [0, 1, 0], 
                    [-math.sin(theta_y), 0, math.cos(theta_y)]], dtype=np.float)
    r_z = np.array([[math.cos(theta_z), -math.sin(theta_z), 0], 
                    [math.sin(theta_z), math.cos(theta_z), 0], 
                    [0, 0, 1]], dtype=np.float)
    out = np.matmul(r_z, np.matmul(r_y, r_x))
    return out

    
'''
Compose projection matrix from intrinsic and extrisinc matrices
'''
def make_proj_matrix(K, R, T):
    # K: 3x3
    # R: 3x3
    # T: 3x1
    # P = K*[R T]
    # t = np.matmul(-R, T)

    trans = np.hstack((R, T))
    P = np.matmul(K, trans)
    return P


'''
Decompose intrinsic and extrisinc matrices from projection matrix
'''
def decompose_proj(P):
    M =  P[:,:3]
    M_I = np.linalg.inv(M)
    R_I, K_I = np.linalg.qr(M_I)

    R = np.linalg.inv(R_I)
    if np.linalg.det(R) < 0:
        R = -R
        K_I = -K_I
    
    K = np.linalg.inv(K_I)
    t = np.matmul(K_I, P[:,-1])
    t = np.expand_dims(t, axis=-1)
    T = np.matmul(R_I, -t)
    T = np.expand_dims(T, axis=-1)

    return K, R, t, T


'''
Parse rotation angle from rotation matrix
'''
def decompose_rotation_angle(mat, mode='radian'):

    theta_x = math.atan2(mat[2,1], mat[2,2])
    theta_z = math.atan2(mat[1,0], mat[0,0])
    if theta_x > 0:
        theta_y = math.atan2(-mat[2,0], math.sqrt(mat[2,1]**2 + mat[2,2]**2))
    else:
        theta_y = math.atan2(-mat[2,0], -math.sqrt(mat[2,1]**2 + mat[2,2]**2))
    
    theta_y = math.atan2(-mat[2,0], math.sqrt(mat[2,1]**2 + mat[2,2]**2))
    if mode == 'degree':
        theta_x, theta_y, theta_z = rad_2_angle(theta_x), rad_2_angle(theta_y), rad_2_angle(theta_z)
    return theta_x, theta_y, theta_z


'''
Convert radians and angles
'''
def angle_2_rad(x):
    return float(x) / 180. * math.pi


def rad_2_angle(x):
    return float(x) / math.pi * 180.


def make_K_zf():
    K = np.array([[65000/5.5, 0, 512],
                  [0, 65000/5.5, 512],
                  [0, 0, 1]])
    return K


def initialize_variable_zf(image_num):
    x_rotation_step_0 = 2 * math.pi / float(image_num)
    X_angles = np.arange(1, image_num+1) * x_rotation_step_0
    Y_Z_T = np.array([0, 0, 0])
    X0 = np.append(Y_Z_T, X_angles)
    return X0


# def initialize_variable_zf_better(image_num):
#     x_rotation_step_0 = 2 * math.pi / float(image_num)
#     X_angles = np.arange(1, image_num+1) * x_rotation_step_0
#     Y_Z_T = np.zeros(3)
#     X0 = np.append(Y_Z_T, X_angles)
#     return X0


def parse_zf_params(X):
    Y_rotation = X[0]
    Z_rotation = X[1]
    translation = X[2]

    K = make_K_zf()

    r_y = np.array([[math.cos(Y_rotation), 0, math.sin(Y_rotation)], 
                    [0, 1, 0], 
                    [-math.sin(Y_rotation), 0, math.cos(Y_rotation)]], dtype=np.float)
                    
    r_z = np.array([[math.cos(Z_rotation), -math.sin(Z_rotation), 0], 
                    [math.sin(Z_rotation), math.cos(Z_rotation), 0], 
                    [0, 0, 1]], dtype=np.float)

    R_Z_Y = np.matmul(r_z, r_y)
    t = np.array([[0], [math.sin(translation)], [math.cos(translation)]]) * 65000

    return K, R_Z_Y, t


def construct_Ps_zf(X, N):
    X_ratation = X[3:]
    K, R_Z_Y, t = parse_zf_params(X)
    Ps = []
    for idx in range(N):
        P = construct_P_zf(X_ratation[idx], R_Z_Y, t, K)
        Ps.append(P)
    return Ps


def construct_P_zf(X_ratation, R_Z_Y, t, K):
    R_X = np.array([[1, 0, 0], 
                    [0, math.cos(X_ratation), -math.sin(X_ratation)], 
                    [0, math.sin(X_ratation), math.cos(X_ratation)]], dtype=np.float)
    R = np.matmul(R_Z_Y, R_X)
    P = make_proj_matrix(K, R, t)
    return P


def compute_shape(recon_voxels, vertex, facet, N, is_carving=False):
    if is_carving:
        num_voxels = recon_voxels.N
    else:
        num_voxels = np.sum(recon_voxels.V == N)
    
    volume = recon_voxels.Resolution ** 3 * num_voxels
    vertex = np.array(vertex, np.float)
    vertexA = vertex[facet[:,0],:]
    vertexB = vertex[facet[:,1],:]
    vertexC = vertex[facet[:,2],:]  

    A = vertexA-vertexB
    B = vertexA-vertexC
    C = vertexB-vertexC

    EA = np.sqrt(A[:,0]**2+A[:,1]**2+A[:,2]**2)
    EB = np.sqrt(B[:,0]**2+B[:,1]**2+B[:,2]**2)
    EC = np.sqrt(C[:,0]**2+C[:,1]**2+C[:,2]**2)

    p = (EA + EB + EC)/2
    Ar = np.sqrt(p*(p-EA)*(p-EB)*(p-EC))
    surface_area = np.sum(Ar)

    return volume, surface_area


def decompose_P_add_noise(Ps, num_select, noise_level):
    args = {}
    args['Ks'] = []
    args['Rs'] = []
    args['RXs'] = []
    args['RYs'] = []
    args['RZs'] = []
    args['ts'] = []
    args['Ts'] = []
    for P in Ps:
        K, R, t, T = decompose_proj(P)
        X_rotation, Y_rotation, Z_rotation = decompose_rotation_angle(R, mode='degree')
        args['Ks'].append(K)
        args['Rs'].append(R)
        args['RXs'].append(X_rotation)
        args['RYs'].append(Y_rotation)
        args['RZs'].append(Z_rotation)
        args['ts'].append(t)
        args['Ts'].append(T)
    
    X0 = [] # starting point
    X_gt = []
    for idx in range(num_select): 
        X0.append(args['RYs'][idx] + noise_level)
        X_gt.append(args['RYs'][idx])
    X0 = np.array(X0, dtype=np.float)

    return args, X0, X_gt


def get_camera_direction(h, w, Ps):
    x = np.array([[w/2.], [h/2.], [1.0]])
    camera_direction = np.zeros((3, len(Ps)))
    for idx, P in enumerate(Ps):
        K, R, _, _ = decompose_proj(P)
        X = np.matmul(np.linalg.inv(K), x)
        X = np.matmul(R.T, X)
        camera_direction[:, idx] = X.reshape(-1) / np.linalg.norm(X)
    return camera_direction
    

if __name__ == "__main__":
    
    P = np.array([[3.9924, 39.4177, -0.7633, 3.9592], 
                  [-14.4302, -0.9414, -27.4510, -14.4294], 
                  [0.0122, -0.00014575, -0.00056931, 0.0122]])

    K, R, t, T = decompose_proj(P)
    # print(P)
    # print("*"*10)
    # print(K)
    # print("*"*10)
    # print(R)
    # print("*"*10)
    # print(t)
    # print("*"*10)
    # print(T)
    # print("*"*10)

    R_X, R_Y, R_Z = decompose_rotation_angle(R, mode='degree')
    R_recon = make_rotation_matrix(R_X, R_Y, R_Z, mode='degree')
    P_recon = make_proj_matrix(K, R_recon, t)
    print(R)
    print(R_recon)
