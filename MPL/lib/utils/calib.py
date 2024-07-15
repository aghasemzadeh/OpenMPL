
'''
This file contains functions for camera calibration and skew lines distance calculation.
'''

import numpy as np
import copy

def find_3_points_on_ray(u,v,fx,fy,ox,oy):
    '''
    u, v: pixel coordinates shape (b,)
    '''
    assert u.shape == v.shape
    
    x = np.stack([(u-ox)/fx, np.ones(u.shape), (fy/fx)*((u-ox)/(v-oy))])
    y = np.stack([(v-oy)/fy, (fx/fy)*((v-oy)/(u-ox)), np.ones(u.shape)])
    z = np.stack([np.ones(u.shape), fx/(u-ox), fy/(v-oy)])
    P = np.stack([x, y, z])
    P = np.transpose(P)
    return P

def cam_to_world(point_cam, R, T):
    # takes batches of points and converts them to world coordinates
    # point_cam: (b, n, 3) batch, number of points, xyz
    # build E
    R = np.array(R)
    to_conc = np.ones((point_cam.shape[0], point_cam.shape[1], 1))     # this for concatenating 
    p = np.concatenate([point_cam, to_conc], axis=2)  # convert to homogenous         (b, 3, 4) batch, number of points, homogenous coordinates
    p = np.expand_dims(p, axis=3)
    R_temp = np.vstack([R, np.zeros((1,3))])
    T_temp = np.vstack([T, 1.0])
    E = np.hstack([R_temp, T_temp])
    E_inv = np.linalg.inv(E)
    E_inv = np.stack([E_inv]*point_cam.shape[1])                      # create E_inv for 3 points
    E_inv = np.stack([E_inv]*p.shape[0])             # create E_inv for all matchings
    point_world = E_inv @ p
    point_world_squeezed = np.squeeze(point_world, axis=-1)
    point_world = point_world_squeezed[:,:,:] / point_world[:,:,3]  # convert back to cartesian (check that x_homo[2] > 0)
    point_world = point_world[:,:,:3]
    return point_world

def world_to_cam(point_world, R, T):
    # takes batches of points and converts them to world coordinates
    # point_world: (b, n, 3) batch, number of points, xyz
    # build E
    R = np.array(R)
    to_conc = np.ones((point_world.shape[0], point_world.shape[1], 1))     # this for concatenating 
    p = np.concatenate([point_world, to_conc], axis=2)  # convert to homogenous         (b, n, 4) batch, number of points, homogenous coordinates
    p = np.expand_dims(p, axis=3)
    R_temp = np.vstack([R, np.zeros((1,3))])
    T_temp = np.vstack([T, 1.0])
    E = np.hstack([R_temp, T_temp])
    E = np.stack([E]*point_world.shape[1])                      # create E for 17 points
    E = np.stack([E]*p.shape[0])             # create E for all matchings
    point_cam = E @ p
    point_cam_squeezed = np.squeeze(point_cam, axis=-1)
    point_cam = point_cam_squeezed[:,:,:] / point_cam[:,:,3]  # convert back to cartesian (check that x_homo[2] > 0)
    point_cam = point_cam[:,:,:3]
    return point_cam

def cam_to_image(point_cam, K):
    # takes batches of points and converts them to image 2D coordinates
    # point_cam: (b, n, 3) batch, number of points, xyz
    
    K = np.array(K)
    to_conc = np.ones((point_cam.shape[0], point_cam.shape[1], 1))     # this for concatenating 
    p = np.concatenate([point_cam, to_conc], axis=2)  # convert to homogenous         (b, n, 4) batch, number of points, homogenous coordinates
    p = np.expand_dims(p, axis=3)
    M = np.hstack([K, np.zeros((3, 1))])
    M = np.stack([M]*point_cam.shape[1])                      # create M for 17 points
    M = np.stack([M]*p.shape[0])             # create M for all matchings
    
    point_img = M @ p
    point_img_squeezed = np.squeeze(point_img, axis=-1)
    point_img = point_img_squeezed[:,:,:] / point_img[:,:,2]  # convert back to cartesian (check that x_homo[1] > 0)
    point_img = point_img[:,:,:2]
    return point_img



def find_line_with_3points(points):
    K = np.array(
        [[points[0][1], points[0][2], -1],
        [points[1][1], points[1][2], -1],
        [points[2][1], points[2][2], -1]])
    print(K.shape)
    Y = np.array(
        [-points[0][0], -points[1][0], -points[2][0]])
    print(Y.shape)
    K_inv = np.linalg.inv(K)
    X = K_inv @ Y
    return X
    
def distance_between_two_skew_lines(points_l1, points_l2):
    
    nominator = np.array(
    [[points_l2[:, 0, 0] - points_l1[:, 0, 0], points_l2[:, 0, 1] - points_l1[:, 0, 1], points_l2[:, 0, 2] - points_l1[:, 0, 2]],
     [points_l1[:, 1, 0] - points_l1[:, 2, 0], points_l1[:, 1, 1] - points_l1[:, 2, 1], points_l1[:, 1, 2] - points_l1[:, 2, 2]],
     [points_l2[:, 1, 0] - points_l2[:, 2, 0], points_l2[:, 1, 1] - points_l2[:, 2, 1], points_l2[:, 1, 2] - points_l2[:, 2, 2]]]
    )

    nominator = np.transpose(nominator,(2,0,1))

    det = np.linalg.det(nominator)
    A = (points_l1[:, 1, 1] - points_l1[:, 2, 1]) * (points_l2[:, 1, 2] - points_l2[:, 2, 2]) - (points_l2[:, 1, 1] - points_l2[:, 2, 1]) * (points_l1[:, 1, 2] - points_l1[:, 2, 2])
    B = (points_l1[:, 1, 2] - points_l1[:, 2, 2]) * (points_l2[:, 1, 0] - points_l2[:, 2, 0]) - (points_l2[:, 1, 2] - points_l2[:, 2, 2]) * (points_l1[:, 1, 0] - points_l1[:, 2, 0])
    C = (points_l1[:, 1, 0] - points_l1[:, 2, 0]) * (points_l2[:, 1, 1] - points_l2[:, 2, 1]) - (points_l2[:, 1, 0] - points_l2[:, 2, 0]) * (points_l1[:, 1, 1] - points_l1[:, 2, 1])    

    
    denominator = np.sqrt(A**2 + B**2 + C**2)

    
    return np.abs(det/denominator)


def smart_pseudo_remove_weight(target, weight, meta, epipolar_error_threshold=5):
    '''
    The purpose is to calculate the epipolar error between each pair of cameras.
    And then, if the epipolar error is larger than 5, we set the weight of the corresponding keypoint to 0.
    
    target: list, element: (17, 3)
    weight: list, element: (17,)
    meta: list, element: dict
    '''
    n_views = len(target)
    n_kps = meta[0]['joints_2d'].shape[0]
    target = copy.deepcopy(target)
    weight = copy.deepcopy(weight)
    meta = copy.deepcopy(meta)
    
    epipolar_error_by_cams = np.zeros((n_views, n_kps))
    for i in range(n_views):
        for j in range(n_views):
            if i >= j:
                continue
            else:
                cam_0_fx = meta[i]['camera']['fx']
                cam_0_fy = meta[i]['camera']['fy']
                cam_0_ox = meta[i]['camera']['cx']
                cam_0_oy = meta[i]['camera']['cy']
                u, v = (meta[i]['joints_2d'][:,0].reshape(-1,), meta[i]['joints_2d'][:,1].reshape(-1,))
                P0 = find_3_points_on_ray(u,v,cam_0_fx,cam_0_fy,cam_0_ox,cam_0_oy)               # output shape (b,3,3)
    
                cam_1_fx = meta[j]['camera']['fx']
                cam_1_fy = meta[j]['camera']['fy']
                cam_1_ox = meta[j]['camera']['cx']
                cam_1_oy = meta[j]['camera']['cy']
                u, v = (meta[j]['joints_2d'][:,0].reshape(-1,), meta[j]['joints_2d'][:,1].reshape(-1,))
    
                P1 = find_3_points_on_ray(u,v,cam_1_fx,cam_1_fy,cam_1_ox,cam_1_oy)
    
                cam_0_R = meta[i]['camera']['R']
                cam_0_t = meta[i]['camera']['t']
                cam_1_R = meta[j]['camera']['R']
                cam_1_t = meta[j]['camera']['t']
    
                P0 = cam_to_world(P0, cam_0_R, cam_0_t)
                P1 = cam_to_world(P1, cam_1_R, cam_1_t)
    
                epipolar_error = distance_between_two_skew_lines(P0,P1)
                
                epipolar_error_by_cams[i] += epipolar_error * meta[i]['joints_2d_conf'].reshape(-1,)
                epipolar_error_by_cams[j] += epipolar_error * meta[j]['joints_2d_conf'].reshape(-1,)
    
    epipolar_error_by_cams = epipolar_error_by_cams / (n_views-1)
    
    for i in range(n_views):
        weight[i][epipolar_error_by_cams[i] > epipolar_error_threshold] = 0
    return weight
    