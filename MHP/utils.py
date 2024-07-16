
import os
from os import path as osp
import numpy as np
import torch
from tqdm import tqdm
import pickle
import trimesh
import argparse
import glob 
import torch
import json
import copy

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

# load camera calib info
def load_all_cameras_h36m(calib_path, actors):
    """
    loads all cameras from the calibration file and returns a dictionary
    containing the camera parameters for each camera
    each item has a list showing different camera setups (actors)
    """
    calib_file = os.path.join(calib_path, 'camera_data.pkl')
    with open(calib_file, 'rb') as f:
        camera_data = pickle.load(f)
    cams = range(1, 5)
    cameras = {cam_id:[] for cam_id in cams}
    for cam_setup in actors:
        for cam_id in cams:
            camera = camera_data[(cam_setup, cam_id)]
            camera_dict = {}
            camera_dict['camera_setup'] = cam_setup
            camera_dict['camera_id'] = cam_id
            camera_dict['R'] = camera[0]
            camera_dict['T'] = camera[1] / 1000
            camera_dict['t'] = - np.linalg.inv(camera_dict['R'].T) @ camera_dict['T']
            camera_dict['fx'] = camera[2][0].squeeze()
            camera_dict['fy'] = camera[2][1].squeeze()
            camera_dict['cx'] = camera[3][0].squeeze()
            camera_dict['cy'] = camera[3][1].squeeze()
            camera_dict['k'] = camera[4]
            camera_dict['p'] = camera[5]
            camera_dict['K'] = np.array([[camera_dict['fx'], 0, camera_dict['cx']], 
                                            [0, camera_dict['fy'], camera_dict['cy']], 
                                        [0, 0, 1]])
            cameras[cam_id].append(camera_dict)
    return cameras

def compatible_cams(cameras):
    for k,cam in cameras.items():    
        cam['K'] = np.matrix(cam['K'])
        cam['distCoef'] = np.array(cam['distCoef'])
        cam['k'] = np.array([cam['distCoef'][0], cam['distCoef'][1], cam['distCoef'][4]])
        cam['p'] = np.array([cam['distCoef'][2], cam['distCoef'][3]])
        cam['R'] = np.matrix(cam['R'])
        cam['t'] = np.array(cam['t']).reshape((3,1)) / 100
        cam['T'] = np.array(-cam['R'].T @ np.array(cam['t']).reshape((3,1)))
        cam['fx'] = cam['K'][0,0]
        cam['fy'] = cam['K'][1,1]
        cam['cx'] = cam['K'][0,2]
        cam['cy'] = cam['K'][1,2]
    return cameras

def load_all_cameras_cmu(calib_root, cmu_calibs):
    """
    loads all cameras from the calibration file and returns a dictionary
    containing the camera parameters for each camera
    each item has a list showing different camera setups (actors)
    """
    cams = range(31)
    cameras_all_calibs = {cam_id: [] for cam_id in cams}
    for cam_setup in cmu_calibs:
        calib_file = osp.join(calib_root, cam_setup, 'calibration_{}.json'.format(cam_setup))
        with open(calib_file, 'r') as f:
            calibration_cat = json.load(f)
        cameras = {cam['node']:cam for cam in calibration_cat['cameras'] if cam['type']=='hd'}
        cameras = compatible_cams(cameras)
        for cam_id, cam in cameras.items():
            cameras_all_calibs[cam_id].append(cam)
    return cameras_all_calibs

# def load_all_20_cameras():
#     calib_file = 'cameras_20.json'
#     with open(calib_file, 'r') as f:
#         calibration_cat = json.load(f)
#     cameras = {i:cam for i, cam in enumerate(calibration_cat['cameras'])}
#     for cam_id, cam in cameras.items():
#         cam[]


MMPOSE2H36M = {
    1: 12,  # rhip
    2: 14,  # rkne
    3: 16,  # rank
    4: 11,  # lhip
    5: 13,  # lkne
    6: 15,  # lank
    9: 0,   # nose
    11: 5,  # lsho
    12: 7,  # lelb
    13: 9,  # lwri
    14: 6,  # rsho
    15: 8,  # relb
    16: 10, # rwri
}


def mmpose2h36m(mmpose, scores):
    h36m = np.zeros((17, 2))
    conf = np.zeros((17, 1))
    for i in range(17):
        try:
            h36m[i] = mmpose[MMPOSE2H36M[i]]
            conf[i] = scores[MMPOSE2H36M[i]]
        except KeyError:
            h36m[i] = np.array([0, 0])
            conf[i] = 0
    
    # calculate head
    head = mmpose[0:5].mean(axis=0)
    h36m[10] = head
    conf[10] = scores[0:5].mean(axis=0)
    
    # calculate neck
    neck = mmpose[3:7].mean(axis=0)
    h36m[8] = neck
    conf[8] = scores[3:7].mean(axis=0)
    
    # calculate root
    root = mmpose[11:13].mean(axis=0)
    h36m[0] = root
    conf[0] = scores[11:13].mean(axis=0)
    
    # calculate belly
    belly = np.mean([neck, root], axis=0)
    h36m[7] = belly
    conf[7] = np.mean([conf[8], conf[0]], axis=0)
    
    return h36m, conf

def calculate_mpjpe(pred, gt):
    if len(pred.shape) == 2:
        pjpe = np.sqrt(((pred - gt) ** 2).sum(axis=1))
        return pjpe.mean()
    else:
        pjpe = np.sqrt(((pred - gt) ** 2).sum(axis=2))
        return pjpe.mean(axis=1)
    
def calculate_scale_factor_1d(source_points, dest_points):
    distances_source = np.abs(np.diff(source_points, axis=0))
    distances_dest = np.abs(np.diff(dest_points, axis=0))

    scale_factor = distances_dest.mean() / distances_source.mean()
    return scale_factor.item()

def run_mmpose(image, mmpose_inferencer, convert_to_h36m=True):
    result_mmpose_generator = mmpose_inferencer(image, show=False, device='cuda:0') #, show_progress=False)
    result_mmpose = next(result_mmpose_generator)
    predictions_mmpose = result_mmpose['predictions'][0][0] # first image - first prediction
    keypoints = np.array(predictions_mmpose['keypoints'])
    keypoint_scores = np.array(predictions_mmpose['keypoint_scores'])
    if convert_to_h36m:
        h36m_points_mmpose, h36m_scores_mmpose = mmpose2h36m(keypoints, keypoint_scores)
        return h36m_points_mmpose, h36m_scores_mmpose
    else:
        return keypoints, keypoint_scores.reshape(-1, 1)

def amass_vertices_to_joints(vertices, h36m_jregressor, R, t, K, lower_body_reversed=True):
    joints = torch.einsum('bik,ji->bjk', [vertices, h36m_jregressor])
    joints_copy = joints.clone()
    if lower_body_reversed:
        # to account for the mistake in the regressor (lower body reversed)
        joints[:, 1:4] = joints_copy[:, 4:7]
        joints[:, 4:7] = joints_copy[:, 1:4]
    joints_cam = world_to_cam(joints, R, t)
    points_image = cam_to_image(joints_cam, K)
    return points_image, joints.cpu().numpy()

def find_most_aligned_joints(points_image_amass, keypoints_most_aligned_with_h36m, imw, imh):
    # find the most aligned joints
    most_aligned_joints = []
    for joint in keypoints_most_aligned_with_h36m:
        joint_amass = points_image_amass[joint]
        if joint_amass[0] < 0 or joint_amass[0] > imw or joint_amass[1] < 0 or joint_amass[1] > imh:
            continue
        most_aligned_joints.append(joint)
    if len(most_aligned_joints) == 1:
        if most_aligned_joints[0] == keypoints_most_aligned_with_h36m[0]:
            most_aligned_joints.append(keypoints_most_aligned_with_h36m[1])
        else:
            most_aligned_joints.append(keypoints_most_aligned_with_h36m[0])
    return most_aligned_joints

def filter_wrong_mmpose_detections(h36m_points_mmpose, h36m_scores_mmpose, points_image_amass, imw, imh):
    keypoints_outside = np.logical_or(np.logical_or(points_image_amass[:, 0] < 0, points_image_amass[:, 0] > imw),
                                        np.logical_or(points_image_amass[:, 1] < 0, points_image_amass[:, 1] > imh))
    h36m_scores_mmpose[keypoints_outside] = 0
    return h36m_scores_mmpose

def fit_skeleton_to_amass(skeleton, amass_joints_2d, most_reliable_joints, mmpose_confs, imw, imh, apply_joint_clip, fit_using_most_aligned=False):
    if most_reliable_joints == []:
        return np.zeros((17, 2)), np.zeros((17, 1))
    scale_factor_x = calculate_scale_factor_1d(skeleton[most_reliable_joints, 0], amass_joints_2d[most_reliable_joints, 0])
    scale_factor_y = calculate_scale_factor_1d(skeleton[most_reliable_joints, 1], amass_joints_2d[most_reliable_joints, 1])
    scale_factor = np.array([scale_factor_x, scale_factor_y])
    if fit_using_most_aligned:
        len_most_reliable_joints = len(most_reliable_joints)
        all_possible = (np.stack([skeleton] * len_most_reliable_joints) - skeleton[most_reliable_joints, None]) * scale_factor + amass_joints_2d[most_reliable_joints, None]
        amass_joints_2d_ = np.stack([amass_joints_2d] * len_most_reliable_joints)
    else:
        all_possible = (np.stack([skeleton] * 17) - skeleton[range(17), None]) * scale_factor + amass_joints_2d[range(17), None]
        amass_joints_2d_ = np.stack([amass_joints_2d] * 17)
    all_mpjpes = calculate_mpjpe(all_possible, amass_joints_2d_)
    best_kp = all_mpjpes.argmin()
    best_match_joints = all_possible[best_kp]
    # clip best match joints to image size 
    to_zero = np.logical_or(np.logical_or(best_match_joints[:, 0] < 0, best_match_joints[:, 0] > imw), np.logical_or(best_match_joints[:, 1] < 0, best_match_joints[:, 1] > imh))
    mmpose_confs_ = mmpose_confs.copy()
    if apply_joint_clip:
        mmpose_confs_[to_zero] = 0
        best_match_joints[:, 0] = np.clip(best_match_joints[:, 0], 0, imw - 1)
        best_match_joints[:, 1] = np.clip(best_match_joints[:, 1], 0, imh - 1)
    return best_match_joints, mmpose_confs_

def locate_mesh_in_room(vertices, room_min_x, room_max_x, room_min_y, room_max_y, room_min_z, room_max_z, h36m_or_cmu, rotate=False):
    if h36m_or_cmu == 'triangulate_3d':
        vertices[:] = vertices[:] - vertices[0, :]
        # vertices_copy = vertices.clone()
        # vertices[:, 1] = -vertices[:, 2]   # swap y and z
        # vertices[:, 2] = vertices_copy[:, 1]  # swap y and z
        return vertices
    augmentation_3d = torch.cat([torch.rand(1, 1) * (room_max_x - room_min_x) + room_min_x, torch.rand(1, 1) * (room_max_y - room_min_y) + room_min_y, torch.rand(1, 1) * (room_max_z - room_min_z) + room_min_z], dim=1)
    if room_min_z != 0 and room_max_z != 0:
        vertices = vertices - vertices[0, :]
    else:
        vertices[:, 0] = vertices[:, 0] - vertices[0, 0]
        vertices[:, 1] = vertices[:, 1] - vertices[0, 1]
    if rotate:
        rotation = np.random.rand(1) * 360
        vertices = rotate_pose(vertices[None], rotation, axis='z').squeeze()
        # vertices = vertices.type(torch.float32)
    vertices_augmented = vertices + augmentation_3d
    vertices_augmented = vertices_augmented.type(torch.float32)
    if h36m_or_cmu == 'cmu':
        vertices_augmented_copy = vertices_augmented.clone()
        vertices_augmented[:, 1] = -vertices_augmented[:, 2]   # swap y and z
        vertices_augmented[:, 2] = vertices_augmented_copy[:, 1]  # swap y and z
    return vertices_augmented


def rotate_pose(poses, angles_degrees, axis='y'):
    # Convert angles to radians
    angles_radians = np.radians(angles_degrees)

    # Number of poses in the batch
    b = poses.shape[0]
    rotation_matrix = np.zeros((3, 3, b))
    # Initialize rotation matrix based on the specified axis
    if axis == 'x':
        rotation_matrix[0, 0, :] = 1
        rotation_matrix[1, 1, :] = np.cos(angles_radians)
        rotation_matrix[1, 2, :] = -np.sin(angles_radians)
        rotation_matrix[2, 1, :] = np.sin(angles_radians)
        rotation_matrix[2, 2, :] = np.cos(angles_radians)
    elif axis == 'y':
        rotation_matrix[0, 0, :] = np.cos(angles_radians)
        rotation_matrix[0, 2, :] = np.sin(angles_radians)
        rotation_matrix[1, 1, :] = 1
        rotation_matrix[2, 0, :] = -np.sin(angles_radians)
        rotation_matrix[2, 2, :] = np.cos(angles_radians)
    elif axis == 'z':
        rotation_matrix[0, 0, :] = np.cos(angles_radians)
        rotation_matrix[0, 1, :] = -np.sin(angles_radians)
        rotation_matrix[1, 0, :] = np.sin(angles_radians)
        rotation_matrix[1, 1, :] = np.cos(angles_radians)
        rotation_matrix[2, 2, :] = 1
    else:
        raise ValueError("Invalid axis. Please choose 'x', 'y', or 'z'.")
    # Rotate the poses
    rotated_poses = np.matmul(poses, rotation_matrix.transpose((2,1,0)))

    return rotated_poses


def add_camera_between(cameras, cam_id_new, cam_id, cam_id_2, avg_on_axis=[0, 1, 2]):
    cameras[cam_id_new] = copy.deepcopy(cameras[cam_id])
    for i, _ in enumerate(cameras[cam_id_new]):
        cameras[cam_id_new][i]['camera_id'] = cam_id_new
        for j in avg_on_axis:    
            cameras[cam_id_new][i]['T'][j] = (cameras[cam_id][i]['T'][j] + cameras[cam_id_2][i]['T'][j]) / 2
        cameras[cam_id_new][i]['R'] = (cameras[cam_id][i]['R'] + cameras[cam_id_2][i]['R']) / 2
        cameras[cam_id_new][i]['t'] = -np.array(np.dot(np.linalg.inv(cameras[cam_id_new][i]['R'].T), cameras[cam_id_new][i]['T']))

    return cameras