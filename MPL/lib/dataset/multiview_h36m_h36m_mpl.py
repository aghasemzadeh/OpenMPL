# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import pickle
import collections
import random
import torch
import copy
import cv2
from utils.calib import smart_pseudo_remove_weight
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform, affine_transform_pts
from utils.calib import world_to_cam, cam_to_image, cam_to_world
from utils.utils_amass import rotate_pose

from dataset.joints_dataset_mpl import JointsDataset_MPL
import logging
import json
logger = logging.getLogger(__name__)

downsample = 16

# # set random seed
# np.random.seed(0)
# random.seed(0)

class MultiView_H36M_H36M_MPL(JointsDataset_MPL):

    """
    The purpose is to train on H36M 3D data based on the camera system from CMU Panoptic.
    the 3D data is taken from h36M, and the 2D is created based on camera parameters from CMU Panoptic.
    """
    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        self.num_joints = 17
        self.actual_joints = {
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'neck',
            9: 'nose',
            10: 'head',
            11: 'lsho',
            12: 'lelb',
            13: 'lwri',
            14: 'rsho',
            15: 'relb',
            16: 'rwri'
        }
        self.val_on_train = cfg.DATASET.VAL_ON_TRAIN
        self.is_train = is_train
            
        self.camera_manual_order = cfg.DATASET.CAMERA_MANUAL_ORDER
        print('camera_manual_order',self.camera_manual_order)
        
        self.dataset_type = cfg.DATASET.DATASET_TYPE
        
        if self.h36m_old_datasets:
            dataset_path = osp.join(self.root_2, 'h36m', 'annot')
            if self.use_mmpose:
                dataset_path = osp.join(self.root_2, 'h36m', 'annot_mmpose')
        else:
            dataset_folder_name = 'datasets_mmpose' if self.use_mmpose else 'datasets' 
            dataset_folder_name_2 = self.h36m_dataset_name + '_' + self.mmpose_type if self.use_mmpose else self.h36m_dataset_name
            dataset_path = osp.join(self.root_2, 'h36m', self.dir_mpl_data, dataset_folder_name, dataset_folder_name_2)
        if cfg.DATASET.CROP:
            anno_file = osp.join(dataset_path, 'h36m_{}.pkl'.format(image_set))
        else:
            anno_file = osp.join(dataset_path, 'h36m_{}_uncrop.pkl'.format(image_set))

        self.db = self.load_db(anno_file)
        if not cfg.DATASET.WITH_DAMAGE:
            print('before filter', len(self.db))
            self.db = [db_rec for db_rec in self.db if not self.isdamaged(db_rec)]
            print('after filter', len(self.db))
            
        self.db = self.make_cameras_compatible(self.db)

        self.u2a_mapping = super().get_mapping()
        super().do_mapping()

        self.grouping = self.get_group_h36m(self.db)
        self.group_size = len(self.grouping)
        logger.info('=> {} num samples: {}'.format(image_set, self.group_size))
        
        
        actors = self.h36m_calib_actors
        self.h36m_cameras = self.load_all_cameras_h36m(calib_file=osp.join(self.root, 'camera_data.pkl'), actors=actors)
        
        # if self.train_on_all_cameras and is_train:
        #     pass
        # else:
        self.h36m_cameras = {view: v for view, v in self.h36m_cameras.items() if view in self.views}
            
        self.n_all_cameras = len(self.h36m_cameras)    
        self.all_camera_ids = list(self.h36m_cameras.keys())
        self.n_camera_setups = len(self.h36m_cameras[self.all_camera_ids[0]])

        
        self.R_noise_value = cfg.DATASET.R_NOISE_VALUE
        self.T_noise_value = cfg.DATASET.T_NOISE_VALUE
        if cfg.DATASET.APPLY_NOISE_CAMERAS and self.is_train:
            self.add_noise_to_cameras()
        
    def add_noise_to_camera(self, camera):
        # add noise to camera params
        camera['R'] = camera['R'] + np.random.normal(0, 1.0, camera['R'].shape) * self.R_noise_value
        camera['T'] = camera['T'] + np.random.normal(0, 1.0, camera['T'].shape) * self.T_noise_value
        return camera
    
    def load_all_cameras_h36m(self, calib_file, actors):
        """
        loads all cameras from the calibration file and returns a dictionary
        containing the camera parameters for each camera
        each item has a list showing different camera setups (actors)
        """
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
                camera_dict['T'] = camera[1]
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
    
    def load_all_cameras_cmu(self, cmu_calibs):
        cams = range(31)
        cameras_all_calibs = {cam_id: [] for cam_id in cams}
        for cam_setup in cmu_calibs:
            calib_file = osp.join(self.root, cam_setup, 'calibration_{}.json'.format(cam_setup))
            with open(calib_file, 'r') as f:
                calibration_cat = json.load(f)
            cameras = {cam['node']:cam for cam in calibration_cat['cameras'] if cam['type']=='hd'}
            cameras = self.compatible_cams(cameras)
            for cam_id, cam in cameras.items():
                cameras_all_calibs[cam_id].append(cam)
        return cameras_all_calibs
        
    def add_noise_to_cameras(self):
        # add noise to cameras
        for db_rec in self.db:
            db_rec['camera'] = self.add_noise_to_camera(db_rec['camera'])

    def index_to_action_names(self):
        return None

    def load_db(self, dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
            return dataset
        
    def make_cameras_compatible(self, db):
        for data in db:
            camera = data['camera']
            if 'K' not in camera.keys():
                camera['fx'] = camera['fx'][0] 
                camera['fy'] = camera['fy'][0]
                camera['cx'] = camera['cx'][0]
                camera['cy'] = camera['cy'][0]
                camera['K'] = np.array([[camera['fx'], 0, camera['cx']]
                                        ,[0, camera['fy'], camera['cy']]
                                    ,[0, 0, 1]])    
                camera['t'] = - np.linalg.inv(camera['R'].T) @ camera['T']
            
        return db


    def get_group_h36m(self, db):
        grouping = {}
        nitems = len(db)
        # if self.use_cmu_cameras_on_h36m:
        views = [1, 2, 3, 4]
        CAM_IX = {x-1: i for i, x in enumerate(views[:self.n_views])}
        # else:
        #     views = self.views
        #     CAM_IX = {x-1: i for i, x in enumerate(self.views)}
        for i in range(nitems):
            keystr = self.get_key_str_h36m(db[i])
            try:
                camera_id = CAM_IX[db[i]['camera_id']]
            except KeyError:
                continue
            if keystr not in grouping:
                grouping[keystr] = [-1] * len(views[:self.n_views])
            grouping[keystr][camera_id] = i
            
        filtered_grouping = []
        for _, v in grouping.items():
            if np.all(np.array(v) != -1):
                filtered_grouping.append(v)
                
        # if self.use_cmu_cameras_on_h36m and len(filtered_grouping) > 0:
        #     n_views_to_add = self.n_views - len(filtered_grouping[0])
        #     for i, group in enumerate(filtered_grouping):
        #         for j in range(n_views_to_add):
        #             filtered_grouping[i].append(-1)
                

        if self.is_train:
            filtered_grouping = filtered_grouping[::5]
        else:
            filtered_grouping = filtered_grouping[::64]
            
        if self.train_n_samples is not None and self.is_train:
            filtered_grouping = filtered_grouping[:self.train_n_samples]
        if self.test_n_samples is not None and not self.is_train:
            filtered_grouping = filtered_grouping[:self.test_n_samples]

        return filtered_grouping
    
    def __getitem__(self, idx):
        input, target, weight, meta = [], [], [], []
        item = self.grouping[idx][0]
        joints_3d_camera = copy.deepcopy(self.db[item]['joints_3d_camera'])
        camera = self.db[item]['camera'].copy()
        joints_3d = cam_to_world(joints_3d_camera[None], camera['R'], camera['t']).squeeze()  # (17, 3) in world coordinate
        pose_3d = joints_3d / 1000 # convert to meters

        joints_3d_org = pose_3d.copy()
        if self.centeralize_root_first and self.is_train:     # only x and y
            pose_3d[:, 0] = pose_3d[:, 0] - pose_3d[0, 0]  # subtract root joint
            pose_3d[:, 2] = pose_3d[:, 2] - pose_3d[0, 2]  # subtract root joint
            
        if self.normalize_room:
            room_x_scale = (self.room_max_x - self.room_min_x) / 2
            room_y_scale = (self.room_max_y - self.room_min_y) / 2
            room_center = np.array([(self.room_max_x + self.room_min_x) / 2, 0, (self.room_max_y + self.room_min_y) / 2])
            
        if not self.output_in_meter:
            pose_3d = pose_3d * 1000           # (17, 3)   in world coordinate. *100 to convert to cm
        
        if self.rotate and self.is_train:
            rotation = np.random.rand(1) * 360
            pose_3d = rotate_pose(pose_3d[None], rotation, axis='y')[0]
        if self.no_augmentation_3d or not self.is_train:
            augmentation_3d = np.zeros((1, 3)) 
        else:
            if self.flip:     # flipping the pose is wrong here because the face cannot be flipped (#TODO in amass preprocess)
                flipping = np.random.rand(1)
                if flipping > 0.5:
                    pose_3d_copy = pose_3d.copy()
                    pose_3d[self.joints_right] = pose_3d_copy[self.joints_left]
                    pose_3d[self.joints_left] = pose_3d_copy[self.joints_right]
                    
            augmentation_3d = np.concatenate([np.random.rand(1, 1) * (self.room_max_x - self.room_min_x) + self.room_min_x, np.random.rand(1, 1) * (self.room_max_y - self.room_min_y) + self.room_min_y, np.zeros((1, 1))], axis=1)
        
        pose_3d = pose_3d + augmentation_3d
        
        camera_setup_to_use = np.random.random_integers(0, self.n_camera_setups-1)
        
        # if self.train_on_all_cameras and self.is_train:
        #     camera_ids = np.random.choice(self.all_camera_ids, self.n_views, replace=False)
        # else:
        camera_ids = np.array(self.views)
            
        for camera_id in camera_ids:
            i, t, w, m = self.getitem(pose_3d, camera_id, joints_3d_org, camera_setup_to_use)
            if self.normalize_room:
                t = self.normalize_pose3d_coordinates(t, room_center, room_x_scale, room_y_scale)
                m['rays'] = self.normalize_pose3d_coordinates(m['rays'], room_center, room_x_scale, room_y_scale)
                m['cam_center'] = self.normalize_pose3d_coordinates(m['cam_center'], room_center, room_x_scale, room_y_scale)
                m['room_scaled'] = True
                m['room_x_scale'] = max(room_x_scale, room_y_scale)
                m['room_y_scale'] = max(room_x_scale, room_y_scale)
                m['room_center'] = room_center
                m['room_scaled_equal'] = True
            if self.switch_x_z:
                t = t[:, [2, 1, 0]]
            elif self.switch_z_x_y:
                t = t[:, [2, 0, 1]]
            elif self.switch_y_z_x:
                t = t[:, [1, 2, 0]]
            elif self.switch_x_y:
                t = t[:, [1, 0, 2]]
            elif self.switch_y_z:
                t = t[:, [0, 2, 1]]
            input.append(i)
            target.append(t)
            weight.append(w)
            meta.append(m)
        if self.APPLY_SMART_PSEUDO_TRAINING and self.is_train:
            weight = smart_pseudo_remove_weight(target, weight, meta, self.epipolar_error_threshold)
        if 'camera' in meta[0].keys():
            for m in meta:
                del m['camera']
                del m['joints_2d_conf']
        return input, target, weight, meta
        
        
        # if self.centeralize_root_first and self.is_train:     # only x and y
        #     pose_3d[:, 0] = pose_3d[:, 0] - pose_3d[0, 0]  # subtract root joint
        #     pose_3d[:, 1] = pose_3d[:, 1] - pose_3d[0, 1]  # subtract root joint
        # for camera_id, item in enumerate(items):
        #     i, t, w, m = self.getitem(item, camera_id)
        #     input.append(i)
        #     target.append(t)
        #     weight.append(w)
        #     meta.append(m)
        # if self.APPLY_SMART_PSEUDO_TRAINING and self.is_train:
        #     weight = smart_pseudo_remove_weight(target, weight, meta, self.epipolar_error_threshold)
        # if 'camera' in meta[0].keys():
        #     for m in meta:
        #         del m['camera']
        #         del m['joints_2d_conf']
        # return input, target, weight, meta
        
    def getitem(self, joints_3d, camera_id, joints_3d_org=None, camera_setup_to_use=0):
        camera = self.h36m_cameras[camera_id][camera_setup_to_use].copy()
        if self.output_in_meter:
            camera['T'] = camera['T'] / 1000
            camera['t'] = camera['t'] / 1000
            
        if self.apply_noise_cameras and self.is_train:
            noise = np.random.normal(0, 1, camera['T'].shape) * self.t_noise_value
            camera['t'] = camera['t'] + noise
            noise = np.random.normal(0, 1, camera['R'].shape) * self.R_noise_value
            camera['R'] = camera['R'] + noise
            camera['T'] = -camera['R'].T @ camera['t'].squeeze()
            
        joints_3d_cam = world_to_cam(joints_3d[None,:,:], camera['R'], camera['t'])  # (17, 3) in camera coordinate
        joints_2d = cam_to_image(joints_3d_cam, camera['K'])[0]  
        
        if self.target_normalized_3d:
            joints_3d = joints_3d_org
            
        joints = joints_2d.copy()
        # joints = db_rec['joints_2d'].copy()             # (17, 2)   in original image scale (1000, 1000)
        # print('joints', joints)
        joints_org = joints.copy()
        joints_vis = np.ones((17, 1))        # (17, 3)   0,0,0 or 1,1,1
        
        noise_vis = np.zeros((17, 2))
        noise_penalize_conf = np.ones((17, 1))
        if self.APPLY_NOISE and self.is_train:
            noise = np.random.normal(0, 1, joints.shape) * self.NOISE_LEVEL
            noise_vis = noise.copy()
            joints = joints + noise
            # penalize the confidence of the noisy joints
            # penalize_conf = np.exp(-np.abs(noise.sum(axis=1)) / 2)
            if self.penalize_confidence == 'exp_error':
                a = self.penalize_confidence_a
                b = self.penalize_confidence_b
                noise_value = np.sqrt((noise ** 2).sum(axis=1))
                penalize_conf = a * np.exp(- b * noise_value)
            elif self.penalize_confidence == 'linear':
                a = self.penalize_confidence_a
                b = self.penalize_confidence_b
                noise_value = np.sqrt((noise ** 2).sum(axis=1))
                penalize_conf = a * noise_value + b
            elif self.penalize_confidence == 'exp_sqrt':
                penalize_conf = np.exp(-np.sqrt((noise ** 2).sum(axis=1)) / 2)
            else:
                penalize_conf = np.ones((17,))
            noise_penalize_conf = penalize_conf.copy()
            joints_vis = joints_vis * penalize_conf[:, None]
            # joints_vis = joints_vis * np.repeat(penalize_conf[:, None], 3, axis=1)
            
        center = np.array(np.random.rand(2,) * 250 + 250).copy()      # (2, )     (cx, cy)  in original image scale
        scale = np.array(np.random.rand(2,) + 2).copy()        # (2, )     (s1, s2) random number between 2 and 3
        rotation = 0
        
        if self.no_augmentation:
            center = np.array([self.image_size[0]/2, self.image_size[1]/2]).copy()      # (2, )     (cx, cy)  in original image scale
            scale = np.array([1, 1]).copy()        # (2, )     (s1, s2) random number between 2 and 3
            rotation = 0
        
        if self.inputs_normalized and self.normalize_cameras:
            cam_center_image = np.array([camera['cx'], camera['cy']])      # (2, )     (cx, cy)  in original image scale
            cam_center_image = self.normalize_screen_coordinates(cam_center_image, self.image_size[0], self.image_size[1])
            camera['cx'] = cam_center_image[0]
            camera['cy'] = cam_center_image[1]
            focal_length = np.array([camera['fx'], camera['fy']])      # (2, )     (fx, fy)  in original image scale
            focal_length = focal_length / self.image_size[0] * 2
            camera['fx'] = focal_length[0]
            camera['fy'] = focal_length[1]
            
            R = camera['R'].copy()
            K = np.array([
                [focal_length[0], 0, cam_center_image[0]],
                [0, focal_length[1], cam_center_image[1]],
                [0, 0, 1.],
            ])
        else:
            R = camera['R'].copy()
            K = np.array([
                [float(camera['fx']), 0, float(camera['cx'])],
                [0, float(camera['fy']), float(camera['cy'])],
                [0, 0, 1.],
            ])
        if self.use_t:
            T = camera['t'].copy()
        else:
            T = camera['T'].copy()
        
        Rt = np.zeros((3, 4))
        Rt[:, :3] = R
        Rt[:, 3] = -R @ T.squeeze()
        if self.use_t:
            cam_center = torch.Tensor(T.T)    # Tensor, (1, 3) camera center in world coordinate
        else:
            cam_center = torch.Tensor(camera['T'].T)    # Tensor, (1, 3) camera center in world coordinate
            
        if self.image_size[0] == 256:
            scale = scale * 4.0     # the images are hd, so we need to scale them down
        # scale = scale * 2.0     # the images are hd, so we need to scale them down
            
        # affine transformation matrix
        trans = get_affine_transform(center, scale, rotation, self.image_size)              # (2, 3)
        trans_inv = get_affine_transform(center, scale, rotation, self.image_size, inv=1)   # (2, 3)
        
        if self.no_augmentation:
            trans = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            trans_inv = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        
        

        cropK = np.concatenate((trans, np.array([[0., 0., 1.]])), 0).dot(K)     # augmented K (for 256 * 256)
        # cropK = K
        KRT = cropK.dot(Rt)                 # (3,4)    camera matrix (intrinsic & extrinsic)

        # preprocess image  (1000, 1000) - > (256, 256)
        # input = cv2.warpAffine(
        #     data_numpy,
        #     trans, (int(self.image_size[0]), int(self.image_size[1])),
        #     flags=cv2.INTER_LINEAR)             # numpy image

        # if self.transform:
        #     input = self.transform(input)

        # preprocess label  (1000, 1000) - > (256, 256)
        if self.clip_joints:
            if not self.no_augmentation:
                for i in range(self.num_joints):
                    joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)        # (17, 2) in (256, 256) scale
                    if (np.min(joints[i, :2]) < 0 or
                            joints[i, 0] >= self.image_size[0] or
                            joints[i, 1] >= self.image_size[1]):
                        joints_vis[i, :] = 0
            joints_vis[:, 0] = np.where(0 < joints[:, 0], joints_vis[:, 0], 0)
            joints_vis[:, 0] = np.where(joints[:, 0] < self.image_size[0] - 1, joints_vis[:, 0], 0)
            joints_vis[:, 0] = np.where(0 < joints[:, 1], joints_vis[:, 0], 0)
            joints_vis[:, 0] = np.where(joints[:, 1] < self.image_size[1] - 1, joints_vis[:, 0], 0)
            joints[:, 0] = np.clip(joints[:, 0], 0, self.image_size[0] - 1)
            joints[:, 1] = np.clip(joints[:, 1], 0, self.image_size[1] - 1)
        else:
            for i in range(self.num_joints):
                if joints_vis[i, 0] > 0.0:
                    if not self.no_augmentation:
                        joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)        # (17, 2) in (256, 256) scale
                    if (np.min(joints[i, :2]) < 0 or
                            joints[i, 0] >= self.image_size[0] or
                            joints[i, 1] >= self.image_size[1]):
                        joints_vis[i, :] = 0
                        joints[i, :] = 0        # to avoid further errors when downscaling
                else:
                    joints[i, :] = 0            # to avoid further errors when downscaling
            
        if self.APPLY_NOISE_MISSING and self.is_train:
            missing = np.random.uniform(0, 1, joints_vis.shape[0])
            mask = np.ones_like(joints_vis)
            mask[missing < self.MISSING_LEVEL] = 0
            joints_vis = joints_vis * mask
            joints = joints * mask[:, 0:1]
            
        # ========================= 3D Target ===========================
        # joints_3d = db_rec['joints_3d'].copy()
        # joints_3d_conf = db_rec['joints_3d_conf'].copy()
        # joints_3d_conf[joints_3d_conf < 0] = 0
        
        joints_3d_conf = np.ones((17, 1))
        joints_3d = torch.from_numpy(joints_3d).float()
        # joints_3d_conf = torch.from_numpy(joints_vis[:, 0]).float()
        joints_3d_conf = torch.from_numpy(joints_3d_conf).float()
        
        # # ========================== 2D joints scalar ====================================
        # if self.APPLY_NOISE and self.is_train:
        #     noise = np.random.normal(0, 1, joints.shape) * self.NOISE_LEVEL
        #     joints = joints + noise
           
        

        # ========================== 3D ray vectors ====================================
        # (256/down * 256/down, 3)
        if self.inputs_normalized:
            joints = self.normalize_screen_coordinates(joints, self.image_size[0], self.image_size[1])
            joints_org = self.normalize_screen_coordinates(joints_org, self.image_size[0], self.image_size[1])
            noise_vis = self.normalize_screen_coordinates(noise_vis, self.image_size[0], self.image_size[1])
            
        joints_ds = joints / self.downsample
        coords_ray = self.create_3d_ray_coords(camera, trans_inv, joints_ds)
        # coords_ray = coords_ray.reshape(int(np.sqrt(coords_ray.shape[0])), int(np.sqrt(coords_ray.shape[0])), 3)[joints_ds[:, 0].astype(int), joints_ds[:, 1].astype(int), :]  # (17, 3)
        # coords_ray = coords_ray.reshape(self.image_size[0] // self.downsample, self.image_size[1] // self.downsample, 3)[joints_ds[:, 0].astype(int), joints_ds[:, 1].astype(int), :]  # (17, 3)
        
            
        joints = np.concatenate([joints, joints_vis[:, 0:1]], axis=1)     # (17, 3) in (256, 256) scale

        joints = torch.from_numpy(joints).float()          # (17, 3) in (256, 256) scale    
        joints_org = torch.from_numpy(joints_org).float()          # (17, 2) 
        noise_vis = torch.from_numpy(noise_vis).float()          # (17, 2)
        noise_penalize_conf = torch.from_numpy(noise_penalize_conf).float()      
        # ==========================  Meta Info ==========================
        meta = {
            'scale': scale,
            'center': center,
            'rotation': rotation,
            'joints_2d': joints_2d,   # (17, 2) in origin image (1000, 1000)
            'joints_2d_transformed': joints,    # (17, 2) in input image (256, 256)
            'joints_vis': joints_vis,
            'source': 'amass',
            
            'cam_center': cam_center,   # (1, 3) in world coordinate
            'rays': coords_ray,         # (256/down * 256/down, 3)  in world coordinate
            'KRT': KRT,                 # (3, 4) for augmented image
            'K': K,
            'RT': Rt,

            'img-path': '',                      # str
            'subject_id': '',             # string 11
            'cam_id': camera_id,                # string 01
            'joints_2d_org': joints_org,     # (17, 2) in origin image (1000, 1000) without added noise
            'noise_vis': noise_vis,
            'noise_penalize_conf': noise_penalize_conf,
        }
        if self.APPLY_SMART_PSEUDO_TRAINING and self.is_train:
            meta.update({
                'joints_2d_conf': 0,
                'camera': camera, # original camera info
            })
        return joints, joints_3d, joints_3d_conf, meta
    
    # def getitem(self, idx, camera_id):
    #     db_rec = copy.deepcopy(self.db[idx])

    #     # ==================================== Image ====================================
    #     image_dir = 'images.zip@' if self.data_format == 'zip' else ''
    #     if db_rec['source'] == 'cmu_panoptic':
    #         image_file = osp.join(self.root_2, image_dir,
    #                           db_rec['image'])
    #     else:    
    #         image_file = osp.join(self.root_2, db_rec['source'], image_dir, 'images',
    #                             db_rec['image'])
    #     # print('image_file', image_file)
    #     if self.data_format == 'zip':
    #         from utils import zipreader
    #         data_numpy = zipreader.imread(
    #             image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    #     else:
    #         data_numpy = cv2.imread(
    #             image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        

    #     if db_rec['source'] == 'cmu_panoptic':
    #         pass
    #     else:        
    #         data_numpy = data_numpy[:1000]                  # According to ET

    #     # ==================================== Label ====================================
    #     joints_3d = db_rec['joints_3d'].copy() / 10            # (17, 3)   in world coordinate /10 is for converting mm to cm (for cmu panoptic)
    #     joints_3d = joints_3d - joints_3d[0] + self.cmu_center  # (17, 3)   in world coordinate
    #     camera = self.cmu_cameras[camera_id]
    #     joints = world_to_cam(joints_3d[None,:,:], camera['R'], camera['t'])  # (17, 3) in camera coordinate
    #     joints = cam_to_image(joints, camera['K'])[0]        # (17, 2) in original image scale (1000, 1000)
        
    #     # joints = db_rec['joints_2d'].copy()             # (17, 2)   in original image scale (1000, 1000)
    #     # print('joints', joints)
    #     if self.APPLY_NOISE and self.is_train:
    #         noise = np.random.normal(0, 1, joints.shape) * self.NOISE_LEVEL
    #         joints = joints + noise
    #     # raise
    #     joints_vis = db_rec['joints_vis'].copy()        # (17, 3)   0,0,0 or 1,1,1
    #     if self.APPLY_NOISE_MISSING and self.is_train:
    #         missing = np.random.uniform(0, 1, joints_vis.shape[0])
    #         mask = np.ones_like(joints_vis)
    #         mask[missing < self.MISSING_LEVEL] = 0
    #         joints_vis = joints_vis * mask

    #     center = np.array(db_rec['center']).copy()      # (2, )     (cx, cy)  in original image scale
    #     scale = np.array(db_rec['scale']).copy()        # (2, )     (s1, s2)
    #     rotation = 0

    #     # ==================================== Camera  ====================================
    #     # camera = 
    #     # camera matrix
    #     R = camera['R'].copy()
    #     K = np.array([
    #         [float(camera['fx']), 0, float(camera['cx'])],
    #         [0, float(camera['fy']), float(camera['cy'])],
    #         [0, 0, 1.],
    #     ])
    #     T = camera['T'].copy()
    #     Rt = np.zeros((3, 4))
    #     Rt[:, :3] = R
    #     Rt[:, 3] = -R @ T.squeeze()
    #     cam_center = torch.Tensor(camera['T'].T)    # Tensor, (1, 3) camera center in world coordinate

    #     # fix the system error of camera
    #     # distCoeffs = np.array(
    #     #     [float(i) for i in [camera['k'][0], camera['k'][1], camera['p'][0], camera['p'][1], camera['k'][2]]])
    #     # data_numpy = cv2.undistort(data_numpy, K, distCoeffs)
    #     # joints = cv2.undistortPoints(joints[:, None, :], K, distCoeffs, P=K).squeeze()
    #     # center = cv2.undistortPoints(np.array(center)[None, None, :], K, distCoeffs, P=K).squeeze()

    #     # ==================================== Preprocess ====================================
    #     # augmentation factor
    #     if self.is_train:
    #         sf = self.scale_factor
    #         rf = self.rotation_factor
    #         scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
    #         rotation = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
    #             if random.random() <= 0.6 else 0
    #     # if db_rec['source'] == 'cmu_panoptic':
    #     scale = scale * 4.0     # the images are hd, so we need to scale them down
            
    #     # affine transformation matrix
    #     trans = get_affine_transform(center, scale, rotation, self.image_size)              # (2, 3)
    #     trans_inv = get_affine_transform(center, scale, rotation, self.image_size, inv=1)   # (2, 3)

    #     cropK = np.concatenate((trans, np.array([[0., 0., 1.]])), 0).dot(K)     # augmented K (for 256 * 256)
    #     KRT = cropK.dot(Rt)                 # (3,4)    camera matrix (intrinsic & extrinsic)

    #     # preprocess image  (1000, 1000) - > (256, 256)
    #     input = cv2.warpAffine(
    #         data_numpy,
    #         trans, (int(self.image_size[0]), int(self.image_size[1])),
    #         flags=cv2.INTER_LINEAR)             # numpy image

    #     if self.transform:
    #         input = self.transform(input)

    #     # preprocess label  (1000, 1000) - > (256, 256)
    #     for i in range(self.num_joints):
    #         if joints_vis[i, 0] > 0.0:
    #             joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)        # (17, 2) in (256, 256) scale
    #             if (np.min(joints[i, :2]) < 0 or
    #                     joints[i, 0] >= self.image_size[0] or
    #                     joints[i, 1] >= self.image_size[1]):
    #                 joints_vis[i, :] = 0
    #                 joints[i, :] = 0        # to avoid further errors when downscaling
    #         else:
    #             joints[i, :] = 0            # to avoid further errors when downscaling

    #     # # ========================== heatmap ==========================
    #     # target, target_weight = self.generate_target(joints, joints_vis)

    #     # target = torch.from_numpy(target)                   # (17, 64, 64) heatmap
    #     # target_weight = torch.from_numpy(target_weight)
        
    #     # ========================= 3D Target ===========================
    #     # joints_3d = db_rec['joints_3d'].copy()
    #     # joints_3d_conf = db_rec['joints_3d_conf'].copy()
    #     # joints_3d_conf[joints_3d_conf < 0] = 0
        
    #     joints_3d_conf = np.ones((17, 1))
    #     joints_3d = torch.from_numpy(joints_3d).float()
    #     joints_3d_conf = torch.from_numpy(joints_3d_conf).float()
        
    #     # ========================== 2D joints scalar ====================================
    #     if self.APPLY_NOISE and self.is_train:
    #         noise = np.random.normal(0, 1, joints.shape) * self.NOISE_LEVEL
    #         joints = joints + noise
           
        

    #     # ========================== 3D ray vectors ====================================
    #     # (256/down * 256/down, 3)
    #     coords_ray = self.create_3d_ray_coords(camera, trans_inv)
    #     joints_ds = joints // downsample
    #     coords_ray = coords_ray.reshape(int(np.sqrt(coords_ray.shape[0])), int(np.sqrt(coords_ray.shape[0])), 3)[joints_ds[:, 0].astype(int), joints_ds[:, 1].astype(int), :]  # (17, 3)

    #     joints = torch.from_numpy(joints).float()          # (17, 2) in (256, 256) scale    
    #     # ==========================  Meta Info ==========================
    #     meta = {
    #         'scale': scale,
    #         'center': center,
    #         'rotation': rotation,
    #         'joints_2d': db_rec['joints_2d'],   # (17, 2) in origin image (1000, 1000)
    #         'joints_2d_transformed': joints,    # (17, 2) in input image (256, 256)
    #         'joints_vis': joints_vis,
    #         'source': db_rec['source'],
            
    #         'cam_center': cam_center,   # (1, 3) in world coordinate
    #         'rays': coords_ray,         # (256/down * 256/down, 3)  in world coordinate
    #         'KRT': KRT,                 # (3, 4) for augmented image
    #         'K': K,
    #         'RT': Rt,

    #         'img-path': image_file,                      # str
    #         'subject_id': db_rec['subject'],             # string 11
    #         'cam_id': db_rec['camera_id']                # string 01
    #     }
    #     if self.APPLY_SMART_PSEUDO_TRAINING and self.is_train:
    #         meta.update({
    #             'joints_2d_conf': db_rec['joints_2d_conf'] if 'joints_2d_conf' in db_rec.keys() else 0,
    #             'camera': camera, # original camera info
    #         })
    #     return joints, joints_3d, joints_3d_conf, meta

    def __len__(self):
        return self.group_size

    def get_key_str(self, datum):
        return 's_{:02}_pose_{}_imgid_{:08}'.format(
            datum['subject'], datum['pose_id'],
            datum['image_id'])
        
    def get_key_str_h36m(self, datum):
        return 's_{:02}_act_{:02}_subact_{:02}_imgid_{:06}'.format(
            datum['subject'], datum['action'], datum['subaction'],
            datum['image_id'])
        
    def isdamaged(self, db_rec):
        # from https://github.com/yihui-he/epipolar-transformers/blob/4da5cbca762aef6a89d37f889789f772b87d2688/data/datasets/joints_dataset.py#L174
        #damaged seq
        #'Greeting-2', 'SittingDown-2', 'Waiting-1'
        if db_rec['subject'] == 9:
            if db_rec['action'] != 5 or db_rec['subaction'] != 2:
                if db_rec['action'] != 10 or db_rec['subaction'] != 2:
                    if db_rec['action'] != 13 or db_rec['subaction'] != 1:
                        return False
        else:
            return False
        return True

    # def evaluate(self, pred, *args, **kwargs):
    #     pred = pred.copy()

    #     headsize = self.image_size[0] / 10.0
    #     threshold = 0.5

    #     u2a = self.u2a_mapping
    #     a2u = {v: k for k, v in u2a.items() if v != '*'}
    #     a = list(a2u.keys())
    #     u = list(a2u.values())
    #     indexes = list(range(len(a)))
    #     indexes.sort(key=a.__getitem__)
    #     sa = list(map(a.__getitem__, indexes))
    #     su = np.array(list(map(u.__getitem__, indexes)))    # [ 0  1  2  3  4  5  6  7  9 11 12 14 15 16 17 18 19]

    #     gt = []
    #     for items in self.grouping:
    #         for item in items:
    #             gt.append(self.db[item]['joints_3d'][su, :2])       # (17, 3) in original scale
    #     gt = np.array(gt)           # (num_sample, 17, 3) 
    #     pred = pred[:, su, :2]      # (num_sample, 17, 3) 
        
    #     distance_mm_per_keypoint = {kp: [] for _, kp in self.actual_joints.items()}
    #     distance = np.sqrt(np.sum((gt - pred)**2, axis=2))
        
    #     for i, kp in self.actual_joints.items():
    #         distance_mm_per_keypoint[kp] = distance[:, i]
        
    #     distance_mm_per_keypoint = {k: np.mean(v) for k, v in distance_mm_per_keypoint.items()}
    #     str_per_kp = "\n"
    #     for k, v in distance_mm_per_keypoint.items():
    #         str_per_kp = str_per_kp + k + '\t{}\n'.format(v)
    #     logger.info('3D MPJPE per keypoint: {}'.format(str_per_kp))
    #     # detected = (distance <= headsize * threshold)

    #     # joint_detection_rate = np.sum(detected, axis=0) / np.float(gt.shape[0])
    #     mpjpe = np.mean(distance, axis=0)

    #     name_values = collections.OrderedDict()
    #     joint_names = self.actual_joints
    #     for i in range(len(a2u)):
    #         name_values[joint_names[sa[i]]] = mpjpe[i]
    #     return name_values, np.mean(distance)

