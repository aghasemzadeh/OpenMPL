# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import copy
import random
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform, affine_transform_pts
from utils.utils_amass import rotate_pose
from utils.calib import cam_to_world, cam_to_image, world_to_cam
import collections
import logging
import pickle
import json

logger = logging.getLogger(__name__)

downsample = 16

# set random seed
# np.random.seed(0)

class JointsDataset_MPL(Dataset):

    def __init__(self, cfg, subset, is_train, transform=None):
        self.is_train = is_train
        self.subset = subset
        self.views = cfg.DATASET.TRAIN_VIEWS if is_train else cfg.DATASET.TEST_VIEWS
        if self.views is None:
            self.views = list(range(1, 5))
        self.n_views = len(self.views)
        
        self.run_on_all_cameras = cfg.DATASET.TRAIN_ON_ALL_CAMERAS if is_train else cfg.DATASET.TEST_ON_ALL_CAMERAS
        if self.run_on_all_cameras:
            self.n_views = cfg.DATASET.N_VIEWS_TRAIN_TEST_ALL
            
        self.all_views_cmu = list(range(31))
        self.all_views_cmu.remove(20)
        
        self.views_in_amass = [3, 6, 12, 13, 23]
        
        
        self.use_mmpose = False
        if cfg.DATASET.USE_MMPOSE_TRAIN and is_train:
            self.use_mmpose = True
        elif (cfg.DATASET.USE_MMPOSE_VAL or cfg.DATASET.USE_MMPOSE_VAL) and not is_train:
            self.use_mmpose = True
            
        self.mmpose_type = cfg.DATASET.TRAIN_MMPOSE_TYPE if is_train else cfg.DATASET.TEST_MMPOSE_TYPE
        
        self.dataset_type = cfg.DATASET.DATASET_TYPE
        
        self.h36m_old_datasets = cfg.DATASET.TRAIN_USE_H36M_OLD_DATASETS if is_train else cfg.DATASET.TEST_USE_H36M_OLD_DATASETS
        self.h36m_dataset_name = cfg.DATASET.TRAIN_H36M_DATASET_NAME if is_train else cfg.DATASET.TEST_H36M_DATASET_NAME
        if is_train:
            self.filter_groupings = cfg.DATASET.FILTER_GROUPINGS and cfg.DATASET.TRAIN_FILTER_GROUPINGS
        else:
            self.filter_groupings = cfg.DATASET.FILTER_GROUPINGS and cfg.DATASET.TEST_FILTER_GROUPINGS
        
            
        self.use_3d_triangulated_mmpose = cfg.DATASET.MIX_3D_AMASS_WITH_TRIANGULATED_MMPOSE_TRAIN if is_train else cfg.DATASET.USE_3D_TRIANGULATED_MMPOSE_TEST
        self.mix_3d_amass_with_triangulated_mmpose = cfg.DATASET.MIX_3D_AMASS_WITH_TRIANGULATED_MMPOSE_TRAIN if is_train else cfg.DATASET.MIX_3D_AMASS_WITH_TRIANGULATED_MMPOSE_TEST
        self.mix_smart_3d_amass_with_triangulated_mmpose = cfg.DATASET.MIX_SMART_3D_AMASS_WITH_TRIANGULATED_MMPOSE_TRAIN if is_train else cfg.DATASET.MIX_SMART_3D_AMASS_WITH_TRIANGULATED_MMPOSE_TEST
        self.epipolar_error_acceptance_threshold = cfg.DATASET.EPIPOLAR_ERROR_ACCEPTANCE_THRESHOLD
        self.keypoints_to_mix_amass_with_3d_triangulated_mmpose = cfg.DATASET.KEYPOINTS_TO_MIX_AMASS_WITH_3D_TRIANGULATED_MMPOSE
        self.filter_cmu_wrong_cases = cfg.DATASET.TRAIN_FILTER_CMU_WRONG_CASES if is_train else cfg.DATASET.TEST_FILTER_CMU_WRONG_CASES
            
        self.mix_gt_with_mmpose = cfg.DATASET.MIX_GT_WITH_MMPOSE_WHEN_USE_MMPOSE
        self.mix_amass_with_mmpose = cfg.DATASET.MIX_AMASS_WITH_MMPOSE_WHEN_USE_MMPOSE or self.mix_gt_with_mmpose
        self.keypoints_to_mix = cfg.DATASET.KEYPOINTS_TO_MIX
        
        self.target_normalized_3d = cfg.DATASET.TARGET_NORMALIZED_3D
        
        self.room_min_x = cfg.DATASET.ROOM_MIN_X
        self.room_max_x = cfg.DATASET.ROOM_MAX_X
        self.room_min_y = cfg.DATASET.ROOM_MIN_Y
        self.room_max_y = cfg.DATASET.ROOM_MAX_Y
        self.room_center = cfg.DATASET.ROOM_CENTER
        
        self.root = cfg.DATASET.ROOT
        self.root_2 = cfg.DATASET.ROOT_2DATSET
        self.root_3 = cfg.DATASET.ROOT_3DATSET
        self.dir_mpl_data = cfg.DATASET.DIR_MPL_DATA
        if cfg.DATASET.ROOT_TRAIN is not None and is_train:
            self.root = cfg.DATASET.ROOT_TRAIN
        if cfg.DATASET.ROOT_TEST is not None and not is_train:
            self.root = cfg.DATASET.ROOT_TEST
        self.data_format = cfg.DATASET.DATA_FORMAT
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.image_size = cfg.NETWORK.IMAGE_SIZE
        self.heatmap_size = cfg.NETWORK.HEATMAP_SIZE
        self.sigma = cfg.NETWORK.SIGMA
        self.transform = transform
        self.db = []

        self.APPLY_NOISE = cfg.DATASET.APPLY_NOISE if is_train else cfg.DATASET.APPLY_NOISE_TEST
        self.NOISE_LEVEL = cfg.DATASET.NOISE_LEVEL
        
        self.APPLY_NOISE_MISSING = cfg.DATASET.APPLY_NOISE_MISSING
        self.MISSING_LEVEL = cfg.DATASET.MISSING_LEVEL
        
        self.apply_noise_cameras = cfg.DATASET.APPLY_NOISE_CAMERAS
        self.R_noise_value = cfg.DATASET.R_NOISE_VALUE
        self.t_noise_value = cfg.DATASET.T_NOISE_VALUE
        
        self.APPLY_SMART_PSEUDO_TRAINING = cfg.TRAIN.SMART_PSEUDO_TRAINING if is_train else False
        self.epipolar_error_threshold = cfg.TRAIN.EPIPOLAR_ERROR_THRESHOLD
        self.downsample = cfg.DOWNSAMPLE
        
        self.centeralize_root_first = cfg.DATASET.CENTERALIZE_ROOT_FIRST
        self.no_augmentation_3d = cfg.DATASET.NO_AUGMENTATION_3D
        
        self.output_in_meter = cfg.DATASET.OUTPUT_IN_METER
        self.no_augmentation = cfg.DATASET.NO_AUGMENTATION
        self.clip_joints = cfg.DATASET.CLIP_JOINTS
        self.inputs_normalized = cfg.DATASET.INPUTS_NORMALIZED
        self.normalize_cameras = cfg.DATASET.NORMALIZE_CAMERAS
        self.normalize_room = cfg.DATASET.NORMALIZE_ROOM
        self.flip = cfg.DATASET.FLIP_3D
        self.rotate = cfg.DATASET.ROTATE_3D
        
        self.use_grid = True if cfg.DATASET.USE_GRID else False
        self.bug_test = cfg.DATASET.BUG_TEST_3D_EMB
        self.switch_x_z = cfg.DATASET.SWITCH_X_Z
        self.switch_x_y = cfg.DATASET.SWITCH_X_Y
        self.switch_y_z = cfg.DATASET.SWITCH_Y_Z
        self.switch_z_x_y = cfg.DATASET.SWITCH_Z_X_Y
        self.switch_y_z_x = cfg.DATASET.SWITCH_Y_Z_X
        self.amass_data_no_axis_swap = cfg.DATASET.AMASS_DATA_NO_AXIS_SWAP
        self.use_t = cfg.DATASET.USE_T
        
        self.amass_val_located = cfg.DATASET.AMASS_VAL_LOCATED
        
        self.CMU_KEYPOINT_STANDARD = cfg.DATASET.CMU_KEYPOINT_STANDARD
        
        self.cmu_calib = cfg.DATASET.CMU_CALIB
        self.cmu_calibs_train = cfg.DATASET.TRAIN_CMU_CALIB
        self.cmu_calibs_val = cfg.DATASET.TEST_CMU_CALIB
        self.h36m_calib_actors = cfg.DATASET.TRAIN_H36M_CALIB_ACTORS if is_train else cfg.DATASET.TEST_H36M_CALIB_ACTORS
        
        self.use_helper_cameras = cfg.DATASET.USE_HELPER_CAMERAS
        self.views_helper = cfg.DATASET.TRAIN_VIEWS_HELPER
        if self.views_helper is not None:
            self.n_views_helper = len(self.views_helper)
        self.root_views_helper = cfg.DATASET.ROOT_VIEWS_HELPER
        
        self.train_n_samples = cfg.DATASET.TRAIN_N_SAMPLES
        self.test_n_samples = cfg.DATASET.TEST_N_SAMPLES
        
        self.penalize_confidence = cfg.DATASET.PENALIZE_CONFIDENCE
        self.penalize_confidence_a = cfg.DATASET.PENALIZE_CONFIDENCE_A
        self.penalize_confidence_b = cfg.DATASET.PENALIZE_CONFIDENCE_B
        self.penalize_factor = cfg.DATASET.PENALIZE_CONFIDENCE_FACTOR
        self.only_keep_inside_room = cfg.DATASET.ONLY_KEEP_INSIDE_ROOM
        self.only_keep_if_in_calibs_actors = cfg.DATASET.ONLY_KEEP_IF_IN_CALIBS_ACTORS
        
        self.train_on_all_amass = cfg.DATASET.TRAIN_ON_ALL_AMASS
        
        self.place_person_in_center = cfg.DATASET.TRAIN_PLACE_PERSON_IN_CENTER if is_train else cfg.DATASET.TEST_PLACE_PERSON_IN_CENTER
        self.bring_amass_root_to_room_center = cfg.DATASET.BRING_AMASS_ROOT_TO_ROOM_CENTER
        
        self.intrinsic_to_meters = cfg.DATASET.INTRINSIC_TO_METERS_IF_OUTPUT_IN_METER
        
        self.use_h36m_cameras_on_cmu = cfg.DATASET.USE_H36M_CAMERAS_ON_CMU
        self.use_cmu_cameras_on_cmu = cfg.DATASET.USE_CMU_CAMERAS_ON_CMU
        self.use_cmu_cameras_on_h36m = cfg.DATASET.USE_CMU_CAMERAS_ON_H36M
        
        self.use_amass_old_datasets = cfg.DATASET.TRAIN_USE_AMASS_OLD_DATASETS if is_train else cfg.DATASET.TEST_USE_AMASS_OLD_DATASETS
        self.use_amass_new_datasets_with_old_way = cfg.DATASET.TRAIN_USE_AMASS_NEW_DATASETS_WITH_OLD_WAY if is_train else cfg.DATASET.TEST_USE_AMASS_NEW_DATASETS_WITH_OLD_WAY
        self.flip_lower_body_kp = cfg.DATASET.FLIP_LOWER_BODY_KP_TEST if not is_train else False
        self.num_joints = 17
        union_joints = {
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'thorax',
            9: 'neck',
            10: 'upper neck',
            11: 'nose',
            12: 'head',
            13: 'head top',
            14: 'lsho',
            15: 'lelb',
            16: 'lwri',
            17: 'rsho',
            18: 'relb',
            19: 'rwri'
        }

        self.union_joints = {
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
        
        self.close_joints = {
            'rhip': 'lhip',
            'lkne': 'rkne',
            'rank': 'lank',
            'rsho': 'lsho',
            'relb': 'lelb',
            'nose': 'head',
        }
        
        # self.amass_normalization_const = {
        #     'means': [-0.40025026059636765, -0.9625251337422954, -0.5074277076365897],
        #     'vars': [0.24167735219571976, 0.20952544156186956, 0.3594625811074876],
        # }
        
        self.actual_joints = {}
        self.u2a_mapping = {}

        # grid coordinate. For

        _y, _x = torch.meshgrid(torch.arange(self.image_size[0] // self.downsample),
                                torch.arange(self.image_size[1] // self.downsample))
        grid = torch.stack([_x, _y], dim=-1)  # Tensor, size:(32, 32, 2) val: 0-32
        grid = grid * self.downsample + self.downsample / 2.0 - 0.5  # Tensor, size:(32, 32, 2), val: 0-256
        self.grid = grid.view(-1, 2)  # Tensor, size:(hw, 2), val: 0-256

    def get_mapping(self):
        union_keys = list(self.union_joints.keys())
        union_values = list(self.union_joints.values())

        mapping = {k: '*' for k in union_keys}
        for k, v in self.actual_joints.items():
            idx = union_values.index(v)
            key = union_keys[idx]
            mapping[key] = k
        return mapping
    
    def compatible_cams(self, cameras):
        for k,cam in cameras.items():    
            cam['K'] = np.matrix(cam['K'])
            cam['distCoef'] = np.array(cam['distCoef'])
            cam['k'] = np.array([cam['distCoef'][0], cam['distCoef'][1], cam['distCoef'][4]])
            cam['p'] = np.array([cam['distCoef'][2], cam['distCoef'][3]])
            cam['R'] = np.matrix(cam['R'])
            cam['t'] = np.array(cam['t']).reshape((3,1))
            cam['T'] = np.array(-cam['R'].T @ np.array(cam['t']).reshape((3,1)))
            cam['fx'] = cam['K'][0,0]
            cam['fy'] = cam['K'][1,1]
            cam['cx'] = cam['K'][0,2]
            cam['cy'] = cam['K'][1,2]
        return cameras

    def load_amass_new(self, anno_file):
        with open(anno_file, 'rb') as f:
            db = pickle.load(f)
        if 'triangulated_3d_mmpose' in db.keys():
            if db['triangulated_3d_mmpose'] is None:
                del db['triangulated_3d_mmpose']
        if self.use_mmpose:
            joints_2d_mmpose = db['joints_2d_mmpose']
            nan_indices = np.argwhere(np.isnan(joints_2d_mmpose))
            to_remove = np.ones((joints_2d_mmpose.shape[0],), dtype=bool)
            to_remove[nan_indices[:, 0]] = False
            for k in db.keys():
                db[k] = db[k][to_remove]
        db_2d = {}
        for k in db.keys():
            if k != 'joints_3d':
                db_2d[k] = db[k]
        if self.mix_smart_3d_amass_with_triangulated_mmpose:
            db = db['joints_3d']
        elif self.mix_3d_amass_with_triangulated_mmpose:
            db = db['joints_3d']
            db[:, self.keypoints_to_mix_amass_with_3d_triangulated_mmpose] = db_2d['triangulated_3d_mmpose'][:, self.keypoints_to_mix_amass_with_3d_triangulated_mmpose]
        elif self.use_3d_triangulated_mmpose:
            db = db_2d['triangulated_3d_mmpose']
        else:
            db = db['joints_3d']
        
        try:
            views_used = db_2d['views_used']
            self.views_in_amass = list(views_used[0])
        except:
            pass
        return db, db_2d
    
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
            calib_file = osp.join(self.root_3, cam_setup, 'calibration_{}.json'.format(cam_setup))
            with open(calib_file, 'r') as f:
                calibration_cat = json.load(f)
            cameras = {cam['node']:cam for cam in calibration_cat['cameras'] if cam['type']=='hd'}
            cameras = self.compatible_cams(cameras)
            for cam_id, cam in cameras.items():
                cameras_all_calibs[cam_id].append(cam)
        return cameras_all_calibs
    
    def do_mapping(self):
        mapping = self.u2a_mapping
        for item in self.db:
            joints = item['joints_2d']
            joints_vis = item['joints_vis']

            njoints = len(mapping)
            joints_union = np.zeros(shape=(njoints, 2))
            joints_union_vis = np.zeros(shape=(njoints, 3))

            for i in range(njoints):
                if mapping[i] != '*':
                    index = int(mapping[i])
                    joints_union[i] = joints[index]
                    joints_union_vis[i] = joints_vis[index]
            item['joints_2d'] = joints_union
            item['joints_vis'] = joints_union_vis

    def _get_db(self):
        raise NotImplementedError
    
    def filter_db(self, db):
        if self.use_h36m_cameras_on_cmu:
            views = [3, 6, 12, 13, 23]
            views = views[:self.n_views]
        else:
            views = self.views
        db_filtered = []
        for i in db:
            if i['camera_id'] not in views:
                continue
            else:
                db_filtered.append(i)
        return db_filtered

    def evaluate(self, pred, *args, **kwargs):
        pred = pred.copy()

        headsize = self.image_size[0] / 10.0
        threshold = 0.5

        u2a = self.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = list(a2u.values())
        indexes = list(range(len(a)))
        indexes.sort(key=a.__getitem__)
        sa = list(map(a.__getitem__, indexes))
        su = np.array(list(map(u.__getitem__, indexes)))    # [ 0  1  2  3  4  5  6  7  9 11 12 14 15 16 17 18 19]

        gt = []
        for items in self.grouping:
            # for item in items:
            item = items[0]
            gt.append(self.db[item]['joints_3d'][su, :])       # (17, 3) in original scale
        gt = np.array(gt)           # (num_sample, 17, 3) 
        pred = pred[:, su, :]      # (num_sample, 17, 3) 
        # if self.relative_evaluation:
        #     gt = gt - gt[:, 0:1, :]    # (num_sample, 17, 3)
        #     pred = pred - pred[:, 0:1, :]  # (num_sample, 17, 3)
        distance_mm_per_keypoint = {kp: [] for _, kp in self.actual_joints.items()}
        distance = np.sqrt(np.sum((gt - pred)**2, axis=2))
        
        for i, kp in self.actual_joints.items():
            distance_mm_per_keypoint[kp] = distance[:, i]
        
        distance_mm_per_keypoint = {k: np.mean(v) for k, v in distance_mm_per_keypoint.items()}
        str_per_kp = "\n"
        for k, v in distance_mm_per_keypoint.items():
            str_per_kp = str_per_kp + k + '\t{}\n'.format(v)
        logger.info('3D MPJPE per keypoint: {}'.format(str_per_kp))
        # detected = (distance <= headsize * threshold)

        # joint_detection_rate = np.sum(detected, axis=0) / np.float(gt.shape[0])
        mpjpe = np.mean(distance, axis=0)

        name_values = collections.OrderedDict()
        joint_names = self.actual_joints
        for i in range(len(a2u)):
            name_values[joint_names[sa[i]]] = mpjpe[i]
        return name_values, np.mean(distance)

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx, camera_id=None, camera_setup_to_use=None):
        db_rec = copy.deepcopy(self.db[idx])

        # ==================================== Image ====================================
        image_dir = 'images.zip@' if self.data_format == 'zip' else ''
        if db_rec['source'] == 'cmu_panoptic':
            image_file = osp.join(self.root, image_dir,
                              db_rec['image'])
        else:    
            image_file = osp.join(self.root, db_rec['source'], image_dir, 'images',
                                db_rec['image'])
        # print('image_file', image_file)
        # if self.data_format == 'zip':
        #     from utils import zipreader
        #     data_numpy = zipreader.imread(
        #         image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # else:
        #     data_numpy = cv2.imread(
        #         image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        

        # if db_rec['source'] == 'cmu_panoptic':
        #     pass
        # else:        
        #     data_numpy = data_numpy[:1000]                  # According to ET

        # ==================================== Label ====================================
        joints = db_rec['joints_2d'].copy()             # (17, 2)   in original image scale (1000, 1000)
        if self.flip_lower_body_kp:
            joints_copy = joints.copy()
            joints[4:7] = joints_copy[1:4]
            joints[1:4] = joints_copy[4:7]
        # print('joints', joints)
        # if self.APPLY_NOISE and self.is_train:
        #     noise = np.random.normal(0, 1, joints.shape) * self.NOISE_LEVEL
        #     joints = joints + noise
        # raise
        if 'joints_2d_conf' in db_rec:
            joints_vis = db_rec['joints_2d_conf'].copy()        # (17, 3)   0,0,0 or 1,1,1
            if self.flip_lower_body_kp:
                joints_vis_copy = joints_vis.copy()
                joints_vis[4:7] = joints_vis_copy[1:4]
                joints_vis[1:4] = joints_vis_copy[4:7]
            
            # if db_rec['source'] == 'h36m':  # as mmpose is not consistent with the original h36m, we need to fix it. here we reduce the confidence of root, rhip, and lhip to take this difference into account
            #     ROOT = self.keypoints_inverse_dict['root']
            #     LHIP = self.keypoints_inverse_dict['lhip']
            #     RHIP = self.keypoints_inverse_dict['rhip']
        else:
            joints_vis = np.ones((17, 1))
        

        center = np.array(db_rec['center']).copy()      # (2, )     (cx, cy)  in original image scale
        scale = np.array(db_rec['scale']).copy()        # (2, )     (s1, s2)
        rotation = 0

        # ==================================== Camera  ====================================
        if self.use_h36m_cameras_on_cmu:
            camera = self.h36m_cameras[camera_id][camera_setup_to_use].copy()
            if self.output_in_meter:
                camera['T'] = camera['T'] / 1000
                camera['t'] = camera['t'] / 1000
                
        elif self.use_cmu_cameras_on_h36m or self.use_cmu_cameras_on_cmu:
            camera = self.cmu_cameras[camera_id][camera_setup_to_use].copy()
            if self.output_in_meter:
                camera['T'] = camera['T'] / 100
                camera['t'] = camera['t'] / 100
        else:
            camera = db_rec['camera'].copy()  
        # ========================= 3D Target ===========================
        if db_rec['source'] == 'cmu_panoptic':
            joints_3d = db_rec['joints_3d'].copy()      # (17, 3) in world coordinate
            if self.flip_lower_body_kp:
                joints_3d_copy = joints_3d.copy()
                joints_3d[4:7] = joints_3d_copy[1:4]
                joints_3d[1:4] = joints_3d_copy[4:7]
            if self.place_person_in_center: # it is for debug purpose
                if self.use_h36m_cameras_on_cmu:
                    raise ValueError('Not implemented')
                joints_3d[:, 0] = joints_3d[:, 0] - joints_3d[0, 0]
                joints_3d[:, 2] = joints_3d[:, 2] - joints_3d[0, 2]
                joints_3d_camera = world_to_cam(joints_3d[None], camera['R'], camera['t']).squeeze()  # (17, 3) in camera coordinate
                joints_2d = cam_to_image(joints_3d_camera[None], camera['K']).squeeze()  # (17, 2) in image coordinate
                joints = joints_2d[:, :2]
            if self.output_in_meter:
                joints_3d = joints_3d / 100
            if self.use_h36m_cameras_on_cmu:
                # swap y and z
                joints_3d_copy = joints_3d.copy()
                joints_3d[:, 1] = joints_3d[:, 2]   # swap y and z
                joints_3d[:, 2] = -joints_3d_copy[:, 1]  # swap y and z
                joints_3d_camera = world_to_cam(joints_3d[None], camera['R'], camera['t']).squeeze()  # (17, 3) in camera coordinate
                joints_2d = cam_to_image(joints_3d_camera[None], camera['K']).squeeze()  # (17, 2) in image coordinate
                joints = joints_2d[:, :2]
            if self.use_cmu_cameras_on_cmu:
                joints_3d_camera = world_to_cam(joints_3d[None], camera['R'], camera['t']).squeeze()  # (17, 3) in camera coordinate
                joints_2d = cam_to_image(joints_3d_camera[None], camera['K']).squeeze()  # (17, 2) in image coordinate
                joints = joints_2d[:, :2]
        elif db_rec['source'] == 'h36m':
            # Attention: the world joints of h36m are defected --> use camera coords and transform them to world
            joints_3d_camera = db_rec['joints_3d_camera'].copy()      # (17, 3) in camera coordinate
            joints_3d = cam_to_world(joints_3d_camera[None], camera['R'], camera['t']).squeeze()  # (17, 3) in world coordinate
            if self.flip_lower_body_kp:
                joints_3d_copy = joints_3d.copy()
                joints_3d[4:7] = joints_3d_copy[1:4]
                joints_3d[1:4] = joints_3d_copy[4:7]
            # bring the center of the world coordinate to the center of the room
            if self.place_person_in_center: # it is for debug purpose
                joints_3d[:, 0] = joints_3d[:, 0] - joints_3d[0, 0]
                joints_3d[:, 1] = joints_3d[:, 1] - joints_3d[0, 1]
                joints_3d_camera = world_to_cam(joints_3d[None], camera['R'], camera['t']).squeeze()  # (17, 3) in camera coordinate
                joints_2d = cam_to_image(joints_3d_camera[None], camera['K']).squeeze()  # (17, 2) in image coordinate
                joints = joints_2d[:, :2]
            if self.output_in_meter:
                joints_3d = joints_3d / 1000
                
            if self.use_cmu_cameras_on_h36m:
                joints_3d_copy = joints_3d.copy()
                joints_3d[:, 1] = -joints_3d[:, 2]   # swap y and z
                joints_3d[:, 2] = joints_3d_copy[:, 1]  # swap y and z
                joints_3d_camera = world_to_cam(joints_3d[None], camera['R'], camera['t']).squeeze()  # (17, 3) in camera coordinate
                joints_2d = cam_to_image(joints_3d_camera[None], camera['K']).squeeze()  # (17, 2) in image coordinate
                joints = joints_2d[:, :2]
            # joints_3d[:, 0] = joints_3d[:, 0] - self.room_center[0]
            # joints_3d[:, 2] = joints_3d[:, 2] - self.room_center[2]
                    
        if self.output_in_meter:
            if db_rec['source'] == 'cmu_panoptic':
                camera['T'] = camera['T'] / 100
                camera['t'] = camera['t'] / 100
            else:
                camera['T'] = camera['T'] / 1000
                camera['t'] = camera['t'] / 1000
                # if self.intrinsic_to_meters:
                #     camera['K'] = camera['K'] / 1000
                #     camera['fx'] = camera['fx'] / 1000
                #     camera['fy'] = camera['fy'] / 1000
                #     camera['cx'] = camera['cx'] / 1000
                #     camera['cy'] = camera['cy'] / 1000
                # bring the center of the world coordinate to the center of the room
                # camera['T'][0] = camera['T'][0] - self.room_center[0]
                # camera['T'][2] = camera['T'][2] - self.room_center[2]
                # camera['t'] = - np.linalg.inv(camera['R'].T) @ camera['T']
        # camera matrix
        joints_org = joints.copy()
        noise_vis = np.zeros((17, 2))
        noise_penalize_conf = np.ones((17, 1))
        # ========================== 2D joints scalar ====================================
        if self.APPLY_NOISE:
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
        # cam_center = torch.Tensor(camera['T'].T)    # Tensor, (1, 3) camera center in world coordinate

        # fix the system error of camera
        # distCoeffs = np.array(
        #     [float(i) for i in [camera['k'][0], camera['k'][1], camera['p'][0], camera['p'][1], camera['k'][2]]])
        # data_numpy = cv2.undistort(data_numpy, K, distCoeffs)
        # joints = cv2.undistortPoints(joints[:, None, :], K, distCoeffs, P=K).squeeze()
        # center = cv2.undistortPoints(np.array(center)[None, None, :], K, distCoeffs, P=K).squeeze()

        # ==================================== Preprocess ====================================
        # augmentation factor
        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            rotation = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0
        if db_rec['source'] == 'cmu_panoptic' and self.image_size[0] == 256:
            scale = scale * 4.0     # the images are hd, so we need to scale them down
            
        # affine transformation matrix
        trans = get_affine_transform(center, scale, rotation, self.image_size)              # (2, 3)
        trans_inv = get_affine_transform(center, scale, rotation, self.image_size, inv=1)   # (2, 3)
        
        if self.no_augmentation:
            trans = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            trans_inv = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        cropK = np.concatenate((trans, np.array([[0., 0., 1.]])), 0).dot(K)     # augmented K (for 256 * 256)
        KRT = cropK.dot(Rt)                 # (3,4)    camera matrix (intrinsic & extrinsic)

        # preprocess image  (1000, 1000) - > (256, 256)
        # input = cv2.warpAffine(
        #     data_numpy,
        #     trans, (int(self.image_size[0]), int(self.image_size[1])),
        #     flags=cv2.INTER_LINEAR)             # numpy image

        # if self.transform:
        #     input = self.transform(input)

        # preprocess label  (1000, 1000) - > (256, 256)
        # for i in range(self.num_joints):
        #     if joints_vis[i, 0] > 0.0:
        #         joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)        # (17, 2) in (256, 256) scale
        #         if (np.min(joints[i, :2]) < 0 or
        #                 joints[i, 0] >= self.image_size[0] or
        #                 joints[i, 1] >= self.image_size[1]):
        #             joints_vis[i, :] = 0
        #             joints[i, :] = 0        # to avoid further errors when downscaling
        #     else:
        #         joints[i, :] = 0            # to avoid further errors when downscaling
        
        if self.clip_joints:
            if not self.no_augmentation:
                for i in range(self.num_joints):
                    joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)        # (17, 2) in (256, 256) scale
                    if (np.min(joints[i, :2]) < 0 or
                            joints[i, 0] >= self.image_size[0] or
                            joints[i, 1] >= self.image_size[1]):
                        joints_vis[i, :] = 0
            # penalize confidence if kp goes out of image
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

        # # ========================== heatmap ==========================
        # target, target_weight = self.generate_target(joints, joints_vis)

        # target = torch.from_numpy(target)                   # (17, 64, 64) heatmap
        # target_weight = torch.from_numpy(target_weight)
        
        if self.APPLY_NOISE_MISSING and self.is_train:
            missing = np.random.uniform(0, 1, joints_vis.shape[0])
            mask = np.ones_like(joints_vis)
            mask[missing < self.MISSING_LEVEL] = 0
            joints_vis = joints_vis * mask
            joints = joints * mask[:, 0:1]
        

        if db_rec['source'] == 'cmu_panoptic':
            joints_3d_conf = db_rec['joints_3d_conf'].copy()
            if self.flip_lower_body_kp:
                joints_3d_conf_copy = joints_3d_conf.copy()
                joints_3d_conf[4:7] = joints_3d_conf_copy[1:4]
                joints_3d_conf[1:4] = joints_3d_conf_copy[4:7]
        else:
            joints_3d_conf = np.ones_like(joints_3d[:, 0])
        joints_3d_conf[joints_3d_conf < 0] = 0
        
        joints_3d = torch.from_numpy(joints_3d).float()
        joints_3d_conf = torch.from_numpy(joints_3d_conf).float()
        
        
           
        

        # ========================== 3D ray vectors ====================================
        # (256/down * 256/down, 3)
        if self.inputs_normalized:
            joints = self.normalize_screen_coordinates(joints, self.image_size[0], self.image_size[1])
            joints_org = self.normalize_screen_coordinates(joints_org, self.image_size[0], self.image_size[1])
            noise_vis = self.normalize_screen_coordinates(noise_vis, self.image_size[0], self.image_size[1])
            
        joints_ds = joints / self.downsample
        coords_ray = self.create_3d_ray_coords(camera, trans_inv, joints_ds=joints_ds)
        # coords_ray = coords_ray.reshape(int(np.sqrt(coords_ray.shape[0])), int(np.sqrt(coords_ray.shape[0])), 3)[joints_ds[:, 0].astype(int), joints_ds[:, 1].astype(int), :]  # (17, 3)

            
        joints = np.concatenate([joints, joints_vis[:, 0:1]], axis=1)     # (17, 3) in (256, 256) scale
        
        joints = torch.from_numpy(joints).float()          # (17, 2) in (256, 256) scale   
        joints_org = torch.from_numpy(joints_org).float()          # (17, 2) 
        noise_vis = torch.from_numpy(noise_vis).float()          # (17, 2)
        noise_penalize_conf = torch.from_numpy(noise_penalize_conf).float()      
        # joints_3d_conf = db_rec['joints_3d_conf'].copy().squeeze()
        # joints_3d_conf[joints_3d_conf < 0, ] = 0
        # joints_3d_conf = torch.from_numpy(joints_3d_conf).float()
        # ==========================  Meta Info ==========================
        meta = {
            'scale': scale,
            'center': center,
            'rotation': rotation,
            'joints_2d': db_rec['joints_2d'],   # (17, 2) in origin image (1000, 1000)
            'joints_2d_transformed': joints,    # (17, 2) in input image (256, 256)
            'joints_vis': joints_vis,
            'joints_3d_conf': joints_3d_conf, # (17, 1) confidence of 3d GT joints
            'source': db_rec['source'],
            
            'cam_center': cam_center,   # (1, 3) in world coordinate
            'rays': coords_ray,         # (256/down * 256/down, 3)  in world coordinate
            'KRT': KRT,                 # (3, 4) for augmented image
            'K': K,
            'RT': Rt,

            'img-path': image_file,                      # str
            'subject_id': db_rec['subject'],             # string 11
            'cam_id': db_rec['camera_id'],                # string 01
            'image': db_rec['image'],  
            'joints_2d_org': joints_org,     # (17, 2) in origin image (1000, 1000) without added noise
            'noise_vis': noise_vis,
            'noise_penalize_conf': noise_penalize_conf,                 
        }
        if self.APPLY_SMART_PSEUDO_TRAINING and self.is_train:
            meta.update({
                'joints_2d_conf': db_rec['joints_2d_conf'] if 'joints_2d_conf' in db_rec.keys() else 0,
                'camera': camera, # original camera info
            })
        return joints, joints_3d, joints_3d_conf, meta

    def generate_target(self, joints_3d, joints_vis):
        target, weight = self.generate_heatmap(joints_3d, joints_vis)
        return target, weight
    
    def normalize_screen_coordinates(self, X, w, h): 
        assert X.shape[-1] == 2
        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        return (X/w)*2 - [1, h/w]
    
    def normalize_pose3d_coordinates(self, X, center, x_width, y_width): 
        assert X.shape[-1] == 3
        w = max(x_width, y_width)
        return (X - center) / w
        

    def generate_heatmap(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros(
            (self.num_joints, self.heatmap_size[1], self.heatmap_size[0]),
            dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            feat_stride = self.image_size / self.heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0
                continue

            size = 2 * tmp_size + 1     # 13
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * self.sigma**2))

            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight

    def create_3d_ray_coords(self, camera, trans_inv, joints_ds):
        multiplier = 1.0                        # avoid numerical instability
        if self.downsample != 1 and self.use_grid:
            grid = self.grid.clone()                # Tensor,   (hw, 2), val in 0-256
            grid = grid.reshape(self.image_size[1] // self.downsample, self.image_size[0] // self.downsample, 2)[joints_ds[:, 0].astype(int), joints_ds[:, 1].astype(int), :]  # (17, 2)
            # transform to original image R.T.dot(x.T) + T
            coords = affine_transform_pts(grid.numpy(), trans_inv)  # array, size: (hw, 2), val: 0-1000
        else:
            coords = joints_ds

        if np.isscalar(camera['fx']):
            coords[:, 0] = (coords[:, 0] - camera['cx']) / camera['fx'] * multiplier
            coords[:, 1] = (coords[:, 1] - camera['cy']) / camera['fy'] * multiplier
        elif camera['fx'].shape == ():
            coords[:, 0] = (coords[:, 0] - camera['cx']) / camera['fx'] * multiplier
            coords[:, 1] = (coords[:, 1] - camera['cy']) / camera['fy'] * multiplier
        else:
            coords[:, 0] = (coords[:, 0] - camera['cx'][0]) / camera['fx'][0] * multiplier      # array
            coords[:, 1] = (coords[:, 1] - camera['cy'][0]) / camera['fy'][0] * multiplier
        

        # (hw, 3) 3D points in cam coord
        coords_cam = np.concatenate((coords,
                                     multiplier * np.ones((coords.shape[0], 1))), axis=1)   # array

        if self.use_t:
            coords_world = (camera['R'].T @ coords_cam.T + camera['t']).T  # (hw, 3)    in world coordinate    array
        else:
            coords_world = (camera['R'].T @ coords_cam.T + camera['T']).T  # (hw, 3)    in world coordinate    array
        coords_world = torch.from_numpy(coords_world).float()  # (hw, 3)
        if self.bug_test:
            coords_world[:, :] = 0
        return coords_world
