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
import json
import collections
import random
import torch
import copy
import cv2
from utils.calib import smart_pseudo_remove_weight
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform, affine_transform_pts
from utils.calib import world_to_cam, cam_to_image
from utils.utils_amass import rotate_pose

from dataset.joints_dataset_mpl import JointsDataset_MPL
import logging
logger = logging.getLogger(__name__)

downsample = 16

# set random seed
# np.random.seed(1)
# random.seed(1)

class MultiView_AMASS_CMU_Panoptic_MPL(JointsDataset_MPL):

    """
    The purpose is to train on H36M 3D data based on the camera system from CMU Panoptic.
    the 3D data is taken from h36M, and the 2D is created based on camera parameters from CMU Panoptic.
    """
    def __init__(self, cfg, image_set, is_train, transform=None, is_mmpose=False):
        super().__init__(cfg, image_set, is_train, transform)
        self.num_joints = 17
        
        if self.CMU_KEYPOINT_STANDARD == 'h36m':
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
        elif self.CMU_KEYPOINT_STANDARD == 'coco':
            self.actual_joints = {
                0: 'nose',
                1: 'leye',
                2: 'reye',
                3: 'lear',
                4: 'rear',
                5: 'lsho',
                6: 'rsho',
                7: 'lelb',
                8: 'relb',
                9: 'lwri',
                10: 'rwri',
                11: 'lhip',
                12: 'rhip',
                13: 'lkne',
                14: 'rkne',
                15: 'lank',
                16: 'rank'
            }
            self.union_joints = self.actual_joints
        
        self.joints_right = [1, 2, 3, 14, 15, 16]
        self.joints_left = [4, 5, 6, 11, 12, 13]

        self.val_on_train = cfg.DATASET.VAL_ON_TRAIN
        self.is_train = is_train
        
        self.use_mmpose = False
        if is_mmpose:
            self.use_mmpose = True
        else:
            if cfg.DATASET.USE_MMPOSE_TRAIN and is_train:
                self.use_mmpose = True
            elif (cfg.DATASET.USE_MMPOSE_VAL or cfg.DATASET.USE_MMPOSE_TEST) and not is_train:
                self.use_mmpose = True
        
        
        self.inputs_normalized = cfg.DATASET.INPUTS_NORMALIZED
        
        self.no_augmentation = cfg.DATASET.NO_AUGMENTATION
        self.output_in_meter = cfg.DATASET.OUTPUT_IN_METER
        self.clip_joints = cfg.DATASET.CLIP_JOINTS
            
        self.camera_manual_order = cfg.DATASET.CAMERA_MANUAL_ORDER
        print('camera_manual_order',self.camera_manual_order)
        
        self.dataset_type = cfg.DATASET.DATASET_TYPE
        self.amass_dataset_type = cfg.DATASET.AMASS_DATASET_TYPE
        
        
        cmu_calibs = self.cmu_calibs_train if is_train else self.cmu_calibs_val
        if self.cmu_calib is not None:  # to take old configs into account
            cmu_calibs = [self.cmu_calib]
        self.cmu_cameras = self.load_all_cameras(cmu_calibs)
        if self.run_on_all_cameras:
            pass
        else:
            # self.cmu_cameras_kept = []
            # for view in self.views:
            #     self.cmu_cameras_kept.append(self.cmu_cameras[view])
            # self.cmu_cameras = self.cmu_cameras_kept
            self.cmu_cameras = {view: v for view, v in self.cmu_cameras.items() if view in self.views}
            
        self.n_all_cameras = len(self.cmu_cameras)    
        self.all_camera_ids = list(self.cmu_cameras.keys())
        self.n_camera_setups = len(self.cmu_cameras[self.all_camera_ids[0]])
        # if self.train_on_all_cameras and is_train:
        #     self.cmu_cameras = self.load_all_cameras(calib_file=osp.join(self.root, '171026_pose1', 'calibration_171026_pose1.json'))
        #     self.n_all_cameras = len(self.cmu_cameras)
        # else:
        #     self.db_cmu = self.load_db(anno_file)
        #     self.db_cmu = self.filter_db(self.db_cmu)   # remove images that are with other cameras
        #     self.grouping = self.get_group(self.db_cmu)
        #     self.cmu_cameras = []
        #     self.cmu_center = self.db_cmu[0]['joints_3d'][0]      # (3, ) in world coordinate (root joint)
        #     for i in self.grouping[0]:
        #         self.cmu_cameras.append(self.db_cmu[i]['camera'])
        
        
            # with open('cmu_cameras.pkl', 'wb') as f:
            #     pickle.dump(self.cmu_cameras, f)
            # raise
        
        # if self.amass_val_located and not self.is_train:
        #     anno_file = osp.join(self.root_2, 'd3_joints_validation_located.npy')
        # else:
        #     if self.amass_dataset_type is not None:
        #         anno_file = osp.join(self.root_2, self.amass_dataset_type, 'd3_joints_{}.npy'.format(image_set))
        #     else:
        #         anno_file = osp.join(self.root_2, 
        #                             'd3_joints_{}.npy'.format(image_set))
        
        # self.db = np.load(anno_file, allow_pickle=True)
        # if not self.is_train and not self.amass_val_located:
        #     self.db = self.locate_person_in_the_room_val(self.db)
        
        # if cfg.DATASET.TRAIN_N_SAMPLES is not None and is_train:
        #     self.db = self.db[:cfg.DATASET.TRAIN_N_SAMPLES]
        # if cfg.DATASET.TEST_N_SAMPLES is not None and not is_train:
        #     self.db = self.db[:cfg.DATASET.TEST_N_SAMPLES]
        
        if self.use_amass_old_datasets:
            if self.amass_val_located and not self.is_train:
                anno_file = osp.join(self.root_2, 'd3_joints_validation_h36m_located.npy')
            else:
                if self.amass_dataset_type is not None:
                    anno_file = osp.join(self.root_2, self.amass_dataset_type, 'd3_joints_{}.npy'.format(image_set))
                else:
                    anno_file = osp.join(self.root_2, 
                                        'd3_joints_{}.npy'.format(image_set))
            # anno_file = osp.join(self.root_2, 'd3_joints_train.npy')
            
            self.db = np.load(anno_file, allow_pickle=True)
            if self.train_on_all_amass and is_train:
                anno_file_val = osp.join(self.root_2, 'd3_joints_validation.npy')
                self.db_val = np.load(anno_file_val, allow_pickle=True)
                self.db = np.concatenate([self.db, self.db_val], axis=0)
            if not self.is_train and not self.amass_val_located:
                self.db = self.locate_person_in_the_room_val(self.db)
                
            if cfg.DATASET.TRAIN_N_SAMPLES is not None and is_train:
                self.db = self.db[:cfg.DATASET.TRAIN_N_SAMPLES]
            if cfg.DATASET.TEST_N_SAMPLES is not None and not is_train:
                self.db = self.db[:cfg.DATASET.TEST_N_SAMPLES]
            logger.info('=> {} load {} samples'.format(image_set, len(self.db)))
        else:
            # if self.amass_val_located and not self.is_train:
            #     # TODO
            #     pass
            # else:
            if self.amass_dataset_type is not None:
                anno_file = osp.join(self.root_2, self.amass_dataset_type, 'amass_mmpose_joints_{}.pkl'.format(image_set))
            else:
                anno_file = osp.join(self.root_2, 
                                    'amass_mmpose_joints_{}.pkl'.format(image_set))
            self.db, self.db_2d = self.load_amass_new(anno_file)
            
            if cfg.DATASET.TRAIN_N_SAMPLES is not None and is_train:
                self.db = self.db[:cfg.DATASET.TRAIN_N_SAMPLES]
            if cfg.DATASET.TEST_N_SAMPLES is not None and not is_train:
                self.db = self.db[:cfg.DATASET.TEST_N_SAMPLES]
            for k in self.db_2d.keys():
                if cfg.DATASET.TRAIN_N_SAMPLES is not None and is_train:
                    self.db_2d[k] = self.db_2d[k][:cfg.DATASET.TRAIN_N_SAMPLES]
                if cfg.DATASET.TEST_N_SAMPLES is not None and not is_train:
                    self.db_2d[k] = self.db_2d[k][:cfg.DATASET.TEST_N_SAMPLES]
            logger.info('=> {} load {} samples'.format(image_set, len(self.db)))

        self.u2a_mapping = super().get_mapping()
        # logger.info('=> {} num samples: {}'.format(image_set, len(self.db)))
        # super().do_mapping()
        
    def locate_person_in_the_room_val(self, db):
        if not self.output_in_meter:
            room_max_x = self.room_max_x / 100
            room_min_x = self.room_min_x / 100
            room_max_y = self.room_max_y / 100
            room_min_y = self.room_min_y / 100
        else:
            room_max_x = self.room_max_x
            room_min_x = self.room_min_x
            room_max_y = self.room_max_y
            room_min_y = self.room_min_y
        db[:, :, 0] = db[:, :, 0] - db[:, 0:1, 0]  # subtract root joint
        db[:, :, 1] = db[:, :, 1] - db[:, 0:1, 1]  # subtract root joint
        
        if self.rotate:
            rotation = np.random.rand(len(db)) * 360
            db = rotate_pose(db, rotation, axis='z')
                
        if self.no_augmentation_3d:
            location_3d = np.zeros((db.shape[0], 3))
        else:
            location_3d = np.concatenate([np.random.rand(db.shape[0], 1) * (room_max_x - room_min_x) + room_min_x, np.random.rand(db.shape[0], 1) * (room_max_y - room_min_y) + room_min_y, np.zeros((db.shape[0], 1))], axis=1)
        db += location_3d[:, None, :]
        return db
    
    def index_to_action_names(self):
        return None
    
    def load_all_cameras(self, cmu_calibs):
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

    def load_db(self, dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
            return dataset

    def get_group(self, db):
        grouping = {}
        nitems = len(db)    
        CAM_IX = {x: i for i, x in enumerate(self.views)}
        for i in range(nitems):
            keystr = self.get_key_str(db[i])
            camera_id = CAM_IX[db[i]['camera_id']]
            if keystr not in grouping:
                grouping[keystr] = [-1] * len(self.views)
            grouping[keystr][camera_id] = i

        filtered_grouping = []
        for _, v in grouping.items():
            if np.all(np.array(v) != -1):
                filtered_grouping.append(v)
                

        return filtered_grouping

    
    def __getitem__(self, idx):
        input, target, weight, meta = [], [], [], []
        pose_3d = copy.deepcopy(self.db[idx])
        if self.mix_smart_3d_amass_with_triangulated_mmpose:
            pose_3d_tri = copy.deepcopy(self.db_2d['triangulated_3d_mmpose'][idx])
            error_triang = np.sqrt(np.sum((pose_3d[self.keypoints_to_mix_amass_with_3d_triangulated_mmpose] - pose_3d_tri[self.keypoints_to_mix_amass_with_3d_triangulated_mmpose]) ** 2, axis=1))
            error_tring_all = np.zeros((17), dtype=bool)
            error_tring_all[self.keypoints_to_mix_amass_with_3d_triangulated_mmpose] = error_triang < self.epipolar_error_acceptance_threshold
            pose_3d[error_tring_all] = pose_3d_tri[error_tring_all]
        joints_3d_org = copy.deepcopy(pose_3d)
        if self.centeralize_root_first and self.is_train:     # only x and y
            if not self.use_amass_old_datasets:
                pose_3d[:, 0] = pose_3d[:, 0] - pose_3d[0, 0]  # subtract root joint
                pose_3d[:, 2] = pose_3d[:, 2] - pose_3d[0, 2]  # subtract root joint
            else:
                pose_3d[:, 0] = pose_3d[:, 0] - pose_3d[0, 0]  # subtract root joint
                pose_3d[:, 1] = pose_3d[:, 1] - pose_3d[0, 1]  # subtract root joint
        
        if self.amass_data_no_axis_swap:
            if self.normalize_room:
                room_x_scale = (self.room_max_x - self.room_min_x) / 2
                room_y_scale = (self.room_max_y - self.room_min_y) / 2
                room_center = np.array([(self.room_max_x + self.room_min_x) / 2, (self.room_max_y + self.room_min_y) / 2, 0])
            pass
        else:
            pose_3d_copy = pose_3d.copy()
            pose_3d[:, 1] = -pose_3d[:, 2]   # swap y and z
            pose_3d[:, 2] = pose_3d_copy[:, 1]  # swap y and z
            if self.normalize_room:
                room_x_scale = (self.room_max_x - self.room_min_x) / 2
                room_y_scale = (self.room_max_y - self.room_min_y) / 2
                room_center = np.array([(self.room_max_x + self.room_min_x) / 2, 0, (self.room_max_y + self.room_min_y) / 2])
        
        # pose_3d[:, 0] = pose_3d[:, 0] - pose_3d[0, 0]  # subtract root joint
        # pose_3d[:, 2] = pose_3d[:, 2] - pose_3d[0, 2]  # subtract root joint
        
        if not self.output_in_meter:
            pose_3d = pose_3d * 100           # (17, 3)   in world coordinate. *100 to convert to cm
            # generate 2 random numbers between -100 and 100 for x and z and add them to all joints
        if self.rotate and self.is_train:
            rotation = np.random.rand(1) * 360
            if self.amass_data_no_axis_swap:
                pose_3d = rotate_pose(pose_3d[None], rotation, axis='z')[0]
            else:
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
                    
            
            if self.amass_data_no_axis_swap:
                augmentation_3d = np.concatenate([np.random.rand(1, 1) * (self.room_max_x - self.room_min_x) + self.room_min_x, np.random.rand(1, 1) * (self.room_max_y - self.room_min_y) + self.room_min_y, np.zeros((1, 1))], axis=1)
            else:    
                augmentation_3d = np.concatenate([np.random.rand(1, 1) * (self.room_max_x - self.room_min_x) + self.room_min_x, np.zeros((1, 1)), np.random.rand(1, 1) * (self.room_max_y - self.room_min_y) + self.room_min_y], axis=1)
            pose_3d = pose_3d + augmentation_3d
        
        # camera_setup_to_use = np.random.random_integers(0, self.n_camera_setups-1)
        # if self.train_on_all_cameras and self.is_train:
        #     camera_ids = np.random.choice(self.all_camera_ids, self.n_views, replace=False)
        # else:
        #     camera_ids = self.views
        
        if self.use_amass_old_datasets or self.use_amass_new_datasets_with_old_way:
            camera_setup_to_use = np.random.random_integers(0, self.n_camera_setups-1)
        else:
            camera_setup_to_use = self.db_2d['camera_setup_used'][idx]
            
        if self.run_on_all_cameras:
            camera_ids = np.random.choice(self.all_camera_ids, self.n_views, replace=False)
        else:
            camera_ids = np.array(self.views)
        
        for camera_id in camera_ids:
            i, t, w, m = self.getitem(pose_3d, camera_id, joints_3d_org, camera_setup_to_use, idx=idx)
            if self.normalize_room:
                t = self.normalize_pose3d_coordinates(t, room_center, room_x_scale, room_y_scale)
                m['rays'] = self.normalize_pose3d_coordinates(m['rays'], room_center, room_x_scale, room_y_scale)
                m['cam_center'] = self.normalize_pose3d_coordinates(m['cam_center'], room_center, room_x_scale, room_y_scale)
                m['room_scaled'] = True
                m['room_x_scale'] = max(room_x_scale, room_y_scale)
                m['room_y_scale'] = max(room_x_scale, room_y_scale)
                m['room_center'] = room_center
                m['room_scaled_equal'] = True
                # t[:, 0] = t[:, 0] / room_x_scale
                # t[:, 1] = t[:, 1] / room_y_scale
                # m['rays'][:, 0] = m['rays'][:, 0] / room_x_scale
                # m['rays'][:, 1] = m['rays'][:, 1] / room_y_scale
                # m['cam_center'][:, 0] = m['cam_center'][:, 0] / room_x_scale
                # m['cam_center'][:, 1] = m['cam_center'][:, 1] / room_y_scale
                # m['room_scaled'] = True
                # m['room_x_scale'] = room_x_scale
                # m['room_y_scale'] = room_y_scale
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
    
    """ debug code
        import matplotlib.pyplot as plt
        fig = plt.figure()
        body_edges = np.array([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16],[8,9],[9,10]])
        for i, input_ in enumerate(input):
            sub = int('23{}'.format(i+1))
            ax = fig.add_subplot(sub)
            to_plot = input_
            ax.scatter(to_plot[:,0], to_plot[:,1])
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
            for edge in body_edges:
                ax.plot(to_plot[edge,0], to_plot[edge,1], color='r')
        plt.savefig('test14.png')
        
        
        
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d
        body_edges = np.array([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16],[8,9],[9,10]])
        fig = plt.figure() 
        ax = fig.add_subplot(111, projection='3d')
        to_plot = target[0]
        ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:, 2], marker='o', s=20, c='b')
        for edge in body_edges:
            ax.plot(to_plot[edge,0], to_plot[edge,1], to_plot[edge,2], color='g')

        max_range = np.array([to_plot[:, 0].max() - to_plot[:, 0].min(), to_plot[:, 1].max() - to_plot[:, 1].min(), to_plot[:, 2].max() - to_plot[:, 2].min()]).max() / 2.0
        x_mean = to_plot[:, 0].mean()
        y_mean = to_plot[:, 1].mean()
        z_mean = to_plot[:, 2].mean()
        ax.set_xlim(x_mean - max_range, x_mean + max_range)
        ax.set_ylim(y_mean - max_range, y_mean + max_range)
        ax.set_zlim(z_mean - max_range, z_mean + max_range)
        plt.savefig('test15.png')
    """
    
    def getitem(self, joints_3d, camera_id, joints_3d_org=None, camera_setup_to_use=0, idx=None):
        

        # ==================================== Label ====================================
        
        
        # joints_3d = joints_3d - joints_3d[0] + self.cmu_center  # (17, 3)   in world coordinate
        camera = self.cmu_cameras[camera_id][camera_setup_to_use].copy()
        if self.output_in_meter:
            camera['T'] = camera['T'] / 100
            camera['t'] = camera['t'] / 100
            
        if self.apply_noise_cameras and self.is_train:
            noise = np.random.normal(0, 1, camera['T'].shape) * self.t_noise_value
            camera['t'] = camera['t'] + noise
            noise = np.random.normal(0, 1, camera['R'].shape) * self.R_noise_value
            camera['R'] = camera['R'] + noise
            camera['T'] = -camera['R'].T @ camera['t'].squeeze()
            
        if self.use_amass_old_datasets or self.use_amass_new_datasets_with_old_way:    
            joints_3d_cam = world_to_cam(joints_3d[None,:,:], camera['R'], camera['t'])  # (17, 3) in camera coordinate
            joints_2d = cam_to_image(joints_3d_cam, camera['K'])[0]        # (17, 2) in original image scale (1000, 1000)
            
        else:
            view_finder = self.views_in_amass.index(camera_id)
            if self.use_mmpose:
                joints_2d = copy.deepcopy(self.db_2d['joints_2d_mmpose'][idx][view_finder])
                if self.mix_amass_with_mmpose:
                    joints_2d[self.keypoints_to_mix] = copy.deepcopy(self.db_2d['joints_2d_amass'][idx][view_finder][self.keypoints_to_mix])
                if np.isnan(joints_2d).any():
                    joints_2d = np.zeros_like(joints_2d)
            else:
                joints_2d = copy.deepcopy(self.db_2d['joints_2d_amass'][idx][view_finder])
        
        if self.target_normalized_3d:
            joints_3d = joints_3d_org
        
        joints = joints_2d.copy()
        # joints = db_rec['joints_2d'].copy()             # (17, 2)   in original image scale (1000, 1000)
        # print('joints', joints)
        joints_org = joints.copy()
        joints_vis = np.ones((17, 1))        # (17, 3)   0,0,0 or 1,1,1
        if not self.use_amass_old_datasets and self.use_mmpose:
            joints_vis = copy.deepcopy(self.db_2d['confs_2d_mmpose'][idx][view_finder])
            if self.mix_amass_with_mmpose:
                joints_vis[self.keypoints_to_mix] = 1
        elif self.use_amass_new_datasets_with_old_way and self.use_mmpose:
            raise 'disable use_mmpose with use_amass_new_datasets_with_old_way'
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
            
        # if self.APPLY_NOISE and self.is_train:
        #     noise = np.random.normal(0, 1, joints.shape) * self.NOISE_LEVEL
        #     joints = joints + noise
        #     # penalize the confidence of the noisy joints
        #     if self.penalize_confidence == 'exp_abs_sum':
        #         penalize_conf = self.penalize_factor * np.exp(-np.abs(noise.sum(axis=1)) / 2)
        #     elif self.penalize_confidence == 'exp_norm':
        #         penalize_conf = self.penalize_factor * np.exp(-np.linalg.norm(noise, axis=1) / 2)
        #     elif self.penalize_confidence == '1-norm':
        #         penalize_conf = 1 - self.penalize_factor * np.linalg.norm(noise, axis=1)
        #     joints_vis = joints_vis * penalize_conf[:, None]
        #     joints_vis = np.clip(joints_vis, 0, 1)
        # raise
        # joints_vis = db_rec['joints_vis'].copy()        # (17, 3)   0,0,0 or 1,1,1
        # if self.APPLY_NOISE_MISSING and self.is_train:
        #     missing = np.random.uniform(0, 1, joints_vis.shape[0])
        #     mask = np.ones_like(joints_vis)
        #     mask[missing < self.MISSING_LEVEL] = 0
        #     joints_vis = joints_vis * mask

        center = np.array(np.random.rand(2,) * 250 + 250).copy()      # (2, )     (cx, cy)  in original image scale
        scale = np.array(np.random.rand(2,) + 2).copy()        # (2, )     (s1, s2) random number between 2 and 3
        rotation = 0
        
        if self.no_augmentation:
            center = np.array([self.image_size[0]/2, self.image_size[1]/2]).copy()      # (2, )     (cx, cy)  in original image scale
            scale = np.array([1, 1]).copy()        # (2, )     (s1, s2) random number between 2 and 3
            rotation = 0

        # ==================================== Camera  ====================================
        # camera matrix
        # normalize the camera matrix too
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

        # fix the system error of camera
        # distCoeffs = np.array(
        #     [float(i) for i in [camera['k'][0], camera['k'][1], camera['p'][0], camera['p'][1], camera['k'][2]]])
        # data_numpy = cv2.undistort(data_numpy, K, distCoeffs)
        # joints = cv2.undistortPoints(joints[:, None, :], K, distCoeffs, P=K).squeeze()
        # center = cv2.undistortPoints(np.array(center)[None, None, :], K, distCoeffs, P=K).squeeze()

        # ==================================== Preprocess ====================================
        # # augmentation factor
        # if self.is_train:
        #     sf = self.scale_factor
        #     rf = self.rotation_factor
        #     scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        #     rotation = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
        #         if random.random() <= 0.6 else 0
        # if db_rec['source'] == 'cmu_panoptic':
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

        # # ========================== heatmap ==========================
        # target, target_weight = self.generate_target(joints, joints_vis)

        # target = torch.from_numpy(target)                   # (17, 64, 64) heatmap
        # target_weight = torch.from_numpy(target_weight)
        
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
    
    
    

    def __len__(self):
        return len(self.db)

    def get_key_str(self, datum):
        return 's_{:02}_pose_{}_imgid_{:08}'.format(
            datum['subject'], datum['pose_id'],
            datum['image_id'])
        
    def get_key_str_h36m(self, datum):
        return 's_{:02}_act_{:02}_subact_{:02}_imgid_{:06}'.format(
            datum['subject'], datum['action'], datum['subaction'],
            datum['image_id'])

    def evaluate(self, pred, *args, **kwargs):
        pred = pred.copy()

        u2a = self.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = list(a2u.values())
        indexes = list(range(len(a)))
        indexes.sort(key=a.__getitem__)
        sa = list(map(a.__getitem__, indexes))
        su = np.array(list(map(u.__getitem__, indexes)))    # [ 0  1  2  3  4  5  6  7  9 11 12 14 15 16 17 18 19]

        gt = []
        for item in self.db:
            gt.append(item[su, :])       # (17, 3) in original scale
        gt = np.array(gt) * 100         # (num_sample, 17, 3)  *100 to convert to cm
        gt = gt - gt[:, 0:1]
        pred = pred - pred[:, 0:1]
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