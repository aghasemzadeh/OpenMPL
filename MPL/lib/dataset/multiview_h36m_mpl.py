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
from utils.calib import smart_pseudo_remove_weight
from utils.calib import cam_to_world

from dataset.joints_dataset_mpl import JointsDataset_MPL
import logging
import json

logger = logging.getLogger(__name__)

# # set random seed
# np.random.seed(0)
# random.seed(0)

class MultiViewH36M_MPL(JointsDataset_MPL):

    def __init__(self, cfg, image_set, is_train, transform=None, is_mmpose=False):
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
        self.keypoints_inverse_dict = {v: k for k, v in self.actual_joints.items()}
        self.camera_manual_order = cfg.DATASET.CAMERA_MANUAL_ORDER
        print('camera_manual_order',self.camera_manual_order)
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
            
        if self.h36m_old_datasets:
            dataset_path = osp.join(self.root, 'h36m', 'annot')
            if self.use_mmpose:
                dataset_path = osp.join(self.root, 'h36m', 'annot_mmpose')
        else:
            dataset_folder_name = 'datasets_mmpose' if self.use_mmpose else 'datasets' 
            dataset_folder_name_2 = self.h36m_dataset_name + '_' + self.mmpose_type if self.use_mmpose else self.h36m_dataset_name
            dataset_path = osp.join(self.root, 'h36m', self.dir_mpl_data, dataset_folder_name, dataset_folder_name_2)
        if cfg.DATASET.CROP:
            # if self.use_mmpose:
            #     anno_file = osp.join(self.root, 'h36m', 'annot_mmpose',
            #                         'h36m_{}.pkl'.format(image_set))
            # else:
            #     anno_file = osp.join(self.root, 'h36m', 'annot',
            #                         'h36m_{}.pkl'.format(image_set))
            anno_file = osp.join(dataset_path, 'h36m_{}.pkl'.format(image_set))
        else:
            # if self.use_mmpose:
            #     anno_file = osp.join(self.root, 'h36m', 'annot_mmpose',
            #                         'h36m_{}_uncrop.pkl'.format(image_set))
            # else:
            #     anno_file = osp.join(self.root, 'h36m', 'annot',
            #                         'h36m_{}_uncrop.pkl'.format(image_set))
            anno_file = osp.join(dataset_path, 'h36m_{}_uncrop.pkl'.format(image_set))

        self.db = self.load_db(anno_file)
        if not cfg.DATASET.WITH_DAMAGE:
            print('before filter', len(self.db))
            self.db = [db_rec for db_rec in self.db if not self.isdamaged(db_rec)]
            print('after filter', len(self.db))
        
        self.db = self.make_cameras_compatible(self.db)
        if self.only_keep_inside_room:
            self.db_ = []
            for i, db_rec in enumerate(self.db):
                joints_3d_camera = db_rec['joints_3d_camera'].copy()      # (17, 3) in camera coordinate
                camera = db_rec['camera'].copy()  
                joints_3d = cam_to_world(joints_3d_camera[None], camera['R'], camera['t']).squeeze()  # (17, 3) in world coordinate
                if self.output_in_meter:
                    joints_3d = joints_3d / 1000
                if self.room_min_x < joints_3d[0, 0] < self.room_max_x and self.room_min_y < joints_3d[0, 1] < self.room_max_y:
                    self.db_.append(db_rec)
            self.db = self.db_
        if self.only_keep_if_in_calibs_actors:
            # only keep the data that is in the calibs
            # self.h36m_calib_actors_val
            self.db_ = []
            for i, db_rec in enumerate(self.db):
                if db_rec['subject'] in self.h36m_calib_actors:
                    self.db_.append(db_rec)
            self.db = self.db_
                

        self.u2a_mapping = super().get_mapping()
        super().do_mapping()

        self.grouping = self.get_group(self.db)
        self.group_size = len(self.grouping)
        
        logger.info('=> load {} samples'.format(self.group_size))
        
        if self.use_cmu_cameras_on_h36m:
            cmu_calibs = self.cmu_calibs_train if is_train else self.cmu_calibs_val
            if isinstance(cmu_calibs, str):
                cmu_calibs = [cmu_calibs]
            self.cmu_cameras = self.load_all_cameras_cmu(cmu_calibs)
            
            self.cmu_cameras = {view: v for view, v in self.cmu_cameras.items() if view in self.views}
            self.n_all_cameras = len(self.cmu_cameras)    
            self.all_camera_ids = list(self.cmu_cameras.keys())
            self.n_camera_setups = len(self.cmu_cameras[self.all_camera_ids[0]])
        
        
        # f = open('filtered_grouping_keys_{}_new.txt'.format(image_set), 'w')
        # for k in self.grouping:
        #     for i in k:
        #         f.write(self.db[i]['image'])
        #         f.write('\n')
        # f.close()
        # print('txt file written')
        
        self.R_noise_value = cfg.DATASET.R_NOISE_VALUE
        self.T_noise_value = cfg.DATASET.T_NOISE_VALUE
        if cfg.DATASET.APPLY_NOISE_CAMERAS and self.is_train:
            self.add_noise_to_cameras()
            
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
    
    
        
    def add_noise_to_camera(self, camera):
        # add noise to camera params
        camera['R'] = camera['R'] + np.random.normal(0, 1.0, camera['R'].shape) * self.R_noise_value
        camera['T'] = camera['T'] + np.random.normal(0, 1.0, camera['T'].shape) * self.T_noise_value
        return camera
        
    def add_noise_to_cameras(self):
        # add noise to cameras
        for db_rec in self.db:
            db_rec['camera'] = self.add_noise_to_camera(db_rec['camera'])

    def index_to_action_names(self):
        return {
            2: 'Direction',
            3: 'Discuss',
            4: 'Eating',
            5: 'Greet',
            6: 'Phone',
            7: 'Photo',
            8: 'Pose',
            9: 'Purchase',
            10: 'Sitting',
            11: 'SittingDown',
            12: 'Smoke',
            13: 'Wait',
            14: 'WalkDog',
            15: 'Walk',
            16: 'WalkTwo'
        }

    def load_db(self, dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
            return dataset

    def get_group(self, db):
        grouping = {}
        nitems = len(db)
        # if self.camera_manual_order is not False:
        #     CAM_IX = {cmo: i for i, cmo in enumerate(self.camera_manual_order)}
        # else:
        #     CAM_IX = {i: i for i in range(4)}
        # print('CAM_IX', CAM_IX)
        # for i in range(nitems):
        #     keystr = self.get_key_str(db[i])
        #     camera_id = CAM_IX[db[i]['camera_id']]
        #     if keystr not in grouping:
        #         grouping[keystr] = [-1, -1, -1, -1]
        #     grouping[keystr][camera_id] = i
        if self.use_cmu_cameras_on_h36m:
            views = [1, 2, 3, 4]
            CAM_IX = {x-1: i for i, x in enumerate(views[:self.n_views])}
        else:
            views = self.views
            CAM_IX = {x-1: i for i, x in enumerate(self.views)}
        for i in range(nitems):
            keystr = self.get_key_str(db[i])
            try:
                camera_id = CAM_IX[db[i]['camera_id']]
            except KeyError:
                continue
            if keystr not in grouping:
                grouping[keystr] = [-1] * len(views)
            grouping[keystr][camera_id] = i
            
        filtered_grouping = []
        for _, v in grouping.items():
            if np.all(np.array(v) != -1):
                filtered_grouping.append(v)
                
        if self.use_cmu_cameras_on_h36m and len(filtered_grouping) > 0:
            n_views_to_add = self.n_views - len(filtered_grouping[0])
            for i, group in enumerate(filtered_grouping):
                for j in range(n_views_to_add):
                    filtered_grouping[i].append(-1)
                
        if self.filter_groupings:
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
        items = self.grouping[idx]
        if self.normalize_room:
            room_x_scale = (self.room_max_x - self.room_min_x) / 2
            room_y_scale = (self.room_max_y - self.room_min_y) / 2
        
        camera_setup_to_use = None
        if self.use_cmu_cameras_on_h36m:    
            camera_setup_to_use = np.random.random_integers(0, self.n_camera_setups-1)
            camera_ids = np.array(self.views)
        for ix, item in enumerate(items):
            camera_id = None
            if self.use_cmu_cameras_on_h36m:
                camera_id = camera_ids[ix]
            i, t, w, m = super().__getitem__(item, camera_id, camera_setup_to_use)
            if self.normalize_room:
                t[:, 0] = t[:, 0] / room_x_scale
                t[:, 1] = t[:, 1] / room_y_scale
                m['rays'][:, 0] = m['rays'][:, 0] / room_x_scale
                m['rays'][:, 1] = m['rays'][:, 1] / room_y_scale
                m['cam_center'][:, 0] = m['cam_center'][:, 0] / room_x_scale
                m['cam_center'][:, 1] = m['cam_center'][:, 1] / room_y_scale
                m['room_scaled'] = True
                m['room_x_scale'] = room_x_scale
                m['room_y_scale'] = room_y_scale
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

    def __len__(self):
        return self.group_size

    def get_key_str(self, datum):
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