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

from torch.utils.data import Dataset
from dataset.joints_dataset_mpl import JointsDataset_MPL
import logging
import json
import glob
import os
import xml.etree.ElementTree as ET
import torch

logger = logging.getLogger(__name__)

# # set random seed
# np.random.seed(0)
# random.seed(0)

class MultiViewInference_OpenMPLPoser_MPL(Dataset):

    def __init__(self, cfg, transform=None):
        super().__init__()
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
        
        self.cameras_path = cfg.DATASET.INFERENCE_CAMERAS_PATH
        self.data_dir = cfg.DATASET.INFERENCE_DATA_DIR
        self.openmplposer_cameras = self.load_all_cameras(self.cameras_path)

        self.image_size = cfg.NETWORK.IMAGE_SIZE
        self.use_t = cfg.DATASET.USE_T
        self.clip_joints = cfg.DATASET.CLIP_JOINTS

        self.kp_visiblity_th = cfg.DATASET.KP_VISIBILITY_TH
        self.zero_tokens_for_missing_joints = cfg.DATASET.ZERO_TOKENS_FOR_MISSING_JOINTS

        # self.filename_list = glob.glob(f'{self.data_dir}')

        # logger.info('=> load {} files from {}'.format(len(self.filename_list), self.data_dir))


        with open(self.data_dir, 'rb') as f:
            data = pickle.load(f)
        self.db = data['yolo_keypoints']

        for k, v in self.db.items():
            if k == 'yolo_version':
                continue
            self.db[k] = v.numpy()

        logger.info('=> load {} frames from {}'.format(len(self.db['vcam0']), self.data_dir))


    def load_all_cameras(self, calibs):

        def parse_matrix(node):
            rows = int(node.find('rows').text)
            cols = int(node.find('cols').text)
            data_text = node.find('data').text.strip().replace('\n', ' ')
            data = list(map(float, data_text.split()))
            return np.array(data).reshape((rows, cols))
        
        cams = range(1, 4)
        cameras_all_calibs = {cam_id: [] for cam_id in cams}
        xml_files = glob.glob(osp.join(calibs, '*.xml'))
        distCoef =  [-0.287016,0.182978,1.91352e-06,0.000618877,-0.0471994] # from cmu panoptic

        for xml_file in xml_files:

            # Parse the XML
            tree = ET.parse(xml_file)
            root = tree.getroot()

            K = parse_matrix(root.find('Intrinsics'))
            P = parse_matrix(root.find('CameraMatrix'))

            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]

            R = P[:, :3]
            T = P[:, 3:]

            R = R.T

            F = np.diag([-1, 1, 1])
            R = R.copy()
            R = F @ R
            k = np.array([distCoef[0], distCoef[1], distCoef[4]])
            p = np.array([distCoef[2], distCoef[3]])

            camera_dict = {
                'camera_setup': 0,
                'camera_id': int(xml_file.split('/')[-1].split('_')[-1].replace('.xml', '')),
                'K': np.array(K),
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'k': k,
                'p': p,
                'R': np.array(R),
                'T': np.array(T),
                't': -np.linalg.inv(R.T) @ T,
            }
            cameras_all_calibs[camera_dict['camera_id']].append(camera_dict)
        return cameras_all_calibs

    def __getitem__(self, idx):
        input, target, weight, meta = [], [], [], []

        camera_ids = self.openmplposer_cameras.keys()

        for camera_id in camera_ids:
            i, t, w, m = self.getitem(idx, camera_id)
            
            input.append(i)
            target.append(t)
            weight.append(w)
            meta.append(m)

        return input, target, weight, meta



    def getitem(self, idx, camera_id):

        camera = self.openmplposer_cameras[camera_id][0].copy()

        joints = self.db['vcam{}'.format(camera_id - 1)][idx].copy()  # (17, 2)
        joints_vis = self.db['confidences'][idx, :, camera_id - 1].copy().reshape(-1, 1)  # (17, 1)

        R = camera['R']
        T = camera['T']
        t = camera['t']
        K = camera['K']

        if self.clip_joints:
            joints_vis[:, 0] = np.where(0 < joints[:, 0], joints_vis[:, 0], 0)
            joints_vis[:, 0] = np.where(joints[:, 0] < self.image_size[0] - 1, joints_vis[:, 0], 0)
            joints_vis[:, 0] = np.where(0 < joints[:, 1], joints_vis[:, 0], 0)
            joints_vis[:, 0] = np.where(joints[:, 1] < self.image_size[1] - 1, joints_vis[:, 0], 0)
            joints[:, 0] = np.clip(joints[:, 0], 0, self.image_size[0] - 1)
            joints[:, 1] = np.clip(joints[:, 1], 0, self.image_size[1] - 1)
        else:
            for i in range(self.num_joints):
                if joints_vis[i, 0] > 0.0:
                    if (np.min(joints[i, :2]) < 0 or
                            joints[i, 0] >= self.image_size[0] or
                            joints[i, 1] >= self.image_size[1]):
                        joints_vis[i, :] = 0
                        joints[i, :] = 0        # to avoid further errors when downscaling
                else:
                    joints[i, :] = 0            # to avoid further errors when downscaling

        joints = self.normalize_screen_coordinates(joints, self.image_size[0], self.image_size[1])

        coords_ray = self.create_3d_ray_coords(camera, joints)
        # joints_vis = np.ones((self.num_joints, 1), dtype=np.float32)  # all visible

        joints = np.concatenate((joints, joints_vis), axis=1)  # (17, 3)

        # do logical and on joints_2d_confs on axis 1
        joints_2d_confs = np.where(joints_vis > self.kp_visiblity_th, 1, 0)
        joints_2d_confs = np.all(joints_2d_confs, axis=1, keepdims=True)   # (17, 1, 1)
        if self.zero_tokens_for_missing_joints:
            joints *= joints_2d_confs
            
        joints = torch.from_numpy(joints).float()  # (17, 3)
        cam_center = torch.Tensor(camera['T'].T)

        meta = {
            'camera_id': camera_id,
            'frame_id': idx,
            'rays': coords_ray,  # (hw, 3)
            'cam_center': cam_center,  # (3,)
        }
        return joints, [], [], meta


    def create_3d_ray_coords(self, camera, joints_ds):
        multiplier = 1.0                        # avoid numerical instability

        coords = joints_ds.copy()

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
        return coords_world

    def normalize_screen_coordinates(self, X, w, h): 
        assert X.shape[-1] == 2
        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        return (X/w)*2 - [1, h/w]
    
    def __len__(self):
        return len(self.db['vcam0'])

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