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
import logging
import copy
from utils.calib import smart_pseudo_remove_weight

from dataset.joints_dataset_mpl import JointsDataset_MPL


logger = logging.getLogger(__name__)

# # set random seed
# np.random.seed(0)
# random.seed(0)

class MultiViewCMUpanoptic_MPL(JointsDataset_MPL):

    def __init__(self, cfg, image_set, is_train, transform=None, is_mmpose=False):
        super().__init__(cfg, image_set, is_train, transform)
        self.num_joints = 17
        self.CMU_KEYPOINT_STANDARD = cfg.DATASET.CMU_KEYPOINT_STANDARD
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
        self.val_on_train = cfg.DATASET.VAL_ON_TRAIN
        self.is_train = is_train
            
        self.dataset_type = cfg.DATASET.DATASET_TYPE
        self.use_mmpose = False
        if is_mmpose:
            self.use_mmpose = True
        else:
            if cfg.DATASET.USE_MMPOSE_TRAIN and is_train:
                self.use_mmpose = True
            elif (cfg.DATASET.USE_MMPOSE_VAL or cfg.DATASET.USE_MMPOSE_TEST) and not is_train:
                self.use_mmpose = True
        
        self.cmu_filtered_new = cfg.DATASET.TRAIN_CMU_FILTERED_NEW if is_train else cfg.DATASET.TEST_CMU_FILTERED_NEW
        self.cmu_old_datasets = cfg.DATASET.TRAIN_USE_CMU_OLD_DATASETS if is_train else cfg.DATASET.TEST_USE_CMU_OLD_DATASETS
        self.cmu_dataset_name = cfg.DATASET.TRAIN_CMU_DATASET_NAME if is_train else cfg.DATASET.TEST_CMU_DATASET_NAME
        
        if self.cmu_old_datasets:
            dataset_path = osp.join(self.root, self.dir_mpl_data, self.dataset_type.replace('annot', 'annot_filtered'))
            if self.use_mmpose:
                dataset_path = osp.join(self.root, self.dir_mpl_data, self.dataset_type.replace('annot', 'datasets_mmpose/annot_mmpose_filtered'))
        else:
            dataset_folder_name = 'datasets_mmpose' if self.use_mmpose else 'datasets' 
            dataset_folder_name_2 = self.cmu_dataset_name + '_' + self.mmpose_type if self.use_mmpose else self.cmu_dataset_name
            dataset_path = osp.join(self.root, self.dir_mpl_data, dataset_folder_name, dataset_folder_name_2)
        if self.val_on_train:
            anno_file = osp.join(dataset_path, 'cmu_panoptic_train.pkl')
            
        elif cfg.DATASET.CROP:
            anno_file = osp.join(dataset_path, 'cmu_panoptic_{}.pkl'.format(image_set))
        else:
            raise ValueError('Not implemented yet')

        self.db = self.load_db(anno_file)
        if self.filter_cmu_wrong_cases:
            self.db = self.filter_faulty_cases(self.db)
        if not self.run_on_all_cameras:
            self.db = self.filter_db(self.db)   # remove images that are with other cameras

        self.u2a_mapping = super().get_mapping()
        super().do_mapping()

        self.grouping = self.get_group(self.db)
        if cfg.DATASET.TRAIN_N_SAMPLES is not None and is_train:
            self.grouping = self.grouping[:cfg.DATASET.TRAIN_N_SAMPLES]
        if cfg.DATASET.TEST_N_SAMPLES is not None and not is_train:
            self.grouping = self.grouping[:cfg.DATASET.TEST_N_SAMPLES]
        self.group_size = len(self.grouping)
        logger.info('=> {} num samples: {}'.format(image_set, self.group_size))
        
        if self.use_h36m_cameras_on_cmu:
            actors = self.h36m_calib_actors
            self.h36m_cameras = self.load_all_cameras_h36m(calib_file=osp.join(self.root_3, 'camera_data.pkl'), actors=actors)
            self.n_all_cameras = len(self.h36m_cameras)    
            self.all_camera_ids = list(self.h36m_cameras.keys())
            self.n_camera_setups = len(self.h36m_cameras[self.all_camera_ids[0]])
            
        
        if self.use_cmu_cameras_on_cmu:
            cmu_calibs = self.cmu_calibs_train if is_train else self.cmu_calibs_val
            if isinstance(cmu_calibs, str):
                cmu_calibs = [cmu_calibs]
            self.cmu_cameras = self.load_all_cameras_cmu(cmu_calibs)
            
            self.cmu_cameras = {view: v for view, v in self.cmu_cameras.items() if view in self.views}
            self.n_all_cameras = len(self.cmu_cameras)    
            self.all_camera_ids = list(self.cmu_cameras.keys())
            self.n_camera_setups = len(self.cmu_cameras[self.all_camera_ids[0]])
        
        self.R_noise_value = cfg.DATASET.R_NOISE_VALUE
        self.T_noise_value = cfg.DATASET.T_NOISE_VALUE
        if cfg.DATASET.APPLY_NOISE_CAMERAS and self.is_train:
            self.add_noise_to_cameras()
        
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
        return None

    def load_db(self, dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
            return dataset
    
    

    def get_group(self, db):
        grouping = {}
        nitems = len(db)
        if self.use_h36m_cameras_on_cmu:
            views = [3, 6, 12, 13, 23]
            CAM_IX = {x: i for i, x in enumerate(views[:self.n_views])}
        if self.run_on_all_cameras:
            views = self.all_views_cmu
            CAM_IX = {x: i for i, x in enumerate(views)}
        else:
            views = self.views
            CAM_IX = {x: i for i, x in enumerate(self.views)}
        for i in range(nitems):
            keystr = self.get_key_str(db[i])
            camera_id = CAM_IX[db[i]['camera_id']]
            if keystr not in grouping:
                grouping[keystr] = [-1] * len(views)
            grouping[keystr][camera_id] = i

        filtered_grouping = []
        for _, v in grouping.items():
            if np.all(np.array(v) != -1):
                filtered_grouping.append(v)
        # if self.use_mmpose:
        #     pass
        # else:
        #     if self.val_on_train:
        #         filtered_grouping = filtered_grouping[::2]
        #         filtered_grouping = random.sample(filtered_grouping, 300)
        #     elif self.is_train:
        #         filtered_grouping = filtered_grouping[::2]     
        #     else:
        #         filtered_grouping = filtered_grouping[::6]
                
        # if self.train_n_samples is not None and self.is_train:
        #     filtered_grouping = filtered_grouping[:self.train_n_samples]
        # if self.test_n_samples is not None and not self.is_train:
        #     filtered_grouping = filtered_grouping[:self.test_n_samples]

        return filtered_grouping

    def __getitem__(self, idx):
        input, target, weight, meta = [], [], [], []
        # items = self.grouping[idx]
        items = copy.deepcopy(self.grouping[idx])
        if self.run_on_all_cameras:
            views = np.random.choice(self.all_views_cmu, self.n_views, replace=False)
            items = [items[self.all_views_cmu.index(v)] for v in views]
        if self.normalize_room:
            room_x_scale = (self.room_max_x - self.room_min_x) / 2
            room_y_scale = (self.room_max_y - self.room_min_y) / 2
            room_center = np.array([(self.room_max_x + self.room_min_x) / 2, 0, (self.room_max_y + self.room_min_y) / 2])
        
        camera_setup_to_use = None
        if self.use_h36m_cameras_on_cmu or self.use_cmu_cameras_on_cmu:
            camera_setup_to_use = np.random.random_integers(0, self.n_camera_setups-1)
            camera_ids = np.array(self.views)
        for ix, item in enumerate(items):
            camera_id = None
            if self.use_h36m_cameras_on_cmu or self.use_cmu_cameras_on_cmu:
                camera_id = camera_ids[ix]
            i, t, w, m = super().__getitem__(item, camera_id, camera_setup_to_use)
            if self.normalize_room:
                t = self.normalize_pose3d_coordinates(t, room_center, room_x_scale, room_y_scale)
                m['rays'] = self.normalize_pose3d_coordinates(m['rays'], room_center, room_x_scale, room_y_scale)
                m['cam_center'] = self.normalize_pose3d_coordinates(m['cam_center'], room_center, room_x_scale, room_y_scale)
                m['room_scaled'] = True
                m['room_x_scale'] = max(room_x_scale, room_y_scale)
                m['room_y_scale'] = max(room_x_scale, room_y_scale)
                m['room_center'] = room_center
                m['room_scaled_equal'] = True
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
        return 's_{:02}_pose_{}_imgid_{:08}'.format(
            datum['subject'], datum['pose_id'],
            datum['image_id'])
        
    def filter_faulty_cases(self, db):
        # self.wrong_cases = ['171204_pose5_00006400', '171204_pose5_00004253', '171204_pose5_00002510', '171204_pose5_00026417', '171204_pose6_00014295', '171204_pose6_00009655', '171204_pose5_00025555', '171204_pose5_00001828', '171204_pose6_00021737', '171204_pose5_00021103', '171204_pose5_00007966', '171204_pose5_00005346', '171204_pose5_00012126', '171204_pose5_00008043', '171204_pose5_00003483', '171204_pose5_00022525', '171204_pose5_00007483', '171204_pose5_00022065', '171204_pose5_00022590', '171204_pose5_00006210', '171204_pose5_00003514', '171204_pose5_00002006', '171204_pose5_00005567', '171204_pose5_00001643', '171204_pose5_00026524', '171204_pose5_00003854', '171204_pose5_00015569', '171204_pose5_00007831', '171204_pose5_00025627', '171204_pose5_00025113', '171204_pose5_00015234', '171204_pose5_00004945', '171204_pose5_00007316', '171204_pose5_00003704', '171204_pose5_00025254', '171204_pose5_00007990', '171204_pose5_00015836', '171204_pose5_00022435', '171204_pose5_00006448', '171204_pose5_00024363', '171204_pose5_00015546', '171204_pose6_00010708', '171204_pose5_00003874', '171204_pose5_00015834', '171204_pose5_00021178', '171204_pose5_00005233', '171204_pose5_00015398', '171204_pose5_00025068', '171204_pose5_00021110', '171204_pose5_00008465', '171204_pose5_00007417', '171204_pose5_00019873', '171204_pose5_00002527', '171204_pose5_00007173', '171204_pose5_00013912', '171204_pose5_00010032', '171204_pose5_00002565', '171204_pose5_00016406', '171204_pose5_00008901', '171204_pose5_00023402']
        self.wrong_cases = ['171204_pose5_00000478', '171204_pose5_00007177', '171204_pose5_00010040', '171204_pose5_00025087', '171204_pose5_00004253', '171204_pose5_00007831', '171204_pose5_00002650', '171204_pose5_00005567', '171204_pose5_00003874', '171204_pose5_00025216', '171204_pose5_00026524', '171204_pose5_00025207', '171204_pose5_00025068', '171204_pose5_00003491', '171204_pose5_00013912', '171204_pose5_00008043', '171204_pose5_00024363', '171204_pose5_00020806', '171204_pose5_00005346', '171204_pose5_00005508', '171204_pose5_00026417', '171204_pose5_00003579', '171204_pose5_00008901', '171204_pose5_00023402', '171204_pose5_00002292', '171204_pose5_00002006', '171204_pose6_00001382', '171204_pose5_00003914', '171204_pose5_00003854', '171204_pose5_00022473', '171204_pose5_00003514', '171204_pose5_00002175', '171204_pose5_00006400', '171204_pose5_00022525', '171204_pose5_00006449', '171204_pose5_00007352', '171204_pose5_00025555', '171204_pose5_00022251', '171204_pose5_00007337', '171204_pose5_00003417', '171204_pose5_00019545', '171204_pose5_00007966', '171204_pose5_00015546', '171204_pose5_00023900', '171204_pose5_00026850', '171204_pose5_00015398', '171204_pose5_00015234', '171204_pose5_00007483', '171204_pose5_00025113', '171204_pose5_00008465', '171204_pose5_00022590', '171204_pose5_00007990', '171204_pose5_00025254', '171204_pose5_00015834', '171204_pose5_00001828', '171204_pose5_00025659', '171204_pose5_00007324', '171204_pose5_00021110', '171204_pose5_00015569', '171204_pose6_00014295', '171204_pose5_00007316', '171204_pose5_00004945', '171204_pose5_00022435', '171204_pose5_00015753', '171204_pose5_00022410', '171204_pose5_00008696', '171204_pose5_00003880', '171204_pose6_00009655', '171204_pose5_00020505', '171204_pose5_00026822', '171204_pose6_00022094', '171204_pose6_00014635', '171204_pose5_00019873', '171204_pose5_00020363', '171204_pose5_00003571', '171204_pose5_00008629', '171204_pose5_00007398', '171204_pose5_00008627', '171204_pose5_00003483', '171204_pose5_00007417', '171204_pose5_00007885', '171204_pose5_00022065', '171204_pose5_00016406', '171204_pose5_00025264', '171204_pose5_00021178', '171204_pose5_00002565', '171204_pose5_00002527', '171204_pose5_00007834', '171204_pose5_00021103', '171204_pose5_00025917', '171204_pose5_00003704', '171204_pose5_00006210', '171204_pose5_00001643', '171204_pose5_00006448', '171204_pose5_00005233', '171204_pose5_00023507', '171204_pose5_00013911', '171204_pose5_00012126', '171204_pose5_00020703', '171204_pose5_00025627', '171204_pose5_00002510', '171204_pose5_00015836', '171204_pose5_00007173', '171204_pose5_00002149', '171204_pose6_00021737', '171204_pose6_00010708', '171204_pose5_00010032', '171204_pose6_00001274']
        if self.cmu_dataset_name == 'annot_standard_coco_7train_2val_all_cams_no_bad_annot_all_cams_filtered_5_64':
            self.wrong_cases = ['171204_pose5_00026493', '171204_pose5_00004921', '171204_pose5_00003748', '171204_pose5_00008113', '171204_pose5_00002424', '171204_pose5_00020129', '171204_pose5_00026021', '171204_pose6_00002462', '171204_pose5_00002490', '171204_pose5_00026176', '171204_pose5_00022694', '171204_pose5_00021101', '171204_pose5_00007250', '171204_pose5_00025726', '171204_pose5_00005393', '171204_pose5_00003825', '171204_pose5_00020465', '171204_pose6_00017843', '171204_pose6_00018819', '171204_pose5_00003601', '171204_pose6_00013567', '171204_pose5_00006529', '171204_pose5_00002685', '171204_pose5_00021623', '171204_pose5_00021468', '171204_pose5_00001385', '171204_pose5_00003554', '171204_pose5_00023617', '171204_pose5_00001868', '171204_pose5_00003475', '171204_pose5_00014649', '171204_pose5_00022705', '171204_pose5_00025730', '171204_pose6_00017441', '171204_pose5_00014691', '171204_pose5_00021601', '171204_pose5_00003963', '171204_pose5_00024474', '171204_pose5_00004940', '171204_pose5_00020119', '171204_pose5_00025699', '171204_pose5_00015152', '171204_pose5_00020881', '171204_pose5_00025468', '171204_pose5_00008733', '171204_pose5_00008597', '171204_pose5_00007893', '171204_pose5_00007307', '171204_pose5_00002804', '171204_pose5_00005376', '171204_pose5_00026150', '171204_pose5_00001856', '171204_pose5_00001502', '171204_pose5_00015198', '171204_pose5_00007246', '171204_pose5_00021730', '171204_pose5_00001592', '171204_pose6_00010843', '171204_pose5_00021704', '171204_pose5_00005932', '171204_pose6_00015383', '171204_pose5_00007333', '171204_pose5_00021744', '171204_pose5_00015535', '171204_pose6_00014148', '171204_pose5_00002486', '171204_pose5_00017658', '171204_pose5_00008465', '171204_pose5_00003671', '171204_pose5_00004956', '171204_pose6_00014160', '171204_pose5_00020438', '171204_pose5_00015175', '171204_pose5_00022075', '171204_pose6_00014516', '171204_pose5_00025659', '171204_pose5_00007541', '171204_pose5_00026129', '171204_pose5_00008297', '171204_pose5_00021645', '171204_pose5_00022169', '171204_pose5_00004828', '171204_pose5_00007474', '171204_pose6_00000447', '171204_pose5_00007600', '171204_pose5_00007209', '171204_pose5_00001860', '171204_pose5_00026172', '171204_pose6_00001335', '171204_pose5_00020733', '171204_pose5_00025679', '171204_pose5_00007459', '171204_pose5_00002860', '171204_pose5_00025842', '171204_pose5_00022618', '171204_pose5_00007367', '171204_pose5_00004802', '171204_pose5_00007532', '171204_pose5_00001833', '171204_pose5_00004343', '171204_pose5_00008287', '171204_pose5_00025055', '171204_pose5_00003571', '171204_pose5_00017665', '171204_pose5_00008734', '171204_pose5_00025635', '171204_pose5_00002367', '171204_pose5_00005561', '171204_pose5_00020775', '171204_pose5_00026166', '171204_pose5_00026630', '171204_pose5_00003904', '171204_pose5_00008180', '171204_pose5_00026465', '171204_pose5_00007376', '171204_pose5_00024131', '171204_pose5_00004320', '171204_pose5_00026629', '171204_pose5_00002536', '171204_pose5_00007257', '171204_pose5_00004206', '171204_pose5_00003929', '171204_pose5_00007303', '171204_pose5_00022687', '171204_pose5_00014696', '171204_pose5_00020180', '171204_pose5_00008657', '171204_pose5_00026721', '171204_pose5_00004823', '171204_pose5_00022154', '171204_pose5_00007308', '171204_pose5_00012894', '171204_pose5_00026665', '171204_pose5_00005246', '171204_pose5_00022609', '171204_pose5_00007804', '171204_pose5_00003842', '171204_pose5_00015186', '171204_pose5_00025727', '171204_pose5_00022646', '171204_pose5_00025605', '171204_pose5_00005935', '171204_pose5_00015128', '171204_pose5_00008475', '171204_pose5_00001929', '171204_pose5_00004299', '171204_pose5_00004929', '171204_pose5_00003546', '171204_pose5_00003952', '171204_pose5_00001208', '171204_pose5_00022030', '171204_pose5_00006535', '171204_pose5_00021740', '171204_pose5_00003971', '171204_pose5_00015867', '171204_pose5_00002632', '171204_pose5_00022152']            
        elif self.cmu_dataset_name == 'annot_standard_coco_7train_2val_all_cams_no_bad_annot_no_vis_discard_filtered_5_64':
            self.wrong_cases = ['171204_pose5_00000478', '171204_pose5_00002175', '171204_pose5_00007834', '171204_pose5_00015753', '171204_pose5_00025917', '171204_pose5_00007177', '171204_pose5_00010040', '171204_pose5_00022410', '171204_pose5_00006449', '171204_pose5_00025087', '171204_pose5_00007352', '171204_pose5_00002650', '171204_pose5_00025216', '171204_pose5_00022251', '171204_pose5_00007337', '171204_pose5_00003417', '171204_pose5_00008696', '171204_pose5_00019545', '171204_pose5_00025207', '171204_pose5_00003880', '171204_pose5_00023507', '171204_pose5_00013911', '171204_pose5_00023900', '171204_pose5_00026850', '171204_pose5_00020505', '171204_pose5_00020703', '171204_pose6_00022094', '171204_pose5_00003491', '171204_pose5_00026822', '171204_pose6_00014635', '171204_pose5_00020363', '171204_pose5_00003571', '171204_pose5_00008629', '171204_pose5_00020806', '171204_pose5_00007398', '171204_pose5_00015836', '171204_pose5_00008627', '171204_pose5_00007885', '171204_pose5_00002149', '171204_pose5_00005508', '171204_pose5_00025264', '171204_pose5_00003579', '171204_pose5_00001828', '171204_pose5_00025659', '171204_pose5_00002292', '171204_pose5_00007324', '171204_pose6_00001274', '171204_pose6_00001382', '171204_pose5_00003914', '171204_pose5_00022473']
        db_new = []
        for i in range(len(db)):
            image_name = db[i]['image']
            if '{}_{}'.format(image_name.split('/')[0], image_name.split('/')[-1].split('_')[-1].replace('.jpg','')) in self.wrong_cases:
                continue
            db_new.append(db[i])
        
        return db_new
