# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
from easydict import EasyDict as edict

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
    
config = edict()

config.OUTPUT_DIR = 'output'
config.LOG_DIR = 'log'
config.DATA_DIR = ''
config.MODEL = 'multiview_transpose'
config.GPUS = '0,1'
config.WORKERS = 8
config.PRINT_FREQ = 100
config.WANDB = True
config.WANDB_LOG_IMG = False
config.WANDB_PROJECT = 'mpl'
config.WANDB_ENTITY = 'saghasemzadeh'
config.LOG_IMAGE_FREQ = 1000
config.LOG_WANDB_DIR = '/globalscratch/users/a/b/abolfazl/PPT_files/wandb/'
config.MPJPE_PER_KEYPOINT = True
config.TRI_2D_GT = False
config.TARGET_COORDS = False
config.DOWNSAMPLE = 16
config.SEED = 0
config.VALIDATE_ON_TWO_DATASETS = False
config.NOT_CONSIDER_SOME_KP_IN_EVAL = None
config.SLURM_JOB_ID = None

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# common params for NETWORK
config.NETWORK = edict()
config.NETWORK.NAME = 'pose_hrnet'
config.NETWORK.INIT_WEIGHTS = True
config.NETWORK.PRETRAINED = ''
config.NETWORK.NUM_JOINTS = 17
config.NETWORK.TAG_PER_JOINT = True
config.NETWORK.TARGET_TYPE = 'gaussian'
config.NETWORK.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
config.NETWORK.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
config.NETWORK.PATCH_SIZE = [64, 64]
config.NETWORK.SIGMA = 2
config.NETWORK.HIDDEN_HEATMAP_DIM = -1
config.NETWORK.TRANSFORMER_DEPTH = 2
config.NETWORK.TRANSFORMER_HEADS = 2
config.NETWORK.TRANSFORMER_MLP_RATIO = 2
config.NETWORK.POS_EMBEDDING_TYPE = 'learnable' 
config.NETWORK.DIM = 2
config.NETWORK.MULTI_TRANSFORMER_DEPTH = [12, 12]
config.NETWORK.MULTI_TRANSFORMER_HEADS = [16, 16]
config.NETWORK.MULTI_DIM = [48, 48]
config.NETWORK.INIT = False
config.NETWORK.NUM_BRANCHES = 1
config.NETWORK.BASEconfigHANNEL = 32
config.NETWORK.EXTRA = edict()
config.NETWORK.NO_VISUAL_TOKEN_FUSION = False
config.NETWORK.POSE_3D_EMB_LEARNABLE = False

config.NETWORK.TRANSFORMER_DROP_RATE = 0
config.NETWORK.TRANSFORMER_ATTN_DROP_RATE = 0
config.NETWORK.TRANSFORMER_DROP_PATH_RATE = 0.1
config.NETWORK.TRANSFORMER_ADD_CONFIDENCE_INPUT = False
config.NETWORK.TRANSFORMER_MULT_CONFIDENCE_EMB = False
config.NETWORK.TRANSFORMER_CONCAT_CONFIDENCE_EMB = False
config.NETWORK.TRANSFORMER_CONFIDENCE_INPUT_AS_THIRD = False
config.NETWORK.TRANSFORMER_LINEAR_WEIGHTED_MEAN = False
config.NETWORK.TRANSFORMER_ADD_3D_POS_ENCODING_IN_SPATIAL = False
config.NETWORK.TRANSFORMER_INPUT_RAYS_AS_TOKEN = False
config.NETWORK.TRANSFORMER_ADD_3D_POS_ENCODING_TO_RAYS = False
config.NETWORK.TRANSFORMER_CONF_ATTENTION_UNCERTAINTY_WEIGHT = False
config.NETWORK.TRANSFORMER_MULTIPLE_SPATIAL_BLOCKS = False
config.NETWORK.TRANSFORMER_NO_SPT = False
config.NETWORK.TRANSFORMER_NO_FPT = False
config.NETWORK.TRANSFORMER_CONFIDENCE_IN_FPT = False
config.NETWORK.TRANSFORMER_OUTPUT_HEAD_DEEP = False
config.NETWORK.TRANSFORMER_OUTPUT_HEAD_KADKHOD = False
config.NETWORK.TRANSFORMER_OUTPUT_HEAD_HIDDEN_DIM = 1024

config.NETWORK.TRANSFORMER_FPT_BLOCKS_VIEW_KEYPOINT_TOKENS = False


config.NETWORK.INIT_WEIGHTS_FROM = 'scratch'

config.NETWORK.MODEL_SIMPLE_DEEPER = False
config.NETWORK.MODEL_SIMPLE_KADKHODA = False
config.NETWORK.SIMPLE_MLP_HIDDEN_SIZE = 1024



config.LOSS = edict()
config.LOSS.TYPE = 'MPJPE'
config.LOSS.USE_TARGET_WEIGHT = True
config.LOSS.WEIGHT_AXIS = None

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = '../data/h36m/'
config.DATASET.ROOT_2DATSET = '../data/h36m/'
config.DATASET.ROOT_3DATSET = '../data/h36m/'
config.DATASET.ROOT_TRAIN = None
config.DATASET.ROOT_TEST = None
config.DATASET.DIR_MPL_DATA = 'MPL_data'
config.DATASET.TRAIN_DATASET = 'mixed_dataset'
config.DATASET.TEST_DATASET = 'multi_view_h36m'
config.DATASET.TRAIN_SUBSET = 'train'
config.DATASET.TEST_SUBSET = 'validation'
config.DATASET.ROOTIDX = 0
config.DATASET.DATA_FORMAT = 'jpg'
config.DATASET.BBOX = 2000
config.DATASET.CROP = True
config.DATASET.WITH_DAMAGE = True
config.DATASET.APPLY_NOISE = False
config.DATASET.APPLY_NOISE_TEST = False
config.DATASET.NOISE_LEVEL = 1.0
config.DATASET.APPLY_NOISE_CAMERAS = False
config.DATASET.R_NOISE_VALUE = 0.0
config.DATASET.T_NOISE_VALUE = 0.0
config.DATASET.APPLY_NOISE_MISSING = False
config.DATASET.MISSING_LEVEL = 0.0          # between 0 and 1. 0.0 means no missing
config.DATASET.USE_MMPOSE_TRAIN = False
config.DATASET.USE_MMPOSE_VAL = False
config.DATASET.USE_MMPOSE_TEST = False
config.DATASET.USE_3D_TRIANGULATED_MMPOSE_TRAIN = False
config.DATASET.USE_3D_TRIANGULATED_MMPOSE_TEST = False
config.DATASET.MIX_3D_AMASS_WITH_TRIANGULATED_MMPOSE_TRAIN = False
config.DATASET.MIX_3D_AMASS_WITH_TRIANGULATED_MMPOSE_TEST = False
config.DATASET.MIX_SMART_3D_AMASS_WITH_TRIANGULATED_MMPOSE_TRAIN = False
config.DATASET.MIX_SMART_3D_AMASS_WITH_TRIANGULATED_MMPOSE_TEST = False
config.DATASET.EPIPOLAR_ERROR_ACCEPTANCE_THRESHOLD = [0.06, 0.06, 0.09, 0.09]   # in meters! corresponds to KEYPOINTS_TO_MIX_AMASS_WITH_3D_TRIANGULATED_MMPOSE
config.DATASET.KEYPOINTS_TO_MIX_AMASS_WITH_3D_TRIANGULATED_MMPOSE = [5, 6, 11, 12]

config.DATASET.MIX_AMASS_WITH_MMPOSE_WHEN_USE_MMPOSE = False
config.DATASET.MIX_GT_WITH_MMPOSE_WHEN_USE_MMPOSE = False       # if true, it will mix the gt with mmpose when use mmpose (keypoints_to_mix)
config.DATASET.KEYPOINTS_TO_MIX = [0, 1, 4]

config.DATASET.CAMERA_MANUAL_ORDER = False
config.DATASET.VAL_ON_TRAIN = False
config.DATASET.DATASET_TYPE = 'annot_different_scene_same_cams'   # annot_different_scene_same_cams, annot_same_scene_different_cams, annot_different_scene_different_cams
config.DATASET.TARGET_NORMALIZED_3D = False
config.DATASET.INPUTS_NORMALIZED = False
config.DATASET.NORMALIZE_CAMERAS = False
config.DATASET.NORMALIZE_ROOM = False
config.DATASET.USE_T = False
config.DATASET.NO_AUGMENTATION = False
config.DATASET.NO_AUGMENTATION_3D = False
config.DATASET.OUTPUT_IN_METER = False
config.DATASET.CLIP_JOINTS = False

config.DATASET.TRAIN_VIEWS = None
config.DATASET.TEST_VIEWS = None
config.DATASET.ROOM_MIN_X = -100
config.DATASET.ROOM_MAX_X = 100
config.DATASET.ROOM_MIN_Y = -100
config.DATASET.ROOM_MAX_Y = 100
config.DATASET.ROOM_CENTER = [0, 0, 0]
config.DATASET.CENTERALIZE_ROOT_FIRST = False
config.DATASET.FLIP_3D = False
config.DATASET.ROTATE_3D = False
config.DATASET.USE_GRID = True
config.DATASET.BUG_TEST_3D_EMB = False
config.DATASET.TRAIN_ON_ALL_CAMERAS = False
config.DATASET.TEST_ON_ALL_CAMERAS = False
config.DATASET.N_VIEWS_TRAIN_TEST_ALL = 2
config.DATASET.AMASS_VAL_LOCATED = False
config.DATASET.AMASS_DATASET_TYPE = None

config.DATASET.TRAIN_N_SAMPLES = None
config.DATASET.TEST_N_SAMPLES = None
config.DATASET.SWITCH_X_Z = False
config.DATASET.SWITCH_X_Y = False
config.DATASET.SWITCH_Y_Z = False
config.DATASET.SWITCH_Z_X_Y = False
config.DATASET.SWITCH_Y_Z_X = False
config.DATASET.AMASS_DATA_NO_AXIS_SWAP = False
config.DATASET.CMU_CALIB = None
config.DATASET.TRAIN_CMU_CALIB = ['171204_pose4']
config.DATASET.TEST_CMU_CALIB = ['171204_pose4']
config.DATASET.TRAIN_H36M_CALIB_ACTORS = [9]
config.DATASET.TEST_H36M_CALIB_ACTORS = [9]

config.DATASET.PENALIZE_CONFIDENCE = 'no'  # exp_error, linear, exp_sqrt
config.DATASET.PENALIZE_CONFIDENCE_FACTOR = 1.0
config.DATASET.PENALIZE_CONFIDENCE_A = 0.83206918
config.DATASET.PENALIZE_CONFIDENCE_B = 0.00850464

config.DATASET.TRAIN_CMU_FILTERED_NEW = True
config.DATASET.TEST_CMU_FILTERED_NEW = False

config.DATASET.TRAIN_USE_CMU_OLD_DATASETS = True
config.DATASET.TEST_USE_CMU_OLD_DATASETS = True
config.DATASET.TRAIN_CMU_DATASET_NAME = 'annot_standard_7train_2val_filtered_5_64'
config.DATASET.TEST_CMU_DATASET_NAME = 'annot_standard_7train_2val_filtered_5_64'
config.DATASET.TRAIN_MMPOSE_TYPE = 'mmpose_rtm_coco'
config.DATASET.TEST_MMPOSE_TYPE = 'mmpose_rtm_coco'
config.DATASET.TRAIN_USE_AMASS_OLD_DATASETS = True
config.DATASET.TEST_USE_AMASS_OLD_DATASETS = True

config.DATASET.TRAIN_USE_H36M_OLD_DATASETS = True
config.DATASET.TEST_USE_H36M_OLD_DATASETS = True
config.DATASET.TRAIN_H36M_DATASET_NAME = 'annot_filtered_5_64'
config.DATASET.TEST_H36M_DATASET_NAME = 'annot_filtered_5_64'
config.DATASET.FILTER_GROUPINGS = True
config.DATASET.TRAIN_FILTER_GROUPINGS = True
config.DATASET.TEST_FILTER_GROUPINGS = True

config.DATASET.TRAIN_FILTER_CMU_WRONG_CASES = False
config.DATASET.TEST_FILTER_CMU_WRONG_CASES = False

config.DATASET.TRAIN_USE_AMASS_NEW_DATASETS_WITH_OLD_WAY = False
config.DATASET.TEST_USE_AMASS_NEW_DATASETS_WITH_OLD_WAY = False

config.DATASET.ONLY_KEEP_INSIDE_ROOM = False
config.DATASET.ONLY_KEEP_IF_IN_CALIBS_ACTORS = False

config.DATASET.TRAIN_ON_ALL_AMASS = False

config.DATASET.TRAIN_PLACE_PERSON_IN_CENTER = False
config.DATASET.TEST_PLACE_PERSON_IN_CENTER = False

config.DATASET.BRING_AMASS_ROOT_TO_ROOM_CENTER = False
config.DATASET.INTRINSIC_TO_METERS_IF_OUTPUT_IN_METER = False

config.DATASET.USE_HELPER_CAMERAS = False
config.DATASET.TRAIN_VIEWS_HELPER = None
config.DATASET.ROOT_VIEWS_HELPER = None     # path to cmu panoptic dataset

config.DATASET.USE_H36M_CAMERAS_ON_CMU = False
config.DATASET.USE_CMU_CAMERAS_ON_CMU = False
config.DATASET.USE_CMU_CAMERAS_ON_H36M = False

config.DATASET.FLIP_LOWER_BODY_KP_TEST = False

config.DATASET.REDUCE_CONFIDENCE_CLOSE_JOINTS = False


config.DATASET.CMU_KEYPOINT_STANDARD = 'h36m' # h36m, coco

# training data augmentation
config.DATASET.SCALE_FACTOR = 0
config.DATASET.ROT_FACTOR = 0

# train
config.TRAIN = edict()
config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 110]
config.TRAIN.LR = 0.001

config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 140

config.TRAIN.RESUME = False

config.TRAIN.BATCH_SIZE = 8
config.TRAIN.SHUFFLE = True

config.TRAIN.DO_FUSION = True
config.TRAIN.SMART_PSEUDO_TRAINING = False
config.TRAIN.EPIPOLAR_ERROR_THRESHOLD = 5

# testing
config.TEST = edict()
config.TEST.BATCH_SIZE = 8
config.TEST.STATE = ''
config.TEST.POST_PROCESS = False
config.TEST.SHIFT_HEATMAP = False
config.TEST.USE_GT_BBOX = False
config.TEST.IMAGE_THRE = 0.1
config.TEST.NMS_THRE = 0.6
config.TEST.OKS_THRE = 0.5
config.TEST.IN_VIS_THRE = 0.0
config.TEST.BBOX_FILE = ''
config.TEST.BBOX_THRE = 1.0
config.TEST.MATCH_IOU_THRE = 0.3
config.TEST.DETECTOR = 'fpn_dcn'
config.TEST.DETECTOR_DIR = ''
config.TEST.MODEL_FILE = ''
config.TEST.HEATMAP_LOCATION_FILE = 'predicted_heatmaps.h5'
config.TEST.PRED_GT_LOCATION_FILE = 'preds_gt.pkl'
config.TEST.DO_FUSION = True

# debug
config.DEBUG = edict()
config.DEBUG.DEBUG = True
config.DEBUG.SAVE_BATCH_IMAGES_GT = True
config.DEBUG.SAVE_BATCH_IMAGES_PRED = True
config.DEBUG.SAVE_HEATMAPS_GT = True
config.DEBUG.SAVE_HEATMAPS_PRED = True

# pictorial structure
config.PICT_STRUCT = edict()
config.PICT_STRUCT.FIRST_NBINS = 16
config.PICT_STRUCT.PAIRWISE_FILE = ''
config.PICT_STRUCT.RECUR_NBINS = 2
config.PICT_STRUCT.RECUR_DEPTH = 10
config.PICT_STRUCT.LIMB_LENGTH_TOLERANCE = 150
config.PICT_STRUCT.GRID_SIZE = 2000
config.PICT_STRUCT.DEBUG = False
config.PICT_STRUCT.TEST_PAIRWISE = False
config.PICT_STRUCT.SHOW_ORIIMG = False
config.PICT_STRUCT.SHOW_CROPIMG = False
config.PICT_STRUCT.SHOW_HEATIMG = False


def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array(
                [eval(x) if isinstance(x, str) else x for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array(
                [eval(x) if isinstance(x, str) else x for x in v['STD']])
    if k == 'NETWORK':
        if 'HEATMAP_SIZE' in v:
            if isinstance(v['HEATMAP_SIZE'], int):
                v['HEATMAP_SIZE'] = np.array(
                    [v['HEATMAP_SIZE'], v['HEATMAP_SIZE']])
            else:
                v['HEATMAP_SIZE'] = np.array(v['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file, 'r') as f:
        exp_config = edict(yaml.load(f, Loader=Loader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def update_dir(model_dir, log_dir, data_dir):
    if model_dir:
        config.OUTPUT_DIR = model_dir

    if log_dir:
        config.LOG_DIR = log_dir

    if data_dir:
        config.DATA_DIR = data_dir

    config.DATASET.ROOT = os.path.join(config.DATA_DIR, config.DATASET.ROOT)

    config.TEST.BBOX_FILE = os.path.join(config.DATA_DIR, config.TEST.BBOX_FILE)

    config.NETWORK.PRETRAINED = os.path.join(config.DATA_DIR,
                                             config.NETWORK.PRETRAINED)


def get_model_name(cfg):
    name = '{model}'.format(
        model=cfg.MODEL)
    deconv_suffix = 'd1'
    full_name = '{height}x{width}_{name}_{deconv_suffix}'.format(
        height=cfg.NETWORK.IMAGE_SIZE[1],
        width=cfg.NETWORK.IMAGE_SIZE[0],
        name=name,
        deconv_suffix=deconv_suffix)

    return name, full_name


if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])

