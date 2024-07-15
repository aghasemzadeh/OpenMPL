# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.loss import *
from core.function_mpl import validate
from utils.utils import create_logger

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument(
        '--frequent',
        help='frequency of logging',
        default=config.PRINT_FREQ,
        type=int)
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument(
        '--state',
        help='the state of model which is used to test (best or final)',
        default='final',
        type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--model-file', help='model state file', type=str)
    parser.add_argument(
        '--flip-test', help='use flip test', action='store_true')
    parser.add_argument(
        '--post-process', help='use post process', action='store_true')
    parser.add_argument(
        '--shift-heatmap', help='shift heatmap', action='store_true')

    # philly
    parser.add_argument(
        '--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument(
        '--dataDir', help='data directory', type=str, default='')
    parser.add_argument(
        '--data-format', help='data format', type=str, default='')
    parser.add_argument(
        '--enable-wandb', help='enable wandb', action='store_true', default=False
    )
    parser.add_argument(
        '--test-dataset', help='test dataset', type=str, default=None
    )
    parser.add_argument(
        '--use-mmpose-val', help='use mmpose val', action='store_true', default=None
    )
    parser.add_argument(
        '--not-use-mmpose-val', help='not use mmpose val', action='store_false', dest='use_mmpose_val',default=None
    )
    parser.add_argument(
        '--batch-size', help='batch-size', type=int, default=None
    )
    parser.add_argument(
        '--filter-cmu-wrong-cases', help='filter cmu wrong cases', action='store_true', default=False
    )
    parser.add_argument(
        '--test-cmu-dataset-name', help='test cmu dataset name', type=str, default=None
    )
    parser.add_argument(
        '--test-mmpose-type', help='test mmpose type', type=str, default=None
    )
    parser.add_argument(
        '--n-samples', help='n samples', type=int, default=None
    )

    args = parser.parse_args()

    update_dir(args.modelDir, args.logDir, args.dataDir)

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.data_format:
        config.DATASET.DATA_FORMAT = args.data_format
    if args.workers:
        config.WORKERS = args.workers
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.state:
        config.TEST.STATE = args.state
    if not args.enable_wandb:
        config.WANDB = False
    if args.use_mmpose_val is not None:
        config.DATASET.USE_MMPOSE_VAL = args.use_mmpose_val
    if args.test_dataset is not None:
        config.DATASET.TEST_DATASET = args.test_dataset
    if args.batch_size is not None:
        config.TEST.BATCH_SIZE = args.batch_size
    if args.filter_cmu_wrong_cases:
        config.DATASET.TEST_FILTER_CMU_WRONG_CASES = args.filter_cmu_wrong_cases
    if args.test_cmu_dataset_name is not None:
        config.DATASET.TEST_CMU_DATASET_NAME = args.test_cmu_dataset_name
    if args.test_mmpose_type is not None:
        config.DATASET.TEST_MMPOSE_TYPE = args.test_mmpose_type
    if args.n_samples is not None:
        config.DATASET.TEST_N_SAMPLES = args.n_samples


def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # backbone_model = eval('models.' + config.BACKBONE_MODEL + '.get_pose_net')(
    #     config, is_train=False)

    model = eval('models.' + config.MODEL + '.get_multiview_mpl_net')(config, is_train=False)
    
    if config.TEST.MODEL_FILE:
        if config.TEST.MODEL_FILE == 'no_fine_tuning':
            pass
        else:
            logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
            model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        model_path = 'model_best.pth.tar' if config.TEST.STATE.startswith('best') else 'final_state.pth.tar'
        model_path = 'checkpoint.pth.tar' if config.TEST.STATE == 'checkpoint' else model_path
        model_state_file = os.path.join(final_output_dir, model_path)
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file),  strict=False)

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    # criterion = Weighted_MPJPE().cuda()
    # criterion = MPJPE().cuda()
    if config.LOSS.TYPE == 'Weighted_MPJPE':
        criterion = Weighted_MPJPE().cuda()
    elif config.LOSS.TYPE == 'MPJPE':
        criterion = MPJPE(config).cuda()
    elif config.LOSS.TYPE == 'JointsL1Loss':
        criterion = KeypointsPoseL1Loss(config).cuda()
    elif config.LOSS.TYPE == 'JointsMSELoss':
        criterion = KeypointsPoseMSELoss(config).cuda()
    elif config.LOSS.TYPE == 'MPJPE_KADKHODA':
        criterion = MPJPE_KADKHODA(config).cuda()
    else:
        raise ValueError('Loss type not supported')

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    valid_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    # evaluate on validation set
    validate(config, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)


if __name__ == '__main__':
    main()
