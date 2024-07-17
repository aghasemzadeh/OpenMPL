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
import shutil

import numpy as np
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import *
from core.function_mpl import train
from core.function_mpl import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint, load_checkpoint
from utils.utils import create_logger, create_logger_sweep
import dataset
import models

import wandb
import yaml


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    parser.add_argument(
        '--frequent',
        help='frequency of logging',
        default=config.PRINT_FREQ,
        type=int)
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--train-batch-size', help='training batch size', type=int)
    parser.add_argument('--test-batch-size', help='test batch size', type=int)
    parser.add_argument('--apply-noise', help='apply noise during training', type=int)
    parser.add_argument('--noise-level', help='level of noise', type=float)
    parser.add_argument('--apply-noise-missing', help='apply noise during training', type=int)
    parser.add_argument('--missing-level', help='level of noise', type=float)
    parser.add_argument('--normalize-cameras', help='normalize cameras', type=int)
    parser.add_argument('--normalize-room', help='normalize room', type=int)
    parser.add_argument('--penalize-confidence', help='penalize confidence', type=str)
    parser.add_argument('--penalize-confidence-a', help='penalize confidence a', type=float)
    parser.add_argument('--penalize-confidence-b', help='penalize confidence b', type=float)
    parser.add_argument('--dim', help='dimension of the model', type=int)
    parser.add_argument('--transformer-heads', help='number of heads', type=int)
    parser.add_argument('--transformer-depth', help='depth of encoders', type=int)
    parser.add_argument('--pose-3d-encoding-learnable', help='pose 3d encoding learnable', type=int)
    parser.add_argument('--add-confidence-input', help='add confidence input', type=int)
    parser.add_argument('--input-rays-as-token', help='input rays as token', type=int)
    parser.add_argument('--exp-name', help='experiment name', type=str)
    parser.add_argument('--lr', help='initial lr', type=float)
    parser.add_argument('--slurm-job-id', help='slurm job id', type=str, default=None)

    parser.add_argument(
        '--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument(
        '--dataDir', help='data directory', type=str, default='')
    parser.add_argument(
        '--data-format', help='data format', type=str, default='')
    args = parser.parse_args()
    update_dir(args.modelDir, args.logDir, args.dataDir)
    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.data_format:
        config.DATASET.DATA_FORMAT = args.data_format
    if type(args.workers) == int:
        config.WORKERS = args.workers
    
    if args.train_batch_size:
        config.TRAIN.BATCH_SIZE = args.train_batch_size
    if args.test_batch_size:
        config.TEST.BATCH_SIZE = args.test_batch_size
    if type(args.apply_noise) == int:
        config.DATASET.APPLY_NOISE = bool(args.apply_noise)
    if type(args.noise_level) == float:
        config.DATASET.NOISE_LEVEL = args.noise_level
    if type(args.apply_noise_missing) == int:
        config.DATASET.APPLY_NOISE_MISSING = bool(args.apply_noise_missing)
    if type(args.missing_level) == float:
        config.DATASET.MISSING_LEVEL = args.missing_level
    if type(args.normalize_cameras) == int:
        config.DATASET.NORMALIZE_CAMERAS = bool(args.normalize_cameras)
    if type(args.normalize_room) == int:
        config.DATASET.NORMALIZE_ROOM = bool(args.normalize_room)
    if args.penalize_confidence:
        config.LOSS.PENALIZE_CONFIDENCE = args.penalize_confidence
    if args.penalize_confidence_a:
        config.LOSS.PENALIZE_CONFIDENCE_A = args.penalize_confidence_a
    if args.penalize_confidence_b:
        config.LOSS.PENALIZE_CONFIDENCE_B = args.penalize_confidence_b
    if args.dim:
        config.NETWORK.DIM = args.dim
    if args.transformer_heads:
        config.NETWORK.TRANSFORMER_HEADS = args.transformer_heads
    if args.transformer_depth:
        config.NETWORK.TRANSFORMER_DEPTH = args.transformer_depth
    if type(args.pose_3d_encoding_learnable) == int:
        config.NETWORK.POSE_3D_EMB_LEARNABLE = bool(args.pose_3d_encoding_learnable)
    if type(args.add_confidence_input) == int:
        config.NETWORK.POSEFORMER_ADD_CONFIDENCE_INPUT = bool(args.add_confidence_input)
    if type(args.input_rays_as_token) == int:
        config.NETWORK.POSEFORMER_INPUT_RAYS_AS_TOKEN = bool(args.input_rays_as_token)
    if args.lr:
        config.TRAIN.LR = args.lr
    if args.slurm_job_id:
        config.SLURM_JOB_ID = args.slurm_job_id
        
def update_config_file(conf, args):
    if args.train_batch_size:
        conf["TRAIN"]["BATCH_SIZE"] = args.train_batch_size
    if args.test_batch_size:
        conf["TEST"]["BATCH_SIZE"] = args.test_batch_size
    if type(args.apply_noise) == int:
        conf["DATASET"]["APPLY_NOISE"] = bool(args.apply_noise)
    if type(args.noise_level) == float:
        conf["DATASET"]["NOISE_LEVEL"] = args.noise_level
    if type(args.apply_noise_missing) == int:
        conf["DATASET"]["APPLY_NOISE_MISSING"] = bool(args.apply_noise_missing)
    if type(args.missing_level) == float:
        conf["DATASET"]["MISSING_LEVEL"] = args.missing_level
    if type(args.normalize_cameras) == int:
        conf["DATASET"]["NORMALIZE_CAMERAS"] = bool(args.normalize_cameras)
    if type(args.normalize_room) == int:
        conf["DATASET"]["NORMALIZE_ROOM"] = bool(args.normalize_room)
    if args.penalize_confidence:
        conf["LOSS"]["PENALIZE_CONFIDENCE"] = args.penalize_confidence
    if args.penalize_confidence_a:
        conf["LOSS"]["PENALIZE_CONFIDENCE_A"] = args.penalize_confidence_a
    if args.penalize_confidence_b:
        conf["LOSS"]["PENALIZE_CONFIDENCE_B"] = args.penalize_confidence_b
    if args.dim:
        conf["NETWORK"]["DIM"] = args.dim
    if args.transformer_heads:
        conf["NETWORK"]["TRANSFORMER_HEADS"] = args.transformer_heads
    if args.transformer_depth:
        conf["NETWORK"]["TRANSFORMER_DEPTH"] = args.transformer_depth
    if type(args.pose_3d_encoding_learnable) == int:
        conf["NETWORK"]["POSE_3D_EMB_LEARNABLE"] = bool(args.pose_3d_encoding_learnable)
    if type(args.add_confidence_input) == int:
        conf["NETWORK"]["POSEFORMER_ADD_CONFIDENCE_INPUT"] = bool(args.add_confidence_input)
    if type(args.input_rays_as_token) == int:
        conf["NETWORK"]["POSEFORMER_INPUT_RAYS_AS_TOKEN"] = bool(args.input_rays_as_token)
    if args.lr:
        conf["TRAIN"]["LR"] = args.lr
    return conf

def main():

    args = parse_args()
    reset_config(config, args)

    seed_torch(seed=config.SEED)
    if args.exp_name:
        logger, final_output_dir, tb_log_dir = create_logger_sweep(
            config, args.cfg, args.exp_name, 'train')
    else:
        logger, final_output_dir, tb_log_dir = create_logger(
            config, args.cfg, 'train')

    # logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    
    # check if validate on 2 datasets
    if config.VALIDATE_ON_TWO_DATASETS:
        assert config.DATASET.USE_MMPOSE_VAL == False, "Cannot validate on 2 datasets if USE_MMPOSE_VAL is True. The first dataset is always the normal one, and the second is mmpose based."
        validate_on_two_datasets = True
        logger.info('=> validating on two datasets! (2D GT and MMPOSE) This uses more memory and is slower.')
    else:
        validate_on_two_datasets = False
    

    # =========================== Model ===========================
    model = eval('models.' + config.MODEL + '.get_multiview_mpl_net')(config, is_train=True)
    print(model)

    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', config.MODEL + '.py'),
        final_output_dir)
    # shutil.copy2(args.cfg, final_output_dir)
    # save config to final_output_dir
    if args.exp_name:
        cfg_name = args.exp_name + '.yaml'
    else:
        cfg_name = os.path.basename(args.cfg)
    with open(args.cfg, 'r') as f:
        config_in_args = yaml.load(f, Loader=yaml.FullLoader)
    config_in_args = update_config_file(config_in_args, args)
    with open(os.path.join(final_output_dir, cfg_name), 'w') as f:
        yaml.dump(config_in_args, f, default_flow_style=False)
    logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }


    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    logger.info('=> setting loss type: {}'.format(config.LOSS.TYPE))
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

    optimizer = get_optimizer(config, model)
    start_epoch = config.TRAIN.BEGIN_EPOCH
    wandb_id = None
    wandb_name = None
    if config.TRAIN.RESUME:
        try: 
            start_epoch, model, optimizer, wandb_id = load_checkpoint(model, optimizer, final_output_dir)
        except:
            start_epoch, model, optimizer = load_checkpoint(model, optimizer,
                                                        final_output_dir)
        
    # =========================== WandB ===========================
    if config.WANDB:
        if wandb_id is None:
            wandb_id = wandb.util.generate_id()
            if args.exp_name:
                wandb_name = final_output_dir.split('/')[-1]
            else:
                wandb_name = args.cfg.split('/')[-1].replace('.yaml', '')
        wandb.init(
            id=wandb_id,
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            config=config,
            resume="allow",
            name=wandb_name,
            dir=config.LOG_WANDB_DIR,
            save_code=False,
        )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR)

    # ================================ Data loading code ================================
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.' + config.DATASET.TRAIN_DATASET)(
        config, config.DATASET.TRAIN_SUBSET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    valid_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    if validate_on_two_datasets:
        valid_dataset_2 = eval('dataset.' + config.DATASET.TEST_DATASET)(
            config, config.DATASET.TEST_SUBSET, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), is_mmpose=True)

    drop_last = True
    if config.DATASET.TRAIN_N_SAMPLES is not None:
        drop_last = False if config.DATASET.TRAIN_N_SAMPLES < config.TRAIN.BATCH_SIZE else True
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        drop_last=drop_last,
        pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    if validate_on_two_datasets:
        valid_loader_2 = torch.utils.data.DataLoader(
            valid_dataset_2,
            batch_size=config.TEST.BATCH_SIZE * len(gpus),
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=True)

    best_perf = 0.0
    best_model = False
    if start_epoch >= config.TRAIN.END_EPOCH:
        epoch = start_epoch
        logger.info('=> start_epoch is greater than or equal to end_epoch. Only running Validation.')
        perf_indicator = validate(config, valid_loader, valid_dataset, model,
                                  criterion, final_output_dir, writer_dict, epoch=epoch)
        if config.WANDB:
            wandb.log({
                'valid/3d_perf': perf_indicator,
                'epoch': epoch,
            })
        
        if validate_on_two_datasets:
            perf_indicator_2 = validate(config, valid_loader_2, valid_dataset_2, model,
                                  criterion, final_output_dir, writer_dict, epoch=epoch, is_mmpose=True)
            if config.WANDB:
                wandb.log({
                    'valid/3d_perf_mmpose': perf_indicator_2,
                    'epoch': epoch,
                })
            perf_indicator = perf_indicator_2
            
    for epoch in range(start_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()
        if config.WANDB:
            wandb.log({
                'train/lr_epoch': lr_scheduler.get_lr()[0],
                'train/lr_optimizer_epoch': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
            })

        train(config, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, writer_dict)

        perf_indicator = validate(config, valid_loader, valid_dataset, model,
                                  criterion, final_output_dir, writer_dict, epoch=epoch)
        if config.WANDB:
            wandb.log({
                'valid/3d_perf': perf_indicator,
                'epoch': epoch,
            })
        
        if validate_on_two_datasets:
            perf_indicator_2 = validate(config, valid_loader_2, valid_dataset_2, model,
                                  criterion, final_output_dir, writer_dict, epoch=epoch, is_mmpose=True)
            if config.WANDB:
                wandb.log({
                    'valid/3d_perf_mmpose': perf_indicator_2,
                    'epoch': epoch,
                })
            perf_indicator = perf_indicator_2

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': get_model_name(config),
            'state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
            'wandb_id': wandb_id,
            'random_seed_state': random.getstate(),
            'numpy_random_seed_state': np.random.get_state(),
            'torch_random_seed_state': torch.get_rng_state(),
        }, best_model, final_output_dir)
        # save the number of epoch passed in a text file
        with open(os.path.join(final_output_dir, 'epoch.txt'), 'w') as f:
            f.write(str(epoch + 1))
            
    # save the number of epoch passed in a text file
    with open(os.path.join(final_output_dir, 'epoch.txt'), 'w') as f:
        f.write(str(epoch + 1))

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
