# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import h5py
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import get_model_name
from core.evaluate import accuracy, calc_distance_per_dim, calc_mpjpe, print_per_kp
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from core.utils_plot import plot_3d_points, plot_2d_points, plot_3d_points_plotly
import random 
import pickle
import json
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

logger = logging.getLogger(__name__)


def get_keep_ratio(final_ratio, ite, total):
    # Start from 1.0, gradually decrease to the given ratio
    ratio = 1.0 - (1 - final_ratio) * ite / total
    return ratio




NUM_VIEW = {'multiview_h36m': 4, 'multiview_skipose': 6}

# pick the neighboring camera, same as Epipolar Transformers
cam_rank = {
    'multiview_h36m':
        {
            0: 2,  # cam1 -> cam3
            1: 3,  # cam2 -> cam4
            2: 0,  # cam3 -> cam1
            3: 1   # cam4 -> cam2
        },
    'multiview_skipose':
        {
            0: 1,
            1: 0,
            2: 3,
            3: 2,
            4: 5,
            5: 4
        }
}


cam_pair = {
    'multiview_h36m': [[0, 2], [1, 3]],
    'multiview_skipose':[[0, 1], [2, 3], [4, 5]],
}


def get_epipolar_field(points1, center1, points2, center2, power=10, eps=1e-10):
    # Points1 / Points2: (B, N, 3)
    # Center1 / Center2: (B, 1, 3)
    # power: a higher value will generate a sharpen map along the epipolar line
    # Return: ()
    num_p1 = points1.shape[1]  # N1 = H * W

    # norm vector of  space C1C2P1 (Eq 3 in paper)
    vec_c1_c2 = center2 - center1 + eps         # (B, 1, 3)
    vec_c1_p1 = points1 - center1               # (B, N1, 3)
    space_norm_vec = torch.cross(vec_c1_p1, vec_c1_c2.repeat(1, num_p1, 1), dim=2) # (B, N, 3) x (B, N, 3) -> (B, N, 3)
    space_norm_vec_norm = F.normalize(space_norm_vec, dim=2, p=2)  # (B, N1, 3)

    vec_c2_p2 = points2 - center2  # (B, N2, 3)
    vec_c2_p2_norm = F.normalize(vec_c2_p2, dim=2, p=2)  # (B, N2, 3)

    # Eq 4 in paper
    cos = torch.bmm(space_norm_vec_norm, vec_c2_p2_norm.transpose(2, 1))    # (B, N1, 3) * (B, 3, N2) -> (B, N1, N2)

    field = 1 - cos.abs()
    field = field ** power
    field[field < 1e-5] = 1e-5      # avoid 0
    return field


def train(config, data, model, criterion, optim, epoch, output_dir,
          writer_dict):

    # total Epoch 
    total_epoch = config.TRAIN.END_EPOCH
    total_iter = len(data) * total_epoch
    cur_iter = epoch * len(data)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss_x = AverageMeter()
    loss_y = AverageMeter()
    loss_z = AverageMeter()
    avg_acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, weight, meta) in enumerate(data):
        # one subject, one action, in different views
        # input:    list, length:4, (bs, 3, 256, 256)       4 views
        # target:   list, length:4, (bs, 17, 64, 64)        4 views
        # weight:   list, length:4, (bs, 17, 1)             4 views
        data_time.update(time.time() - end)

        
        # ============= sample two views =============
        centers = [meta[idx]['cam_center'].float() for idx in range(len(input))]
        rays = [meta[idx]['rays'].float() for idx in range(len(input))]

        # ================== model forward ==================
        ratio = get_keep_ratio(0.7, cur_iter, total=total_iter)
        ratio = 0.7 
        if config.NETWORK.TRANSFORMER_OUTPUT_HEAD_KADKHOD:
            output, x_intermediate = model(input, centers=centers, rays=rays)
        else:
            output = model(input, centers=centers, rays=rays)  # list, (B, num_joints, H=64, W=64)
        
        

        # ================== Loss on the final heatmap (64 * 64) ==================

        t = target[0]
        w = weight[0]
        t = t.cuda(non_blocking=True)
        w = w.cuda(non_blocking=True)
        if config.NETWORK.TRANSFORMER_OUTPUT_HEAD_KADKHOD:
            loss, loss_axis = criterion(x_intermediate + [output], t, w)
        else:
            loss, loss_axis = criterion(output, t, w)
        # target = t
        
        # ================== log input and output in wandb ==================
        if config.WANDB and config.WANDB_LOG_IMG:
            if i % config.LOG_IMAGE_FREQ == 0:
                # 3d target of the last sample in the batch
                # fig_target = plt.figure()
                # plot_3d_points(fig_target, t[-1].cpu().numpy())
                
                fig_target = plot_3d_points_plotly(t[-1].cpu().numpy())
                
                # fig_pred = plt.figure()
                # plot_3d_points(fig_pred, output[-1].detach().cpu().numpy())
                fig_pred = plot_3d_points_plotly(output[-1].detach().cpu().numpy())
                joints_2d_org = [meta[idx]['joints_2d_org'].cpu().numpy() for idx in range(len(input))]
                
                figs_2d = []
                for view in range(len(target)):
                    fig = plt.figure()
                    plot_2d_points(fig, input[view][-1, :, :2].cpu().numpy(), 
                                   input[view][-1, :, 2].cpu().numpy(), 
                                   to_plot_org=joints_2d_org[view][-1])
                    figs_2d.append(fig)
                wandb.log({"train/3d_target": fig_target,
                           "train/3d_pred": fig_pred,
                           "epoch": epoch,
                           })
                for i, fig in enumerate(figs_2d):
                    wandb.log({"train/2d_input_view{}".format(i): fig,
                               "epoch": epoch,})
                cols = ["3d_target", "3d_pred"]
                for view in range(len(target)):
                    cols.append("2d_input_view{}".format(view))
                table = wandb.Table(columns=cols)
                path_to_plotly_html_fig_target = "./plotly_figure_fig_target.html"
                fig_target.write_html(path_to_plotly_html_fig_target, auto_play=False)
                path_to_plotly_html_fig_pred = "./plotly_figure_fig_pred.html"
                fig_pred.write_html(path_to_plotly_html_fig_pred, auto_play=False)
                row = [wandb.Html(path_to_plotly_html_fig_target), wandb.Html(path_to_plotly_html_fig_pred)]
                for fig in figs_2d:
                    row.append(wandb.Image(fig))
                table.add_data(*row)
                wandb.log({"train/3d_2d_table": table, "epoch": epoch})
                print('wandb training image logged!')
                
                
        
        # _, mpjep__ = evaluate(output.detach().cpu().numpy(), target.detach().cpu().numpy(), data.dataset.actual_joints, config)
        # print('mpjep__:', mpjep__)


        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.update(loss.item(), len(input) * input[0].size(0))
        loss_x.update(loss_axis[0].item(), len(input) * input[0].size(0))
        loss_y.update(loss_axis[1].item(), len(input) * input[0].size(0))
        loss_z.update(loss_axis[2].item(), len(input) * input[0].size(0))

        # # ================== accuracy based on heatmap (64 * 64) ==================
        # nviews = len(output)
        # acc = [None] * nviews
        # cnt = [None] * nviews
        # pre = [None] * nviews
        # for j in range(nviews):
        #     _, acc[j], cnt[j], pre[j] = accuracy(
        #         output[j].detach().cpu().numpy(),
        #         target[j].detach().cpu().numpy(), target_coords=config.TARGET_COORDS)
        # acc = np.mean(acc)
        # cnt = np.mean(cnt)
        # avg_acc.update(acc, cnt)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Loss_x {loss_x.val:.5f} ({loss_x.avg:.5f})\t' \
                  'Loss_y {loss_y.val:.5f} ({loss_y.avg:.5f})\t' \
                  'Loss_z {loss_z.val:.5f} ({loss_z.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\tRatio {ratio:.3f}' \
                  'Memory {memory:.1f}'.format(
                      epoch, i, len(data), ratio, batch_time=batch_time,
                      speed=len(input) * input[0].size(0) / batch_time.val,
                      data_time=data_time, loss=losses, loss_x=loss_x, loss_y=loss_y, loss_z=loss_z,
                      acc=avg_acc, memory=gpu_memory_usage, ratio=ratio)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', avg_acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1
            
            if config.WANDB:
                wandb.log({
                    'train/train_loss_iter': losses.val,
                    'train/train_loss_avg': losses.avg,
                    'train/train_loss_x_iter': loss_x.val,
                    'train/train_loss_x_avg': loss_x.avg,
                    'train/train_loss_y_iter': loss_y.val,
                    'train/train_loss_y_avg': loss_y.avg,
                    'train/train_loss_z_iter': loss_z.val,
                    'train/train_loss_z_avg': loss_z.avg,
                    # 'train/train_accuracy_iter': avg_acc.val,
                    # 'train/train_accuracy_avg': avg_acc.avg,
                    
                })

            for k in range(len(input)):
                view_name = 'view_{}'.format(k + 1)
                prefix = '{}_{}_{:08}'.format(
                    os.path.join(output_dir, 'train'), view_name, i)
                # save_debug_images(config, input[k], meta[k], target[k],
                #                   pre[k] * 4, output[k], prefix)

    # epoch loss summary
    msg = 'Summary Epoch: [{0}]\tLoss ({loss.avg:.5f})\tLoss_x ({loss_x.avg:.5f})\tLoss_y ({loss_y.avg:.5f})\tLoss_z ({loss_z.avg:.5f})\tAccuracy {acc.avg:.3f}'.format(epoch,
                                                                                                                                                                        loss=losses,
                                                                                                                                                                        loss_x=loss_x,
                                                                                                                                                                        loss_y=loss_y,
                                                                                                                                                                        loss_z=loss_z,
                                                                                                                                                                        acc=avg_acc)
    logger.info(msg)
    
    if config.WANDB:
        wandb.log({
            'train/train_loss_epoch': losses.avg,
            'train/train_loss_x_epoch': loss_x.avg,
            'train/train_loss_y_epoch': loss_y.avg,
            'train/train_loss_z_epoch': loss_z.avg,
            # 'train/train_accuracy_epoch': avg_acc.avg,
            'epoch': epoch,
        })


def validate(config,
             loader,
             dataset,
             model,
             criterion,
             output_dir,
             writer_dict=None,
             epoch=None,
             is_mmpose=False):

    model.eval()
    batch_time = AverageMeter()
    inference_time = AverageMeter()
    losses = AverageMeter()
    loss_x = AverageMeter()
    loss_y = AverageMeter()
    loss_z = AverageMeter()
    avg_acc = AverageMeter()

    n_view = 6 if config.DATASET.TEST_DATASET == 'multiview_skipose' else 4
    n_view = 5 if config.DATASET.TEST_DATASET.startswith('multiview_cmu_panoptic') else n_view
    n_view = len(config.DATASET.TEST_VIEWS) if config.DATASET.TEST_VIEWS is not None else n_view
    nsamples = len(dataset)

    njoints = config.NETWORK.NUM_JOINTS                 # 17
    height = int(config.NETWORK.HEATMAP_SIZE[0])        # 64
    width = int(config.NETWORK.HEATMAP_SIZE[1])         # 64
    all_preds = np.zeros((nsamples, njoints, 3), dtype=np.float32)      # (#sample, 17, 3)
    all_gts = np.zeros((nsamples, njoints, 3), dtype=np.float32)      # (#sample, 17, 3)
    all_heatmaps = np.zeros(
        (nsamples, njoints, height, width), dtype=np.float32)           # (#sample, 17,64, 64)
    
    all_3d_confs = np.ones((nsamples, njoints), dtype=np.float32)      # (#sample, 17) 3D confidence of CMU annotations
    fnames = []
    
    validation_name = 'val' if not is_mmpose else 'val_mmpose'

    idx = 0
    with torch.no_grad():
        end = time.time()
        
        for i, (input, target, weight, meta) in enumerate(loader):
            # input:    list, length:4, (bs, 3, 256, 256)       4 views
            # target:   list, length:4, (bs, 17, 64, 64)        4 views
            # weight:   list, length:4, (bs, 17, 1)             4 views

            # ======================== combinations of input ========================
            batch_size = input[0].shape[0]
            if 'image' in meta[0]:
                fnames += meta[0]['image']

            rays = [meta[j]['rays'].float()  for j in range(len(input))  ] 
            centers = [meta[j]['cam_center'].float() for j in range(len(input))]
            start_inference = time.time()
            if config.NETWORK.TRANSFORMER_OUTPUT_HEAD_KADKHOD:
                output, x_intermediate = model(input, centers=centers, rays=rays)
            else:
                output = model(input, centers=centers, rays=rays)
            inference_time.update(time.time() - start_inference)
            """
                debug code for plotting pred and target in 3D
                
                import matplotlib.pyplot as plt
                from mpl_toolkits import mplot3d
                body_edges = np.array([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16],[8,9],[9,10]])
                fig = plt.figure() 
                ax = fig.add_subplot(111, projection='3d')
                to_plot = target[0][0].cpu().numpy()
                ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:, 2], marker='o', s=20, c='b')
                for edge in body_edges:
                    ax.plot(to_plot[edge,0], to_plot[edge,1], to_plot[edge,2], color='g')

                to_plot = output[0].cpu().numpy()
                ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:, 2], marker='o', s=20, c='r')
                for edge in body_edges:
                    ax.plot(to_plot[edge,0], to_plot[edge,1], to_plot[edge,2], color='orange')

                max_range = np.array([to_plot[:, 0].max() - to_plot[:, 0].min(), to_plot[:, 1].max() - to_plot[:, 1].min(), to_plot[:, 2].max() - to_plot[:, 2].min()]).max() / 2.0
                x_mean = to_plot[:, 0].mean()
                y_mean = to_plot[:, 1].mean()
                z_mean = to_plot[:, 2].mean()
                ax.set_xlim(x_mean - max_range, x_mean + max_range)
                ax.set_ylim(y_mean - max_range, y_mean + max_range)
                ax.set_zlim(z_mean - max_range, z_mean + max_range)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.view_init(0, 90)
                plt.savefig('test25.png')
            """

            
            # ======================== Loss calculation ========================
            
            t = target[0]
            w = weight[0]
            t = t.cuda(non_blocking=True)
            w = w.cuda(non_blocking=True)
            if config.NETWORK.TRANSFORMER_OUTPUT_HEAD_KADKHOD:
                loss, loss_axis = criterion(x_intermediate + [output], t, w)
            else:
                loss, loss_axis = criterion(output, t, w)
            # target = t
            losses.update(loss.item(), len(input) * input[0].size(0))
            loss_x.update(loss_axis[0].item(), len(input) * input[0].size(0))
            loss_y.update(loss_axis[1].item(), len(input) * input[0].size(0))
            loss_z.update(loss_axis[2].item(), len(input) * input[0].size(0))
            
            if config.WANDB and config.WANDB_LOG_IMG:
                if i % config.LOG_IMAGE_FREQ == 0:
                    # 3d target of the last sample in the batch
                    # fig_target = plt.figure()
                    # plot_3d_points(fig_target, t[-1].cpu().numpy())
                    
                    fig_target = plot_3d_points_plotly(t[-1].cpu().numpy())
                    
                    # fig_pred = plt.figure()
                    # plot_3d_points(fig_pred, output[-1].detach().cpu().numpy())
                    fig_pred = plot_3d_points_plotly(output[-1].detach().cpu().numpy())
                    
                    figs_2d = []
                    for view in range(len(target)):
                        fig = plt.figure()
                        plot_2d_points(fig, input[view][-1, :, :2].cpu().numpy(), input[view][-1, :, 2].cpu().numpy())
                        figs_2d.append(fig)
                    wandb.log({"{}/3d_target".format(validation_name): fig_target,
                            "{}/3d_pred".format(validation_name): fig_pred,
                            "epoch": epoch,
                            })
                    for i, fig in enumerate(figs_2d):
                        wandb.log({"{}/2d_input_view{}".format(validation_name, i): fig,
                                "epoch": epoch,})
                    cols = ["3d_target", "3d_pred"]
                    for view in range(len(target)):
                        cols.append("2d_input_view{}".format(view))
                    table = wandb.Table(columns=cols)
                    path_to_plotly_html_fig_target = "./plotly_figure_fig_target.html"
                    fig_target.write_html(path_to_plotly_html_fig_target, auto_play=False)
                    path_to_plotly_html_fig_pred = "./plotly_figure_fig_pred.html"
                    fig_pred.write_html(path_to_plotly_html_fig_pred, auto_play=False)
                    row = [wandb.Html(path_to_plotly_html_fig_target), wandb.Html(path_to_plotly_html_fig_pred)]
                    for fig in figs_2d:
                        row.append(wandb.Image(fig))
                    table.add_data(*row)
                    wandb.log({"{}/3d_2d_table".format(validation_name): table, "epoch": epoch})
                    
                    print('wandb {} image logged!'.format(validation_name))

            # ================== accuracy based on heatmap (64 * 64) ==================
            # nviews = len(output)
            # acc = [None] * nviews
            # cnt = [None] * nviews
            # pre = [None] * nviews
            # for j in range(nviews):
            #     _, acc[j], cnt[j], pre[j] = accuracy(
            #         output[j].detach().cpu().numpy(),
            #         target[j].detach().cpu().numpy())       # threshold: 64 / 10 * 0.5
            # acc = np.mean(acc)
            # cnt = np.mean(cnt)
            # avg_acc.update(acc, cnt)

            batch_time.update(time.time() - end)
            end = time.time()

            # ======================== Save prediction (heatmap + coords.) ========================
            preds = np.zeros((batch_size, njoints, 3), dtype=np.float32)     # (bs * #view, 17, 3)
            # heatmaps = np.zeros(
            #     (nimgs, njoints, height, width), dtype=np.float32)      # (bs * #view, 17, 64, 64)
            # for k, o, m in zip(range(nviews), output, meta):
            #     # o: (bs, 17, 64, 64)
            #     pred, maxval = get_final_preds(config,
            #                                    o.clone().cpu().numpy(),
            #                                    m['center'].numpy(),
            #                                    m['scale'].numpy())
            #     # pred:   (bs, num_joints=17, 2)    coordinate in original image (1000, 1000)
            #     # maxval: (bs, num_joints=17, 1)    peak value on heatmap
            #     pred = pred[:, :, 0:2]          # (bs, 17, 2)
            #     pred = np.concatenate((pred, maxval), axis=2)       # (bs, 17, 3)
            #     preds[k::nviews] = pred
            #     heatmaps[k::nviews] = o.clone().cpu().numpy()       # (bs, 17, 64, 64)

            preds = output.clone().cpu().numpy()
            gts = t.clone().cpu().numpy()
            if 'room_scaled' in meta[0]:
                if 'room_scaled_equal' in meta[0]:
                    room_scale = meta[0]['room_x_scale'][0].item()
                    room_center = meta[0]['room_center'][0].clone().cpu().numpy()
                    preds = preds * room_scale + room_center
                    gts = gts * room_scale + room_center
                else:
                    room_x_scale = meta[0]['room_x_scale'][0].item()
                    room_y_scale = meta[0]['room_y_scale'][0].item()
                    preds[:, :, 0] = preds[:, :, 0] * room_x_scale
                    preds[:, :, 1] = preds[:, :, 1] * room_y_scale
                    gts[:, :, 0] = gts[:, :, 0] * room_x_scale
                    gts[:, :, 1] = gts[:, :, 1] * room_y_scale
            all_preds[idx:idx + batch_size] = preds                      # (bs * #view, 17, 3) in original image
            # all_heatmaps[idx:idx + nimgs] = heatmaps
            all_gts[idx:idx + batch_size] = gts    # (bs * #view, 17, 3) in original image
            if 'joints_3d_conf' in meta[0]:
                all_3d_confs[idx:idx + batch_size] = meta[0]['joints_3d_conf'].clone().cpu().numpy().squeeze()
            idx += batch_size

            # # ======================== Log ========================
            if i % config.PRINT_FREQ == 0:
            # if True:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                       'Loss_x {loss_x.val:.5f} ({loss_x.avg:.5f})\t' \
                       'Loss_y {loss_y.val:.5f} ({loss_y.avg:.5f})\t' \
                       'Loss_z {loss_z.val:.5f} ({loss_z.avg:.5f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(loader), batch_time=batch_time,
                          loss=losses, loss_x=loss_x, loss_y=loss_y, loss_z=loss_z, acc=avg_acc)
                logger.info(msg)

                for k in range(len(output)):
                    view_name = 'view_{}'.format(k + 1)
                    prefix = '{}_{}_{:08}'.format(
                        os.path.join(output_dir, 'validation'), view_name, i)
                    # save_debug_images(config, input[k], meta[k], target[k],
                    #                   pre[k] * 4, output[k], prefix)
                    
            if config.WANDB:
                wandb.log({
                    '{}/val_loss_iter'.format(validation_name): losses.val,
                    '{}/val_loss_avg'.format(validation_name): losses.avg,
                    '{}/val_loss_x_iter'.format(validation_name): loss_x.val,
                    '{}/val_loss_x_avg'.format(validation_name): loss_x.avg,
                    '{}/val_loss_y_iter'.format(validation_name): loss_y.val,
                    '{}/val_loss_y_avg'.format(validation_name): loss_y.avg,
                    '{}/val_loss_z_iter'.format(validation_name): loss_z.val,
                    '{}/val_loss_z_avg'.format(validation_name): loss_z.avg,
                    # 'val/val_accuracy_iter': avg_acc.val,
                    # 'val/val_accuracy_avg': avg_acc.avg,
                })

        #
        if is_mmpose:
            logger.info('\n****************************** ! Validating on MMPOSE ! ************************* ')
        msg = '----Test----: [{0}/{1}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                'Loss_x {loss_x.val:.4f} ({loss_x.avg:.4f})\t' \
                'Loss_y {loss_y.val:.4f} ({loss_y.avg:.4f})\t' \
                'Loss_z {loss_z.val:.4f} ({loss_z.avg:.4f})\t' \
                'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    loss=losses,
                    loss_x=loss_x,
                    loss_y=loss_y,
                    loss_z=loss_z,
                    acc=avg_acc)
        logger.info(msg)
        
        msg = 'Inference Time: sum (avg) {inference_time.sum:.3f}s ({inference_time.avg:.3f}s)\t' \
                'Speed {speed:.1f} samples/s'.format(
                    inference_time=inference_time,
                    speed=nsamples / inference_time.sum,
                    )
        logger.info(msg)
        
        # write speed to a file
        with open(os.path.join(output_dir, 'speed.txt'), 'w') as f:
            f.write(str(nsamples / inference_time.sum))
            
        
        if config.WANDB:
            wandb.log({
                '{}/val_loss_epoch'.format(validation_name): losses.avg,
                '{}/val_loss_x_epoch'.format(validation_name): loss_x.avg,
                '{}/val_loss_y_epoch'.format(validation_name): loss_y.avg,
                '{}/val_loss_z_epoch'.format(validation_name): loss_z.avg,
                'epoch': epoch,
                # 'val/val_accuracy_epoch': avg_acc.avg,
            })

        # ======================= save all heatmaps and joint locations =======================
        u2a = dataset.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = np.array(list(a2u.values()))

        save_file = config.TEST.HEATMAP_LOCATION_FILE
        file_name = os.path.join(output_dir, save_file)
        file = h5py.File(file_name, 'w')
        # file['heatmaps'] = all_heatmaps[:, u, :, :]
        file['locations'] = all_preds[:, u, :]
        file['gts'] = all_gts[:, u, :]
        file['joint_names_order'] = a
        file.close()
        
        save_file = config.TEST.PRED_GT_LOCATION_FILE
        test_name = config.DATASET.TEST_DATASET
        state = config.TEST.STATE
        if is_mmpose:
            use_mmpose = 'mmpose'
        else:
            use_mmpose = 'mmpose' if config.DATASET.USE_MMPOSE_VAL else 'org'
        file_name = os.path.join(output_dir, '{}_{}_{}_{}.pkl'.format(save_file.replace('.pkl', ''), test_name, use_mmpose, state))
        with open(file_name, 'wb') as f:
            pickle.dump({'pred': all_preds[:, u, :], 'gt': all_gts[:, u, :]}, f)
        
        # save gts and preds to a separate file with frame names
        if len(fnames) > 0:
            fnames = [fname.split('/')[0] + '_' + fname.split('/')[-1].split('_')[-1].replace('.jpg','') for fname in fnames]
            # to_dump = {}
            # for fname, pred_dump, gt_dump, conf_3d in zip(fnames, all_preds[:, u, :], all_gts[:, u, :], all_3d_confs[:, u]):
            #     to_dump[fname] = {'GT': gt_dump, 'pred': pred_dump, 'conf_3d': conf_3d}
            to_dump = {
                'fnames': fnames,
                'pred': all_preds,
                'gt': all_gts,
            }
            file_name = os.path.join(output_dir, '{}_{}_{}_{}_dict.pkl'.format(save_file.replace('.pkl', ''), test_name, use_mmpose, state))
            with open(file_name, 'wb') as f:
                pickle.dump(to_dump, f)

        # ======================== evaluate MPJPE  ========================
        logger.info('Start 3D eval ....')
        per_action = True if config.DATASET.TEST_DATASET.startswith('multiview_h36m') else False
        logger.info(' -------------------- Relative --------------------')
        # >>>>>>>>>>>>>>>>>>>> relative evaluation
        name_value, perf_indicator = evaluate(all_preds[:, u, :], all_gts[:, u, :], dataset.actual_joints, config, output_dir, conf_3d=all_3d_confs, relative_evaluation=True, per_action=per_action, fnames=fnames, epoch=epoch, use_mmpose=use_mmpose, validation_name=validation_name)
        names = name_value.keys()
        values = name_value.values()
        num_values = len(name_value)
        _, full_arch_name = get_model_name(config)
        logger.info('| Arch ' +
                    ' '.join(['| {}'.format(name) for name in names]) + ' |')
        logger.info('|---' * (num_values + 1) + '|')
        logger.info('| ' + full_arch_name + ' ' +
                    ' '.join(['| {:.3f}'.format(value) for value in values]) +
                    ' |')
        logger.info('Evaluate {}'.format(str(perf_indicator)))
        # <<<<<<<<<<<<<<<<<<<< relative evaluation
        
        logger.info(' -------------------- Absolute --------------------')
        
        # name_value, perf_indicator = dataset.evaluate(all_preds)
        name_value, perf_indicator = evaluate(all_preds[:, u, :], all_gts[:, u, :], dataset.actual_joints, config, output_dir, conf_3d=all_3d_confs, epoch=epoch, per_action=per_action, fnames=fnames, use_mmpose=use_mmpose, validation_name=validation_name)
        names = name_value.keys()
        values = name_value.values()
        num_values = len(name_value)
        _, full_arch_name = get_model_name(config)
        logger.info('| Arch ' +
                    ' '.join(['| {}'.format(name) for name in names]) + ' |')
        logger.info('|---' * (num_values + 1) + '|')
        logger.info('| ' + full_arch_name + ' ' +
                    ' '.join(['| {:.3f}'.format(value) for value in values]) +
                    ' |')
        logger.info('Evaluate {}'.format(str(perf_indicator)))
        perf_indicator = 1 / perf_indicator
        logger.info('perf_indicator: {}'.format(perf_indicator))

    return perf_indicator

def index_to_action_names_h36m():
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

def evaluate(pred, gt, actual_joints, config, output_dir, conf_3d=None, relative_evaluation=False, epoch=None, per_action=False, fnames=None, use_mmpose=None, validation_name='val'):
        pred = pred.copy()
        gt = gt.copy()
        
        if config.DATASET.OUTPUT_IN_METER:
            pred = pred * 100
            gt = gt * 100
        
        if relative_evaluation:
            gt = gt - gt[:, 0:1, :]    # (num_sample, 17, 3)
            pred = pred - pred[:, 0:1, :]  # (num_sample, 17, 3)
            
        if conf_3d is not None:
            gt[conf_3d <= 0] = np.nan
            pred[conf_3d <= 0] = np.nan
            
        mode = 'relative' if relative_evaluation else 'absolute'
        pjpe, mpjpe = calc_mpjpe(gt, pred, mode=mode)
        
        name_values = collections.OrderedDict()
        joint_names = actual_joints
        for i in range(len(joint_names)):
            name_values[joint_names[i]] = pjpe[i]
            
        logger.info('3D MPJPE (cm): {}'.format(mpjpe))
        to_log = print_per_kp(pjpe, actual_joints.values())
        logger.info('3D MPJPE per keypoint: {}'.format(to_log))
        
        distance_per_dim_per_kp, distance_per_dim = calc_distance_per_dim(pred, gt) 
        logger.info('Distance per dimension: {}'.format(distance_per_dim))
        to_log = print_per_kp(distance_per_dim_per_kp, actual_joints.values())
        logger.info('Distance per dimension per keypoint: {}'.format(to_log))
        
        if config.WANDB:
            if relative_evaluation:
                wandb.log({
                    '{}/val_rel_3d_mpjpe_epoch'.format(validation_name): mpjpe,
                    '{}/val_rel_distance_x_epoch'.format(validation_name): distance_per_dim[0],
                    '{}/val_rel_distance_y_epoch'.format(validation_name): distance_per_dim[1],
                    '{}/val_rel_distance_z_epoch'.format(validation_name): distance_per_dim[2],
                    'epoch': epoch,
                })
            else:
                wandb.log({
                    '{}/val_3d_mpjpe_epoch'.format(validation_name): mpjpe,
                    '{}/val_distance_x_epoch'.format(validation_name): distance_per_dim[0],
                    '{}/val_distance_y_epoch'.format(validation_name): distance_per_dim[1],
                    '{}/val_distance_z_epoch'.format(validation_name): distance_per_dim[2],
                    'epoch': epoch,
                })
            
                log_wandb_per_kp = {'kp_{}/{}_{}_3d_mpjpe_epoch'.format(kp, validation_name, kp): v for kp, v in zip(actual_joints.values(), pjpe)}
                log_wandb_per_kp.update({'kp_{}/{}_{}_distance_x_epoch'.format(kp, validation_name, kp): v for kp, v in zip(actual_joints.values(), distance_per_dim_per_kp[:, 0])})
                log_wandb_per_kp.update({'kp_{}/{}_{}_distance_y_epoch'.format(kp, validation_name, kp): v for kp, v in zip(actual_joints.values(), distance_per_dim_per_kp[:, 1])})
                log_wandb_per_kp.update({'kp_{}/{}_{}_distance_z_epoch'.format(kp, validation_name, kp): v for kp, v in zip(actual_joints.values(), distance_per_dim_per_kp[:, 2])})
                log_wandb_per_kp.update({'epoch': epoch})
                wandb.log(log_wandb_per_kp)
            
        test_name = config.DATASET.TEST_DATASET
        if use_mmpose is None:
            use_mmpose = 'mmpose' if config.DATASET.USE_MMPOSE_VAL else 'org'
        absolute_or_relative = 'relative' if relative_evaluation else 'absolute'
        state = config.TEST.STATE
        pkl_file = os.path.join(output_dir, 'mpjpe_{}_{}_{}_{}.pkl'.format(test_name, use_mmpose, state, absolute_or_relative))
        with open(pkl_file, 'wb') as f:
            pickle.dump(name_values, f)
        
        
        if per_action:
            action_names = index_to_action_names_h36m()
            actions = np.array([int(fname.split('_')[3]) for fname in fnames])
            mpjpe_per_action = {}
            pjpe_per_action = {}
            distance_per_dim_per_action = {}
            for action in action_names:
                idx = actions == action
                if np.sum(idx) > 0:
                    gt_action = gt[idx]
                    pred_action = pred[idx]
                    pjpe_action, mpjpe_action = calc_mpjpe(gt_action, pred_action, mode=mode)
                    distance_per_dim_per_kp_action, distance_per_dim_action = calc_distance_per_dim(pred_action, gt_action) 
                    mpjpe_per_action[action] = mpjpe_action
                    pjpe_per_action[action] = pjpe_action
                    distance_per_dim_per_action[action] = distance_per_dim_action
            to_log = print_per_kp(mpjpe_per_action.values(), action_names.values())
            logger.info('3D MPJPE per action: {}'.format(to_log))
            pkl_file = os.path.join(output_dir, 'mpjpe_perAction_{}_{}_{}_{}.pkl'.format(test_name, use_mmpose, state, absolute_or_relative))
            to_dump = {k: v for k, v in zip(action_names.values(), mpjpe_per_action.values())}
            with open(pkl_file, 'wb') as f:
                pickle.dump(to_dump, f)
            # for action in action_names:
            #     to_log = print_per_kp(pjpe_per_action[action], actual_joints.values())
            #     logger.info('3D MPJPE per keypoint for action {}: {}'.format(action, to_log))
            if config.WANDB:
                if relative_evaluation:
                    log_wandb_per_action = {'action_{}/{}_rel_3d_mpjpe_epoch'.format(action_names[action], validation_name): v for action, v in mpjpe_per_action.items()}
                    log_wandb_per_action.update({'epoch': epoch})
                    wandb.log(log_wandb_per_action)
                else:
                    log_wandb_per_action = {'action_{}/{}_3d_mpjpe_epoch'.format(action_names[action], validation_name): v for action, v in mpjpe_per_action.items()}
                    log_wandb_per_action.update({'action_{}/{}_{}_distance_x_epoch'.format(action_names[action], validation_name, action_names[action]): v[0] for action, v in distance_per_dim_per_action.items()})
                    log_wandb_per_action.update({'action_{}/{}_{}_distance_y_epoch'.format(action_names[action], validation_name, action_names[action]): v[1] for action, v in distance_per_dim_per_action.items()})
                    log_wandb_per_action.update({'action_{}/{}_{}_distance_z_epoch'.format(action_names[action], validation_name, action_names[action]): v[2] for action, v in distance_per_dim_per_action.items()})
                    log_wandb_per_action.update({'epoch': epoch})
                    wandb.log(log_wandb_per_action)
                    
                    
                
            #         log_wandb_per_kp = {'kp_{}/val_{}_3d_mpjpe_epoch'.format(kp, kp): v for kp, v in zip(actual_joints.values(), pjpe)}
            #         log_wandb_per_kp.update({'kp_{}/val_{}_distance_x_epoch'.format(kp, kp): v for kp, v in zip(actual_joints.values(), distance_per_dim_per_kp[:, 0])})
            #         log_wandb_per_kp.update({'kp_{}/val_{}_distance_y_epoch'.format(kp, kp): v for kp, v in zip(actual_joints.values(), distance_per_dim_per_kp[:, 1])})
            #         log_wandb_per_kp.update({'kp_{}/val_{}_distance_z_epoch'.format(kp, kp): v for kp, v in zip(actual_joints.values(), distance_per_dim_per_kp[:, 2])})
            #         log_wandb_per_kp.update({'epoch': epoch})
            #         wandb.log(log_wandb_per_kp)
                
        return name_values, mpjpe 


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

