# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from core.inference import get_max_preds


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists

def calc_dists_target_coords(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            normed_preds = preds[n, c, :] / normalize
            normed_targets = target[n, c, :] / normalize
            dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
    return dists

def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5, target_coords=False):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if target_coords:
        pred, _ = get_max_preds(output)     # (B, 17, 2)
        target, _ = get_max_preds(target)
        dists = calc_dists_target_coords(pred, target, norm)
    else:
        if hm_type == 'gaussian':
            pred, _ = get_max_preds(output)     # (B, 17, 2)
            target, _ = get_max_preds(target)
            h = output.shape[2]
            w = output.shape[3]
            norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10      # (64, 64) / 10
        dists = calc_dists(pred, target, norm)
    

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1
    if cnt == 0:
        return acc, 0, cnt, pred
    avg_acc = avg_acc / cnt
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


def calc_mpjpe(output, target, mode='absolute', not_consider_kp=None):
    """
    output: (B, 17, 3)
    target: (B, 17, 3)
    mode: 'absolute' or 'relative'
    
    return: (17,), scalar
    """
    if mode == 'absolute':
        pjpe = np.sqrt(np.nansum((output - target)**2, axis=2)).mean(axis=0)
        if not_consider_kp is not None:
            pjpe_without_some_kp = np.delete(pjpe, not_consider_kp)
            mpjpe = pjpe_without_some_kp.mean()
            return pjpe, mpjpe
        return pjpe, pjpe.mean()
    elif mode == 'relative':
        output_rel = output - output[:, 0:1, :]
        target_rel = target - target[:, 0:1, :]
        pjpe = np.sqrt(np.nansum((output_rel - target_rel)**2, axis=2)).mean(axis=0)
        if not_consider_kp is not None:
            pjpe_without_some_kp = np.delete(pjpe, not_consider_kp)
            mpjpe = pjpe_without_some_kp.mean()
            return pjpe, mpjpe
        return pjpe, pjpe.mean()
    
    
def calc_distance_per_dim(output, target):
    """
    output: (B, 17, 3)
    target: (B, 17, 3)
    
    return: (17, 3), (3,)
    """
    distance = np.nanmean(np.abs(output - target), axis=0)
    return distance, distance.mean(axis=0)

def print_per_kp(error_per_kp, kp_names):
    str_per_kp = "\n"
    for error, kp in zip(error_per_kp, kp_names):
        str_per_kp = str_per_kp + kp + '\t\t\t{}\n'.format(error)
    return str_per_kp

