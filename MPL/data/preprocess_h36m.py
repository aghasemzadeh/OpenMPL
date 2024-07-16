import json
import os
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import copy
import argparse
# import torch



img_size = (1000, 1000)

MMPOSE2H36M = {
    1: 12,  # rhip
    2: 14,  # rkne
    3: 16,  # rank
    4: 11,  # lhip
    5: 13,  # lkne
    6: 15,  # lank
    9: 0,   # nose
    11: 5,  # lsho
    12: 7,  # lelb
    13: 9,  # lwri
    14: 6,  # rsho
    15: 8,  # relb
    16: 10, # rwri
}


def mmpose2h36m(mmpose, scores):
    h36m = np.zeros((17, 2))
    conf = np.zeros((17, 1))
    for i in range(17):
        try:
            h36m[i] = mmpose[MMPOSE2H36M[i]]
            conf[i] = scores[MMPOSE2H36M[i]]
        except KeyError:
            h36m[i] = np.array([0, 0])
            conf[i] = 0
    
    # calculate head
    head = mmpose[0:5].mean(axis=0)
    h36m[10] = head
    conf[10] = scores[0:5].mean(axis=0)
    
    # calculate neck
    neck = mmpose[3:7].mean(axis=0)
    h36m[8] = neck
    conf[8] = scores[3:7].mean(axis=0)
    
    # calculate root
    root = mmpose[11:13].mean(axis=0)
    h36m[0] = root
    conf[0] = scores[11:13].mean(axis=0)
    
    # calculate belly
    belly = np.mean([neck, root], axis=0)
    h36m[7] = belly
    conf[7] = np.mean([conf[8], conf[0]], axis=0)
    
    return h36m, conf

def preprocess_h36m_mmpose(args):
    data = args[0]
    dir_h36m_mmpose = args[1]
    keypoints_standard = args[2]
    dataset = copy.deepcopy(data)
    json_mmpose = os.path.join(dir_h36m_mmpose, 
                                data['image'].replace('.jpg', '.json'))
    try:
        with open(json_mmpose, 'r') as f:
            mmpose_data = json.load(f)
    except FileNotFoundError:
        return []
        
    if keypoints_standard == 'h36m':
        joints_2d, joints_2d_conf = mmpose2h36m(np.array(mmpose_data[0]['keypoints']), np.array(mmpose_data[0]['keypoint_scores']))
    elif keypoints_standard == 'coco':
        joints_2d = np.array(mmpose_data[0]['keypoints'])
        joints_2d_conf = np.array(mmpose_data[0]['keypoint_scores']).reshape((-1,1))
    # dataset['mmpose_2d'] = True
    dataset['joints_2d'] = joints_2d
    dataset['joints_2d_conf'] = joints_2d_conf
    
    return dataset


def get_key_str(datum):
    return 's_{:02}_act_{:02}_subact_{:02}_imgid_{:06}'.format(
        datum['subject'], datum['action'], datum['subaction'],
        datum['image_id'])
        
def filter_group(data, views, skip_step):
    grouping = {}
    nitems = len(data)
    views = views
    CAM_IX = {x-1: i for i, x in enumerate(views)}
    for i in range(nitems):
        keystr = get_key_str(data[i])
        try:
            camera_id = CAM_IX[data[i]['camera_id']]
        except KeyError:
            continue
        if keystr not in grouping:
            grouping[keystr] = [-1] * len(views)
        grouping[keystr][camera_id] = i
    
    filtered_grouping = []
    for _, v in grouping.items():
        if np.all(np.array(v) != -1):
            filtered_grouping.append(v)
    
    filtered_grouping = filtered_grouping[::skip_step]
    
    return filtered_grouping

def check_if_file_in_keys(args):
    data = args[0]
    keys = args[1]
    # keys = remaining_keys
    pose_action = data['image'].split('/')[-4]
    frame_id = data['image'].split('/')[-1].split('_')[-1]
    for j in keys:
        if j.startswith(pose_action) and j.endswith(frame_id):
            return True
    return False
   
def isdamaged(db_rec):
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

def create_dataset(running_modes, skip_step, pkl_dir, pkl_dir_filtered, train_or_val, write_kept_keys, kept_keys_path, pkl_dir_filtered_mmpose, mmpose_output_path, keypoints_standard):
        
    # --------- filter the data based on the requested frames
    if 'filter' in running_modes:
        if 'preprocess' not in running_modes:
            print('Reading {} dataset from {}'.format(train_or_val, os.path.join(pkl_dir, 'h36m_{}.pkl'.format(train_or_val))))
            with open(os.path.join(pkl_dir, 'h36m_{}.pkl'.format(train_or_val)), 'rb') as f:
                dataset_ = pickle.load(f)
                
        dataset_ = [db_rec for db_rec in dataset_ if not isdamaged(db_rec)]
        remaining_idx = filter_group(dataset_, [1, 2, 3, 4], skip_step)
        remaining_keys = []
        dataset_filtered = []
        if write_kept_keys:
            f = open(kept_keys_path.replace('.txt', '_{}.txt'.format(train_or_val)), 'w')
            for k in tqdm(remaining_idx, desc='filtering {} data'.format(train_or_val)):
                for i in k:
                    f.write(dataset_[i]['image'])
                    remaining_keys.append(dataset_[i]['image'])
                    dataset_filtered.append(dataset_[i])
                    f.write('\n')
            f.close()
            
        print('\n filter keys stored in {}'.format(kept_keys_path.replace('.txt', '_{}.txt'.format(train_or_val))))
                
        print('{} dataset filtered: {}, org {} data: {}'.format(train_or_val, train_or_val, len(dataset_filtered), len(dataset_)))
        with open(os.path.join(pkl_dir_filtered, 'h36m_{}.pkl'.format(train_or_val)), 'wb') as f:
            pickle.dump(dataset_filtered, f)
        
    # --------- replace 2D joints with mmpose detections
    if 'mmpose' in running_modes:
        if 'filter' not in running_modes:
            print('Reading {} dataset from {}'.format(train_or_val, os.path.join(pkl_dir_filtered, 'h36m_{}.pkl'.format(train_or_val))))
            with open(os.path.join(pkl_dir_filtered, 'h36m_{}.pkl'.format(train_or_val)), 'rb') as f:
                dataset_filtered = pickle.load(f)
                
        dataset_filtered_mmpose = []
        mmpose_output_path_list = [mmpose_output_path] * len(dataset_filtered)
        keypoints_standard_list = [keypoints_standard] * len(dataset_filtered)
        pool = Pool()
        for result in tqdm(pool.imap(preprocess_h36m_mmpose, zip(dataset_filtered, mmpose_output_path_list, keypoints_standard_list)), total=len(dataset_filtered), desc='preprocessing {} mmpose data'.format(train_or_val)):
            dataset_filtered_mmpose.append(result)
        
        dataset_filtered_mmpose = [x for x in dataset_filtered_mmpose if len(x) > 0]
        print('{} data mmpose: {}, org {} data: {}'.format(train_or_val, len(dataset_filtered_mmpose), train_or_val, len(dataset_filtered)))
        with open(os.path.join(pkl_dir_filtered_mmpose, 'h36m_{}.pkl'.format(train_or_val)), 'wb') as f:
            pickle.dump(dataset_filtered_mmpose, f)
        
    return 'done'
        
    
def parse_args():
    parser = argparse.ArgumentParser(description='prepare h36m dataset')
    parser.add_argument('h36m_dir', help='directory of the h36m dataset')
    parser.add_argument('dataset', help='name of dataset')   # eg annot_same_scene_different_cams, annot_different_scene_same_cams_2val_pose')
    parser.add_argument('--running-modes', nargs='+', default=['filter', 'mmpose'])
    parser.add_argument('--keypoints-standard', default='h36m', help='standard for 2d keypoints', choices=['h36m', 'coco'])
    parser.add_argument('--run-for-sets', nargs='+', default=['train', 'validation'])
    parser.add_argument('--skip-step-train', type=int, default=2)
    parser.add_argument('--skip-step-val', type=int, default=6)
    parser.add_argument('--write-kept-keys', action='store_true')
    parser.add_argument('--kept-keys-path', default='./filtered_grouping_keys.txt')
    parser.add_argument('--mmpose-dataset-name', default='mmpose')
    parser.add_argument('--mmpose-output-path', default='./mmpose_outputs')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    
    args = parse_args()
    dir_h36m = args.h36m_dir
    dataset_name = args.dataset
    running_modes = args.running_modes
    run_for_sets = args.run_for_sets


    train_skip_step = args.skip_step_train
    val_skip_step = args.skip_step_val
    write_kept_keys = args.write_kept_keys
    kept_keys_path = args.kept_keys_path
    mmpose_dataset_name = args.mmpose_dataset_name
    mmpose_output_path = args.mmpose_output_path
    
    keypoints_standard = args.keypoints_standard
    pkl_dir_filtered_mmpose = None
    
    pkl_dir = os.path.join(dir_h36m, dataset_name)
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)
    pkl_dir_filtered = None
    if 'filter' in running_modes or 'mmpose' in running_modes:
        pkl_dir_filtered = os.path.join(dir_h36m, 'MPL_data', 'datasets', '{}_filtered_{}_{}'.format(dataset_name, train_skip_step, val_skip_step))
        if not os.path.exists(pkl_dir_filtered):
            os.makedirs(pkl_dir_filtered)
            
    if 'mmpose' in running_modes:
        pkl_dir_filtered_base = os.path.basename(pkl_dir_filtered)
        pkl_dir_filtered_mmpose = os.path.join(dir_h36m, 'MPL_data', 'datasets_mmpose', '{}_{}'.format(pkl_dir_filtered_base, mmpose_dataset_name))
        if not os.path.exists(pkl_dir_filtered_mmpose):
            os.makedirs(pkl_dir_filtered_mmpose)
    
    if 'train' in run_for_sets:    
        create_dataset(running_modes, train_skip_step, pkl_dir, pkl_dir_filtered, 'train', write_kept_keys, kept_keys_path, pkl_dir_filtered_mmpose, mmpose_output_path, keypoints_standard)
    if 'validation' in run_for_sets:
        create_dataset(running_modes, val_skip_step, pkl_dir, pkl_dir_filtered, 'validation', write_kept_keys, kept_keys_path, pkl_dir_filtered_mmpose, mmpose_output_path, keypoints_standard)
    
   