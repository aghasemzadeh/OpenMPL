import json
import os
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import copy
import argparse
# import torch

# for detecting the bbox of humans
# from mmdet.apis import DetInferencer


## to be in line with SOTA
training_categories = [
    '171026_pose1',
    '171026_pose2',
    '171026_pose3',
    '171204_pose1',
    '171204_pose2',
    '171204_pose3',
    '171204_pose4',
]

val_categories = [
    '171204_pose5',
    '171204_pose6',
]

training_cameras = [
    3,
    6,
    12,
    13,
    23,
]

val_cameras = [
    3,
    6,
    12,
    13,
    23,
]

# val_cameras = [
#     1,
#     7,
#     8,
#     25,
#     26,
# ]

img_size = (1920, 1080)

joints_cmu = {
    0: 'neck',
    1: 'nose',
    2: 'root',
    3: 'lsho',
    4: 'lelb',
    5: 'lwri',
    6: 'lhip',
    7: 'lkne',
    8: 'lank',
    9: 'rsho',
    10: 'relb',
    11: 'rwri',
    12: 'rhip',
    13: 'rkne',
    14: 'rank',
    15: 'leye',
    16: 'lear',
    17: 'reye',
    18: 'rear'
}
joints_cmu_inverse_dict = {v: k for k, v in joints_cmu.items()}

joints_h36m = {
    0: 'root',
    1: 'rhip',
    2: 'rkne',
    3: 'rank',
    4: 'lhip',
    5: 'lkne',
    6: 'lank',
    # 7: 'belly',
    8: 'neck',
    9: 'nose',
    # 10: 'head',
    11: 'lsho',
    12: 'lelb',
    13: 'lwri',
    14: 'rsho',
    15: 'relb',
    16: 'rwri'
}

joints_coco = {
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

# mmdet for bbox detection
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def projectPoints(X, K, R, t, Kd):
    """ Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
    
    Roughly, x = K*(R*X + t) + distortion
    
    See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """
    
    x = np.asarray(R*X + t)
    
    x[0:2,:] = x[0:2,:]/x[2,:]
    
    r = x[0,:]*x[0,:] + x[1,:]*x[1,:]
    
    x[0,:] = x[0,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[2]*x[0,:]*x[1,:] + Kd[3]*(r + 2*x[0,:]*x[0,:])
    x[1,:] = x[1,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[3]*x[0,:]*x[1,:] + Kd[2]*(r + 2*x[1,:]*x[1,:])

    x[0,:] = K[0,0]*x[0,:] + K[0,1]*x[1,:] + K[0,2]
    x[1,:] = K[1,0]*x[0,:] + K[1,1]*x[1,:] + K[1,2]
    
    return x

def world_to_cam(point_world, R, T):
    # takes batches of points and converts them to world coordinates
    # point_world: (b, n, 3) batch, number of points, xyz
    # build E
    R = np.array(R)
    to_conc = np.ones((point_world.shape[0], point_world.shape[1], 1))     # this for concatenating 
    p = np.concatenate([point_world, to_conc], axis=2)  # convert to homogenous         (b, n, 4) batch, number of points, homogenous coordinates
    p = np.expand_dims(p, axis=3)
    R_temp = np.vstack([R, np.zeros((1,3))])
    T_temp = np.vstack([T, 1.0])
    E = np.hstack([R_temp, T_temp])
    E = np.stack([E]*point_world.shape[1])                      # create E for 17 points
    E = np.stack([E]*p.shape[0])             # create E for all matchings
    point_cam = E @ p
    point_cam_squeezed = np.squeeze(point_cam, axis=-1)
    point_cam = point_cam_squeezed[:,:,:] / point_cam[:,:,3]  # convert back to cartesian (check that x_homo[2] > 0)
    point_cam = point_cam[:,:,:3]
    return point_cam

def _infer_box(pose3d, camera, rootIdx):
    root_joint = pose3d[rootIdx, :]
    tl_joint = root_joint.copy()
    tl_joint[0] -= 1000.0
    tl_joint[1] -= 900.0
    br_joint = root_joint.copy()
    br_joint[0] += 1000.0
    br_joint[1] += 1100.0
    tl_joint = np.reshape(tl_joint, (1, 3))
    br_joint = np.reshape(br_joint, (1, 3))

    tl2d = _weak_project(tl_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()

    br2d = _weak_project(br_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()
    return np.array([tl2d[0], tl2d[1], br2d[0], br2d[1]])


def _weak_project(pose3d, fx, fy, cx, cy):
    pose2d = pose3d[:, :2] / pose3d[:, 2:3]
    pose2d[:, 0] *= fx
    pose2d[:, 1] *= fy
    pose2d[:, 0] += cx
    pose2d[:, 1] += cy
    return pose2d


def cmu_to_h36m(pose_cmu):
    pose_h36m = np.zeros((17, 4))
    for i in range(17):
        if i == 7 or i == 10:
            continue
        pose_h36m[i] = pose_cmu[joints_cmu_inverse_dict[joints_h36m[i]]]
    pose_cmu[pose_cmu[:, 3] == -1, 3] = np.finfo(float).eps   # change the confidence of missing joints to 0
    head = np.average(pose_cmu[15:19,:3], axis=0, weights=pose_cmu[15:19,3]).reshape((1,3))
    head_conf = np.average(pose_cmu[15:19,3], weights=pose_cmu[15:19,3].astype(bool)).reshape((1,1))
    head = np.hstack([head, head_conf])
    belly = np.average(np.array([pose_cmu[0, :3], pose_cmu[2, :3]]), axis=0, weights=np.array([pose_cmu[0, 3], pose_cmu[2, 3]])).reshape((1,3))
    belly_conf = np.average(np.array([pose_cmu[0, 3], pose_cmu[2, 3]]), weights=np.array([pose_cmu[0, 3], pose_cmu[2, 3]]).astype(bool)).reshape((1,1))
    belly = np.hstack([belly, belly_conf])
    pose_h36m[7] = belly
    pose_h36m[10] = head
    return pose_h36m

def cmu_to_coco(pose_cmu):
    pose_coco = np.zeros((17, 4))
    pose_cmu[pose_cmu[:, 3] == -1, 3] = np.finfo(float).eps   # change the confidence of missing joints to 0
    for i in range(17):
        pose_coco[i] = pose_cmu[joints_cmu_inverse_dict[joints_coco[i]]]
    return pose_coco

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

def preprocess_cmu_mmpose(args):
    data = args[0]
    dir_cmu_mmpose = args[1]
    keypoints_standard = args[2]
    dataset = copy.deepcopy(data)
    json_mmpose = os.path.join(dir_cmu_mmpose, 
                                data['image'].split('/')[-4],   # pose_action
                                data['image'].split('/')[-2],   # camera
                                data['image'].split('/')[-1].replace('.jpg', '.json'))
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
    dataset['mmpose_2d'] = True
    dataset['joints_2d'] = joints_2d
    dataset['joints_2d_conf'] = joints_2d_conf
    dataset['keypoint_standard'] = keypoints_standard
    
    return dataset

def compatible_cams(cameras):
    for k,cam in cameras.items():    
        cam['K'] = np.matrix(cam['K'])
        cam['distCoef'] = np.array(cam['distCoef'])
        cam['R'] = np.matrix(cam['R'])
        cam['t'] = np.array(cam['t']).reshape((3,1))
        cam['T'] = np.array(-cam['R'].T @ np.array(cam['t']).reshape((3,1)))
        cam['fx'] = cam['K'][0,0]
        cam['fy'] = cam['K'][1,1]
        cam['cx'] = cam['K'][0,2]
        cam['cy'] = cam['K'][1,2]
    return cameras
    

def process_frame(args):
    frame_info = args['frame_info']
    annot_info = args['annot_info']
    calib = args['calib']
    keypoints_standard = args['keypoints_standard']
    dont_discard_less_than_5_visible_joints = args.get('dont_discard_less_than_5_visible_joints', True)
    # print('dont_discard_less_than_5_visible_joints: {}'.format(dont_discard_less_than_5_visible_joints))
    with open(annot_info, 'r') as f:
        annot = json.load(f)
    which_cam = int(frame_info.split('/')[-2].split('_')[1])
    camera = calib[which_cam]
    K = np.matrix(camera['K'])
    distCoef = np.array(camera['distCoef'])
    camera['k'] = np.array([distCoef[0], distCoef[1], distCoef[4]])
    camera['p'] = np.array([distCoef[2], distCoef[3]])
    R = np.matrix(camera['R'])
    t = np.array(camera['t']).reshape((3,1))
    if len(annot['bodies']) == 0:
        # print('empty frame')
        return {}
    skel_cmu = np.array(annot['bodies'][0]['joints19']).reshape((-1,4))
    if (skel_cmu[:,3] < 0.1).any():
        return {}
    
    if keypoints_standard == 'h36m':
        skel = cmu_to_h36m(skel_cmu)
    elif keypoints_standard == 'coco':
        skel = cmu_to_coco(skel_cmu)
    
    pt = projectPoints(skel[:,:3].T, K, R, t, distCoef)
    
    joints_2d = pt[:2,:].T
    invis = (joints_2d[:,0] < 0 ) + (joints_2d[:,0] > img_size[0]) + (joints_2d[:,1] < 0) + (joints_2d[:,1] > img_size[1])
    vis = 1 - invis
    
    # if number of visible joints is less than 5, discard
    if np.sum(vis) < 5 and not dont_discard_less_than_5_visible_joints:
        print('not enough visible keypoints: {}'.format(np.sum(vis)))
        return {}
    vis = np.repeat(vis.reshape(-1,1), 3, axis=1)   # for compatibility with h36m
    
    # skel_camera = world_to_cam(skel[None, :, 0:3], R, t)[0]
    # box = _infer_box(skel_camera, camera, 2)
    
    max_x = np.max(joints_2d[vis[:,0] == True, 0], initial=0)
    min_x = np.min(joints_2d[vis[:,0] == True, 0], initial=0)
    max_y = np.max(joints_2d[vis[:,0] == True, 1], initial=0)
    min_y = np.min(joints_2d[vis[:,0] == True, 1], initial=0)
    box = np.array([min_x, min_y, max_x, max_y])
    
    
    # box = None
    # detection_inferencer = DetInferencer(model='rtmdet_tiny_8xb32-300e_coco', device=device)
    # mmdet_output = detection_inferencer(os.path.join(dir_cmu_panoptic, frame_info))    
    
    
    center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
    scale = ((box[2] - box[0]) / 200.0, (box[3] - box[1]) / 200.0)
    
    dataset = {
        'image': frame_info,
        'joints_2d': joints_2d,
        'joints_vis': vis,
        'joints_3d': skel[:,:3],
        'joints_3d_conf': skel[:,3].reshape((-1,1)),
        'source': 'cmu_panoptic',
        'subject': 0,
        'pose_id': frame_info.split('/')[0],
        'image_id': int(frame_info.split('/')[-1].split('_')[-1].replace('.jpg', '')),
        'camera_id': which_cam,
        'center': center,
        'scale': scale,
        'box': box,
        'camera': camera,
    }
    return dataset

def get_key_str(datum):
    return 's_{:02}_pose_{}_imgid_{:08}'.format(
        datum['subject'], datum['pose_id'],
        datum['image_id'])
        
def filter_group(data, views, skip_step):
    grouping = {}
    nitems = len(data)
    CAM_IX = {x: i for i, x in enumerate(views)}
    for i in range(nitems):
        keystr = get_key_str(data[i])
        camera_id = CAM_IX[data[i]['camera_id']]
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
   
def create_dataset(cats, cams, running_modes, skip_step, pkl_dir, pkl_dir_filtered, dir_cmu_panoptic, train_or_val, write_kept_keys, kept_keys_path, pkl_dir_filtered_mmpose, mmpose_output_path, keypoints_standard, filter_from_file, dont_discard_less_than_5_visible_joints):
    if 'preprocess' in running_modes:
        calib = {}
        for d in cats:
            with open(os.path.join(dir_cmu_panoptic, d, 'calibration_{}.json'.format(d)), 'r') as f:
                calibration_cat = json.load(f)
            
            cameras = {cam['node']:cam for cam in calibration_cat['cameras'] if cam['type']=='hd' and cam['node'] in cams}
            # Convert data into numpy arrays for convenience
            calib[d] = compatible_cams(cameras)
        
        
        frame_list = []
        for cat in tqdm(cats, desc='Fetching {} frames'.format(train_or_val)):
            # if len(frame_list) >= 10:
            #     break
            for cam in cams:
                try:
                    for frame in os.listdir(os.path.join(dir_cmu_panoptic, cat, 'hdImgs', '00_{0:02d}'.format(cam))):
                        frame_ix = frame.split('.')[0].split('_')[-1]
                        annot_file = os.path.join(dir_cmu_panoptic, cat, 'hdPose3d_stage1_coco19', 'body3DScene_{}.json'.format(frame_ix))
                        if frame.endswith('.jpg') and os.path.isfile(annot_file):
                            # print('found')
                            frame_list.append({
                                'frame_info': os.path.join(cat, 'hdImgs', '00_{0:02d}'.format(cam), frame),
                                'annot_info': annot_file,
                                'calib': calib[cat],
                                'keypoints_standard': keypoints_standard,
                                'dont_discard_less_than_5_visible_joints': dont_discard_less_than_5_visible_joints
                                })
                except FileNotFoundError:
                    continue
        # frame_list.sort(key=lambda x: x['frame_info'])
        print('Total frames to process: {}'.format(len(frame_list)))
        
        dataset_ = []
        pool = Pool()
        for result in tqdm(pool.imap(process_frame, frame_list), total=len(frame_list), desc='Processing {} frames'.format(train_or_val)):
            if result != {}:
                dataset_.append(result)
        
        with open(os.path.join(pkl_dir, 'cmu_panoptic_{}.pkl'.format(train_or_val)), 'wb') as f:
            pickle.dump(dataset_, f)
        
    # --------- filter the data based on the requested frames
    if 'filter' in running_modes:
        if 'preprocess' not in running_modes:
            print('Reading {} dataset from {}'.format(train_or_val, os.path.join(pkl_dir, 'cmu_panoptic_{}.pkl'.format(train_or_val))))
            with open(os.path.join(pkl_dir, 'cmu_panoptic_{}.pkl'.format(train_or_val)), 'rb') as f:
                dataset_ = pickle.load(f)
                
        if filter_from_file:
            with open(kept_keys_path.replace('.txt', '_{}.txt'.format(train_or_val)), 'r') as f:
                keys = f.readlines()
            keys = [x.strip() for x in keys]
            args = [(x, keys) for x in dataset_]
            remaining_idx = []
            pool = Pool()
            for result in tqdm(pool.imap(check_if_file_in_keys, args), total=len(dataset_), desc='filtering {} data'.format(train_or_val)):
                if result:
                    remaining_idx.append(True)
                else:
                    remaining_idx.append(False)
                    
            dataset_filtered = [x for x, y in zip(dataset_, remaining_idx) if y]
        else:
            remaining_idx = filter_group(dataset_, cams, skip_step)
            
            remaining_keys = []
            dataset_filtered = []
            if write_kept_keys:
                f = open(kept_keys_path.replace('.txt', '_{}.txt'.format(train_or_val)), 'w')
            
            for k in tqdm(remaining_idx, desc='filtering {} data'.format(train_or_val)):
                for i in k:
                    if write_kept_keys:
                        f.write(dataset_[i]['image'])   
                    remaining_keys.append(dataset_[i]['image'])
                    dataset_filtered.append(dataset_[i])
                    if write_kept_keys:
                        f.write('\n')
            if write_kept_keys:
                f.close()
            
        print('\n filter keys stored in {}'.format(kept_keys_path.replace('.txt', '_{}.txt'.format(train_or_val))))
                
        print('{} dataset filtered: {}, org {} data: {}'.format(train_or_val, train_or_val, len(dataset_filtered), len(dataset_)))
        with open(os.path.join(pkl_dir_filtered, 'cmu_panoptic_{}.pkl'.format(train_or_val)), 'wb') as f:
            pickle.dump(dataset_filtered, f)
        
    # --------- replace 2D joints with mmpose detections
    if 'mmpose' in running_modes:
        if 'filter' not in running_modes:
            print('Reading {} dataset from {}'.format(train_or_val, os.path.join(pkl_dir_filtered, 'cmu_panoptic_{}.pkl'.format(train_or_val))))
            with open(os.path.join(pkl_dir_filtered, 'cmu_panoptic_{}.pkl'.format(train_or_val)), 'rb') as f:
                dataset_filtered = pickle.load(f)
                
        dataset_filtered_mmpose = []
        mmpose_output_path_list = [mmpose_output_path] * len(dataset_filtered)
        keypoints_standard_list = [keypoints_standard] * len(dataset_filtered)
        pool = Pool()
        for result in tqdm(pool.imap(preprocess_cmu_mmpose, zip(dataset_filtered, mmpose_output_path_list, keypoints_standard_list)), total=len(dataset_filtered), desc='preprocessing {} mmpose data'.format(train_or_val)):
            dataset_filtered_mmpose.append(result)
        
        dataset_filtered_mmpose = [x for x in dataset_filtered_mmpose if len(x) > 0]
        print('{} data mmpose: {}, org {} data: {}'.format(train_or_val, len(dataset_filtered_mmpose), train_or_val, len(dataset_filtered)))
        with open(os.path.join(pkl_dir_filtered_mmpose, 'cmu_panoptic_{}.pkl'.format(train_or_val)), 'wb') as f:
            pickle.dump(dataset_filtered_mmpose, f)
        print('saved mmpose data in {}'.format(os.path.join(pkl_dir_filtered_mmpose, 'cmu_panoptic_{}.pkl'.format(train_or_val))))
        
    return 'done'
        
    
def parse_args():
    parser = argparse.ArgumentParser(description='prepare cmu panoptic dataset')
    parser.add_argument('cmu_dir', help='directory of the cmu panoptic dataset')
    parser.add_argument('dataset', help='name of dataset')   # eg annot_same_scene_different_cams, annot_different_scene_same_cams_2val_pose')
    parser.add_argument('--running-modes', nargs='+', default=['preprocess', 'filter', 'mmpose'])
    parser.add_argument('--keypoints-standard', default='h36m', help='standard for 2d keypoints', choices=['h36m', 'coco'])
    parser.add_argument('--run-for-sets', nargs='+', default=['train', 'validation'])
    # parser.add_argument('--not-run-for-train', action='store_true')
    # parser.add_argument('--not-run-for-val', action='store_true')
    parser.add_argument('--train-cats', nargs='+', default=training_categories)
    parser.add_argument('--val-cats', nargs='+', default=val_categories)
    parser.add_argument('--train-cams', nargs='+', default=training_cameras)
    parser.add_argument('--val-cams', nargs='+', default=val_cameras)
    # parser.add_argument('--only-run-filter', action='store_true')
    # parser.add_argument('--filter-key-file-train', default=None)
    # parser.add_argument('--filter-key-file-val', default=None)
    # parser.add_argument('--filter-data', action='store_true')
    parser.add_argument('--skip-step-train', type=int, default=2)
    parser.add_argument('--skip-step-val', type=int, default=6)
    parser.add_argument('--write-kept-keys', action='store_true')
    parser.add_argument('--kept-keys-path', default='./filtered_grouping_keys.txt')
    parser.add_argument('--filter-from-file', action='store_true')
    parser.add_argument('--mmpose-dataset-name', default='mmpose')
    parser.add_argument('--mmpose-output-path', default='./mmpose_outputs')
    parser.add_argument('--dont-discard-less-than-5-visible-joints', action='store_true')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    
    args = parse_args()
    dir_cmu_panoptic = args.cmu_dir
    dataset_name = args.dataset
    running_modes = args.running_modes
    run_for_sets = args.run_for_sets
    # not_run_for_train = args.not_run_for_train
    # not_run_for_val = args.not_run_for_val
    train_cats = args.train_cats
    val_cats = args.val_cats
    train_cams = args.train_cams
    val_cams = args.val_cams
    if 'all' in train_cams:
        train_cams = list(range(31))
        train_cams.remove(20)   # remove the camera that is not in the dataset
        train_cams.remove(21)
    if 'all' in val_cams:
        val_cams = list(range(31))
        val_cams.remove(20)   # remove the camera that is not in the dataset
        val_cams.remove(21)
    # only_run_filter = args.only_run_filter
    # filter_data = args.filter_data
    train_skip_step = args.skip_step_train
    val_skip_step = args.skip_step_val
    write_kept_keys = args.write_kept_keys
    kept_keys_path = args.kept_keys_path
    mmpose_dataset_name = args.mmpose_dataset_name
    mmpose_output_path = args.mmpose_output_path
    filter_from_file = args.filter_from_file
    dont_discard_less_than_5_visible_joints = args.dont_discard_less_than_5_visible_joints
    
    keypoints_standard = args.keypoints_standard
    pkl_dir_filtered_mmpose = None
    
    print('running on dataset: {}'.format(dataset_name))
    print('keypoints standard: {}'.format(keypoints_standard))
    
    pkl_dir = os.path.join(dir_cmu_panoptic, 'MPL_data', 'datasets', dataset_name)
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir, exist_ok=True)
    pkl_dir_filtered = None
    if 'filter' in running_modes or 'mmpose' in running_modes:
        pkl_dir_filtered = pkl_dir + '_filtered_{}_{}'.format(train_skip_step, val_skip_step)
        if not os.path.exists(pkl_dir_filtered):
            os.makedirs(pkl_dir_filtered, exist_ok=True)
            
    if 'mmpose' in running_modes:
        pkl_dir_filtered_base = os.path.basename(pkl_dir_filtered)
        pkl_dir_filtered_mmpose = os.path.join(dir_cmu_panoptic, 'MPL_data', 'datasets_mmpose', '{}_{}'.format(pkl_dir_filtered_base, mmpose_dataset_name))
        if not os.path.exists(pkl_dir_filtered_mmpose):
            os.makedirs(pkl_dir_filtered_mmpose, exist_ok=True)
    
    if 'train' in run_for_sets:    
        create_dataset(train_cats, train_cams, running_modes, train_skip_step, pkl_dir, pkl_dir_filtered, dir_cmu_panoptic, 'train', write_kept_keys, kept_keys_path, pkl_dir_filtered_mmpose, mmpose_output_path, keypoints_standard, filter_from_file, dont_discard_less_than_5_visible_joints)
    if 'validation' in run_for_sets:
        create_dataset(val_cats, val_cams, running_modes, val_skip_step, pkl_dir, pkl_dir_filtered, dir_cmu_panoptic, 'validation', write_kept_keys, kept_keys_path, pkl_dir_filtered_mmpose, mmpose_output_path, keypoints_standard, filter_from_file, dont_discard_less_than_5_visible_joints)
    
   