import os
from os import path as osp
import numpy as np
import torch
from tqdm import tqdm
import pickle
import trimesh
import argparse
import glob 
from body_visualizer.tools.vis_tools import colors, imagearray2file
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools.vis_tools import show_image
from human_body_prior.tools.omni_tools import log2file, makepath
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
from amass.data.prepare_data import prepare_amass
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from mmpose.apis import MMPoseInferencer
from utils import *

# support_dir = '/data/Git_Repo/TransMoCap/support_data'
support_dir = '/globalscratch/users/a/b/abolfazl/amass_data/support_data'

# Choose the device to run the body model on.
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=0):
    # random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

amass_splits = {
    'validation': ['CMU'],
    'test': ['CMU'],
    'train': ['Eyes_Japan_Dataset',
             'ACCAD',
             'DFaust_67',
             'BMLhandball',
             'BioMotionLab_NTroje',
             'SFU',
             'Transitions_mocap',
             'TCD_handMocap',
             'TotalCapture',
             'KIT',
             'MPI_HDM05',
             'HumanEva',
             'MPI_mosh',
             'BMLmovi',
             'SOMA',
             'MPI_Limits',
             'WEIZMANN',
             'EKUT',
             'SSM_synced',
             'GRAB',
             'DanceDB',
             'HUMAN4D',
             'CNRS'
             ]
}

# get amass train split from input arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # parser.add_argument('--train-datasets', default=None, nargs='+', type=str, help='Train datasets')
    parser.add_argument('--exp', default='amass', type=str, help='Experiment name')
    parser.add_argument('--extra-name', default='', type=str, help='Extra name for the experiment')
    parser.add_argument('--n-splits', default=100, type=int, help='Number of splits')
    # parser.add_argument('--use-cams-from', default='h36m', type=str, help='Use cameras from',
    #                     choices=['h36m', 'cmu', 'both'])
    # parser.add_argument('--calib-file-h36m', default=None, type=str, help='Camera calibration file')
    # parser.add_argument('--calib-root-cmu', default=None, type=str, help='Camera calibration file')
    # parser.add_argument('--actors-h36m', default=[1, 5, 6, 7, 8, 9, 11], nargs='+', type=int, help='Actors to use')
    # parser.add_argument('--calibs-cmu', default=['171204_pose5', '171204_pose6'], nargs='+', type=str, help='Calibrations to use')
    # parser.add_argument('--views-cmu', default=[3, 6, 12, 13, 23], nargs='+', type=int, help='Views to use')
    # parser.add_argument('--room-size', default=[-1.0, 1.0, -1.0, 1.0], nargs='+', type=float, help='Room size [min_x, max_x, min_y, max_y]')
    parser.add_argument('--operation-on', default=['train', 'validation'], nargs='+', type=str, help='Operations on')
    # parser.add_argument('--image-width', default=1000, type=int, help='Image width')
    # parser.add_argument('--image-height', default=1000, type=int, help='Image height')
    # parser.add_argument('--apply-rotation', default=False, action='store_true', help='Apply rotation')
    # parser.add_argument('--apply-joint-clip', default=False, action='store_true', help='Apply joint clip')
    # parser.add_argument('--n-frames', default=-1, type=int, help='Number of frames to use')
    # parser.add_argument('--fit-mmpose-to-amass', default=False, action='store_true', help='Fit skeleton to AMASS')
    # parser.add_argument('--fit-using-most-aligned', default=False, action='store_true', help='Fit using all joints or only the most aligned ones')
    # parser.add_argument('--regressor', default='h36m', type=str, help='Regressor to use', choices=['h36m', 'coco'])

    return parser.parse_args()


class AMASS_DS(Dataset):
    """AMASS: a pytorch loader for unified human motion capture dataset. http://amass.is.tue.mpg.de/"""

    def __init__(self, dataset_dir, num_betas=16, keep_from=None, keep_to=None):

        self.ds = {}
        for data_fname in glob.glob(os.path.join(dataset_dir, '*.pt')):
            k = os.path.basename(data_fname).replace('.pt','')
            self.ds[k] = torch.load(data_fname)
        self.num_betas = num_betas
        if keep_from is not None:
            self.ds = {k: v[keep_from:keep_to] for k, v in self.ds.items()}

    def __len__(self):
       return len(self.ds['trans'])

    def __getitem__(self, idx):
        
        data =  {k: self.ds[k][idx] for k in self.ds.keys()}
        data['root_orient'] = data['pose'][:3]
        data['pose_body'] = data['pose'][3:66]
        data['pose_hand'] = data['pose'][66:]
        data['betas'] = data['betas'][:self.num_betas]
        data['trans'] = data['trans']

        return data


actual_joints = {
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

keypoints_most_aligned_with_h36m = [
    2,  # rkne
    3,  # rank
    5,  # lkne
    9,  # nose
    11, # lsho
    12, # lelb
    13, # lwri
    14, # rsho
    15, # relb
    16, # rwri
]

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


def main():
    seed_torch(0)
    args = parse_args()
    # if args.train_datasets is not None:
    #     amass_splits['train'] = args.train_datasets
    expr_code = args.exp

    # print('Train split has %d datasets.'%len(amass_splits['train']))
    # print('Train datasets:', amass_splits['train'])


    msg = ''' Initial use of standard AMASS dataset preparation pipeline '''

    # amass_dir =  '/data/amass_data_poses/' #'PATH_TO_DOWNLOADED_NPZFILES/*/*_poses.npz'
    amass_dir =  '/globalscratch/users/a/b/abolfazl/amass_data_poses' #'PATH_TO_DOWNLOADED_NPZFILES/*/*_poses.npz'
    
    # if args.regressor == 'h36m':
    #     J_regressor = amass_dir + '/J_regressor_h36m.npy'
    # elif args.regressor == 'coco':
    #     J_regressor = amass_dir + '/J_regressor_coco.npy'
    # else:
    #     raise ValueError('Unknown regressor: {}'.format(args.regressor))
    # work_dir = '/data/Git_Repo/amass/support_data/prepared_data/{}/'.format(expr_code)
    work_dir = '/globalscratch/users/a/b/abolfazl/amass_data/support_data/prepared_data/{}/'.format(expr_code)

    logger = log2file(makepath(work_dir, '%s.log' % (expr_code), isfile=True))
    logger('[%s] AMASS Data Preparation Began.'%expr_code)
    logger(msg)
    
    for subset in args.operation_on:
        # check for files in stage V
        list_files = glob.glob(os.path.join(work_dir, 'stage_V', subset, '*.pkl'))
        list_files.sort()
        list_files = [file for file in list_files if args.extra_name in file]
        list_splits = [int(file.split('/')[-1].split('_')[1]) for file in list_files]
        assert len(list_files) == args.n_splits - 1, 'Number of files in stage V is not {}: {}\t files missing:{}'.format(args.n_splits - 1,len(list_files), list(set(list(range(args.n_splits - 1))) - set(list_splits)))
        joints_3d_all = []
        joints_2d_mmpose_all = []
        confs_2d_mmpose_all = []
        joints_2d_amass_all = []
        triangulated_3d_mmpose_all = []
        camera_setup_used = []
        views_used = []
        for file in list_files:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                print('Loaded:', file)
                joints_3d_all.append(data['joints_3d'])
                joints_2d_mmpose_all.append(data['joints_2d_mmpose'])
                confs_2d_mmpose_all.append(data['confs_2d_mmpose'])
                joints_2d_amass_all.append(data['joints_2d_amass'])
                camera_setup_used.append(data['camera_setup_used'])
                try:
                    triangulated_3d_mmpose_all.append(data['triangulated_3d_mmpose'])
                except:
                    pass
                try:
                    views_used.append(data['views_used'])
                except:
                    pass
                
        joints_3d_all_np = np.concatenate(joints_3d_all, axis=0)
        joints_2d_amass_all_np = np.concatenate(joints_2d_amass_all, axis=0)
        joints_2d_mmpose_all_np = np.concatenate(joints_2d_mmpose_all, axis=0)
        confs_2d_mmpose_all_np = np.concatenate(confs_2d_mmpose_all, axis=0)
        camera_setup_used = np.concatenate(camera_setup_used, axis=0)
        if len(triangulated_3d_mmpose_all) > 0 and triangulated_3d_mmpose_all[0] is not None:
            triangulated_3d_mmpose_all_np = np.concatenate(triangulated_3d_mmpose_all, axis=0)
        else:
            triangulated_3d_mmpose_all_np = None
        if len(views_used) > 0:
            views_used = np.concatenate(views_used, axis=0)
        else:
            len_views = joints_2d_mmpose_all_np.shape[1]
            if len_views == 29:
                print('Warning: Using 29 views by default')
                views_used = list(range(31))
                views_used.remove(20)
                views_used.remove(21)
                views_used = np.array(views_used)
                views_used = np.stack(joints_2d_mmpose_all_np.shape[0] * [views_used])
            elif len_views == 30:
                print('Warning: Using 30 views by default')
                views_used = list(range(31))
                views_used.remove(20)
                views_used = np.array(views_used)
                views_used = np.stack(joints_2d_mmpose_all_np.shape[0] * [views_used])
            else:
                raise ValueError('Unknown number of views:', len_views)
        
        file_name_example = list_files[0].split('/')[-1].replace('split_0', '')
        print('joints_3d_all_np:', joints_3d_all_np.shape)
        
        file_path = osp.join(work_dir, file_name_example)
        
        with open(file_path, 'wb') as f:
            pickle.dump({
                'joints_3d': joints_3d_all_np,
                'joints_2d_mmpose': joints_2d_mmpose_all_np,
                'confs_2d_mmpose': confs_2d_mmpose_all_np,
                'joints_2d_amass': joints_2d_amass_all_np,
                'triangulated_3d_mmpose': triangulated_3d_mmpose_all_np,
                'camera_setup_used': camera_setup_used,
                'views_used': views_used,
            }, f)
        print('Saved dataset to:', file_path)
    
    
            

    
if __name__ == '__main__':
    main()


# python run_mmpose_on_amass.py --exp all_with_mmpose --calib-file-h36m camera_data.pkl --actors-h36m 9 11 --room-size -1 1 -1.5 2 --operation-on validation --apply-rotation 
# python run_mmpose_on_amass.py --exp all_with_mmpose --calib-file-h36m camera_data.pkl --actors-h36m 9 11 --room-size -1 1 -1.5 2 --operation-on train --apply-rotation
# python run_mmpose_on_amass.py --exp all_with_mmpose --use-cams-from cmu --calib-root-cmu cmu_calibs --calibs-cmu  171204_pose5 171204_pose6 --room-size -1 1 -1 1 --operation-on validation --image-width 1920 --image-height 1080 --apply-rotation
# python run_mmpose_on_amass.py --exp all_with_mmpose --use-cams-from cmu --calib-root-cmu cmu_calibs --calibs-cmu  171204_pose5 171204_pose6 --room-size -1 1 -1 1 --operation-on train --image-width 1920 --image-height 1080 --apply-rotation