import os
from os import path as osp
import numpy as np
import torch
from tqdm import tqdm
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
from utils import *


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
    parser.add_argument('--train-datasets', default=None, nargs='+', type=str, help='Train datasets')
    parser.add_argument('--exp', default='amass', type=str, help='Experiment name')
    parser.add_argument('--n-splits', default=100, type=int, help='Number of splits')
    parser.add_argument('--operation-on', default=['train', 'validation'], nargs='+', type=str, help='Operations on')
    parser.add_argument('--work-dir', default='./prepared_data/', type=str, help='Work directory')
    parser.add_argument('--amass-data-dir', default='../../amass_data_poses', type=str, help='PATH_TO_DOWNLOADED_NPZFILES/*/*_poses.npz')

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
    if args.train_datasets is not None:
        amass_splits['train'] = args.train_datasets
    expr_code = args.exp

    print('Train split has %d datasets.'%len(amass_splits['train']))
    print('Train datasets:', amass_splits['train'])


    msg = ''' Initial use of standard AMASS dataset preparation pipeline '''

    # amass_dir =  '/globalscratch/users/a/b/abolfazl/amass_data_poses' #'PATH_TO_DOWNLOADED_NPZFILES/*/*_poses.npz'
    amass_dir = args.amass_data_dir
    
    work_dir = os.path.join(args.work_dir, expr_code)

    logger = log2file(makepath(work_dir, '%s.log' % (expr_code), isfile=True))
    logger('[%s] AMASS Data Preparation Began.'%expr_code)
    logger(msg)

    amass_splits['train'] = list(set(amass_splits['train']).difference(set(amass_splits['test'] + amass_splits['validation'])))

    logger('Train split has %d datasets.'%len(amass_splits['train']))
    logger('Train datasets: {}'.format(amass_splits['train']))

    if osp.join(work_dir, 'stage_III') not in glob.glob(osp.join(work_dir, '*')):
        prepare_amass(amass_splits, amass_dir, work_dir, logger=logger)


    num_betas = 16 # number of body parameters

    
    for subset in args.operation_on:
        split_dir = os.path.join(work_dir, 'stage_III', subset)
        ds = AMASS_DS(dataset_dir=split_dir, num_betas=num_betas)
        len_ds = len(ds)
        print('Train split has %d datapoints.'%len(ds))
        splits = np.linspace(0, len_ds, args.n_splits, endpoint=True).astype(int)
        for i in range(len(splits)):
            if os.path.join(work_dir, 'stage_IV', subset, 'amass_ds_{}.pt'.format(i)) in glob.glob(os.path.join(work_dir, 'stage_IV', subset, '*.pt')):
                continue
            start = splits[i]
            end = splits[i+1] if i < len(splits)-1 else len_ds
            if start == end:
                continue
            print('Saving dataset from %d to %d'%(start, end))
            ds_ = AMASS_DS(dataset_dir=split_dir, num_betas=num_betas, keep_from=start, keep_to=end)
            # save dataset to file
            os.makedirs(os.path.join(work_dir, 'stage_IV', subset), exist_ok=True)
            torch.save(ds_, os.path.join(work_dir, 'stage_IV', subset, 'amass_ds_{}.pt'.format(i)))

    
if __name__ == '__main__':
    main()
