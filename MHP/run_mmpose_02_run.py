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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from mmpose.apis import MMPoseInferencer
from utils import *
from multiviews.triangulate import triangulate_poses


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
    parser.add_argument('--dataset-split-number', default=0, type=int, help='Dataset split')
    parser.add_argument('--train-datasets', default=None, nargs='+', type=str, help='Train datasets')
    parser.add_argument('--exp', default='amass', type=str, help='Experiment name')
    parser.add_argument('--extra-name', default=None, type=str, help='Extra name for the experiment')
    parser.add_argument('--use-cams-from', default='h36m', type=str, help='Use cameras from',
                        choices=['h36m', 'cmu'])
    parser.add_argument('--calib-path-h36m', default=None, type=str, help='Camera calibration path')
    parser.add_argument('--calib-path-cmu', default=None, type=str, help='Camera calibration path')
    parser.add_argument('--actors-h36m', default=[1, 5, 6, 7, 8, 9, 11], nargs='+', type=int, help='Actors to use')
    parser.add_argument('--calibs-cmu', default=['171204_pose5', '171204_pose6'], nargs='+', type=str, help='Calibrations to use')
    parser.add_argument('--views-cmu', default=[3, 6, 12, 13, 23], nargs='+', type=int, help='Views to use')
    parser.add_argument('--room-size', default=[-1.0, 1.0, -1.0, 1.0, 0.0, 0.0], nargs='+', type=float, help='Room size [min_x, max_x, min_y, max_y, min_z, max_z]')
    parser.add_argument('--operation-on', default=['train', 'validation'], nargs='+', type=str, help='Operations on')
    parser.add_argument('--image-width', default=1000, type=int, help='Image width')
    parser.add_argument('--image-height', default=1000, type=int, help='Image height')
    parser.add_argument('--apply-rotation', default=False, action='store_true', help='Apply rotation')
    parser.add_argument('--apply-joint-clip', default=False, action='store_true', help='Apply joint clip')
    parser.add_argument('--n-frames', default=-1, type=int, help='Number of frames to use')
    parser.add_argument('--fit-mmpose-to-amass', default=False, action='store_true', help='Fit skeleton to AMASS')
    parser.add_argument('--fit-using-most-aligned', default=False, action='store_true', help='Fit using all joints or only the most aligned ones')
    parser.add_argument('--regressor', default='h36m', type=str, help='Regressor to use', choices=['h36m', 'coco'])
    parser.add_argument('--triangulate', default=False, action='store_true', help='Triangulate the mmpose keypoints')
    parser.add_argument('--triangulate-th', default=0.85, type=float, help='Triangulation threshold')
    parser.add_argument('--pose2d-model', default='human', type=str, help='2D pose model to use for mmpose')
    parser.add_argument('--support-dir', default='../../amass/support_dir', type=str, help='Support directory for amass')
    parser.add_argument('--work-dir', default='./prepared_data/', type=str, help='Work directory')
    parser.add_argument('--amass-data-dir', default='../../amass_data_poses', type=str, help='PATH_TO_DOWNLOADED_NPZFILES/*/*_poses.npz')
    
    return parser.parse_args()


class AMASS_DS(Dataset):
    """AMASS: a pytorch loader for unified human motion capture dataset. http://amass.is.tue.mpg.de/"""

    def __init__(self, dataset_dir, num_betas=16):

        self.ds = {}
        for data_fname in glob.glob(os.path.join(dataset_dir, '*.pt')):
            k = os.path.basename(data_fname).replace('.pt','')
            self.ds[k] = torch.load(data_fname)
        self.num_betas = num_betas

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


def save_dataset(
                dataset_split_number=0,
                subset=None,
                work_dir=None,
                bm=None,
                faces=None,
                J_regressor=None,
                regressor='h36m',
                num_betas=None,
                cameras=None,
                views=None,
                n_camera_setups=1,
                imw=1000,
                imh=1000,
                room_min_x=-1.0,
                room_max_x=1.0,
                room_min_y=-1.0,
                room_max_y=1.0,
                room_min_z=0.0,
                room_max_z=0.0,
                calibs=[9, 11],
                h36m_or_cmu='h36m',
                apply_rotation=False,
                apply_joint_clip=False,
                number_of_frames=-1,
                fit_using_most_aligned=False,
                fit_mmpose_to_amass=False,
                extra_name=None,
                triangulate=False,
                triangulate_th=0.85,
                pose2d_model='human',
                ):
    calibs_used = '_'.join([str(a) for a in calibs])
    if fit_mmpose_to_amass:
        file_path = work_dir + '/stage_V/' + subset + '/split_{}_{}_amass_mmpose_joints_{}_{}_{}_calibs_{}_{}_fit_mmpose_to_amass_{}_{}_{}_{}.pkl'.format(
            dataset_split_number,
            extra_name if extra_name is not None else '',
            subset, 
            h36m_or_cmu, 
            regressor,
            calibs_used, 
            'rotated' if apply_rotation else 'not_rotated', 
            'joint_clip' if apply_joint_clip else 'no_joint_clip', 
            'fit_using_most_aligned' if fit_using_most_aligned else 'fit_using_all',
            'triangulated' if triangulate else 'not_triangulated',
            'numfram_{}'.format(number_of_frames) if number_of_frames != -1 else '',
            )
    else:
        file_path = work_dir + '/stage_V/' + subset + '/split_{}_{}_amass_mmpose_joints_{}_{}_{}_calibs_{}_no_fit_{}_{}_{}.pkl'.format(
            dataset_split_number,
            extra_name if extra_name is not None else '',
            subset, 
            h36m_or_cmu, 
            regressor,
            calibs_used, 
            'rotated' if apply_rotation else 'not_rotated', 
            'triangulated' if triangulate else 'not_triangulated',
            'numfram_{}'.format(number_of_frames) if number_of_frames != -1 else '',
            )
    if os.path.exists(file_path):
        print('File already exists:', file_path)
        return
        
    split_dir = os.path.join(work_dir, 'stage_III', subset)
    
    

    ds = AMASS_DS(dataset_dir=split_dir, num_betas=num_betas)
    ds = torch.load(os.path.join(work_dir, 'stage_IV', subset, 'amass_ds_{}.pt'.format(dataset_split_number)))
    print('Train split has %d datapoints.'%len(ds))

    batch_size = 1
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=5)
    # prepare dataset
    joints_3d_all = []
    joints_2d_mmpose_all = []
    confs_2d_mmpose_all = []
    joints_2d_amass_all = []
    confs_2d_mmpose_tri_all = []
    triangulated_3d_mmpose_all = []
    camera_setup_used = []
    views_all = []
    #### cameras is initially a dict of lists (view)(camera_setup) --> change it to list of lists (camera_setup)(view)
    cameras_tri = []
    if triangulate:
        for camera_setup in range(n_camera_setups):
            cameras_tri.append([])
            for view in views:
                cameras_tri[-1].append(cameras[view][camera_setup])
    jregressor = torch.Tensor(np.load(J_regressor))
    if fit_mmpose_to_amass:
        mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
        mv.set_cam_trans(trans=[0, 0, 0])
    else:
        mesh_viewers = []
        for camera_setup in range(n_camera_setups):
            mesh_viewers.append([])
            for view in views:
                fx = cameras[view][camera_setup]['fx']
                fy = cameras[view][camera_setup]['fy']
                cx = cameras[view][camera_setup]['cx']
                cy = cameras[view][camera_setup]['cy']
                mv = MeshViewer(width=imw, height=imh, use_offscreen=True, fx=fx, fy=fy, cx=cx, cy=cy)
                mv.set_cam_trans(trans=[0, 0, 0])
                mesh_viewers[-1].append(mv)
        # pass
    # mmpose_inferencer = MMPoseInferencer('human', device='cuda:0')
    mmpose_inferencer = MMPoseInferencer(pose2d_model, device='cuda:0')
    for ix, bdata in tqdm(enumerate(dataloader), total=len(dataloader), desc='Processing dataset {} {} split {}'.format(subset, 
                                                                                                                        extra_name if extra_name is not None else '',
                                                                                                                        dataset_split_number)):
        if number_of_frames !=-1 and ix > number_of_frames:
            break
        # if ix < 10:
        #     continue
        # elif ix > 10:
        #     break
        body_v = bm.forward(**{k:v.to(comp_device) for k,v in bdata.items() if k in ['pose_body', 'pose_hand', 'betas','root_orient', 'trans']}).v
        vertices = torch.Tensor(c2c(body_v))[0]
        vertices = locate_mesh_in_room(vertices, room_min_x, room_max_x, room_min_y, room_max_y, room_min_z, room_max_z, h36m_or_cmu=h36m_or_cmu, rotate=apply_rotation)
        camera_setup_to_use = np.random.random_integers(0, n_camera_setups-1)
        camera_setup_used.append(camera_setup_to_use)
        joints_2d_mmpose_all.append([])
        confs_2d_mmpose_all.append([])
        joints_2d_amass_all.append([])
        confs_2d_mmpose_tri_all.append([])
        views_all.append([])
        for view_ix, view in enumerate(views):
            R = cameras[view][camera_setup_to_use]['R']
            t = cameras[view][camera_setup_to_use]['t']
            K = cameras[view][camera_setup_to_use]['K']
            fx = cameras[view][camera_setup_to_use]['fx']
            fy = cameras[view][camera_setup_to_use]['fy']
            cx = cameras[view][camera_setup_to_use]['cx']
            cy = cameras[view][camera_setup_to_use]['cy']
            if not fit_mmpose_to_amass:
                # mv = MeshViewer(width=imw, height=imh, use_offscreen=True, fx=fx, fy=fy, cx=cx, cy=cy)
                # mv.set_cam_trans(trans=[0, 0, 0])
                mv = mesh_viewers[camera_setup_to_use][view_ix]
            Rt=np.eye(4)
            ##################
            Rt[:3, :3] = -R
            Rt[:3, 3] = -t.T
            Rt[0, :] = -Rt[0, :]        # flip x axis
            vertices_transformed = np.dot(vertices, Rt[:3, :3].T) + Rt[:3, 3]
            body_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
            mv.set_static_meshes([body_mesh])
            # try:
            body_image = mv.render(render_wireframe=False)
            # except:
            #     print('Error in rendering image for index {} and view {}'.format(ix, view))
            #     continue
            # h36m_points_mmpose, h36m_scores_mmpose, coco_points_mmpose, coco_scores_mmpose = run_mmpose(body_image, mmpose_inferencer, convert_to_h36m=)
            if regressor == 'h36m':
                points_mmpose, scores_mmpose = run_mmpose(body_image, mmpose_inferencer, convert_to_h36m=True)
                points_image_amass, joints_3d = amass_vertices_to_joints(vertices[None], jregressor, R, t, K, lower_body_reversed=True)
            elif regressor == 'coco':
                points_mmpose, scores_mmpose = run_mmpose(body_image, mmpose_inferencer, convert_to_h36m=False)
                points_image_amass, joints_3d = amass_vertices_to_joints(vertices[None], jregressor, R, t, K, lower_body_reversed=False)
            points_image_amass = points_image_amass[0]
            joints_3d = joints_3d[0]
            if triangulate:
                scores_mmpose_tri = filter_wrong_mmpose_detections(points_mmpose, scores_mmpose, points_image_amass, imw, imh)    # to filter joints outside image
            if fit_mmpose_to_amass:
                keypoints_most_aligned = find_most_aligned_joints(points_image_amass, keypoints_most_aligned_with_h36m, imw, imh)
                aligned_mmpose_joints, mmpose_confs = fit_skeleton_to_amass(points_mmpose, points_image_amass, keypoints_most_aligned, scores_mmpose, imw, imh, apply_joint_clip, fit_using_most_aligned=fit_using_most_aligned)
                joints_2d_mmpose_all[-1].append(aligned_mmpose_joints)
                confs_2d_mmpose_all[-1].append(mmpose_confs)
                # joints_2d_amass_all[-1].append(points_image_amass)
            else:
                joints_2d_mmpose_all[-1].append(points_mmpose)
                confs_2d_mmpose_all[-1].append(scores_mmpose)
            
            joints_2d_amass_all[-1].append(points_image_amass)
            if triangulate:
                confs_2d_mmpose_tri_all[-1].append(scores_mmpose_tri)
                
            views_all[-1].append(view)
            
        joints_3d_all.append(joints_3d)
        if triangulate:
            triangulated_3d_mmpose = triangulate_poses(cameras_tri[camera_setup_to_use], 
                                                       np.array(joints_2d_mmpose_all[-1]), 
                                                       np.array(confs_2d_mmpose_tri_all[-1]).squeeze(), 
                                                       conf_threshold=triangulate_th)
            triangulated_3d_mmpose_all.append(triangulated_3d_mmpose[0])
    joints_3d_all_np = np.array(joints_3d_all)
    joints_2d_mmpose_all_np = np.array(joints_2d_mmpose_all)
    confs_2d_mmpose_all_np = np.array(confs_2d_mmpose_all)
    joints_2d_amass_all_np = np.array(joints_2d_amass_all)
    if triangulate:
        triangulated_3d_mmpose_all_np = np.array(triangulated_3d_mmpose_all)
    else:
        triangulated_3d_mmpose_all_np = None
    camera_setup_used = np.array(camera_setup_used)
    views_all = np.array(views_all)

    with open(file_path, 'wb') as f:
        pickle.dump({
            'joints_3d': joints_3d_all_np,
            'joints_2d_mmpose': joints_2d_mmpose_all_np,
            'confs_2d_mmpose': confs_2d_mmpose_all_np,
            'joints_2d_amass': joints_2d_amass_all_np,
            'triangulated_3d_mmpose': triangulated_3d_mmpose_all_np,
            'camera_setup_used': camera_setup_used,
            'views_used': views_all,
        }, f)
    print('Saved dataset to:', file_path)

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
    
    if args.regressor == 'h36m':
        J_regressor = amass_dir + '/J_regressor_h36m.npy'
    elif args.regressor == 'coco':
        J_regressor = amass_dir + '/J_regressor_coco.npy'
    else:
        raise ValueError('Unknown regressor: {}'.format(args.regressor))
    
    work_dir = os.path.join(args.work_dir, expr_code)

    logger = log2file(makepath(work_dir, '%s.log' % (expr_code), isfile=True))
    logger('[%s] AMASS Data Preparation Began.'%expr_code)
    logger(msg)

    amass_splits['train'] = list(set(amass_splits['train']).difference(set(amass_splits['test'] + amass_splits['validation'])))

    logger('Train split has %d datasets.'%len(amass_splits['train']))
    logger('Train datasets: {}'.format(amass_splits['train']))

    bm_fname = osp.join(args.support_dir, 'body_models/smplh/male/model.npz')

    num_betas = 16 # number of body parameters

    bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas).to(comp_device)
    faces = c2c(bm.f)
    # print('faces:', faces.shape)

    if args.use_cams_from == 'h36m':
        if args.calib_path_h36m is None:
            raise ValueError('Please provide the camera calibration file for H36M')
        cameras = load_all_cameras_h36m(args.calib_path_h36m, actors=args.actors_h36m)
        logger('Loaded cameras from H36M')
        n_all_cameras = len(cameras) 
        all_camera_ids = list(cameras.keys())
        n_camera_setups = len(cameras[all_camera_ids[0]])  
        views = range(1, n_all_cameras+1)
        calibs = args.actors_h36m
    elif args.use_cams_from == 'cmu':
        if args.calib_path_cmu is None:
            raise ValueError('Please provide the camera calibration root for CMU')
        cameras = load_all_cameras_cmu(args.calib_path_cmu, cmu_calibs=args.calibs_cmu)
        logger('Loaded cameras from CMU')
        n_all_cameras = len(cameras) 
        all_camera_ids = list(cameras.keys())
        n_camera_setups = len(cameras[all_camera_ids[0]])
        views = args.views_cmu
        calibs = args.calibs_cmu
    
    print('Views:', views)
    
    if osp.join(work_dir, 'stage_V') not in glob.glob(osp.join(work_dir, '*')):
            os.makedirs(osp.join(work_dir, 'stage_V'))
    
    for subset in args.operation_on:
        if osp.join(work_dir, 'stage_V', subset) not in glob.glob(osp.join(work_dir, 'stage_V', '*')):
            os.makedirs(osp.join(work_dir, 'stage_V', subset))
        
    

    for subset in args.operation_on:
        save_dataset(
                    dataset_split_number=args.dataset_split_number,
                    subset=subset,
                    work_dir=work_dir,
                    bm=bm,
                    faces=faces,
                    J_regressor=J_regressor,
                    regressor=args.regressor,
                    num_betas=num_betas,
                    cameras=cameras,
                    views=views,
                    n_camera_setups=n_camera_setups,
                    imw=args.image_width,
                    imh=args.image_height,
                    room_min_x=args.room_size[0],
                    room_max_x=args.room_size[1],
                    room_min_y=args.room_size[2],
                    room_max_y=args.room_size[3],
                    room_min_z=args.room_size[4],
                    room_max_z=args.room_size[5],
                    h36m_or_cmu=args.use_cams_from,
                    calibs=calibs,
                    apply_rotation=args.apply_rotation,
                    apply_joint_clip=args.apply_joint_clip,
                    number_of_frames=args.n_frames,
                    fit_using_most_aligned=args.fit_using_most_aligned,
                    fit_mmpose_to_amass=args.fit_mmpose_to_amass,
                    extra_name=args.extra_name,
                    triangulate=args.triangulate,
                    triangulate_th=args.triangulate_th,
                    pose2d_model=args.pose2d_model,
                    )
    
if __name__ == '__main__':
    main()
