#!/bin/bash

#SBATCH --job-name=mmpose
#SBATCH -c 60
#SBATCH -p gpu
##SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=60G
#SBATCH --qos=preemptible

echo "Job on $HOSTNAME"

################ how to process CMU Panoptic dataset ################
# 1. run preprocess_cmu_panoptic.py to generate the .pkl files for train and validation
# 2. run preprocess_cmu_panoptic_filtered.py to filter the unncessary frames
# 3. run preprocess_cmu_panoptic_mmpose.py to generate the .pkl files with mmpose 2D keypoints

# run eval_cmu_panoptic_mmpose.py to evaluate the mmpose 2D keypoints
####################################################################


h36m_dir=/globalscratch/users/a/b/abolfazl/Human36m/H36M-Toolbox/h36m
dataset=annot
mmpose_dataset_name=mmpose_hrnet_coco
mmpose_output_path=/globalscratch/users/a/b/abolfazl/Human36m/Hybrik_parsed/mmpose_outputs_hrnet_coco
kept_keys_path=./h36m_filtered_keys_pose_5_64.txt

# python preprocess_h36m.py $h36m_dir $dataset --running-modes mmpose --run-for-sets validation --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard h36m &
python preprocess_h36m.py $h36m_dir $dataset --running-modes filter --run-for-sets train --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard h36m &
wait


echo "Done"