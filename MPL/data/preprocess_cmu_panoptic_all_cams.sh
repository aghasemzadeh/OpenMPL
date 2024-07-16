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



### 5 cams in coco
cmu_dir=/globalscratch/users/a/b/abolfazl/panoptic-toolbox      # path to the CMU Panoptic dataset
dataset=annot_standard_coco_7train_2val_all_cams_no_bad_annot   # name of the dataset given
mmpose_dataset_name=mmpose_hrnet_coco                           # suffix given to the dataset containing mmpose 2D keypoints
mmpose_output_path=/globalscratch/users/a/b/abolfazl/panoptic-toolbox/mmpose_outputs_hrnet_coco     # path to the directory where the mmpose 2D keypoints are saved
kept_keys_path=./filtered_keys_7train_2val_pose_5_64_all_cams_no_bad_annot.txt                      # path to the file containing the filtered frames
model=td-hm_hrnet-w32_8xb64-210e_coco-384x288                   # name of the mmpose model used for 2D keypoint estimation

python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes preprocess filter --run-for-sets train --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard coco 
txt_file=./filtered_keys_7train_2val_pose_5_64_all_cams_no_bad_annot_train.txt
sh run_mmpose_on_cmu_fromTXT.sh $txt_file $model $mmpose_output_path $cmu_dir
python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes mmpose --run-for-sets train --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard coco 

python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes preprocess filter --run-for-sets validation --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard coco 
txt_file=./filtered_keys_7train_2val_pose_5_64_all_cams_no_bad_annot_validation.txt
sh run_mmpose_on_cmu_fromTXT.sh $txt_file $model $mmpose_output_path $cmu_dir
python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes mmpose --run-for-sets validation --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard coco 


####################################################################

# ### 5 cams in h36m format
# cmu_dir=/globalscratch/users/a/b/abolfazl/panoptic-toolbox
# dataset=annot_standard_coco_7train_2val_all_cams_no_bad_annot_h36m_from_coco_filters
# mmpose_dataset_name=mmpose_hrnet_coco
# mmpose_output_path=/globalscratch/users/a/b/abolfazl/panoptic-toolbox/mmpose_outputs_hrnet_coco
# # kept_keys_path=./filtered_keys_7train_2val_pose_5_64_all_cams_no_bad_annot.txt
# kept_keys_path=/home/ucl/elen/abolfazl/PPT/filtered_keys.txt


# python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes filter mmpose --run-for-sets validation --filter-from-file --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard h36m &
# wait


echo "Done"