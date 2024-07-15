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


# cmu_dir=/globalscratch/users/a/b/abolfazl/panoptic-toolbox
# dataset=annot_standard_coco_7train_2val_all_cams
# mmpose_dataset_name=mmpose_rtm_coco
# mmpose_output_path=/globalscratch/users/a/b/abolfazl/panoptic-toolbox/mmpose_outputs
# kept_keys_path=./filtered_keys_7train_2val_pose_5_64_all_cams.txt


# # python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes mmpose --run-for-sets validation --train-cams all --val-cams all --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard coco &
# python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes mmpose --run-for-sets train --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard coco &
# # python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes mmpose --run-for-sets train --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path &
# # python preprocess_cmu_panoptic.py $cmu_dir $dataset --not-run-for-train --only-run-filter --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path ./filtered_keys_2val_pose_5_64.txt &
# wait

# ### 5 cams in coco
# cmu_dir=/globalscratch/users/a/b/abolfazl/panoptic-toolbox
# dataset=annot_standard_coco_7train_2val_all_cams_no_bad_annot
# mmpose_dataset_name=mmpose_hrnet_coco
# mmpose_output_path=/globalscratch/users/a/b/abolfazl/panoptic-toolbox/mmpose_outputs_hrnet_coco
# kept_keys_path=./filtered_keys_7train_2val_pose_5_64_all_cams_no_bad_annot.txt


# # python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes mmpose --run-for-sets train --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard coco &
# # python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes preprocess filter --run-for-sets validation --train-cams all --val-cams all --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard coco &
# # python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes mmpose --run-for-sets validation --train-cams all --val-cams all --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard coco &
# # python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes preprocess filter --run-for-sets train --train-cams all --val-cams all --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard coco &
# wait

### 5 cams in h36m
cmu_dir=/globalscratch/users/a/b/abolfazl/panoptic-toolbox
dataset=annot_standard_coco_7train_2val_all_cams_no_bad_annot_h36m_from_coco_filters
mmpose_dataset_name=mmpose_hrnet_coco
mmpose_output_path=/globalscratch/users/a/b/abolfazl/panoptic-toolbox/mmpose_outputs_hrnet_coco
# kept_keys_path=./filtered_keys_7train_2val_pose_5_64_all_cams_no_bad_annot.txt
kept_keys_path=/home/ucl/elen/abolfazl/PPT/filtered_keys.txt


python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes filter mmpose --run-for-sets validation --filter-from-file --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard h36m &
wait

# ###### 5 cams in coco no vis discard
# cmu_dir=/globalscratch/users/a/b/abolfazl/panoptic-toolbox
# dataset=annot_standard_coco_7train_2val_all_cams_no_bad_annot_no_vis_discard
# mmpose_dataset_name=mmpose_hrnet_coco
# mmpose_output_path=/globalscratch/users/a/b/abolfazl/panoptic-toolbox/mmpose_outputs_hrnet_coco
# kept_keys_path=./filtered_keys_7train_2val_pose_5_64_all_cams_no_bad_annot_no_vis_discard.txt

# # python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes preprocess filter --run-for-sets train --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard coco --dont-discard-less-than-5-visible-joints &
# python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes mmpose --run-for-sets validation --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard coco --dont-discard-less-than-5-visible-joints &
# # python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes preprocess filter --run-for-sets validation --train-cams all --val-cams all --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard coco &
# # python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes mmpose --run-for-sets validation --train-cams all --val-cams all --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard coco &
# # python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes preprocess filter --run-for-sets train --train-cams all --val-cams all --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard coco &
# wait

# ############## all cameras in coco
# cmu_dir=/globalscratch/users/a/b/abolfazl/panoptic-toolbox
# dataset=annot_standard_coco_7train_2val_all_cams_no_bad_annot_no_vis_discard_all_cams
# mmpose_dataset_name=mmpose_hrnet_coco
# mmpose_output_path=/globalscratch/users/a/b/abolfazl/panoptic-toolbox/mmpose_outputs_hrnet_coco
# kept_keys_path=./filtered_keys_7train_2val_pose_5_64_all_cams_no_bad_annot_no_vis_discard_all_cams.txt


# python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes mmpose --run-for-sets validation --val-cams all --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard coco --dont-discard-less-than-5-visible-joints &
# wait

######### 5 cams in h36m
# cmu_dir=/globalscratch/users/a/b/abolfazl/panoptic-toolbox
# dataset=annot_standard_coco_7train_2val_all_cams_no_bad_annot_h36m_no_vis_discard
# mmpose_dataset_name=mmpose_hrnet_coco
# mmpose_output_path=/globalscratch/users/a/b/abolfazl/panoptic-toolbox/mmpose_outputs_hrnet_coco
# kept_keys_path=./filtered_keys_7train_2val_pose_5_64_all_cams_no_bad_annot_h36m_no_vis_discard.txt

# # python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes preprocess filter --run-for-sets train --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard h36m --dont-discard-less-than-5-visible-joints &
# python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes mmpose --run-for-sets validation --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard h36m --dont-discard-less-than-5-visible-joints &
# # python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes preprocess filter mmpose --run-for-sets train --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard coco &
# # python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes preprocess filter mmpose --run-for-sets validation --train-cams all --val-cams all  --skip-step-train 5 --skip-step-val 64 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard h36m &
# wait


# cmu_dir=/globalscratch/users/a/b/abolfazl/panoptic-toolbox
# dataset=annot_standard_coco_7train_2val_all_cams_no_bad_annot
# mmpose_dataset_name=mmpose_hrnet_coco
# mmpose_output_path=/globalscratch/users/a/b/abolfazl/panoptic-toolbox/mmpose_outputs_hrnet_coco
# kept_keys_path=./filtered_keys_7train_2val_pose_5_32_all_cams_no_bad_annot.txt


# # python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes filter --run-for-sets validation --train-cams all --val-cams all --skip-step-train 5 --skip-step-val 32 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard coco &
# python preprocess_cmu_panoptic.py $cmu_dir $dataset --running-modes mmpose --run-for-sets validation --train-cams all --val-cams all --skip-step-train 5 --skip-step-val 32 --write-kept-keys --kept-keys-path $kept_keys_path --mmpose-dataset-name $mmpose_dataset_name --mmpose-output-path $mmpose_output_path --keypoints-standard coco &
# wait

# python preprocess_cmu_panoptic_filtered.py

# python preprocess_cmu_panoptic_mmpose.py
# python eval_cmu_panoptic_mmpose.py

echo "Done"