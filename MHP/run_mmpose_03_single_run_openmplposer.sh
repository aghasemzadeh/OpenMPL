#!/bin/bash

#SBATCH --job-name=amass_mmpose
#SBATCH -c 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:TeslaA100_80:1
#SBATCH --time=10:00:00
#SBATCH --mem=10G
##SBATCH -w mb-icg102
#SBATCH --qos=preemptible

echo "Job on $HOSTNAME"

support_dir=$1
work_dir=$2
amass_data_dir=$3
calib_path=$4
split_number=$5

python run_mmpose_02_run.py --dataset-split-number $split_number --exp all_with_mmpose --extra-name hrnet --use-cams-from openmplposer --calib-path-openmplposer $calib_path --room-size -3.89 3.89 -3.5 3.5 0 0 --operation-on train --image-width 1920 --image-height 1080 --apply-rotation --regressor coco --triangulate --triangulate-th 0.9 --pose2d-model td-hm_hrnet-w32_8xb64-210e_coco-384x288 --support-dir $support_dir --work-dir $work_dir --amass-data-dir $amass_data_dir 



echo "All done"
