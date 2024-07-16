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

split_number=$1


python run_mmpose_02_run.py --dataset-split-number $split_number --exp all_with_mmpose --extra-name hrnet --views-cmu $cams_cmu --use-cams-from h36m --calib-file-h36m camera_data.pkl --actors-h36m 9 11 --room-size -1 1 -1.5 2 0 0 --operation-on train --image-width 1000 --image-height 1000 --apply-rotation --regressor h36m --pose2d-model td-hm_hrnet-w32_8xb64-210e_coco-384x288

echo "All done"