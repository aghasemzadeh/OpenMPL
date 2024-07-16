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
cams_cmu="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30"

python run_mmpose_02_run.py --dataset-split-number $split_number --exp all_with_mmpose --extra-name hrnet --views-cmu $cams_cmu --use-cams-from cmu --calib-root-cmu cmu_calibs --calibs-cmu  171204_pose5 171204_pose6 --room-size -0.5 -0.1 -0.2 0.2 0 0 --operation-on train --image-width 1920 --image-height 1080 --apply-rotation --regressor coco --triangulate --triangulate-th 0.9 --pose2d-model td-hm_hrnet-w32_8xb64-210e_coco-384x288



echo "All done"