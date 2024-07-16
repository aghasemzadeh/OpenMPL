#!/bin/bash

#SBATCH --job-name=amass_mmpose
#SBATCH -c 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:TeslaA100_80:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=150G
##SBATCH -w mb-icg102
#SBATCH --qos=preemptible

echo "Job on $HOSTNAME"

support_dir=/globalscratch/users/a/b/abolfazl/amass_data/support_data   # path to the support data in amass directory
work_dir=/globalscratch/users/a/b/abolfazl/amass_data/support_data/prepared_data   # path to the support data in amass directory
amass_data_dir=/globalscratch/users/a/b/abolfazl/amass_data_poses #'PATH_TO_DOWNLOADED_NPZFILES/*/*_poses.npz'
cmu_dir=/globalscratch/users/a/b/abolfazl/panoptic-toolbox # path to the CMU Panoptic dataset for calibration
### for preparing the amass dataset (only need to run once for both cmu and h36m)
# python run_mmpose_01_create_dataset.py --work-dir $work_dir --amass-data-dir $amass_data_dir --exp all_with_mmpose --operation-on train
# python run_mmpose_01_create_dataset.py --work-dir $work_dir --amass-data-dir $amass_data_dir --exp all_with_mmpose --operation-on validation


### to run MHP in parallel (if you dont touch the split number in the last step, no need to change these)
sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 0
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 1
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 2
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 3
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 4
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 5
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 6
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 7
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 8
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 9
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 10
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 11
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 12
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 13
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 14
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 15
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 16
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 17
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 18
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 19
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 20
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 21
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 22
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 23
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 24
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 25
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 26
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 27
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 28
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 29
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 30
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 31
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 32
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 33
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 34
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 35
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 36
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 37
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 38
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 39
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 40
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 41
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 42
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 43
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 44
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 45
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 46
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 47
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 48
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 49
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 50
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 51
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 52
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 53
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 54
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 55
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 56
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 57
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 58
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 59
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 60
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 61
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 62
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 63
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 64
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 65
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 66
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 67
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 68
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 69
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 70
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 71
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 72
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 73
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 74
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 75
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 76
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 77
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 78
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 79
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 80
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 81
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 82
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 83
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 84
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 85
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 86
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 87
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 88
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 89
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 90
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 91
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 92
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 93
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 94
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 95
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 96
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 97
# sh run_mmpose_03_single_run_cmu_calibs.sh $support_dir $work_dir $amass_data_dir $cmu_dir 98



# to combine all the results
# python run_mmpose_04_combine.py --exp all_with_mmpose --work-dir $work_dir --extra-name hrnet --operation-on train
# python run_mmpose_04_combine.py --exp all_with_mmpose --work-dir $work_dir --extra-name hrnet --operation-on validation

echo "All done"
