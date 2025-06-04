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
calib_path=/globalscratch/users/a/b/abolfazl/OpenMPLPoser_files/cameras # path to the OpenMPLPoser cameras
### for preparing the amass dataset (only need to run once for both cmu and h36m)
# python run_mmpose_01_create_dataset.py --work-dir $work_dir --amass-data-dir $amass_data_dir --exp all_with_mmpose --operation-on train
# python run_mmpose_01_create_dataset.py --work-dir $work_dir --amass-data-dir $amass_data_dir --exp all_with_mmpose --operation-on validation


### to run MHP in parallel (if you dont touch the split number in the last step, no need to change these)
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 0
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 1
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 2
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 3
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 4
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 5
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 6
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 7
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 8
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 9
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 10
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 11
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 12
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 13
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 14
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 15
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 16
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 17
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 18
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 19
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 20
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 21
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 22
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 23
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 24
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 25
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 26
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 27
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 28
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 29
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 30
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 31
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 32
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 33
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 34
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 35
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 36
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 37
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 38
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 39
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 40
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 41
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 42
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 43
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 44
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 45
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 46
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 47
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 48
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 49
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 50
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 51
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 52
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 53
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 54
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 55
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 56
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 57
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 58
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 59
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 60
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 61
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 62
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 63
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 64
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 65
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 66
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 67
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 68
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 69
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 70
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 71
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 72
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 73
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 74
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 75
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 76
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 77
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 78
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 79
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 80
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 81
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 82
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 83
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 84
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 85
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 86
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 87
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 88
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 89
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 90
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 91
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 92
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 93
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 94
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 95
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 96
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 97
# sbatch run_mmpose_03_single_run_openmplposer.sh $support_dir $work_dir $amass_data_dir $calib_path 98



# to combine all the results
python run_mmpose_04_combine.py --exp all_with_mmpose --work-dir $work_dir --extra-name hrnet --operation-on train
# python run_mmpose_04_combine.py --exp all_with_mmpose --work-dir $work_dir --extra-name hrnet --operation-on validation

echo "All done"
