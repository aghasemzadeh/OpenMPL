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


### for preparing the amass dataset (only need to run once for both cmu and h36m)
# python run_mmpose_01_create_dataset.py --exp all_with_mmpose --operation-on train
# python run_mmpose_01_create_dataset.py --exp all_with_mmpose --operation-on validation


### to run MHP in parallel (if you dont touch the split number in the last step, no need to change these)
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 0
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 1
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 2
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 3
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 4
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 5
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 6
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 7
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 8
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 9
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 10
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 11
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 12
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 13
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 14
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 15
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 16
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 17
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 18
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 19
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 20
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 21
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 22
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 23
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 24
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 25
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 26
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 27
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 28
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 29
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 30
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 31
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 32
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 33
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 34
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 35
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 36
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 37
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 38
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 39
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 40
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 41
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 42
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 43
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 44
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 45
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 46
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 47
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 48
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 49
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 50
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 51
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 52
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 53
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 54
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 55
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 56
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 57
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 58
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 59
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 60
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 61
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 62
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 63
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 64
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 65
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 66
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 67
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 68
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 69
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 70
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 71
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 72
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 73
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 74
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 75
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 76
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 77
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 78
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 79
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 80
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 81
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 82
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 83
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 84
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 85
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 86
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 87
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 88
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 89
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 90
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 91
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 92
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 93
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 94
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 95
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 96
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 97
# sbatch run_mmpose_03_single_run_h36m_calibs.sh 98



# to combine all the results
# python run_mmpose_04_combine.py --exp all_with_mmpose --extra-name hrnet --operation-on train
# python run_mmpose_04_combine.py --exp all_with_mmpose --extra-name hrnet --operation-on validation

echo "All done"
