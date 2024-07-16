#!/bin/bash
#SBATCH --job-name=mmpose
#SBATCH -c 2
#SBATCH -p gpu
##SBATCH --gres=gpu:TeslaA100_80:1
##SBATCH --gres=gpu:TeslaA100:1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=10G
#SBATCH --qos=preemptible

text_file=$1
pose2d=$2
# ouput_folder=$3

# dir_cmu_2d='/globalscratch/users/a/b/abolfazl/panoptic-toolbox/'$ouput_folder
dir_cmu_2d=$3
# root_dir='/globalscratch/users/a/b/abolfazl/panoptic-toolbox'
root_dir=$4

########## pose2d options:
# rtm-coco: human
# mpii: td-reg_res50_rle-8xb64-210e_mpii-256x256
# hrnet-coco: td-hm_hrnet-w32_8xb64-210e_coco-384x288

##########


for image in $(cat "$text_file")
do
    IFS='/' read -ra image_split <<< "$image"
    action=${image_split[-4]}
    camera=${image_split[-2]}
    dir_out1=$dir_cmu_2d/$action/$camera
    # image=$root_dir/$image
    # if $image starts with / then do nothing, else add $root_dir to the beginning of $image
    if [[ $image == /* ]]; then
        # echo "image starts with /"
        image=$image
    else
        image=$root_dir/$image
    fi
    echo python demo/inferencer_demo.py $image --pose2d $pose2d --pred-out-dir $dir_out1
    python demo/inferencer_demo.py $image --pose2d $pose2d --pred-out-dir $dir_out1
done
