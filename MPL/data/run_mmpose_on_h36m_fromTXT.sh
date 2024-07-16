#!/bin/bash

text_file=$1
pose2d=$2
ouput_folder=$3

dir_h36m_2d='/globalscratch/users/a/b/abolfazl/Human36m/Hybrik_parsed/'$ouput_folder
root_dir='/globalscratch/users/a/b/abolfazl/Human36m/Hybrik_parsed/images'

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
    dir_out1=$dir_h36m_2d/$action/$camera
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