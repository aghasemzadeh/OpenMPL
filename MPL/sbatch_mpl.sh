#!/bin/bash
#SBATCH --job-name=mpl
#SBATCH -c 16
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=80G
#SBATCH --qos=preemptible
#SBATCH -x mb-cas001
echo "Job on $HOSTNAME"
# python run/train_mpl.py --cfg /home/ucl/elen/abolfazl/OpenMPL/MPL/configs/openmplposer/mpl_amass/mpl_0_amass_mmpose_hrnet_MultiSPT_FPT_Conf3rd_ConfFPTNo_Raytoken_Learn3dEnc_Add3dEncRays_2views_Seed0.yaml --gpus 0 &
# python run/train_mpl.py --cfg /home/ucl/elen/abolfazl/OpenMPL/MPL/configs/openmplposer/mpl_amass/mpl_1_amass_mmpose_hrnet_MultiSPT_FPT_Conf3rd_ConfFPTNo_Raytoken_Learn3dEnc_Add3dEncRays_2views_Seed0.yaml --gpus 0 &
python run/train_mpl.py --cfg /home/ucl/elen/abolfazl/OpenMPL/MPL/configs/openmplposer/mpl_amass/mpl_2_amass_mmpose_hrnet_MultiSPT_FPT_Conf3rd_ConfFPTNo_Raytoken_Learn3dEnc_Add3dEncRays_2views_Seed0_ZrTkn.yaml --gpus 0 &
wait
