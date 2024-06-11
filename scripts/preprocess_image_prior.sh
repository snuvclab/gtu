#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
conda activate gtu

CLASS_DATA_DIR="/mnt/hdd3/from_byungjun"

# 3. Get OpenPose using version implemented by lllyasviel
conda activate gtu
python -m diffusion_inversion.preprocess_prior \
    --data_path ${CLASS_DATA_DIR} \
    --use_dw_pose
echo Done!
