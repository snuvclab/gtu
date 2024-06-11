#!/bin/bash
set -e
DATA_PATH=$1
MAIN_CAMERA_ID=$2
RESOLUTION_SCALE=$4
TEST_NAME=$5
GPU_ID=$6


mkdir -p output
mkdir -p output/${TEST_NAME}

conda activate gtu

echo "Train BG with bg-regularize"
export CUDA_VISIBLE_DEVICES=$GPU_ID

python -m gtu.main.train_bg \
        -s $DATA_PATH \
        --main_camera ${MAIN_CAMERA_ID} \
        -r ${RESOLUTION_SCALE} \
        --data_device cpu \
        --model_path output/${TEST_NAME} \
        --use_mask \
        --sh_degree 2 \
        --use_bg_reg \
        --random_background \
        --occlusion_mask_path "none"

echo "Done"