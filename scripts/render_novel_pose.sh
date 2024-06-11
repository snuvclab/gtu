#!/bin/bash
set -e
DATA_PATH=$1
MASK_PATH=$2
MAIN_CAMERA_ID=$3
TEST_NAME=$4
EXP_NAME=$5
GPU_ID=$6

python -m gtu.main.novel_pose \
        -s ${DATA_PATH} \
        --mask_path $MASK_PATH \
        --main_camera $MAIN_CAMERA_ID \
        --data_device cpu \
        --model_path output/${TEST_NAME} \
        --opt_smpl \
        --random_background \
        --sh_degree 2 \
        --human_track_method multiview \
        --exp_name ${EXP_NAME} \
        --use_zju_dataset \
        --eval_with_black_bg \
        --no_smpl_view_dir_reg

echo "Done!"