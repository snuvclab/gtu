#!/bin/bash
set -e
DATA_PATH=$1
MASK_PATH=$2
MAIN_CAMERA_ID=$3
RESOLUTION_SCALE=$4
TEST_NAME=$5
EXP_NAME=$6
TARGET_PID=$7   
GPU_ID=$8


TEXTUAL_INVERSION="output_common/"${TEST_NAME}
TEXTUAL_INVERSION=$(pwd)/$TEXTUAL_INVERSION

mkdir -p output_common
mkdir -p ${TEXTUAL_INVERSION}
mkdir -p output
mkdir -p output/${TEST_NAME}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
        
EXP_NAME=${EXP_NAME}"_r"${RESOLUTION_SCALE}"_"${SAMPLE_METHOD}
conda activate gtu
python -m gtu.main.train_combined \
        -s ${DATA_PATH} \
        -r ${RESOLUTION_SCALE} \
        --use_wandb \
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
        --iterations 12010 \
        --test_iterations 12000 \
        --save_iterations 12000 \
        --iter_clip_person_shs 10 \
        --clip_init_smpl_opacity \
        --no_smpl_view_dir_reg  \
        --use_lpips_loss \
        --use_density_reg_loss \
        --skip_aux_test \
        --lambda_rgb_loss 1 \
        --train_wo_bg \
        --target_pid ${TARGET_PID} \
        --preload_human_masks \
        --use_skip_fids_for_indiv_loading

echo "Done"