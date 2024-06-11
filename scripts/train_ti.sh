#!/bin/bash
set -e
DATA_PATH=$1
MASK_PATH=$2
MAIN_CAMERA_ID=$3
RESOLUTION_SCALE=$4
TEST_NAME=$5

TI_EXP_NAME=$6      # name of TI experiment
GPU_ID=$7
PORT=$8


TEXTUAL_INVERSION="output_common/"${TEST_NAME}
TEXTUAL_INVERSION=$(pwd)/$TEXTUAL_INVERSION

mkdir -p output_common
mkdir -p ${TEXTUAL_INVERSION}
mkdir -p output
mkdir -p output/${TEST_NAME}


echo $"Train Textual Inversion with input image + ControlNetv2"
export CUDA_VISIBLE_DEVICES=$GPU_ID
export NCCL_P2P_DISABLE=1

FULL_BODY_PRIOR_PATH="/mnt/hdd3/from_byungjun"      # Update to yours if you use 

conda activate gtu
# 0. prepare TI 
python -m diffusion_inversion.main \
    -s ${DATA_PATH} \
    -r ${RESOLUTION_SCALE} \
    --mask_path $MASK_PATH \
    --data_device cpu \
    --model_path output/${TEST_NAME} \
    --occlusion_mask_path "none" \
    --main_camera $MAIN_CAMERA_ID \
    --ti_save_dir ${TEXTUAL_INVERSION} \
    --human_track_method multiview \
    --optimize_cd \
    --use_inpaint_sd_for_masked_images \
    --crop_mode default \
    --ti_exp_name ${TI_EXP_NAME} --cd_use_controlnet --cd_controlnet_mode v2 \
    --gen_mask

# 1. Do TI
accelerate launch --main_process_port ${PORT} \
    -m diffusion_inversion.main \
    -s ${DATA_PATH} \
    -r ${RESOLUTION_SCALE} \
    --mask_path $MASK_PATH \
    --data_device cpu \
    --model_path output/${TEST_NAME} \
    --occlusion_mask_path "none" \
    --main_camera $MAIN_CAMERA_ID \
    --ti_save_dir ${TEXTUAL_INVERSION} \
    --human_track_method multiview \
    --optimize_cd \
    --use_inpaint_sd_for_masked_images \
    --crop_mode default \
    --ti_exp_name ${TI_EXP_NAME} --cd_use_controlnet --cd_controlnet_mode v2 \
    --use_ti_free_prompt_on_controlnet \
    --cd_bg_loss_weight 0.5 \
    --cd_random_bg \
    --fullbody_prior_path ${FULL_BODY_PRIOR_PATH} \
    --cd_random_bg \
    --cd_use_view_dependent_prompt \
    --cd_get_img_wo_resize \
    --cd_controlnet_weight 0.7

    # --cd_use_fullbody_prior \


# 2. Visualize TI results
python -m diffusion_inversion.main \
    -s ${DATA_PATH} \
    -r ${RESOLUTION_SCALE} \
    --mask_path $MASK_PATH \
    --data_device cpu \
    --model_path output/${TEST_NAME} \
    --occlusion_mask_path ${OCC_MASK_PATH} \
    --main_camera $MAIN_CAMERA_ID \
    --ti_save_dir ${TEXTUAL_INVERSION} \
    --human_track_method multiview \
    --test_cd \
    --test_cd_sds \
    --use_inpaint_sd_for_masked_images \
    --crop_mode default \
    --ti_exp_name ${TI_EXP_NAME} --cd_use_controlnet --cd_controlnet_mode v2 \
    --use_ti_free_prompt_on_controlnet \
    --cd_bg_loss_weight 0.5 \
    --cd_random_bg