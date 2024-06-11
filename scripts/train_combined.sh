#!/bin/bash
set -e
DATA_PATH=$1
MASK_PATH=$2
MAIN_CAMERA_ID=$3
RESOLUTION_SCALE=$4
TEST_NAME=$5
EXP_NAME=$6
TI_EXP_NAME=$7      
GPU_ID=$8


TEXTUAL_INVERSION="output_common/"${TEST_NAME}
TEXTUAL_INVERSION=$(pwd)/$TEXTUAL_INVERSION

mkdir -p output_common
mkdir -p ${TEXTUAL_INVERSION}
mkdir -p output
mkdir -p output/${TEST_NAME}

export CUDA_VISIBLE_DEVICES=$GPU_ID
        
EXP_NAME=${EXP_NAME}"_r"${RESOLUTION_SCALE}"_"${SAMPLE_METHOD}

conda activate gtu
python -m gtu.main.train_combined \
        -s $DATA_PATH \
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
        --use_diffusion_guidance \
        --textual_inversion_path ${TEXTUAL_INVERSION}/${MAIN_CAMERA_ID} \
        --textual_inversion_method ${SAMPLE_METHOD} \
        --check_ti_loaded \
        --use_ti_free_prompt_on_controlnet \
        --dgm_noise_sched time_annealing \
        --dgm_camera_sched defacto \
        --dgm_start_iter 1000 \
        --iterations 12010 \
        --test_iterations 12000 \
        --use_zju_dataset \
        --eval_with_black_bg \
        --lambda_rgb_loss 1000000 \
        --iter_clip_person_shs 10 \
        --dgm_cfg_scale 50 \
        --use_adaptive_rgb_loss \
        --dgm_cfg_rescale_weight 0. \
        --clip_init_smpl_opacity \
        --no_smpl_view_dir_reg \
        --use_lpips_loss \
        --use_density_reg_loss

echo "Done"