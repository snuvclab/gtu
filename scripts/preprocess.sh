#!/bin/bash
set -e
VIDEO_DIR=$1
OUTPUT_DIR=$2
GPU_ID=$3

CODE_ROOT_DIR=$(pwd)
SAM_CHECKPOINT_PATH="checkpoints/sam_vit_h_4b8939.pth"
RESIZE_RATIO=1
SAMPLE_FPS=24
# SAMPLE_FPS=6
mkdir -p ${OUTPUT_DIR}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export NCCL_P2P_DISABLE=1

# 1. Extract frames (you can also modify resize part, too.)
conda activate 4D-humans
python preprocess/extract_frames.py \
    --video_path ${VIDEO_DIR} \
    --output_path ${OUTPUT_DIR} \
    --sample_fps ${SAMPLE_FPS} \
    --resize ${RESIZE_RATIO} \
    --save_jpg_in_addition
echo $"[[STEP 1]] Finished frame extractions"

# 2. Run 4D-humans
cd submodules/4D-Humans
conda activate 4D-humans
python track.py \
        video.source=${OUTPUT_DIR}/images_jpg \
        video.output_dir=${OUTPUT_DIR}/phalp_v2 \
        render.enable=True \
        render.type=HUMAN_MESH
cd ../..
echo $"[[STEP 2]] Finished Humans-4D w/ PHALP tracking"

#    2.1 Render 4D-huamns
conda activate gtu
python -m preprocess.humans4d.visualize \
        --data_dir ${OUTPUT_DIR} \
        --render_smpl 
echo $"[[STEP 2.1]] Finished Humans-4D visualization"

#    2.2 Extract JSON of each people for 
conda activate gtu
python -m preprocess.humans4d.generate_indiv \
        --data_dir ${OUTPUT_DIR} \
        --just_save_crop_json
echo $"[[STEP 2.2]] Finished Humans-4D BBOX extractors"

# 3. Get OpenPose using version implemented by lllyasviel
conda activate gtu
python -m preprocess.simple_openpose \
    --data_path ${OUTPUT_DIR} \
    --use_dw_pose
echo $"[[STEP 3]] Finished OpenPose joint estimators"

# 4. Post-Process in GTU format
conda activate gtu
python -m preprocess.merge_tracklets \
    --data_dir ${OUTPUT_DIR}
    # --viz \
    # --render_smpl 
echo $"[[STEP 4]] Finished Merging Estimations"

# 5. Prepare for getting Occlusion Mask
bash -i scripts/gen_occmask.sh ${OUTPUT_DIR} ${GPU_ID}

# 6. Fit SMPL mesh with OpenPose joints
conda activate gtu
python -m preprocess.smpl_fitting \
    --data_dir ${OUTPUT_DIR} \
    --is_static_camera \
    --render_smpl \
    --debug \
    --use_init_joints_as_pseudo_guide \
    --fit_smpl_hand
echo $"[[STEP 5]] Finished SMPL fitting on OP joints"

# 7.1 Get Masks using SAM
conda activate gtu
python -m preprocess.gen_humannerf_file \
    --data_dir ${OUTPUT_DIR} \
    --save_dir ${OUTPUT_DIR}_hn \
    --sam_checkpoint_path ${SAM_CHECKPOINT_PATH} \
    --use_person_mask

# 7.2 Get Masks to filter (To avoid using Camera-based compositional reconstruction get occmask using depth estimations)
conda activate gtu
python -m preprocess.gen_occmask \
    --data_dir ${OUTPUT_DIR} \
    --save_dir ${OUTPUT_DIR}_hn \
    --save_overview

# 8. Make init bg
conda activate gtu
python -m preprocess.generate_bg \
    --source_path ${OUTPUT_DIR} \
    --human_track_method multiview \
    --cam_name 0