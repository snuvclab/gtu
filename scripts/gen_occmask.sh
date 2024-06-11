#!/bin/bash
set -e
DATA_DIR=$1
GPU_ID=$2

export CUDA_VISIBLE_DEVICES=${GPU_ID}
CODE_ROOT_DIR=$(pwd)

# First get Depth-prediction
conda activate gtu
python -m preprocess.depthanything.do_depth \
    --data_path ${DATA_DIR}

# Second get Panoptic segmentations
GSAM_DIR=${CODE_ROOT_DIR}"/submodules/Grounded-Segment-Anything"

docker run --gpus all --rm --net=host --privileged \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v ${CODE_ROOT_DIR}:${CODE_ROOT_DIR} \
  -v /mnt/hdd:/mnt/hdd \
  -v /mnt/hdd2:/mnt/hdd2 \
  -v /mnt/hdd3:/mnt/hdd3 \
	-e DISPLAY=$DISPLAY \
  -e CUDA_VISIBLE_DEVICES=${GPU_ID} \
	--name=gsa${GPU_ID} \
  --shm-size 128gb \
	--ipc=host gsa:v0 \
  python ${CODE_ROOT_DIR}/preprocess/groundedsam/automatic_label_ram_demo.py \
  --config ${GSAM_DIR}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --ram_checkpoint ${CODE_ROOT_DIR}/checkpoints/ram_swin_large_14m.pth \
  --grounded_checkpoint ${CODE_ROOT_DIR}/checkpoints/groundingdino_swint_ogc.pth \
  --sam_checkpoint ${CODE_ROOT_DIR}/checkpoints/sam_vit_h_4b8939.pth \
  --data_dir ${DATA_DIR} \
  --box_threshold 0.25 \
  --text_threshold 0.2 \
  --iou_threshold 0.5 \
  --device "cuda"
  # --under_cam_dir


# Third get individual's rough masks
docker run --gpus all --rm --net=host --privileged \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v ${CODE_ROOT_DIR}:${CODE_ROOT_DIR} \
  -v /mnt/hdd:/mnt/hdd \
  -v /mnt/hdd2:/mnt/hdd2 \
  -v /mnt/hdd3:/mnt/hdd3 \
	-e DISPLAY=$DISPLAY \
  -e CUDA_VISIBLE_DEVICES=${GPU_ID} \
	--name=gsa${GPU_ID} \
  --shm-size 128gb \
	--ipc=host gsa:v0 \
  python ${CODE_ROOT_DIR}/preprocess/groundedsam/grounded_sam_demo.py \
  --config ${GSAM_DIR}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint ${CODE_ROOT_DIR}/checkpoints/groundingdino_swint_ogc.pth \
  --sam_checkpoint ${CODE_ROOT_DIR}/checkpoints/sam_vit_h_4b8939.pth \
  --data_dir ${DATA_DIR} \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "person" \
  --device "cuda"
