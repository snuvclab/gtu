#!/bin/bash
set -e

DATA_DIR=$1
GPU_ID=$2

mkdir -p ${DATA_DIR}/smplerx
mkdir -p ${DATA_DIR}/smplerx/output
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export LD_LIBRARY_PATH=/home/inhee/miniconda3/envs/smplerx/lib

end_count=$(find "$DATA_DIR/images_jpg" -type f | wc -l)
echo "Number of images: " $end_count
min_frame_id=$(find "$DATA_DIR/images_jpg" -maxdepth 1 -name "*.jpg" -exec basename {} \; | sed 's/[^0-9]//g' | awk '{print $1 + 0}' | sort -n | head -n 1)
echo "Min frame id: " $min_frame_id

conda activate smplerx
cd submodules/SMPLer-X/main

python inference.py \
    --num_gpus 1 \
    --exp_name ${DATA_DIR}/smplerx \
    --pretrained_model smpler_x_s32 \
    --agora_benchmark agora_model \
    --img_path ${DATA_DIR}/images_jpg  \
    --start ${min_frame_id} \
    --end $end_count \
    --output_folder ${DATA_DIR}/smplerx \
    --multi_person \
    --show_verts \
    --show_bbox \
    --save_mesh

ln -s ${DATA_DIR}/images_jpg ${DATA_DIR}/smplerx/orig_img
python render.py \
    --data_path ${DATA_DIR} --seq smplerx \
    --render_biggest_person False

# ffmpeg -y -f image2 -r 15 -i ${DATA_DIR}/smplerx/img/%05d.jpg -vcodec mjpeg -qscale 0 -pix_fmt yuv420p ${DATA_DIR}/smplerx/smplerx_results.mp4