#!/bin/bash

# This code is same as running PHALP (Humans4D version) on in-the-wild videos
DATA_DIR=$1
GPU_ID=$2

conda activate 4d_humans_tracker
cd submodules/4D-Humans


export CUDA_VISIBLE_DEVICES=${GPU_ID}

mkdir -p ${DATA_DIR}/4dhuman


python track.py \
    video.source=${DATA_DIR}/images \
    video.output_dir=${DATA_DIR}/4dhuman \
    render.enable=True \
    render.type=HUMAN_MESH