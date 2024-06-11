#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

CLASS_DATA_DIR=diffusion_inversion/sample_person_photo
mkdir -p ${CLASS_DATA_DIR}

conda activate gtu
python -m diffusion_inversion.retrieve --class_prompt "a photo of person with full body, high quality" --class_data_dir ${CLASS_DATA_DIR} --num_class_images 300