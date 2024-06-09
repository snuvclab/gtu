## Data structure
We manage the data to train our model as follows. Here we show an example of sequence with two people. The directory name `0` means the camera ID (default we use `0`) and directory name `000` and `001` are person IDs (`pid` in our code). 
```bash
# DATA_DIR of joint people information
{data_dir_name}
└── 0
    ├── images
    ├── images_jpg
    ├── mono_depth_viz
    ├── openpose
    ├── panoptic_segmentation
    ├── people_segmentation
    ├── smpl_fitting_results
    ├── phalp_v2
    ├── cameras.pkl
    ├── merge_hids.txt
    ├── mono_depth.pkl
    ├── openpose_estimation.pkl
    ├── op_phalp_merged.pkl
    ├── points3D.ply
    ├── human_points_0.txt
    ├── human_points_1.txt
    ├── results_p0.pkl
    └── results_p1.pkl

# DATA_DIR of individual information
{data_dir_name}_hn
└── 0
    ├── 000
    │   ├── images
    │   ├── masks
    │   ├── occ_masks
    │   └── overview
    ├── 001
    │   ├── images
    │   ├── masks
    │   ├── occ_masks
    │   └── overview
    ├── done.txt
    ├── _images
    └── sorted_pids.pkl
```

## Preprocessing
### Automated scripts
For preprocessing in-the-wild videos, run the commands below.

```bash
# General preprocessing
bash -i scripts/preprocess.sh [VIDEO_FILE_PATH] [DATA_DIR] [GPU_ID]
# Simplified preprocessing (skipping occlusion mask estimation)
bash -i scripts/preprocess_simple.sh [VIDEO_FILE_PATH] [DATA_DIR] [GPU_ID]
```

## Acknowledgements
Our automated in-the-wild video preprocessing code depends on following works. We sincerely thank the authors of
- [controlnet_aux](https://github.com/huggingface/controlnet_aux)
- [DWPose](https://github.com/IDEA-Research/DWPoses)
- [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything)
- [4D-Humans](https://github.com/shubham-goel/4D-Humans)
for their amazing work and codes!
