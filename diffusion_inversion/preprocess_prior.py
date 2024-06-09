"""
    Preproces full-body prios
    (currently assume RGBA inputs)
"""
import argparse
import torch
import random
import numpy as np
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from preprocess.simple_openpose import WrapperDWPose, WrapperOPPose
from utils.draw_op_jnts import op25_to_op18, dwpose_to_op25
from gtu.guidance.joint_utils import filter_invisible_face_joints_w_prompts, get_view_prompt_of_body, combine_prompts


def get_direction_from_fname(img_fname: Path):
    """
    This is a deterministic function that optimizing image directions
    """
    # fid is angle of rotation (viewing right side first)
    fid = int(img_fname.name.split(".")[0])
    if fid > 180:
        is_left = True 
    else:
        is_left = False
        
    # add prompt of it
    if fid <= 30 or fid >= 330:
        prompt = "front"
    elif fid >= 150 and fid <=210:
        prompt = "back"
    else:
        prompt = "side"
    
    
    return is_left, prompt



def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Options for OpenPose estimations.")
    parser.add_argument("--use_dw_pose", action='store_true', help="If then, we use DW Pose Detector for higher accuracy")
    parser.add_argument("--data_path", type=str, help="Path of images to estimate.")
    args = parser.parse_args()


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load Pose estimation modules
    if args.use_dw_pose:
        from controlnet_aux import DWposeDetector
        det_config = "submodules/controlnet_aux/src/controlnet_aux/dwpose/yolox_config/yolox_l_8xb8-300e_coco.py"
        det_ckpt = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth"
        pose_config = "submodules/controlnet_aux/src/controlnet_aux/dwpose/dwpose_config/dwpose-l_384x288.py"
        pose_ckpt = "https://huggingface.co/wanghaofan/dw-ll_ucoco_384/resolve/main/dw-ll_ucoco_384.pth"
        pose_estimator = DWposeDetector(det_config=det_config, det_ckpt=det_ckpt, pose_config=pose_config, pose_ckpt=pose_ckpt, device=device)
        pose_estimator = WrapperDWPose(pose_estimator)
    else:
        pose_estimator = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        pose_estimator = pose_estimator.to(device)
        pose_estimator = WrapperOPPose(pose_estimator)
    
    # Load images to estimate
    data_path = Path(args.data_path) 
    img_lists = []
    for subject_dir in data_path.iterdir():
        if subject_dir.is_dir() and (subject_dir / "color_F").exists():
            img_fnames = sorted(list((subject_dir / "color_F").glob("*.png")))
            
            random_samples = random.sample(range(len(img_fnames)), 2)
            for _idx in random_samples:
                img_lists.append(img_fnames[_idx])
            
    
    # Make directories to save
    viz_save_dir = data_path / 'openpose'
    viz_save_dir.mkdir(exist_ok=True)
    
    img_save_dir = data_path / 'op_imgs'
    img_save_dir.mkdir(exist_ok=True)

    jnts_save_dir = data_path / 'op_jnts'
    jnts_save_dir.mkdir(exist_ok=True)

    # Estimate track infos
    for save_fid, img_fname in tqdm(enumerate(img_lists)):
        img = Image.open(str(img_fname)).convert("RGB")
        img_resolution = min(img.size)

        pose_overlay_img, poses = pose_estimator(img, bbox=None, include_hand=True, include_face=True, detect_resolution=512, image_resolution=img_resolution)
        pose_overlay_img.save(viz_save_dir / f"{save_fid:07}.jpg") # save visualized images

        # Save 2D joints(only full-body (18-D))
        jnts_save_fname = jnts_save_dir / f"{save_fid:07}.npy"
        if (img_save_dir / f"{save_fid:07}.png").exists():
            os.remove(img_save_dir / f"{save_fid:07}.png")
            shutil.copy(img_fname, img_save_dir / f"{save_fid:07}.png")

        body_jnts = poses[0]['body']
        if args.use_dw_pose:
            body_jnts = op25_to_op18(dwpose_to_op25(body_jnts))

        _body_jnts = []
        for jnt in body_jnts:
            if jnt is None:
                _body_jnts.append([-1, -1, 0])
            else:
                _body_jnts.append([jnt[0]*512, jnt[1]*512, jnt[2]])
        body_jnts = np.array(_body_jnts)
        
        
        # Save prompts together
        is_left, body_prompt = get_direction_from_fname(img_fname)
        filtered_op_joints, head_prompt = filter_invisible_face_joints_w_prompts(op_joints=body_jnts[:, :2].copy().tolist(), pj_jnts=body_jnts[:, :2].copy().tolist(), is_left=is_left)
        prompt = ", " + body_prompt + " view"
        prompt += ", " + head_prompt
        npy_save_contents = dict(
            jnts = body_jnts,
            prompts = prompt
        )
        np.save(jnts_save_fname, npy_save_contents, allow_pickle=True)

    print(f"Done! You can find the results in {str(jnts_save_dir)}")


if __name__ == '__main__':
    main()