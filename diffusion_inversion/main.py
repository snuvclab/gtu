#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import importlib
numpy = importlib.import_module('numpy')
numpy.float = numpy.float32
numpy.int = numpy.int32
numpy.bool = numpy.bool_
numpy.unicode = numpy.unicode_
numpy.complex = numpy.complex_
numpy.object = numpy.object_
numpy.str = numpy.dtype.str


import os
import sys
import numpy as np
import shutil
import torch
import cv2
import copy
from typing import List, Union, NamedTuple, Any, Optional, Dict
from random import randint, random
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from omegaconf import OmegaConf
from math import floor, ceil
from PIL import Image
from diffusers import (
    StableDiffusionInpaintPipeline, 
    StableDiffusionPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel, 
    UniPCMultistepScheduler,
)
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel


from gtu.renderer.gaussian_renderer import render_for_diffusion
from gtu.renderer.renderer_wrapper import render_set, render_traj, project_points_to_cam, render_scene_w_human, render_mv_human_in_scene, render_optim_logs, render_visibility_log, render_human_motions
from gtu import arguments
from gtu.dataset.dataloader import load_scene_human
from gtu.dataset.scene import Scene
from gtu.arguments import ModelParams, PipelineParams, OptimizationParams, HumanOptimizationParams
from gtu.guidance.joint_utils import filter_invisible_face_joints_w_prompts, get_view_prompt_of_body

from utils.general_utils import safe_state
from utils.log_utils import print_cli
from utils.image_utils import gen_videos, img_add_text
from utils.jnts_utils import filter_invisible_joints, extract_square_bbox
from utils.draw_op_jnts import draw_op_img
from utils.mask_utils import matting_masks

from gtu.guidance.joint_utils import filter_invisible_face_joints_w_prompts, get_view_prompt_of_body, combine_prompts




def load_sam_estimator():
    from segment_anything import SamPredictor, sam_model_registry

    CHECKPOINT = "/media/ssd2/inhee/download/checkpoints/sam_vit_h_4b8939.pth"
    MODEL = "vit_h"

    sam = sam_model_registry[MODEL](checkpoint=CHECKPOINT)
    sam.to("cuda")
    predictor = SamPredictor(sam)

    return predictor


def get_face_crop_bbox(pj_jnts):
    valid_inds = [0, -4, -3, -2, -1]
    valid_jnts = pj_jnts[valid_inds]
    bbox_offset_ratio = 1.6
    bbox = extract_square_bbox(valid_jnts, offset_ratio=bbox_offset_ratio, get_square=True)

    return bbox



def get_control_image(control_image, pred_rgb, guess_mode, do_classifier_free_guidance=True, controlnet=None, device=None, control_image_processor=None):
    height, width = _default_height_width(None, None, pred_rgb)
    batch_size = pred_rgb.shape[0]
    num_images_per_prompt = 1

    controlnet = controlnet._orig_mod if is_compiled_module(controlnet) else controlnet

    if isinstance(controlnet, ControlNetModel):
        control_image = control_image[0]    # load first image from list
        control_image = prepare_control_image(
            image=control_image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guess_mode=guess_mode,
            control_image_processor=control_image_processor,
        )
    elif isinstance(controlnet, MultiControlNetModel):
        control_images = []

        for control_image_ in control_image:
            control_image_ = prepare_control_image(
                image=control_image_,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
                control_image_processor=control_image_processor,
            )

            control_images.append(control_image_)

        control_image = control_images
    else:
        assert False

    return control_image

def prepare_control_image(
    image,
    width,
    height,
    batch_size,
    num_images_per_prompt,
    device,
    dtype,
    do_classifier_free_guidance=False,
    guess_mode=False,
    control_image_processor = None
):
    image = control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
    image_batch_size = image.shape[0]

    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt

    image = image.repeat_interleave(repeat_by, dim=0)

    image = image.to(device=device, dtype=dtype)

    if do_classifier_free_guidance and not guess_mode:
        image = torch.cat([image] * 2)

    return image

def _default_height_width(height, width, image):
    # NOTE: It is possible that a list of images have different
    # dimensions for each image, so just checking the first image
    # is not _exactly_ correct, but it is simple.
    while isinstance(image, list):
        image = image[0]

    if height is None:
        if isinstance(image, Image.Image):
            height = image.height
        elif isinstance(image, torch.Tensor):
            height = image.shape[2]

        height = (height // 8) * 8  # round down to nearest multiple of 8

    if width is None:
        if isinstance(image, Image.Image):
            width = image.width
        elif isinstance(image, torch.Tensor):
            width = image.shape[3]

        width = (width // 8) * 8  # round down to nearest multiple of 8

    return height, width



def resize_and_center_image(original_image, keypoints, target_resolution=512):
    # Get the original image resolution
    original_height, original_width = original_image.shape[:2]

    # Calculate the scale factor based on the longer side
    max_side = max(original_height, original_width)
    scale_factor = target_resolution / max_side

    # Resize the image
    resized_image = cv2.resize(original_image, None, fx=scale_factor, fy=scale_factor)

    # Create a blank canvas of size 512x512
    canvas_size = 512
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    # Calculate the position to paste the resized image at the center
    start_x = (canvas_size - resized_image.shape[1]) // 2
    start_y = (canvas_size - resized_image.shape[0]) // 2

    # Paste the resized image onto the canvas
    canvas[start_y:start_y + resized_image.shape[0], start_x:start_x + resized_image.shape[1]] = resized_image

    # Offset the keypoints based on the position on the canvas
    if keypoints is not None:
        scaled_keypoints = [(int(x * scale_factor) + start_x, int(y * scale_factor) + start_y) for x, y in keypoints]
    else:
        scaled_keypoints = None

    return canvas, scaled_keypoints
    



def get_crop_img_w_jnts(img, bbox, projected_jnts, rescale: float=1.2, resize: int=512):
    min_x = bbox[0]
    min_y = bbox[1]
    max_x = bbox[2]
    max_y = bbox[3]
    
    _w = int((max_x-min_x)*rescale)
    _h = int((max_y-min_y)*rescale)
    c_x = (min_x + max_x) // 2
    c_y = (min_y + max_y) // 2
    
    w = _w if _w>_h else _h
    h = w

    x = floor(c_x - w//2)
    y = floor(c_y - h//2)

    '''Crop in rectangular shape'''
    '''pad imgs when bbox is out of img'''
    x_front = 0   # offset for the case when we padded in front of the img.
    y_front = 0
    x_back = 0
    y_back = 0
    
    if x<0:
        x_front = -x
    if y<0:
        y_front = -y
    if x+w>= img.shape[1]:
        x_back = x+w-img.shape[1]+1
    if y+h>=img.shape[0]:
        y_back = y+w-img.shape[0]+1

    if x_front+y_front+x_back+y_back > 0:
        ext_img = cv2.copyMakeBorder(img, y_front, y_back, x_front, x_back, cv2.BORDER_CONSTANT, value=(0,0,0))
        x = x + x_front
        y = y + y_front
    else:
        ext_img = img
    cropped_img = ext_img[y:y+h, x:x+h]
    
    if isinstance(projected_jnts, List):
        _projected_jnts = []
        for _jnt in projected_jnts:
            if _jnt is None:
                _projected_jnts.append(_jnt)
            else:
                new_jnt = [0, 0]
                new_jnt[0] = _jnt[0] - (x - x_front)
                new_jnt[1] = _jnt[1] - (y - y_front)
                _projected_jnts.append(new_jnt)
        projected_jnts = _projected_jnts
    else:
        projected_jnts = projected_jnts - np.array([[x - x_front, y - y_front]])


    if resize > 0:
        re_cropped_img = cv2.resize(cropped_img, (resize, resize))
        scale_factor = resize / h
        
        if isinstance(projected_jnts, List):
            _projected_jnts = []
            for _jnt in projected_jnts:
                if _jnt is None:
                    _projected_jnts.append(_jnt)
                else:
                    new_jnt = [0, 0]
                    new_jnt[0] = (_jnt[0] - (h/2)) * scale_factor + resize/2
                    new_jnt[1] = (_jnt[1] - (h/2)) * scale_factor + resize/2
                    _projected_jnts.append(new_jnt)
            re_projected_jnts = _projected_jnts
        else:
            re_projected_jnts = (projected_jnts - np.array([[h/2, h/2]])) * scale_factor + np.array([[resize/2, resize/2]])
    
        return cropped_img, projected_jnts, re_cropped_img, re_projected_jnts
    else:
        return cropped_img, projected_jnts


def get_smallest_bbox(mask):
    # Find the coordinates of non-zero elements in the mask
    non_zero_coords  = np.argwhere(mask)
    if len(non_zero_coords) == 0:
        return None, None, None, None
    # assert len(non_zero_coords) != 0
    rows, cols = non_zero_coords[:, 0], non_zero_coords[:, 1]

    # Calculate the bounding box
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    # Return the bounding box coordinates
    return min_row, min_col, max_row, max_col



def crop_img_with_mask(img, mask, pj_jnts, crop_mode: str='default', pipe=None, ti_pipe=False):
    alpha = (mask[...,None] * 255).astype(np.int8) 
    img = np.concatenate([img, alpha], axis=-1)


    if crop_mode == 'default':
        print_cli("(default) Tightly Cropping visible joints", "info")
        min_x, min_y, max_x, max_y = get_smallest_bbox(mask)    # need to invert as it's cv2 (y,x)
        
        if min_x is None:
            return None, None, None, None, None, None

        raw_cropped_img, raw_projected_jnts, cropped_img, projected_jnts = get_crop_img_w_jnts(img, [min_y, min_x, max_y, max_x], pj_jnts, rescale=1.1, resize=512)
        cropped_img = cropped_img.astype(np.uint8)
        raw_cropped_img = raw_cropped_img.astype(np.uint8)
        
        masked_imgs = cropped_img

        # projected_jnts = filter_invisible_joints(projected_jnts)
        joint_imgs = draw_op_img(projected_jnts, 512)
        joint_imgs = np.array(joint_imgs)[..., ::-1]
        raw_masked_img = cv2.resize(masked_imgs, (raw_cropped_img.shape[0], raw_cropped_img.shape[1])) 
    
    if crop_mode == 'face':
        print_cli("(default) Tightly Cropping Face visible joints", "info")
        
        min_x, min_y, max_x, max_y = get_smallest_bbox(mask)    # need to invert as it's cv2 (y,x)
        face_min_x, face_min_y, face_max_x, face_max_y = get_face_crop_bbox(pj_jnts)
        
        if min_x is None:
            return None, None, None, None, None, None

        # it should be somewhat shared one
        min_x = face_min_x if face_min_x > min_x else min_x
        min_y = face_min_y if face_min_y > min_y else min_y
        max_x = face_max_x if face_max_x < max_x else max_x
        max_y = face_max_y if face_max_y < max_y else max_y

        # get center points
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        side_length = (max_x - min_x) if (max_x - min_x) > (max_y - min_y) else (max_y - min_y)
        min_x = center_x - side_length / 2
        min_y = center_y - side_length / 2
        max_x = center_x + side_length / 2
        max_y = center_y + side_length / 2
        print([min_y, min_x, max_y, max_x])

        raw_cropped_img, raw_projected_jnts, cropped_img, projected_jnts = get_crop_img_w_jnts(img, [min_y, min_x, max_y, max_x], pj_jnts, rescale=1.1, resize=512)
        cropped_img = cropped_img.astype(np.uint8)
        raw_cropped_img = raw_cropped_img.astype(np.uint8)
        
        masked_imgs = cropped_img

        # projected_jnts = filter_invisible_joints(projected_jnts)
        joint_imgs = draw_op_img(projected_jnts, 512)
        joint_imgs = np.array(joint_imgs)[..., ::-1]
        raw_masked_img = cv2.resize(masked_imgs, (raw_cropped_img.shape[0], raw_cropped_img.shape[1]))  
        
    elif crop_mode == 'inpaint':
        print_cli("(inpaint) Using inpainting Pipelines style", "info")
        generator = torch.manual_seed(0)
        
        min_x, min_y, max_x, max_y = get_smallest_bbox(mask)    # need to invert as it's cv2 (y,x)
        if min_x is None:
            return None, None, None, None, None, None
        
        

        raw_cropped_img, raw_projected_jnts, cropped_img, projected_jnts = get_crop_img_w_jnts(img, [min_y, min_x, max_y, max_x], pj_jnts, rescale=1.1, resize=512)
        cropped_img = cropped_img.astype(np.uint8)
        raw_cropped_img = raw_cropped_img.astype(np.uint8)

        # projected_jnts = filter_invisible_joints(projected_jnts)
        op_cond_img = draw_op_img(projected_jnts, 512)
        joint_imgs = np.array(op_cond_img)[..., ::-1]
        # Draw projected jnts here.

        sd_inpaint_img = cropped_img[...,:3][..., [2,1,0]]
        sd_inpaint_img = Image.fromarray(sd_inpaint_img)
        sd_inpaint_mask = 255 - cropped_img[...,3]
        sd_inpaint_mask = Image.fromarray(sd_inpaint_mask)

        with torch.no_grad():
            if ti_pipe:
                prompts = "a photo of a <new1> person"
            else:
                prompts = "a photo of a person"
            

            image = pipe(
                prompt="a photo of a person",
                negative_prompt="art, generated, six fingers, unrealistic, cartoon",
                generator=generator,
                image=sd_inpaint_img,
                mask_image=sd_inpaint_mask,
                control_image=op_cond_img,
                num_inference_steps=20,
                # strength=args.strength,
                guidance_scale=7.5,
                controlnet_conditioning_scale=1.,
            ).images[0]
            
        # Save Mask Concatentated Images
        masked_imgs = np.array(image)[..., ::-1]
        masked_imgs = np.concatenate([masked_imgs, cropped_img[...,-1:]], axis=-1)
        raw_masked_img = cv2.resize(masked_imgs, (raw_cropped_img.shape[0], raw_cropped_img.shape[1]))
        
        

    return masked_imgs, joint_imgs, cropped_img, raw_masked_img, raw_projected_jnts, raw_cropped_img


def load_inpainting_diffusion(use_inpaint_sd: bool=False, use_controlnet: bool=False, use_ti_on_inpaint: bool=False, ti_path=None):
    if use_inpaint_sd:
        torch_dtype = torch.float32
        if use_controlnet:
            controlnet_op = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch_dtype)
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting", 
                controlnet=controlnet_op,
                torch_dtype=torch_dtype,
                safety_checker=None,
            )
        else:
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting", 
                torch_dtype=torch_dtype,
                safety_checker=None,
            )
            
    else:
        torch_dtype = torch.float16
        if use_controlnet:
            controlnet_op = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch_dtype)
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                controlnet=controlnet_op,
                torch_dtype=torch_dtype,
                safety_checker=None,
            )
        else:
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                torch_dtype=torch_dtype,
                safety_checker=None,
            )
            
    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    if use_ti_on_inpaint:
        print("LOADING pretrained textual inversion!")
        pipe.unet.load_attn_procs(
            Path(ti_path), weight_name="pytorch_custom_diffusion_weights.bin"
        )
        pipe.load_textual_inversion(Path(ti_path), weight_name="<new1>.bin")
        print(f"[LOAD] Loading TI from {str(ti_path)}")



    return pipe
        


def do_ti(dataset, pipe, args):
    # Human train settings (change settings if needed here)
    human_train_opt = HumanOptimizationParams()
    human_train_opt.sh_degree = dataset.human_sh_degree
    human_train_opt.view_dir_reg = dataset.smpl_view_dir_reg


    
    arguments.MV_TRAIN_SAMPLE_INTERVAL = 1
    arguments.MV_AUX_SAMPLE_INTERVAL = 1

    # 1. Load train datasets
    scene, people_infos = \
        load_scene_human(
            dataset=dataset,
            pipe=pipe,
            scene_datasets=None,
            iteration=-1,
            exp_name="gen_mask",
            human_tracker_name=None, # option for manual comparison 
            novel_view_traj=[],
            use_diffusion_guidance=False,
            is_train=False,
            for_ti=True,
            fit_scale=False,
            load_aux_mv=False,
            checkpoint=None,
            human_train_opt=human_train_opt
        )

    # apply frame range set from arguments
    if dataset.frames_start >= 0 and dataset.frames_end > 0:
        print("only using limited frames")
        frame_end = len(scene.getTrainCameras()) if dataset.frames_end > len(scene.getTrainCameras()) else dataset.frames_end 
        ARG_FRAMES = list(range(dataset.frames_start, frame_end))
    else:
        ARG_FRAMES = []

    # do TI for first n people
    if dataset.n_person > 0:
        print("only training n_person using Diffusion")
        max_n_person = len(people_infos) if len(people_infos) < dataset.n_person else dataset.n_person
        people_infos = people_infos[:max_n_person]
    

    # 1.5 Make save directories
    if len(args.ti_save_dir) == 0:
        overall_save_dir = Path("output_common") / Path(dataset.model_path).name
    else:
        overall_save_dir = Path(args.ti_save_dir)
    overall_save_dir.mkdir(exist_ok=True)
    if scene.cam_name is None:
        overall_save_dir = overall_save_dir / str(dataset.main_camera)
    else:
        overall_save_dir = overall_save_dir / scene.cam_name
    overall_save_dir.mkdir(exist_ok=True)
    overall_save_dir = overall_save_dir / args.ti_exp_name
    overall_save_dir.mkdir(exist_ok=True)


    # 2. Prepare Diffusion Inpainting masks
    inpaint_pipe = None
    if args.create_additional_data:
        scene_cameras = scene.getTrainCameras()

        for pi in people_infos:
            pid = pi.human_id
            person_save_dir = overall_save_dir / pid
            person_save_dir.mkdir(exist_ok=True)

            mask_save_dir = person_save_dir / 'masked_images'
            mask_save_dir.mkdir(exist_ok=True)
            raw_mask_save_dir = person_save_dir / 'masked_images_raw'
            raw_mask_save_dir.mkdir(exist_ok=True)
            mask_jpg_save_dir = person_save_dir / 'masked_images_jpg'
            mask_jpg_save_dir.mkdir(exist_ok=True)
            op_cond_dir = person_save_dir / 'openpose_conditions'
            op_cond_dir.mkdir(exist_ok=True)
            op_jnts_dir = person_save_dir / 'op_jnts'
            op_jnts_dir.mkdir(exist_ok=True)
            op_overlay_dir = person_save_dir / 'op_overlay'
            op_overlay_dir.mkdir(exist_ok=True)
            align_overlay_dir = person_save_dir / 'align_overlay'
            align_overlay_dir.mkdir(exist_ok=True)
            
            human_mask_fname_dict = dict()
            for _cam in pi.human_scene.getTrainCameras():
                fid = _cam.colmap_id
                mask_fname = _cam.mask_fname
                if mask_fname is None:
                    continue
                human_mask_fname_dict[fid] = mask_fname
            
            valid_fids = args.partial_data_fids
            _v_fids = []
            for fid in valid_fids:
                fid = int(fid)
                if fid not in pi.fids:
                    continue

                if len(ARG_FRAMES) > 0:
                    if fid not in ARG_FRAMES:
                        continue
                else:
                    _v_fids.append(fid)
            valid_fids = _v_fids


            save_idx = args.save_fid_offset


            # Get sam masks
            if args.apply_sam:
                sam_predictor = load_sam_estimator()

                gt_mask_dict = dict()
                sam_mask_dict = dict()
                sam_mask = person_save_dir / 'sam_estimation'
                sam_mask.mkdir(exist_ok=True)
                for cam in tqdm(scene_cameras, desc="Getting SAM masks"):
                    cam_fid = cam.colmap_id
                    if cam_fid in valid_fids and cam_fid in human_mask_fname_dict:
                        gt_img_fname = cam.img_fname
                        gt_img_mask_fname = human_mask_fname_dict[cam_fid]

                        gt_mask = (cv2.imread(str(gt_img_mask_fname), 0)>1).astype(np.uint8)
                        gt_mask = cv2.resize(gt_mask, (cam.image_width, cam.image_height))
                        gt_img = cv2.imread(str(gt_img_fname))
                        gt_img = cv2.resize(gt_img, (cam.image_width, cam.image_height))
                        raw_gt_img = gt_img.copy()

                        # load projected_joints
                        _data_idx = pi.fids.index(cam_fid)
                        pj_jnts = np.stack(pi.misc['projected_op_jnts'][_data_idx], axis=0)     # (18, 2)

                        # Check front occlusion exists
                        if not scene.occ_cam_dict is None:
                            if cam_fid in scene.occ_cam_dict:
                                occ_mask = scene.occ_cam_dict[cam_fid]
                                occ_mask = occ_mask.numpy()
                                occ_mask = cv2.resize(occ_mask, (cam.image_width, cam.image_height))

                                if len(gt_mask.shape) == 2:
                                    gt_mask = gt_mask * occ_mask
                                else:
                                    gt_mask = gt_mask * occ_mask[..., None]

                        # Here we use BLACK background
                        gt_img[gt_mask==0] *= 0

                        # crop image based on mask
                        if len(gt_mask.shape) == 3:
                            gt_mask = (gt_mask.sum(-1) > 0)


                        with torch.no_grad():
                            sam_predictor.set_image(raw_gt_img)
                            sam_pts = pj_jnts
                            sam_pts_conf = np.ones_like(sam_pts[:,0])
                            masks, _, _ = sam_predictor.predict(sam_pts, sam_pts_conf)
                            _mask = masks.sum(axis=0) > 0
                            gt_mask = _mask.astype(np.float32)
                            gt_mask_dict[cam_fid] = gt_mask

                        sam_mask_img = _mask.astype(np.uint8)*255
                        sam_mask_img = np.stack([sam_mask_img, sam_mask_img, sam_mask_img], axis=-1)
                        sam_mask_dict[cam_fid] = sam_mask

                del sam_predictor

                

            # load inpainting module 
            if args.crop_mode == 'inpaint':
                inpaint_pipe = load_inpainting_diffusion(
                                    args.use_inpaint_sd_for_masked_images, 
                                    True, 
                                    args.two_stage_inpaint, 
                                    Path(args.inpaint_ti_dir) / pid
                                    )


            # Load all visible Cameras
            for cam in tqdm(scene_cameras, desc="Processing scene masks (with mask)"):
                cam_fid = cam.colmap_id
                if cam_fid in valid_fids and cam_fid in human_mask_fname_dict:
                    gt_img_fname = cam.img_fname
                    gt_img_mask_fname = human_mask_fname_dict[cam_fid]

                    gt_mask = (cv2.imread(str(gt_img_mask_fname), 0)>1).astype(np.uint8)
                    gt_mask = cv2.resize(gt_mask, (cam.image_width, cam.image_height))
                    gt_img = cv2.imread(str(gt_img_fname))
                    gt_img = cv2.resize(gt_img, (cam.image_width, cam.image_height))
                    raw_gt_img = gt_img.copy()

                    # crop image based on mask
                    if len(gt_mask.shape) == 3:
                        gt_mask = gt_mask[..., 0]
                    
                    # load projected_joints
                    _data_idx = pi.fids.index(cam_fid)
                    pj_jnts = np.stack(pi.misc['projected_op_jnts'][_data_idx], axis=0)     # (18, 2)

                    # Check front occlusion exists
                    if not scene.occ_cam_dict is None:
                        if cam_fid in scene.occ_cam_dict:
                            occ_mask = scene.occ_cam_dict[cam_fid]
                            occ_mask = occ_mask.numpy()
                            occ_mask = cv2.resize(occ_mask, (cam.image_width, cam.image_height))

                            if len(gt_mask.shape) == 2:
                                gt_mask = gt_mask * occ_mask
                            else:
                                gt_mask = gt_mask * occ_mask[..., None]

                    # Here we use BLACK background
                    gt_img[gt_mask==0] *= 0

                    
                    if args.apply_sam:
                        gt_mask = gt_mask_dict[cam_fid]
                        sam_mask_img = gt_mask_dict[cam_fid]


                    if args.alpha_matt_pixels > 0:
                        # apply smoothing on mask 
                        # We define it in "carving way" as lack of information is acceptible while false positive is big defection.
                        
                        # gt_mask, matting_viz = matting_masks(raw_gt_img, gt_mask*255, args.alpha_matt_pixels)
                        # gt_mask = gt_mask / 255.
                        
                        # matting_viz_dir = person_save_dir / 'matting_viz'
                        # matting_viz_dir.mkdir(exist_ok=True)
                        # cv2.imwrite(str(matting_viz_dir / f"{Path(gt_img_fname).name[:-4]}.jpg"), matting_viz)
                        
                        width_of_smoothing = int(args.alpha_matt_pixels)
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width_of_smoothing, width_of_smoothing))

                        # Perform erosion
                        gt_mask = cv2.erode(gt_mask, kernel, iterations=1)

                        # Apply Gaussian blur for smoother edges
                        gt_mask = cv2.GaussianBlur(gt_mask, (width_of_smoothing, width_of_smoothing), iterations=2)


                    # Get masked images
                    masked_img, joint_imgs, cropped_img, raw_masked_img, raw_projected_jnts, raw_cropped_img = crop_img_with_mask(gt_img, gt_mask, pj_jnts, crop_mode = args.crop_mode, pipe=inpaint_pipe, ti_pipe=args.use_inpaint_sd_for_masked_images)
                    if masked_img is None:
                        continue
                    
                    
                    jnts_fg = (joint_imgs.sum(-1, keepdims=True) > 0)
                    img_fg = cropped_img[..., :3] * (1-jnts_fg) + joint_imgs * jnts_fg
                    raw_jnts_img = draw_op_img(pj_jnts, raw_gt_img.shape[:2])
                    raw_jnts_img = np.array(raw_jnts_img)[..., ::-1]
                    jnts_fg = (raw_jnts_img.sum(-1, keepdims=True) > 0)
                    img_fg = raw_gt_img[..., :3] * (1-jnts_fg) + raw_jnts_img * jnts_fg

                    gt_img_fname = Path(gt_img_fname).name
                    save_extension = gt_img_fname.split(".")[-1]
                    save_n_fids = len(gt_img_fname.split(".")[0])
                    for _ in range(args.n_repeat_per_fids):
                        save_idx += 1
                        # make save names 
                        save_fid = str(save_idx).zfill(save_n_fids)
                        
                        # Save jnts as np array
                        np.save(str(op_jnts_dir / f"{save_fid}.npy"), raw_projected_jnts)
                        cv2.imwrite(str(raw_mask_save_dir / f"{save_fid}.png"), raw_masked_img)


                        cv2.imwrite(str(mask_save_dir / f"{save_fid}.png"), masked_img)
                        cv2.imwrite(str(mask_jpg_save_dir / f"{save_fid}.jpg"), masked_img)
                        cv2.imwrite(str(op_cond_dir / f"{save_fid}.png"), joint_imgs)
                        
                        # Debug Alignment
                        cv2.imwrite(str(op_overlay_dir / f"{save_fid}.jpg"), img_fg)
                        
                        # Debug Renderer
                        cv2.imwrite(str(align_overlay_dir / f"{save_fid}.jpg"), img_fg)

                        if args.apply_sam:
                            cv2.imwrite(str(sam_mask / f"{save_fid}.png"), sam_mask_img)

                    # get face zoomed sampling
                    # Get masked images
                    if args.create_face_zoomed_data:
                        masked_img, joint_imgs, cropped_img, raw_masked_img, raw_projected_jnts, raw_cropped_img = crop_img_with_mask(gt_img, gt_mask, pj_jnts, crop_mode = 'face', pipe=inpaint_pipe, ti_pipe=args.use_inpaint_sd_for_masked_images)
                        jnts_fg = (joint_imgs.sum(-1, keepdims=True) > 0)
                        img_fg = cropped_img[..., :3] * (1-jnts_fg) + joint_imgs * jnts_fg
                        raw_jnts_img = draw_op_img(pj_jnts, raw_gt_img.shape[:2])
                        raw_jnts_img = np.array(raw_jnts_img)[..., ::-1]
                        jnts_fg = (raw_jnts_img.sum(-1, keepdims=True) > 0)
                        img_fg = raw_gt_img[..., :3] * (1-jnts_fg) + raw_jnts_img * jnts_fg

                        gt_img_fname = Path(gt_img_fname).name
                        save_extension = gt_img_fname.split(".")[-1]
                        save_n_fids = len(gt_img_fname.split(".")[0])
                        for _ in range(args.n_repeat_per_fids):
                            save_idx += 1
                            # make save names 
                            save_fid = str(save_idx).zfill(save_n_fids)
                            np.save(str(op_jnts_dir / f"{save_fid}.npy"), raw_projected_jnts)
                            cv2.imwrite(str(raw_mask_save_dir / f"{save_fid}.png"), raw_masked_img)

                            cv2.imwrite(str(mask_save_dir / f"{save_fid}.png"), masked_img)
                            cv2.imwrite(str(mask_jpg_save_dir / f"{save_fid}.jpg"), masked_img)
                            cv2.imwrite(str(op_cond_dir / f"{save_fid}.png"), joint_imgs)
                            
                            # Debug Alignment
                            cv2.imwrite(str(op_overlay_dir / f"{save_fid}.jpg"), img_fg)
                            
                            # Debug Renderer
                            cv2.imwrite(str(align_overlay_dir / f"{save_fid}.jpg"), img_fg)

                            if args.apply_sam:
                                cv2.imwrite(str(sam_mask / f"{save_fid}.png"), sam_mask_img)
                

            if args.crop_mode == 'inpaint':
                del inpaint_pipe


    if args.gen_mask:
        scene_cameras = scene.getTrainCameras()

        for pi in people_infos:
            pid = pi.human_id
            person_save_dir = overall_save_dir / pid
            person_save_dir.mkdir(exist_ok=True)

            mask_save_dir = person_save_dir / 'masked_images'
            mask_save_dir.mkdir(exist_ok=True)
            raw_mask_save_dir = person_save_dir / 'masked_images_raw'
            raw_mask_save_dir.mkdir(exist_ok=True)
            op_jnts_dir = person_save_dir / 'op_jnts'
            op_jnts_dir.mkdir(exist_ok=True)
            view_prompts_dir = person_save_dir / 'view_prompts'
            view_prompts_dir.mkdir(exist_ok=True)
            mask_jpg_save_dir = person_save_dir / 'masked_images_jpg'
            mask_jpg_save_dir.mkdir(exist_ok=True)
            op_cond_dir = person_save_dir / 'openpose_conditions'
            op_cond_dir.mkdir(exist_ok=True)
            op_overlay_dir = person_save_dir / 'op_overlay'
            op_overlay_dir.mkdir(exist_ok=True)
            align_overlay_dir = person_save_dir / 'align_overlay'
            align_overlay_dir.mkdir(exist_ok=True)

            
            human_mask_fname_dict = dict()
            for _cam in pi.human_scene.getTrainCameras():
                fid = _cam.colmap_id
                mask_fname = _cam.mask_fname
                if mask_fname is None:
                    continue
                human_mask_fname_dict[fid] = mask_fname
                # print(f"{fid}: {str(mask_fname)}")


            valid_fids = []
            for fid in pi.fids:
                if len(ARG_FRAMES) > 0:
                    if fid not in ARG_FRAMES:
                        continue       
                valid_fids.append(fid)


            # Get sam masks
            if args.apply_sam:
                sam_predictor = load_sam_estimator()
                gt_mask_dict = dict()
                sam_mask_dict = dict()
                sam_mask = person_save_dir / 'sam_estimation'
                sam_mask.mkdir(exist_ok=True)

                for cam in tqdm(scene_cameras, desc="Getting SAM masks"):
                    cam_fid = cam.colmap_id
                    if cam_fid in valid_fids and cam_fid in human_mask_fname_dict:
                        gt_img_fname = cam.img_fname
                        gt_img_mask_fname = human_mask_fname_dict[cam_fid]

                        gt_mask = (cv2.imread(str(gt_img_mask_fname), 0)>1).astype(np.uint8)
                        gt_mask = cv2.resize(gt_mask, (cam.image_width, cam.image_height))
                        gt_img = cv2.imread(str(gt_img_fname))
                        gt_img = cv2.resize(gt_img, (cam.image_width, cam.image_height))
                        raw_gt_img = gt_img.copy()

                        # load projected_joints
                        _data_idx = pi.fids.index(cam_fid)
                        pj_jnts = np.stack(pi.misc['projected_op_jnts'][_data_idx], axis=0)     # (18, 2)

                        # Check front occlusion exists
                        if not scene.occ_cam_dict is None:
                            if cam_fid in scene.occ_cam_dict:
                                occ_mask = scene.occ_cam_dict[cam_fid]
                                occ_mask = occ_mask.numpy()
                                occ_mask = cv2.resize(occ_mask, (cam.image_width, cam.image_height))

                                if len(gt_mask.shape) == 2:
                                    gt_mask = gt_mask * occ_mask
                                else:
                                    gt_mask = gt_mask * occ_mask[..., None]

                        # Here we use BLACK background
                        gt_img[gt_mask==0] *= 0

                        # crop image based on mask
                        if len(gt_mask.shape) == 3:
                            gt_mask = (gt_mask.sum(-1) > 0)

                        with torch.no_grad():
                            sam_predictor.set_image(raw_gt_img)
                            sam_pts = pj_jnts
                            sam_pts_conf = np.ones_like(sam_pts[:,0])
                            masks, _, _ = sam_predictor.predict(sam_pts, sam_pts_conf)
                            _mask = masks.sum(axis=0) > 0
                            gt_mask = _mask.astype(np.float32)
                            gt_mask_dict[cam_fid] = gt_mask

                        sam_mask_img = _mask.astype(np.uint8)*255
                        sam_mask_img = np.stack([sam_mask_img, sam_mask_img, sam_mask_img], axis=-1)
                        sam_mask_dict[cam_fid] = sam_mask

                del sam_predictor

                # load inpainting module 
            if args.crop_mode == 'inpaint':
                inpaint_pipe = load_inpainting_diffusion(
                                    args.use_inpaint_sd_for_masked_images, 
                                    True, 
                                    args.two_stage_inpaint, 
                                    Path(args.inpaint_ti_dir) / pid
                                    )
            

            # Load all visible Cameras
            for cam in tqdm(scene_cameras, desc="Processing scene masks (with mask)"):
                cam_fid = cam.colmap_id
                # print(cam_fid)
                # print(valid_fids)
                # print(human_mask_fname_dict.keys())
                if cam_fid in valid_fids and cam_fid in human_mask_fname_dict:
                    gt_img_fname = cam.img_fname
                    gt_img_mask_fname = human_mask_fname_dict[cam_fid]

                    gt_mask = (cv2.imread(str(gt_img_mask_fname), 0)>1).astype(np.uint8)
                    gt_mask = cv2.resize(gt_mask, (cam.image_width, cam.image_height))
                    gt_img = cv2.imread(str(gt_img_fname))
                    gt_img = cv2.resize(gt_img, (cam.image_width, cam.image_height))
                    raw_gt_img = gt_img.copy()
                    
                    if args.alpha_matt_pixels > 0:
                        # apply smoothing on mask 
                        # We define it in "carving way" as lack of information is acceptible while false positive is big defection.
                        
                        # gt_mask, matting_viz = matting_masks(raw_gt_img, gt_mask*255, args.alpha_matt_pixels)
                        # gt_mask = gt_mask / 255.
                        
                        # matting_viz_dir = person_save_dir / 'matting_viz'
                        # matting_viz_dir.mkdir(exist_ok=True)
                        # cv2.imwrite(str(matting_viz_dir / f"{Path(gt_img_fname).name[:-4]}.jpg"), matting_viz)
                        
                        width_of_smoothing = int(args.alpha_matt_pixels)
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width_of_smoothing, width_of_smoothing))

                        # Perform erosion
                        gt_mask = cv2.erode(gt_mask, kernel, iterations=1)

                        # Apply Gaussian blur for smoother edges
                        gt_mask = cv2.GaussianBlur(gt_mask, (width_of_smoothing, width_of_smoothing), 0)
                    

                    # load projected_joints
                    _data_idx = pi.fids.index(cam_fid)
                    pj_jnts = pi.misc['projected_op_jnts'][_data_idx]
                    view_prompts = pi.misc['body_prompts'][_data_idx]

                    # Check front occlusion exists
                    if not scene.occ_cam_dict is None:
                        if cam_fid in scene.occ_cam_dict:
                            occ_mask = scene.occ_cam_dict[cam_fid]
                            occ_mask = occ_mask.numpy()
                            occ_mask = cv2.resize(occ_mask, (cam.image_width, cam.image_height))

                            if len(gt_mask.shape) == 2:
                                gt_mask = gt_mask * occ_mask
                            else:
                                gt_mask = gt_mask * occ_mask[..., None]

                    # Here we use BLACK background
                    gt_img[gt_mask==0] *= 0

                    # crop image based on mask
                    if len(gt_mask.shape) == 3:
                        gt_mask = (gt_mask.sum(-1) > 0)
                    
                    if args.apply_sam:
                        gt_mask = gt_mask_dict[cam_fid]
                        sam_mask_img = gt_mask_dict[cam_fid]
                        
                    

                    # Get masked images
                    masked_img, joint_imgs, cropped_img, raw_masked_img, raw_projected_jnts, raw_cropped_img = crop_img_with_mask(gt_img, gt_mask, pj_jnts, crop_mode = args.crop_mode, pipe=inpaint_pipe, ti_pipe=args.use_inpaint_sd_for_masked_images)
                    if masked_img is None:
                        continue
                    
                    
                    np.save(str(op_jnts_dir / f"{Path(gt_img_fname).name[:-4]}.npy"), raw_projected_jnts, allow_pickle=True)
                    np.save(str(view_prompts_dir / f"{Path(gt_img_fname).name[:-4]}.npy"), view_prompts, allow_pickle=True)
                    cv2.imwrite(str(raw_mask_save_dir / f"{Path(gt_img_fname).name[:-4]}.png"), raw_masked_img)

                    cv2.imwrite(str(mask_save_dir / Path(gt_img_fname).name[:-4]) + ".png", masked_img)
                    cv2.imwrite(str(mask_jpg_save_dir / Path(gt_img_fname).name[:-4]) + ".jpg", masked_img)
                    cv2.imwrite(str(op_cond_dir / Path(gt_img_fname).name[:-4]) + ".png", joint_imgs)
                    
                    # Debug Alignment
                    jnts_fg = (joint_imgs.sum(-1, keepdims=True) > 0)
                    img_fg = cropped_img[..., :3] * (1-jnts_fg) + joint_imgs * jnts_fg
                    cv2.imwrite(str(op_overlay_dir / Path(gt_img_fname).name[:-4]) + ".jpg", img_fg)
                    
                    # Debug Renderer
                    raw_jnts_img = draw_op_img(pj_jnts, raw_gt_img.shape[:2])
                    raw_jnts_img = np.array(raw_jnts_img)[..., ::-1]
                    jnts_fg = (raw_jnts_img.sum(-1, keepdims=True) > 0)
                    img_fg = raw_gt_img[..., :3] * (1-jnts_fg) + raw_jnts_img * jnts_fg
                    
                    img_fg = img_fg.astype(np.uint8)
                    img_fg = img_add_text(img_fg.copy(), f"prompts: {view_prompts}")
                    cv2.imwrite(str(align_overlay_dir / Path(gt_img_fname).name[:-4]) + ".jpg", img_fg)

                    if args.apply_sam:
                        cv2.imwrite(str(sam_mask / Path(gt_img_fname).name[:-4]) + ".png", sam_mask_img)
            
            
            if args.crop_mode == 'inpaint':
                del inpaint_pipe


    # 2. Train Custom Diffusion
    if args.optimize_cd:
        from diffusion_inversion.train_custom_diffusion import load_default_train_opt, train_cd

        for pi in people_infos:
            pid = pi.human_id
            person_save_dir = overall_save_dir / pid
            person_save_dir.mkdir(exist_ok=True)

            mask_save_dir = person_save_dir / 'masked_images'
            op_cond_dir = person_save_dir / 'openpose_conditions'
            
            cd_train_opt = load_default_train_opt(args.model_name)
            # cd_train_opt.pretrained_model_name_or_path = 
            cd_train_opt.instance_data_dir = str(mask_save_dir)
            cd_train_opt.instance_cond_dir = str(op_cond_dir)
            cd_train_opt.output_dir = str(person_save_dir)
            cd_train_opt.class_data_dir = "./diffusion_inversion/sample_person_photo"
            cd_train_opt.class_prompt = ""      # "human"
            cd_train_opt.num_class_images = 200
            cd_train_opt.instance_prompt = "photo of a <new1> person"  
            cd_train_opt.instance_prompt_wo_token = "photo of a person"  
            cd_train_opt.resolution = 512
            cd_train_opt.train_batch_size = args.cd_batch_size   # when using WITH PRIOR -> bsize 2 raise ERRORS
            cd_train_opt.learning_rate = args.cd_lrs             # default: 1e-5, 1e-6 is more specific version for face optimization
            cd_train_opt.lr_warmup_steps = 0
            cd_train_opt.max_train_steps = args.cd_steps
            cd_train_opt.scale_lr = True
            cd_train_opt.hfip = True                    
            cd_train_opt.modifier_token = "<new1>"
            cd_train_opt.validation_prompt = "photo of a <new1> person"
            cd_train_opt.report_to = "wandb"
            cd_train_opt.no_safe_serialization = True      
            cd_train_opt.use_controlnet = args.cd_use_controlnet   
            cd_train_opt.use_color_jitter = args.cd_use_color_jitter
            cd_train_opt.controlnet_mode = args.cd_controlnet_mode
            cd_train_opt.controlnet_weight = args.cd_controlnet_weight
            cd_train_opt.save_intermediate_for_debug = True     
            cd_train_opt.noaug = args.cd_no_aug
            cd_train_opt.bg_loss_weight = args.cd_bg_loss_weight
            cd_train_opt.random_bg = args.cd_random_bg
            

            cd_train_opt.image_space_loss = args.cd_loss_in_image_space
            cd_train_opt.loss_in_original_img_resolution = args.cd_loss_in_raw_resolution
            cd_train_opt.get_img_wo_resize = args.cd_get_img_wo_resize
            cd_train_opt.mask_cond_image_with_data_mask = args.cd_mask_controlnet_input
            
            if args.cd_get_img_wo_resize :
                op_cond_dir = person_save_dir / 'op_jnts'
                cd_train_opt.instance_cond_dir = str(op_cond_dir)
                mask_save_dir = person_save_dir / 'masked_images_raw'
                cd_train_opt.instance_data_dir = str(mask_save_dir)
                cd_train_opt.masking_cond_image = args.cd_masking_cond_image
                

            if args.cd_use_view_dependent_prompt :
                view_prompt_dir = person_save_dir / 'view_prompts'
                cd_train_opt.view_prompt_dir = str(view_prompt_dir)
            
            # Settings to save VRAMs
            cd_train_opt.enable_xformers_memory_efficient_attention = True
            cd_train_opt.set_grads_to_none = True

            if args.cd_use_prior:
                cd_train_opt.with_prior_preservation = True
                cd_train_opt.real_prior = True
                cd_train_opt.fullbody_prior = False
                cd_train_opt.prior_loss_weight = 1.
                # cd_train_opt.train_batch_size = int(cd_train_opt.train_batch_size/2) if cd_train_opt.train_batch_size > 1 else 1
                cd_train_opt.prior_batch_size = args.cd_prior_batch_size
                
            if args.cd_use_fullbody_prior:
                cd_train_opt.with_prior_preservation = True
                cd_train_opt.real_prior = True
                cd_train_opt.fullbody_prior = True
                cd_train_opt.fullbody_prior_path = args.fullbody_prior_path
                cd_train_opt.prior_loss_weight = 1.
                # cd_train_opt.train_batch_size = int(cd_train_opt.train_batch_size/2) if cd_train_opt.train_batch_size > 1 else 1
                cd_train_opt.prior_batch_size = args.cd_prior_batch_size


            if args.cd_controlnet_mode == "v4":
                raise NotImplementedError()
                cd_train_opt.cd_only_on_controlnet = False

            wandb_exp_name = f"{Path(dataset.model_path).name}_{str(args.ti_exp_name)}_{pid}"
            train_cd(cd_train_opt, wandb_exp_name)


            # Make video of optimization progress
            gen_videos([person_save_dir / 'optim_logs'], is_jpg=True, fps=10, rm_dir=False, regex_fname="0_*.jpg", save_tag="_0")
            gen_videos([person_save_dir / 'optim_logs'], is_jpg=True, fps=10, rm_dir=True, regex_fname="1_*.jpg", save_tag="_1")

            if cd_train_opt.save_intermediate_for_debug:
                if cd_train_opt.with_prior_preservation:
                    gen_videos([person_save_dir / 'debug_logs'], is_jpg=True, fps=30, rm_dir=False, regex_fname="priors_*.jpg", save_tag="_prior")
                gen_videos([person_save_dir / 'debug_logs'], is_jpg=True, fps=30, rm_dir=True, regex_fname="main_*.jpg", save_tag="_main")



    # 3. Test Random View Generations
    # Here we want to test ACTUALLY same situtations using on SDS Loss.
    # Let's sample cameras and prompts for that tasks.
    if args.test_cd_sds:
        from gtu.guidance import DiffusionGuidance
        people_ids = [pi.human_id for pi in people_infos]
        dgm_opt = OmegaConf.load("gtu/guidance/configs/default.yaml")
        dg_log_dir = overall_save_dir / "diffusion_guidance"

        white_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") # for testing
        black_bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        
        for cd_epoch in [-1]: # [1000, 2000]:
            # Prepare Diffusion Guided Module (load diffusion models)
            DGM = DiffusionGuidance(
                opt=dgm_opt, 
                log_dir=dg_log_dir, 
                textual_inversion_path=overall_save_dir.parent, 
                textual_inversion_expname=overall_save_dir.name,
                textual_inversion_in_controlnet=(args.cd_controlnet_mode=='v4'),
                use_ti_free_prompt_on_controlnet = dataset.use_ti_free_prompt_on_controlnet,
                ti_load_epoch = cd_epoch,
                guidance_scale = dataset.dgm_cfg_scale,
                controlnet_weight = dataset.dgm_controlnet_weight,
                lambda_percep=1.0,
                lambda_rgb=0.1,
                random_noise_step = dataset.dgm_random_sample,
                noise_sched = dataset.dgm_noise_sched,
                camera_sched = dataset.dgm_camera_sched,
                do_guid_sched = False,
                sd_version="1.5",
                use_aux_prompt = True,
                use_view_prompt = True
            )
            DGM.prepare_train(
                people_ids, 
                enable_controlnet = True,
                enable_zero123 = False,
                is_inpaint = False,
                do_cfg_rescale = True,
                do_multistep_sds = False
            )
            scene_cameras = scene.getTrainCameras()


            with torch.no_grad():
                for pi in people_infos:
                    # load masks 
                    pid = pi.human_id
                    render_fid = sorted(pi.fids)[0]
                    _data_idx = pi.fids.index(render_fid)
                    uid = 0
                    
                    person_save_dir = dg_log_dir / f"{pi.human_id}_{cd_epoch}"
                    person_save_dir.mkdir(exist_ok=True)

                    # load smpl_param
                    beta = pi.beta
                    if hasattr(pi, 'smpl_params'):
                        smpl_param = pi.smpl_params[_data_idx]
                        smpl_param[-10:] = beta

                        smpl_param = smpl_param.to('cuda').float()
                        smpl_param = smpl_param.unsqueeze(0)
                    else:
                        smpl_param = torch.cat([
                            pi.smpl_scale.reshape(-1),
                            pi.smpl_global_poses[_data_idx],
                            pi.smpl_local_poses[_data_idx],
                            beta
                        ], dim=-1)
                        smpl_param = smpl_param.unsqueeze(0)
                    smpl_deformer = pi.smpl_deformer

                    # cancel out global movement
                    smpl_param[0, 0] = 1.       # Fix scale as 1
                    smpl_param[0, 1:4] *= 0     # remove global translation
                    smpl_param[0, 2] = 0.3     # remove global translation (transl + 0.3 on y direction)
                    smpl_param[0, 4:7] *= 0     # remove global rotation


                    # load person infos
                    cam_dicts, aux_prompt_dicts, aux_neg_prompt_dicts, new_smpl_dicts = DGM.get_test_camera(
                        smpl_param = smpl_param,
                        smpl_deformer = smpl_deformer, 
                        n_cameras=18
                    )
                    # Reset dirs 
                    for k in cam_dicts.keys():
                        cam_save_dir = person_save_dir / k
                        if cam_save_dir.exists():
                            shutil.rmtree(cam_save_dir)


                    # Render images
                    vid_dirs = []
                    for k, v in cam_dicts.items():
                        pos_prompts = aux_prompt_dicts[k]
                        neg_prompts = aux_neg_prompt_dicts[k]
                        smpl_params = new_smpl_dicts[k]

                        cam_save_dir = person_save_dir / k
                        cam_save_dir.mkdir(exist_ok=True)
                        vid_dirs.append(cam_save_dir)
                        
                        idx = 0
                        for cam, pos, neg, _smpl_param in zip(v, pos_prompts, neg_prompts, smpl_params):
                            rendered_output, op_cond, jnt_prompts = render_for_diffusion(
                                mini_cam = cam,
                                pipe = pipe,
                                person_info = pi,
                                smpl_param  = _smpl_param, 
                                uid = uid,
                                bg_color = white_bg,
                            )

                            normal_rendering, op_cond, jnt_prompts = render_for_diffusion(
                                mini_cam = cam,
                                pipe = pipe,
                                person_info = pi,
                                smpl_param  = _smpl_param, 
                                uid = uid,
                                bg_color = white_bg,
                                normal_rendering = True
                            )

                            # set prompt conditioning
                            pos = pos + jnt_prompts
                            print_cli(f"positive: {pos}\nneg:{neg}\n-------", 'debug')

                            DGM.guidance_controlnet.get_text_embeds([pos], [neg])

                            generated_img = DGM.guidance_controlnet.refine(
                                pred_rgb=rendered_output,
                                pid = pid,
                                cond_image = op_cond,
                                guidance_scale=7.5,
                                steps=20,
                                strength=0.
                            )
                            
                            # save images
                            op_cond[0].save(cam_save_dir / f'op_{idx:03}.jpg')

                            generated_img = generated_img.detach().squeeze().cpu().permute(1,2,0).numpy()
                            generated_img = (generated_img * 255).astype(np.uint8)
                            generated_img = generated_img[..., [2,1,0]]    # convert to BGR
                            generated_img = np.concatenate([
                                np.ones((50, generated_img.shape[1], 3), dtype=np.uint8),
                                generated_img
                            ], axis=0)
                            generated_img = img_add_text(generated_img, f"pos: {pos}")
                            cv2.imwrite(str(cam_save_dir / f'generated_{idx:03}.jpg'), generated_img)

                            rendered_output = rendered_output.detach().cpu().squeeze().permute(1,2,0).numpy()
                            rendered_output = (rendered_output * 255).astype(np.uint8)   
                            cv2.imwrite(str(cam_save_dir / f'render_{idx:03}.jpg'), rendered_output)

                            normal_rendering = normal_rendering.detach().cpu().squeeze().permute(1,2,0).numpy()
                            normal_rendering = (normal_rendering * 255).astype(np.uint8)   

                            joint_imgs = np.array(op_cond[0])[..., ::-1]
                            jnts_fg = (joint_imgs.sum(-1, keepdims=True) > 0)
                            normal_rendering = normal_rendering[..., :3] * (1-jnts_fg) + joint_imgs * jnts_fg
                            normal_rendering = normal_rendering.astype(np.uint8)
                            normal_rendering = img_add_text(normal_rendering, f"neg: {neg}")
                            cv2.imwrite(str(cam_save_dir / f'overlay_{idx:03}.jpg'), normal_rendering)

                            idx += 1
                    
                    # Make Videos
                    gen_videos(vid_dirs, is_jpg=True, fps=10, rm_dir=False, regex_fname="op_*.jpg", save_tag="_op")
                    gen_videos(vid_dirs, is_jpg=True, fps=10, rm_dir=False, regex_fname="generated_*.jpg", save_tag="_gen")
                    gen_videos(vid_dirs, is_jpg=True, fps=10, rm_dir=False, regex_fname="render_*.jpg", save_tag="_raw")
                    gen_videos(vid_dirs, is_jpg=True, fps=10, rm_dir=False, regex_fname="overlay_*.jpg", save_tag="_overlay")



    if args.test_cd:
        people_ids = [pi.human_id for pi in people_infos]
        dgm_opt = OmegaConf.load("gtu/guidance/configs/default.yaml")
        test_log_dir = overall_save_dir / "test_inferences"
        test_log_dir.mkdir(exist_ok=True)

        with torch.no_grad():
            for pi in people_infos:
                # load masks 
                pid = pi.human_id
                render_fid = sorted(pi.fids)[0]
                _data_idx = pi.fids.index(render_fid)
                uid = 0

                # Render Images with txt prompts they used.
                validation_prompt = "photo of a <new1> person"  
                wo_cond_prompt = "photo of a person"  
                if args.cd_use_inpainting_diffusion:
                    print("using inpaint")
                    pipe = StableDiffusionInpaintPipeline.from_pretrained(
                        args.model_name, 
                        torch_dtype=torch.float16,
                        safety_checker=None
                    ).to("cuda")
                else:
                    pipe = StableDiffusionPipeline.from_pretrained(
                        args.model_name, 
                        torch_dtype=torch.float16,
                        safety_checker=None
                    ).to("cuda")
                
                #       Load textual inversions
                text_inv_path = overall_save_dir / pid
                pipe.unet.load_attn_procs(
                    Path(text_inv_path), weight_name="pytorch_custom_diffusion_weights.bin"
                )
                pipe.load_textual_inversion(Path(text_inv_path), weight_name="<new1>.bin")


                # Generate Images
                img_set_1 = []
                img_set_2 = []
                for _ in range(4):
                    if args.cd_use_inpainting_diffusion:
                        input_image = torch.zeros(4,3,512,512).float().cuda()
                        mask_image = torch.zeros(4,1,512,512).float().cuda()
                        n_img = 0
                        cnt = 0
                        while(n_img==0):
                            print(cnt)
                            image_1 = pipe(
                                validation_prompt,
                                image=input_image,
                                mask_image=mask_image,
                                negative_prompt="two people, multiple people",
                                num_inference_steps=25,
                                guidance_scale=7.5,
                                eta=1.0
                            ).images
                            n_img = len(image_1)
                            cnt += 1
                            
                            image_2 = pipe(
                                wo_cond_prompt,
                                image=input_image,
                                mask_image=mask_image,
                                negative_prompt="two people, multiple people",
                                num_inference_steps=25,
                                guidance_scale=7.5,
                                eta=1.0,
                            ).images

                        img_1 = [] 
                        for _img in image_1:
                            img_1.append(np.array(_img)[..., ::-1])
                        
                        img_2 = [] 
                        for _img in image_2:
                            img_2.append(np.array(_img)[..., ::-1])

                        img_1 = np.concatenate(img_1, axis=0)
                        img_2 = np.concatenate(img_2, axis=0)

                    else:
                        image_1 = pipe(
                            [validation_prompt] * 4,
                            num_inference_steps=25,
                            guidance_scale=7.5,
                            eta=1.0,
                        ).images

                        image_2 = pipe(
                            [wo_cond_prompt] * 4,
                            num_inference_steps=25,
                            guidance_scale=7.5,
                            eta=1.0,
                        ).images

                        img_1 = [] 
                        for _img in image_1:
                            img_1.append(np.array(_img)[..., ::-1])
                        
                        img_2 = [] 
                        for _img in image_2:
                            img_2.append(np.array(_img)[..., ::-1])

                        img_1 = np.concatenate(img_1, axis=0)
                        img_2 = np.concatenate(img_2, axis=0)
                    
                    img_set_1.append(img_1)
                    img_set_2.append(img_2)

                # final image concat
                img_set_1 = np.concatenate(img_set_1, axis=1)
                img_set_2 = np.concatenate(img_set_2, axis=1)

                img_set_1 = np.concatenate([
                    np.ones((200, img_set_1.shape[1], 3), dtype=np.uint8),
                    img_set_1
                ], axis=0)
                img_set_1 = img_add_text(img_set_1, f"{validation_prompt}")

                img_set_2 = np.concatenate([
                    np.ones((200, img_set_2.shape[1], 3), dtype=np.uint8),
                    img_set_2
                ], axis=0)
                img_set_2 = img_add_text(img_set_2, f"{wo_cond_prompt}")

            
                cv2.imwrite(str(test_log_dir / f"{pid}_w_ti.jpg"), img_set_1)
                cv2.imwrite(str(test_log_dir / f"{pid}_wo_ti.jpg"), img_set_2)

    if args.analyze_cd:
        raise NotImplementedError()
    

    if args.test_mp:
        raise NotImplementedError()
        
            


                    


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--gender', type=str, default='neutral')
    parser.add_argument('--detect_anomaly', action='store_true', default=False)    
    parser.add_argument("--human_camera_paths", nargs="+", type=str, default=[])
    parser.add_argument("--human_track_method", type=str, default ="phalp")
    parser.add_argument("--quiet", action="store_true")


    parser.add_argument("--ti_exp_name", type=str, help="TI experiments name")
    parser.add_argument("--ti_save_dir", type=str, default="")
    parser.add_argument("--mask_resize", type=int, default=512)

    parser.add_argument("--gen_mask", action='store_true')
    parser.add_argument("--optimize_cd", action='store_true')
    parser.add_argument("--test_cd", action='store_true')
    parser.add_argument("--test_cd_sds", action='store_true', help='test cd as similar as using on SDS')
    parser.add_argument("--test_mp", action='store_true', help='test diffusion on Multi-person')
    parser.add_argument("--analyze_cd", action='store_true', help='Analyze the meaning of CD')


    # Generating masked images
    parser.add_argument("--crop_mode", type=str, default="default")
    parser.add_argument("--two_stage_inpaint", action='store_true', help='use two-staged inpainting')
    parser.add_argument("--inpaint_ti_dir", type=str, default="output_common")
    parser.add_argument("--apply_sam", action='store_true', help='use mask obtained by SAM with projected jnts')
    parser.add_argument("--alpha_matt_pixels", default=5, help='ratio of matting pixels considering ')


    # Some CustomDiffusion experiments control options
    parser.add_argument("--cd_batch_size", type=int, default=2)
    parser.add_argument("--cd_steps", type=int, default=1000)
    parser.add_argument("--cd_lrs", type=float, default=5e-6)
    parser.add_argument("--cd_use_prior", action='store_true')
    parser.add_argument("--cd_use_controlnet", action='store_true', help='whether using controlnet on CD training')
    parser.add_argument("--cd_controlnet_weight", type=float, default=0.8)
    parser.add_argument("--cd_use_inpainting_diffusion", action='store_true', help='whether using inpainting diffusion on training')
    parser.add_argument("--cd_use_color_jitter", action='store_true', help='turn on color jittering of CD')
    parser.add_argument("--cd_controlnet_mode", type=str, default="v1")
    parser.add_argument("--cd_no_aug", action='store_true', help='turn off spatial augmentations of CD')
    parser.add_argument("--cd_bg_loss_weight", type=float, default=1.)
    parser.add_argument("--cd_random_bg", action='store_true', help='turn ON that randomly select bg from white/black')
    parser.add_argument("--cd_loss_in_image_space", action='store_true', help='Apply loss after VAE decoder')
    parser.add_argument("--cd_loss_in_raw_resolution", action='store_true', help='Apply loss in original image resolution')
    parser.add_argument("--cd_use_fullbody_prior", action='store_true', help='whether useing fullbody prior for optimization')
    parser.add_argument("--cd_get_img_wo_resize", action='store_true', help='whether getting condition image with better approach')
    parser.add_argument("--cd_prior_batch_size", default=1)
    
    parser.add_argument("--cd_masking_cond_image", action='store_true', help='whether getting condition image with better approach')
    parser.add_argument("--fullbody_prior_path", type=str, default=None)
    parser.add_argument("--cd_use_view_dependent_prompt", action='store_true', help='Use view-dependent prompt conditioning')
    parser.add_argument("--cd_mask_controlnet_input", action='store_true', help='Mask controlnet input with data mask')
    
    

    #   Add some additional images to CD,
    #   This is option for creating datasets,
    parser.add_argument("--create_additional_data", action='store_true', help='whether preprocess additional data or not')
    parser.add_argument("--create_face_zoomed_data", action='store_true', help='whether preprocess additional data or not')
    parser.add_argument("--partial_data_fids", nargs='+', help='fid of partial dataset')
    parser.add_argument("--save_fid_offset", type=int, default=0)
    parser.add_argument("--n_repeat_per_fids", type=int, default=1)
    

    # Some experiments options of diffusion
    parser.add_argument("--use_inpaint_sd_for_masked_images", action='store_true')
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")


    # Some experiments options for testing
    # parser.add_argument("--use_ti_free_prompt_on_controlnet", action='store_true')
    
    args = parser.parse_args(sys.argv[1:])
    
    print("Testing " + args.model_path)

    if False:
        if not (args.crop_mode == 'inpaint'):
            args.cd_random_bg = True            # Just set it as True!
            print("\n[INFO] TURN ON Random Background\n")


    # NEED TO UPDATE BETA from DATASET
    lp_extracted = lp.extract(args)
    # Add human camera paths
    human_camera_paths = args.human_camera_paths
    lp_extracted.human_camera_paths = human_camera_paths
    lp_extracted.human_track_method = args.human_track_method
    lp_extracted.eval = False               # Though it's eval,.for fast loading, we set as follow.

    if args.cd_controlnet_mode not in [
        "v0",       # TI token disconnected on ControlNet
        "v1",       # All connected
        "v2",       # No TI token on ControlNet
        "v3",       # All Connected. Use aligned Conditioning.
        "v4",       # Apply TI token ONLY on ControlNet (+ FineTune ControlNet)
    ]:
        raise TypeError(f"'{args.cd_controlnet_mode}' is invalid mode")
    # else:
    #     if args.cd_use_controlnet:
    #         args.ti_exp_name = args.ti_exp_name + "_" + str(args.cd_controlnet_mode)
            
    if args.cd_batch_size > 1 and args.cd_loss_in_image_space:
        print("\n[INFO] Reducing Batch size to avoid OOM error from image space")
        args.cd_batch_size = int(args.cd_batch_size/2)
        
    

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    do_ti(lp_extracted, pp.extract(args), args)
    # All done
    print("\nTesting complete.")
