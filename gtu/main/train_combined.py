#
# Joint Optimization of Human and Scene
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

import time
import os
import sys
import shutil
import torch
import pandas
import cv2
import numpy as np

from tqdm import tqdm, trange
from omegaconf import OmegaConf
from typing import List, Union, NamedTuple, Any, Optional, Dict
from random import randint, random
from argparse import ArgumentParser
from pathlib import Path


from gtu.renderer.gaussian_renderer import render, render_w_smpl, combined_renderer, diffusion_renderer
from gtu.renderer.torch3d_renderer import render_w_torch3d, render_emf_on_smpl
from gtu.renderer.renderer_wrapper import render_set, render_traj, project_points_to_cam, render_scene_w_human, render_mv_human_in_scene, render_optim_logs, render_visibility_log, render_human_motions, evaluator
from gtu import arguments
from gtu.arguments import ModelParams, PipelineParams, OptimizationParams, HumanOptimizationParams
from gtu.dataset.dataloader import load_scene_human

from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim, depth_loss_dpt, denisty_reg_loss
from utils.image_utils import gen_videos, tensor2cv, get_crop_img
from gtu.dataset.camera_utils import get_top_views_from_camera
from diffusion_inversion.utils import test_dgm_loaded_ti_properly


CLIP_HIGH_RENDERED_RGB = False

FAST_DEBUG=False
if FAST_DEBUG:
    print("\n\n\n\n\nFAST DEBUGGINS is ON now. be aware!\n\n\n\n\n")
VIZ_MASK_THRS = 0.0

### SMPL related hyper-parameters
# Larger value, less detail on SMPL
SMPL_LOSS_SCALER = 1        # (10.27) need to find proper values here


INVERSE_LOSS_WEIGHTING = False
 

# inpainting pipeline
# VIZ_CHECK_ITERS = [1000, 2000, 3000, 4000, 5000]
VIZ_CHECK_ITERS = []
VISIBILITY_LOGGER = True            # If True, do rendering of visibility (debug purpose)


# Aux settings
DGM_AUXILIARY_PROMPTING = True   # If True add more text guides.
DGM_VIEW_DEPENDENT_PROMPTING = True        # If True, do view-dependent prompting (ref-dreamfusion) (our model test with canceled SMPL global SE(3) so fine)
DGM_GUIDANCE_DECAY = False   # Default: false. 


DGM_ENABLE_ZERO123 = False
DGM_ENABLE_SDOP = True
### Debug Option


# Auxiliary options
RENDER_LOG_ITER = 10                # Rendering (for log) iteration
HUMAN_MOTION_ITERS = 100000
SAVE_LOSS_IMGS = True
SAVE_LOSS_ITER = 100


def get_index_in_r(x, y, r=5):
    ind_lists = []

    for i in range(r*2+1):
        for j in range(r*2+1):
            x_ = x - i
            y_ = y - j
            if i**2 + j**2 <= r**2:
                ind_lists.append([x_, y_])
    return ind_lists



def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, fit_scale=False, exp_name='default', novel_view_traj=[], test_aux_mv: bool=True, is_panoptic: bool=True):
    first_iter = 0

    # 0. Some training related settings
    do_fitting_scene = False if dataset.train_wo_bg else True
    do_smpl_fitting_local = True    # Whether fit SMPL pose during optimization
    do_smpl_fitting_global = False  # Whether fit SMPL global orientation during optimization 
    
    # Human train settings (change settings if needed here)
    human_train_opt = HumanOptimizationParams()
    human_train_opt.densify_from_iter = dataset.iter_smpl_densify - dataset.dgm_start_iter  # (start directly)
    human_train_opt.densify_until_iter = dataset.iter_densify_smpl_until
    human_train_opt.opacity_reset_interval = dataset.person_smpl_reset
    human_train_opt.view_dir_reg = dataset.smpl_view_dir_reg
    human_train_opt.sh_degree = dataset.human_sh_degree

    # smpl clipping settings
    human_train_opt.clip_init_smpl_opacity = dataset.clip_init_smpl_opacity
    human_train_opt.smpl_opacity_clip_min = dataset.smpl_opacity_clip_min
    
    print(f"vdir-reg: {getattr(dataset, 'smpl_view_dir_reg')}")
    if human_train_opt.view_dir_reg:
        print("\n\nview-direction regularizer is on now!!! be aware!!!\n\n")
    else:
        print("\n\tVIEW DIR REG OFF NOW\t\n")

    
    # 1. Load train datasets
    scene, scene_gaussians, people_infos, render_cameras, loaded_fitted_smpl = \
        load_scene_human(
            dataset=dataset,
            pipe=pipe,
            scene_datasets=None,
            iteration=-1,
            exp_name=exp_name,
            human_tracker_name=None, # option for manual comparison 
            novel_view_traj=novel_view_traj,
            is_train=True,
            fit_scale=fit_scale,
            load_aux_mv=test_aux_mv,
            checkpoint=checkpoint,
            human_train_opt=human_train_opt,
            skip_loading_bg=dataset.train_wo_bg,
        )
    
    # Copy Current training settings (to reproduce in future)
    shutil.copyfile(__file__, Path(str(scene.model_path)) / 'train.py')
    shutil.copyfile(Path(str(__file__)).parents[1] / 'arguments' / '__init__.py', Path(str(scene.model_path)) / 'argument.py')
    shutil.copyfile(Path(str(__file__)).parents[1] / 'guidance' / '__init__.py', Path(str(scene.model_path)) / 'guidance.py')
    shutil.copyfile(Path(str(__file__)).parents[1] / 'guidance' / 'diffusion_module' / 'controlnet_guidance.py', Path(str(scene.model_path)) / 'controlnet_guidance.py')
    shutil.copyfile(Path(str(__file__)).parents[1] / 'renderer' / 'gaussian_renderer.py', Path(str(scene.model_path)) / 'gs_renderer.py')

    
    
    for pi in people_infos:
        if pi.view_dir_reg:
            print("\n\n\nview dir reg is True for SMPL now!!!!!\n\n\n")

    
    if do_fitting_scene:
        scene_gaussians.training_setup(opt)


    # 4. set minor options
    if dataset.use_lpips_loss:
        from lpips import LPIPS
        lpips = LPIPS(net='vgg').cuda()

        # turn off grad tracking of lpips param
        nets = lpips
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = False

    # set data related to scenes
    scene_gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        scene_gaussians.restore(model_params, opt)
    
    ################  start train  ###############
    if dataset.use_diffusion_guidance or dataset.use_novel_view_density_reg:
        # we also use DGM module to sample camera similarly
        from gtu.guidance import DiffusionGuidance

        people_ids = [pi.human_id for pi in people_infos]
        if dataset.n_person > 0:
            print("only training n_person using Diffusion")
            max_n_person = len(people_ids) if len(people_ids) < dataset.n_person else dataset.n_person
            people_ids = people_ids[:max_n_person]
        
        if dataset.target_pid >= 0:
            hpeople_ids = [dataset.target_pid]
        

        os.makedirs(scene.model_path + "/training", exist_ok=True)
        dg_log_dir = Path(scene.model_path) / "training" / "diffusion_guidance"
        dgm_opt = OmegaConf.load("gtu/guidance/configs/default.yaml")
        dgm_opt.density_start_iter = dataset.iter_smpl_densify
        dgm_opt.density_end_iter = dataset.iter_densify_smpl_until
        dgm_opt.densification_interval = human_train_opt.densification_interval
        dgm_opt.iter_prune_smpl_until = dataset.iter_prune_smpl_until
        dgm_opt.iter_smpl_densify = dataset.iter_smpl_densify
        dgm_opt.scene_extent = float(scene.cameras_extent)

        if dataset.dgm_use_ddim_denoiser:
            print("[INFO] we are using multi-step denoiser (DDIM) now!")

        
        textual_inversion_path = Path(dataset.textual_inversion_path) if dataset.textual_inversion_path != "" else None
        textual_inversion_expname = dataset.textual_inversion_method
        DGM = DiffusionGuidance(
            opt=dgm_opt, 
            log_dir=dg_log_dir, 
            textual_inversion_path=textual_inversion_path, 
            textual_inversion_expname=textual_inversion_expname,
            textual_inversion_in_controlnet=dataset.use_ti_in_controlnet,
            use_ti_free_prompt_on_controlnet = dataset.use_ti_free_prompt_on_controlnet,
            ti_load_epoch = dataset.ti_chkpt_epoch,
            guidance_scale = dataset.dgm_cfg_scale,
            inpaint_guidance_scale = dataset.dgm_inpaint_guidance_scale,
            controlnet_weight = dataset.dgm_controlnet_weight,
            lambda_percep=opt.lambda_dgm_percep,
            lambda_rgb=opt.lambda_dgm_rgb,
            random_noise_step = dataset.dgm_random_sample,
            noise_sched = dataset.dgm_noise_sched,
            camera_sched = dataset.dgm_camera_sched,
            do_guid_sched = DGM_GUIDANCE_DECAY,
            sd_version="1.5",
            use_aux_prompt = DGM_AUXILIARY_PROMPTING,
            use_view_prompt = DGM_VIEW_DEPENDENT_PROMPTING,
            cfg_sched = dataset.dgm_cfg_sched,
        )
        DGM.prepare_train(
            people_ids, 
            enable_controlnet = DGM_ENABLE_SDOP,
            enable_zero123 = DGM_ENABLE_ZERO123,
            is_inpaint = dataset.dgm_use_inpaint and (not dataset.dgm_use_optimizer_masking),
            do_cfg_rescale = dataset.dgm_use_cfg_rescale,
            do_multistep_sds = dataset.dgm_use_ddim_denoiser,
            use_inpaint_unet = dataset.dgm_use_inpaint_unet,
            use_joint_diff = dataset.dgm_use_joint_diff
        )

        CANON_LOSS_MODE = 'all' if dataset.use_canon else None
        CANON_LOSS_MODE = CANON_LOSS_MODE + '_fixed' if (dataset.use_canon_single_camera and not (CANON_LOSS_MODE is None)) else CANON_LOSS_MODE
        if dataset.use_canon_single_camera:
            print("\n[WARNING] USE SINGLE CAMERA + CANON MODE\n")


        if dataset.check_ti_loaded:
            os.makedirs(scene.model_path + "/training", exist_ok=True)
            ti_load_check_save_dir = Path(scene.model_path) / "training" / "check_TI_load"
            ti_load_check_save_dir.mkdir(exist_ok=True)
            if dataset.n_person > 0:
                test_dgm_loaded_ti_properly(DGM, scene, people_infos[:max_n_person], ti_load_check_save_dir, pipe)
            else:
                test_dgm_loaded_ti_properly(DGM, scene, people_infos, ti_load_check_save_dir, pipe)



    else:
        DGM = None
        CANON_LOSS_MODE = None
    
    
    if pipe.use_wandb:
        import wandb
        wandb.init(
            project="diffusion_gs",
            name=exp_name
        )
        setattr(dataset, '_human_opts', human_train_opt)
        setattr(dataset, '_train_opts', opt)
        wandb.config.update(dataset)
        wandb.config.update(opt)

        parsed_dict = wandb.sdk.wandb_helper.parse_config(human_train_opt)
        renamed_dict = dict()
        for k, v in parsed_dict.items():
            renamed_dict[f"human_opt_{k}"] = v
        wandb.config.update(renamed_dict)
        
        # define our custom x axis metric
        wandb.define_metric("loss/step")
        wandb.define_metric("loss/*", step_metric="loss/step")

        # define time axis for metric
        wandb.define_metric("metric/*", step_metric="loss/step")
        wandb.define_metric("dg_loss/*", step_metric="loss/step")
        wandb.define_metric("infos/*", step_metric="loss/step")
        
        if dataset.use_diffusion_guidance:
            for pi in people_infos:
                wandb.define_metric(f"_{pi.human_id}/*", step_metric=f"loss/step")
                
        # Back up SMPL-scale
        log_dict = dict()
        for pi in people_infos:
            log_dict[f"_{pi.human_id}/smpl_scale"] = pi.smpl_scale.item()
        wandb.log(log_dict)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    white_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") # for testing
    black_bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    eval_bg = black_bg if dataset.eval_with_black_bg else white_bg

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    save_video_paths = []
    single_person_opt = True if dataset.target_pid >= 0 else False
    
    # Make top views first
    if len(scene.aux_cam_dict) > 0:
        up_vec = np.array([0, -1, 0]) if is_panoptic else np.array([0, 0, -1])
        top_view_cameras = get_top_views_from_camera(scene.aux_cam_dict, up_vec=up_vec)     

        test_cameras = []
        test_cam_names = []
        for cam_name, cams in  scene.aux_cam_dict.items():
            test_cameras.append(cams[0])
            test_cam_names.append(cam_name)

        if not FAST_DEBUG and not arguments.DEBUG_FAST_LOADING:
            top_view_dicts = render_w_torch3d(
                viewpoint_cameras=top_view_cameras,
                people_infos = people_infos,
                train_camera = scene.getTrainCameras()[0],
                test_cameras = test_cameras,
                render_camera_position = True,
                scene_gaussians = scene_gaussians,
                render_bg_as_pc = True,
            )

    
    # For checking whether dataset loadded properly
    if 0 in testing_iterations: # and False:
        # to check inital alignment
        os.makedirs(scene.model_path + "/training", exist_ok=True)
        if test_aux_mv and len(scene.aux_cam_dict) > 0:
            os.makedirs(scene.model_path + "/training/test_mv", exist_ok=True)
            for i, aux_cam_name in enumerate(test_cam_names):
                aux_cams = scene.aux_cam_dict[aux_cam_name]
                top_views = top_view_dicts[i]
                
                render_set(scene.model_path + "/training/test_mv", f"{aux_cam_name}", 0, aux_cams, scene_gaussians, people_infos, pipe, eval_bg, 
                            is_canon=False, calculate_metric=True, top_views=top_views, turn_off_bg=True) 
                               
            render_scene_w_human(None, None, 0, pipe, exp_name="test", n_camera=1, use_original_world_time=True, uniform_sampler=True, people_infos=people_infos, scene=scene, save_path=scene.model_path + "/training")
        else:
            render_scene_w_human(None, None, 0, pipe, exp_name="test", n_camera=1, use_original_world_time=True, uniform_sampler=True, people_infos=people_infos, scene=scene, save_path=scene.model_path + "/training")
        render_set(scene.model_path + "/training", "test", 0, scene.getTrainCameras(), scene_gaussians, people_infos, pipe, white_bg, is_canon=True, single_person_opt=single_person_opt)


    # Make human camera dict in advance
    human_cam_fdict = dict()
    if dataset.target_pid >= 0:
        human_scene = people_infos[0].human_scene
        
        for human_cam in human_scene.getTrainCameras():
            fid = human_cam.colmap_id
            human_cam_fdict[fid] = human_cam
    
    # viewpoint_cam_fdict = dict()
    # for vcam in scene.getTrainCameras():
    #     viewpoint_cam_fdict[vcam.colmap_id] = vcam
        
    # for fid, human_cam in human_cam_fdict.items():
    #     b_x, b_y, b_w, b_h = human_cam.bbox
    #     gt_image = viewpoint_cam_fdict[fid].original_image
    #     gt_image = gt_image[..., b_y:b_y+b_h, b_x:b_x+b_w][None]
    #     height = human_cam.image_height
    #     width = human_cam.image_width
    #     gt_img = torch.nn.functional.interpolate(gt_image, (height, width), mode="bilinear", align_corners=False)        
    #     gt_img = gt_img[0]
    #     gt_img = tensor2cv(gt_img)
    #     gt_mask = tensor2cv(human_cam.gt_alpha_mask)
    #     occ_mask = tensor2cv(human_cam.occ_mask)
    #     save_img = np.concatenate([gt_img, gt_mask, occ_mask], axis=1)
    #     cv2.imwrite(f"test_debug_{fid:05}.jpg", save_img)
    
    # print(viewpoint_cam_fdict.keys())
    # print(human_cam_fdict.keys())
    # print(people_infos[0].fids)
    
    # import ipdb
    # ipdb.set_trace()
    
    ## Start Training
    for iteration in range(first_iter, opt.iterations + 1):  
        
        iter_start.record()

        if do_fitting_scene:
            scene_gaussians.update_learning_rate(iteration)
        for pi in people_infos:
            if dataset.use_diffusion_guidance:
                if iteration > dataset.dgm_start_iter:
                    pi.gaussians.update_learning_rate(iteration)
            else:
                pi.gaussians.update_learning_rate(iteration)

            if pi.do_trans_grid:
                pi.grid_optimizer.zero_grad()

            if do_smpl_fitting_local:
                pi.local_pose_optimizer.zero_grad()

            if do_smpl_fitting_global:
                pi.global_pose_optimizer.zero_grad()


        for pi in people_infos:
            if (pi.misc['optimized_step'] % dataset.human_sh_degree_enlarge_interval == 0) and (pi.misc['optimized_step'] > 0):
                pi.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        fid = viewpoint_cam.colmap_id 
        
        # If picked camera doesn't include target person, skip
        if dataset.target_pid >= 0:
            while (fid not in human_cam_fdict):
                if not viewpoint_stack == 0:
                    viewpoint_stack = scene.getTrainCameras().copy()
                
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
                fid = viewpoint_cam.colmap_id 

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            torch.autograd.set_detect_anomaly(True)
        
        if dataset.random_background:
            # invert_bg_color = random() > 0.5
            # background = white_bg if invert_bg_color else black_bg
            background = torch.rand(3).float().cuda()
        

        # 1. Reconstruction Loss
        if opt.lambda_rgb_loss > 0:
            if dataset.target_pid >= 0:
                human_cam = human_cam_fdict[fid]
                assert (human_cam.bbox is not None), f"gt_bbox of fid:{fid} not loaded"
                render_pkg = combined_renderer(human_cam, scene_gaussians, people_infos, pipe, background, scaling_modifier = 1.0, override_color = None, render_only_people=dataset.train_wo_bg)
                b_x, b_y, b_w, b_h = human_cam.bbox
                gt_image = viewpoint_cam.original_image.cuda()
                gt_image = gt_image[..., b_y:b_y+b_h, b_x:b_x+b_w][None]
                height = human_cam.image_height
                width = human_cam.image_width
                gt_image = torch.nn.functional.interpolate(gt_image, (height, width), mode="bilinear", align_corners=False)
                gt_image = gt_image[0]

            else:
                render_pkg = combined_renderer(viewpoint_cam, scene_gaussians, people_infos, pipe, background, scaling_modifier = 1.0, override_color = None, render_only_people=dataset.train_wo_bg)
                gt_image = viewpoint_cam.original_image.cuda()  

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            alpha = render_pkg['mask'].squeeze()

            if CLIP_HIGH_RENDERED_RGB:
                image[image > 1] /= image[image > 1] # (turn off handlgin saturated points)
            
            
            
            # (LOADING GTs)
            # 1.1 (mutli-people case)
            if dataset.target_pid < 0:
                if dataset.train_wo_bg or dataset.use_mask_loss:
                    assert (viewpoint_cam.gt_alpha_mask is not None), f"GT alpha mask of '{str(viewpoint_cam.mask_fname)}' not exists"
                    
                    mask = viewpoint_cam.gt_alpha_mask.cuda()
                    mask = 1 - mask         # TODO: I know it's weird, but for individual, loading foreground is default while for scene_cam loading bg(flipped) is default
                    if dataset.reverse_mask:
                        mask = 1 - mask 
                    
                    if dataset.train_wo_bg:
                        gt_image = gt_image * mask + (1-mask) * torch.ones_like(gt_image) * background[:,None,None]
                    
                if (not scene.occ_cam_dict is None) and (fid in scene.occ_cam_dict):
                    height = gt_image.shape[-2]
                    width = gt_image.shape[-1]

                    occ_mask = scene.occ_cam_dict[fid].cuda()[None, None]
                    occ_mask = torch.nn.functional.interpolate(occ_mask, (height, width), mode="bilinear", align_corners=False)
                    occ_mask = occ_mask[0]

                    image = image * occ_mask
                    gt_image = gt_image * occ_mask

                    if dataset.apply_occ_mask_on_density_loss:
                        alpha = alpha * occ_mask.squeeze()
                
            
            # 1.2 (single-person case)
            else:
                assert (len(people_infos) == 1), f"Loaded people is more than 1. {len(people_infos)}"

                if dataset.train_wo_bg or dataset.use_mask_losss:
                    assert (fid in human_cam_fdict), f"{fid} doesn't exist now"
                    human_cam = human_cam_fdict[fid]

                    # When it's obtained from human_cam, as it doesn't know about resolution, need to resize here.
                    height = gt_image.shape[-2]
                    width = gt_image.shape[-1]
                    
                    mask = human_cam.gt_alpha_mask.cuda()[None]
                    mask = torch.nn.functional.interpolate(mask, (height, width), mode="bilinear", align_corners=False)
                    mask = mask[0]

                    if dataset.reverse_mask:
                        print("do reversing")
                        mask = 1 - mask

                    if dataset.train_wo_bg:
                        gt_image = gt_image * mask + (1-mask) * torch.ones_like(gt_image) * background[:, None,None]


                if not (human_cam.occ_mask is None):
                    height = gt_image.shape[-2]
                    width = gt_image.shape[-1]

                    occ_mask = human_cam.occ_mask.squeeze().cuda()[None, None]
                    occ_mask = 1 - occ_mask  # TODO: I know it's weird, but for individual, loading occluding mask is default while for scene_cam loading unoccluded(flipped) is default
                    occ_mask = torch.nn.functional.interpolate(occ_mask, (height, width), mode="bilinear", align_corners=False)
                    occ_mask = occ_mask[0]

                    image = image * occ_mask
                    gt_image = gt_image * occ_mask

                    if dataset.apply_occ_mask_on_density_loss:
                        alpha = alpha * occ_mask.squeeze()

            
            if SAVE_LOSS_IMGS:
                if iteration % SAVE_LOSS_ITER == 0:
                    from utils.image_utils import img_add_text
                    os.makedirs(scene.model_path + "/training", exist_ok=True)
                    os.makedirs(scene.model_path + "/training/loss_imgs", exist_ok=True)
                    save_gt_image = tensor2cv(gt_image.clone().detach())
                    # save_gt_image = img_add_text(save_gt_image, f"gt: {viewpoint_cam.colmap_id}, fid: {fid}")
                    save_image = tensor2cv(image.clone().detach())
                    # save_image = img_add_text(save_image, f"human cam: {human_cam.colmap_id}")
                    save_img = np.concatenate([save_gt_image, save_image], axis=1)
                    cv2.imwrite(str(scene.model_path + f"/training/loss_imgs/{iteration:09}.jpg"), save_img)


            ######            Get img_loss            ######
            # Calculate L1 loss
            Ll1 = l1_loss(image, gt_image)
            # Calculate SSIM loss
            ssim_loss = (1.0 - ssim(image, gt_image))
            # Calculate LPIPS loss
            if dataset.use_lpips_loss:
                lpips_loss = lpips(image[None], gt_image[None]).mean()
            else:
                lpips_loss = 0
                
            # (Optional) if optimized with NR grid
            grid_trans_reg_loss = 0
            if pi.do_trans_grid:
                for pi in people_infos:
                    if fid not in pi.fids:
                        continue
                    _data_idx = pi.fids.index(fid)
                    grid_trans_reg_loss += (pi.smpl_deformer.last_trans[-1] ** 2).mean()
                    grid_trans_reg_loss = grid_trans_reg_loss
                     
            # (Optional) if truned on caluclate density regulairze loss
            density_reg = 0
            if dataset.use_density_reg_loss:
                density_reg = denisty_reg_loss(alpha)
                
            # (Optional) Calculate Mask loss
            mask_loss = 0
            if dataset.use_mask_loss:
                mask_loss = l1_loss(alpha.squeeze(), mask.squeeze())
            ######    End of loss calculation    ######
            

            loss = (1.0 - opt.lambda_dssim) * Ll1 \
                    + opt.lambda_dssim * ssim_loss \
                    + opt.lambda_mask * mask_loss \
                    + opt.lambda_trans_reg * grid_trans_reg_loss \
                    + opt.lambda_lpips * lpips_loss
                    
            # Get image_loss scaler
            lambda_rgb = opt.lambda_rgb_loss
            
            rgb_scaler = 1.
            if dataset.use_diffusion_guidance:
                # Apply strong recon loss ONLY after optimizing with DGM
                if (not (iteration > dataset.dgm_start_iter)) or (iteration % dataset.apply_dgm_every_n != 0):
                    rgb_scaler = 1 / opt.lambda_rgb_loss
                elif dataset.use_adaptive_rgb_loss:
                    dgm_step, max_noise_ratio = DGM.get_noise_level()
                    rgb_scaler = max_noise_ratio ** 2
                lambda_rgb = lambda_rgb * rgb_scaler
                
            loss = loss * lambda_rgb
            loss = loss + opt.lambda_density_reg * density_reg 
                    
            # Do logging
            if pipe.use_wandb:
                log_dict = {
                    "loss/step": iteration,
                    "loss/ssim_loss": lambda_rgb * opt.lambda_dssim * ssim_loss.detach(),
                    "loss/rgb_loss": lambda_rgb * (1.0 - opt.lambda_dssim) * Ll1.detach(),
                    "loss/lpips_loss": lambda_rgb * opt.lambda_lpips * (lpips_loss.detach() if isinstance(lpips_loss, torch.Tensor) else float(lpips_loss)),
                    "loss/mask_loss": lambda_rgb * opt.lambda_mask * (mask_loss.detach() if isinstance(mask_loss, torch.Tensor) else float(mask_loss)),
                    "loss/grid_reg_loss": lambda_rgb * opt.lambda_trans_reg * (grid_trans_reg_loss.detach() if isinstance(grid_trans_reg_loss, torch.Tensor) else float(grid_trans_reg_loss)),
                    "loss/density_reg_loss": lambda_rgb * opt.lambda_density_reg * (density_reg.detach() if isinstance(density_reg, torch.Tensor) else float(density_reg)),
                }
                if dataset.use_diffusion_guidance:
                    log_dict["infos/lambda_rgb"] = lambda_rgb
                    log_dict["loss/raw_ssim_loss"] = opt.lambda_dssim * ssim_loss.detach() 
                    log_dict["loss/raw_rgb_loss"] = (1.0 - opt.lambda_dssim) * Ll1.detach()
                    log_dict["loss/raw_lpips_loss"] = opt.lambda_lpips *  (lpips_loss.detach() if isinstance(lpips_loss, torch.Tensor) else float(lpips_loss))
                    log_dict["loss/raw_mask_loss"] = opt.lambda_mask * (mask_loss.detach() if isinstance(mask_loss, torch.Tensor) else float(mask_loss)),
                    log_dict["loss/raw_grid_reg_loss"] = opt.lambda_trans_reg * (grid_trans_reg_loss.detach() if isinstance(grid_trans_reg_loss, torch.Tensor) else float(grid_trans_reg_loss)),
                    log_dict["loss/raw_density_reg_loss"] = opt.lambda_density_reg * (density_reg.detach() if isinstance(density_reg, torch.Tensor) else float(density_reg)),

                # Also get train metrics together
                with torch.no_grad():
                    try:
                        res_dict = evaluator(
                            rgb = image.detach().unsqueeze(0), 
                            rgb_gt = gt_image.unsqueeze(0)
                        )
                        for k, v in res_dict.items():
                            log_dict[f"metric/train_{k}"] = v.detach().cpu()  
                    except:
                        print("Error calculating evaluation")
        else:
            loss = 0
            Ll1 = 0
            if pipe.use_wandb:
                log_dict = {
                    "loss/step": iteration,
                }
        
        
        # 1.5 Novel View Density Loss (Without SDS case)
        if (not dataset.use_diffusion_guidance) and dataset.use_novel_view_density_reg:
            density_reg_loss_weight = opt.lambda_density_reg
            density_reg_loss_weight = density_reg_loss_weight * lambda_rgb           # reduce the weightings
            reg_losses, dglog_dict = diffusion_renderer(
                                                        DGM, 
                                                        viewpoint_cam, 
                                                        scene_gaussians, 
                                                        people_infos, 
                                                        pipe, 
                                                        background, 
                                                        scaling_modifier=1.0, 
                                                        override_color=None, 
                                                        iteration=iteration, 
                                                        do_optim=False, 
                                                        dgm_loss_weight=opt.lambda_dg_loss, 
                                                        cd_loss_weight=opt.lambda_cd_loss, 
                                                        canon_loss_mode=CANON_LOSS_MODE,
                                                        non_directional_visibility=dataset.dgm_hard_masking,
                                                        ddim_num_step_inferences=dataset.dgm_multi_step_ddim_step,
                                                        ddim_fixed_max_time_step=dataset.dgm_use_fixed_max_step,
                                                        minimum_mask_thrs=dataset.dgm_minimum_mask_thrs,
                                                        masking_optimizer=dataset.dgm_use_optimizer_masking,
                                                        cfg_rescale_weight=dataset.dgm_cfg_rescale_weight,
                                                        density_reg_loss_weight=density_reg_loss_weight,
                                                        grid_trans_reg_loss_weight=opt.lambda_trans_reg,
                                                        only_density_reg_loss=True
                                                        )
            
            loss += reg_losses.squeeze()
            
            for human_id, dg_loss in dglog_dict.items():
                log_dict[f"_{int(human_id):03}/dg_iter"] = DGM.step[human_id]
                for k, v in dg_loss.items():
                    if f"_{int(human_id):03}/{k}" in log_dict:
                        log_dict[f"_{int(human_id):03}/{k}"] += v
                    else:
                        log_dict[f"_{int(human_id):03}/{k}"] = v 
                
                
        if pipe.use_wandb and do_fitting_scene:
            log_dict[f"infos/scene_n_gaussians"] = scene_gaussians.get_n_points

        if (loss.isnan() + loss.isinf()) > 0:
            print("Loss is NaN unexpected behavior")
            for k, v in log_dict.items():
                print(f"log_dict: {k} | {v}")
            assert(0)
            


        # Dream Gaussian loss
        if loss > 0:
            loss.backward()
            
            if iteration > dataset.iter_smpl_densify and (iteration < dataset.iter_densify_smpl_until):
                with torch.no_grad():
                    densify_prune_people_infos(
                                            fid=fid, 
                                            people_infos=people_infos, 
                                            opt=human_train_opt, 
                                            scene_extent=scene.cameras_extent, 
                                            visibility_filter=visibility_filter, 
                                            radii=radii, 
                                            viewspace_point_tensor=viewspace_point_tensor,
                                            scene=scene, 
                                            scene_gaussians=scene_gaussians,
                                            pipe=pipe,
                                            )
                    
            elif iteration > dataset.iter_smpl_densify and (iteration < dataset.iter_prune_smpl_until):
                with torch.no_grad():
                    prune_points_people_infos(
                                            people_infos=people_infos, 
                                            opt=human_train_opt, 
                                            scene_extent=scene.cameras_extent
                                            )
                
                    
            # Optimize it before running DGM (it has probability of destroying the outputs)
            if do_fitting_scene:
                scene_gaussians.optimizer.step()
                scene_gaussians.optimizer.zero_grad(set_to_none = True)

                n_nan = scene_gaussians.prune_infnan_points()
                if n_nan > 0:
                    if not f"infos/scene_gaussians_n_nan" in log_dict:
                        log_dict[f"infos/scene_gaussians_n_nan"] = n_nan
                    else:
                        log_dict[f"infos/scene_gaussians_n_nan"] += n_nan


            for pi in people_infos:
                if fid not in pi.fids:
                    continue
                _data_idx = pi.fids.index(fid)

                pi.gaussians.optimizer.step()
                pi.gaussians.optimizer.zero_grad(set_to_none = True)
                pi.misc['optimized_step'] += 1

                if pi.do_trans_grid:
                    pi.grid_optimizer.step()
                    pi.grid_optimizer.zero_grad()
                    pi.smpl_deformer.last_trans = []

                if do_smpl_fitting_local:
                    pi.local_pose_optimizer.step()
                    pi.local_pose_optimizer.zero_grad()

                if do_smpl_fitting_global:
                    pi.global_pose_optimizer.step()
                    pi.global_pose_optimizer.zero_grad()
                    
                lbs_grid_offset = None
                lbs_grid_scale = None
                if hasattr(pi.smpl_deformer , "lbs_voxel_final"):
                    lbs_grid_offset = pi.smpl_deformer.offset.clone().reshape(1, -1)
                    lbs_grid_scale = pi.smpl_deformer.scale.clone().reshape(1, -1)
                n_nan = pi.gaussians.prune_infnan_points(offset=lbs_grid_offset, scale=lbs_grid_scale)

                if n_nan > 0:
                    if not f"_{int(pi.human_id):03}/n_nan" in log_dict:
                        log_dict[f"_{int(pi.human_id):03}/n_nan"] = n_nan
                    else:
                        log_dict[f"_{int(pi.human_id):03}/n_nan"] += n_nan
            loss = loss.detach()
                

        ### 2. Do Diffusion Guidance
        if dataset.use_diffusion_guidance and (iteration % dataset.apply_dgm_every_n == 0) and (iteration > dataset.dgm_start_iter):
            density_reg_loss_weight = 0 if not dataset.use_density_reg_loss else opt.lambda_density_reg
            density_reg_loss_weight = density_reg_loss_weight * lambda_rgb  
            dg_losses, dglog_dict = diffusion_renderer(
                                                        DGM, 
                                                        viewpoint_cam, 
                                                        scene_gaussians, 
                                                        people_infos, 
                                                        pipe, 
                                                        background, 
                                                        scaling_modifier=1.0, 
                                                        override_color=None, 
                                                        iteration=iteration, 
                                                        do_optim=True, 
                                                        dgm_loss_weight=opt.lambda_dg_loss, 
                                                        cd_loss_weight=opt.lambda_cd_loss, 
                                                        canon_loss_mode=CANON_LOSS_MODE,
                                                        non_directional_visibility=dataset.dgm_hard_masking,
                                                        ddim_num_step_inferences=dataset.dgm_multi_step_ddim_step,
                                                        ddim_fixed_max_time_step=dataset.dgm_use_fixed_max_step,
                                                        minimum_mask_thrs=dataset.dgm_minimum_mask_thrs,
                                                        masking_optimizer=dataset.dgm_use_optimizer_masking,
                                                        cfg_rescale_weight=dataset.dgm_cfg_rescale_weight,
                                                        density_reg_loss_weight=density_reg_loss_weight,
                                                        grid_trans_reg_loss_weight=opt.lambda_trans_reg * lambda_rgb  
                                                        )
            loss += dg_losses.detach().squeeze() if isinstance(dg_losses, torch.Tensor) else dg_losses      # Just for logging

            if pipe.use_wandb:
                log_dict["loss/loss_w_guide"] = loss.detach()
                
                dg_losses = 0
                cd_losses = 0
                for human_id, dg_loss in dglog_dict.items():
                    log_dict[f"dg_loss/{human_id}_dgloss"] = dg_loss['dg_loss'].detach()
                    log_dict[f"_{int(human_id):03}/dg_iter"] = DGM.step[human_id]
                    log_dict[f"_{int(human_id):03}/dg_lambda"] = DGM.dg_lambda[human_id] * opt.lambda_dg_loss

                    dg_losses += dg_loss['dg_loss'].detach().cpu()
                    if 'cd_loss' in dg_loss:
                        log_dict[f"_{int(human_id):03}/cd_lambda"] = DGM.cd_lambda[human_id] * opt.lambda_cd_loss
                        cd_losses += dg_loss['cd_loss'].detach().cpu()

                    for k, v in dg_loss.items():
                        if f"_{int(human_id):03}/{k}" in log_dict:
                            log_dict[f"_{int(human_id):03}/{k}"] += v
                        else:
                            log_dict[f"_{int(human_id):03}/{k}"] = v
                log_dict["loss/controlnet_guide"] = dg_losses
                log_dict["loss/color-consistency"] = cd_losses
        
        log_dict["loss/tot_loss"] = loss.detach() if isinstance(loss, torch.Tensor) else loss
                
    
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * (loss.item() if isinstance(loss, torch.Tensor) else loss) + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                n_gauss = scene_gaussians.get_n_points
                for pi in people_infos:
                    n_gauss += pi.gaussians.get_n_points
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}, N_gauss: {n_gauss}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                # Need to implement saving algorithm here  (torgeth with loading algorithm)
                scene.save(iteration, smpl_params=None, deformer=None, people_infos=people_infos)


            # Save checkpoints
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                # Need to implement saving algorithm here  (torgeth with loading algorithm)
                print("[WARNING] This part is incompleted yet")
                torch.save((scene_gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Do rendering
            if (iteration in testing_iterations):
                os.makedirs(scene.model_path + "/training", exist_ok=True)
                
                if test_aux_mv and len(scene.aux_cam_dict) > 0:
                    os.makedirs(scene.model_path + "/training/test_mv", exist_ok=True)

                    for i, aux_cam_name in enumerate(test_cam_names):
                        aux_cams = scene.aux_cam_dict[aux_cam_name]
                        top_views = top_view_dicts[i]
                        render_set(scene.model_path + "/training/test_mv", f"{aux_cam_name}", iteration, aux_cams, scene_gaussians, people_infos, pipe, eval_bg, 
                                   is_canon=False, calculate_metric=True, top_views=top_views, turn_off_bg=True) 
                    render_scene_w_human(None, None, iteration, pipe, exp_name="test", n_camera=1, use_original_world_time=True, uniform_sampler=True, people_infos=people_infos, scene=scene, save_path=scene.model_path + "/training")
                else:
                    render_scene_w_human(None, None, iteration, pipe, exp_name="test", n_camera=1, use_original_world_time=True, uniform_sampler=True, people_infos=people_infos, scene=scene, save_path=scene.model_path + "/training")
                    
                render_set(scene.model_path + "/training", "test", iteration, scene.getTrainCameras(), scene_gaussians, people_infos, pipe, white_bg, is_canon=True, single_person_opt=single_person_opt)       # turn off is_ablate here

            if iteration % RENDER_LOG_ITER == 0:
                # traing_log steps
                os.makedirs(scene.model_path + "/training", exist_ok=True)
                save_idx = iteration // RENDER_LOG_ITER
                save_video_paths = render_optim_logs(save_idx, iteration, pipe, n_camera=4, n_person=6, n_frame=1, people_infos=people_infos, scene_gaussians=scene_gaussians, save_path=scene.model_path + "/training")

            if iteration % HUMAN_MOTION_ITERS == 0: # or iteration == first_iter:
                os.makedirs(scene.model_path + "/training", exist_ok=True)
                save_path = scene.model_path + "/training"
                render_human_motions(iteration, pipe, people_infos, save_path)
        
        
        # Clip SHs
        with torch.no_grad():
            if dataset.iter_clip_person_shs > 0:
                if iteration % dataset.iter_clip_person_shs:
                    for pi in people_infos:
                        n_clipped = pi.gaussians.clip_invalid_shs()
                        if pipe.use_wandb:
                            human_id = pi.human_id
                            log_dict[f"_{int(human_id):03}/n_clipped"] = n_clipped

            # fix SMPL initial vertices
            if iteration < dataset.iter_fix_smpl_init_verts:
                for pi in people_infos:
                    pi.gaussians.fix_smpl_init_position()       # reset gaussians to have initial points
        
        # Append additional informations in logger
        if pipe.use_wandb:
            for pi in people_infos:
                pid = int(pi.human_id)
                p_n_gaussian = pi.gaussians.get_n_points
                person_iteration = pi.misc['optimized_step']
                log_dict[f"_{pid:03}/n_gaussian"] = p_n_gaussian
                log_dict[f"_{pid:03}/step"] = person_iteration
                
                lbs_grid_offset = None
                lbs_grid_scale = None
                if hasattr(pi.smpl_deformer , "lbs_voxel_final"):
                    lbs_grid_offset = pi.smpl_deformer.offset.clone().reshape(1, -1)
                    lbs_grid_scale = pi.smpl_deformer.scale.clone().reshape(1, -1)
                n_nan = pi.gaussians.prune_infnan_points(offset=lbs_grid_offset, scale=lbs_grid_scale)
                
                if n_nan > 0:
                    if not f"_{int(pi.human_id):03}/n_nan" in log_dict:
                        log_dict[f"_{int(pi.human_id):03}/n_nan"] = n_nan
                    else:
                        log_dict[f"_{int(pi.human_id):03}/n_nan"] += n_nan

                valid_idx = pi.gaussians.denom > 0
                if (valid_idx.sum() > 0) and iteration % 10 == 0:
                    grads = torch.norm(pi.gaussians.xyz_gradient_accum[valid_idx] / pi.gaussians.denom[valid_idx], dim=-1).squeeze()
                    log_dict[f"_{pid:03}/grads_max"] = grads.max()
                    log_dict[f"_{pid:03}/grads_min"] = grads.min()
                    log_dict[f"_{pid:03}/grads_min"] = grads.std()
                    log_dict[f"_{pid:03}/n_valid"] = (valid_idx.sum())
                    log_dict[f"_{pid:03}/valid_ratio"] = (valid_idx.sum()) / p_n_gaussian
                    log_dict[f"_{pid:03}/over_thrs"] = (grads >= human_train_opt.densify_grad_threshold).sum()
                    
        # Do logging
        if pipe.use_wandb:
            wandb.log(log_dict)

    # Make video of optimization visualizer
    gen_videos(save_video_paths, is_jpg=True, fps=30, rm_dir=False)



def densify_prune_people_infos(fid, people_infos: List, opt, scene_extent, visibility_filter, radii, viewspace_point_tensor, scene, scene_gaussians, pipe):
    _idx = 0
    p_id = 0
    
    for pi in people_infos:
        if fid not in pi.fids:
            continue
        
        _gaussians = pi.gaussians
        person_visibility_filter = visibility_filter[_idx:_idx+_gaussians.get_n_points]
        person_radii = radii[_idx:_idx+_gaussians.get_n_points]
        person_viewspace_point_tensor_grad = viewspace_point_tensor[p_id].grad          # [_idx:_idx+pi.gaussians.get_n_points]


        _gaussians.max_radii2D[person_visibility_filter] = torch.max(_gaussians.max_radii2D[person_visibility_filter], person_radii[person_visibility_filter])
        _gaussians.add_densification_stats(None, person_visibility_filter, person_viewspace_point_tensor_grad)
        
        person_iteration = pi.misc['optimized_step']

        _idx += pi.gaussians.get_n_points
        p_id += 1

        if person_iteration > opt.densify_from_iter and person_iteration % opt.densification_interval == 0:
            size_threshold = 20 # if person_iteration > opt.opacity_reset_interval else None

            ####### In fact, it should be 1 / pi.smpl_scale.item(), since we assume unit scale
            ####### Here, we need to apply smpl_scale on gaussians_scale, but can't so divide in extent, which act as thrs
            _gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, 1, size_threshold, delete_large=True)
    
    
def prune_people_infos(people_infos: List, opt):
    for pi in people_infos:
        _gaussians = pi.gaussians
        person_iteration = pi.misc['optimized_step']
        if person_iteration % opt.opacity_reset_interval == 0:
            _gaussians.reset_opacity(only_visible=True)

    

def prune_points_people_infos(people_infos, opt, scene_extent):
    for pi in people_infos:
        person_iteration = pi.misc['optimized_step']
        
        if person_iteration > opt.densify_from_iter and person_iteration % opt.densification_interval == 0:
            # size_threshold = 20
            size_threshold = None
            extent = 5 * pi.smpl_scale.item() #
            pi.gaussians.prune_gaussians(min_opacity=0.005, extent=extent, max_screen_size=size_threshold, min_scale=None)     
        


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--gender', type=str, default='neutral')
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 10_000, 12_000, 15_000])  
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 10_000, 12_000, 15_000])  
    parser.add_argument("--camera_paths", nargs="+", type=str, default=[])
    parser.add_argument("--human_camera_paths", nargs="+", type=str, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--fit_scale', action='store_true', default=False)
    parser.add_argument("--exp_name", type=str, default ="debug")
    parser.add_argument("--human_track_method", type=str, default ="phalp")
    parser.add_argument('--bg_radius', type=float, default=10.)
    parser.add_argument("--use_bg_reg", action='store_true')
    parser.add_argument("--use_zju_dataset", action='store_true')
    parser.add_argument("--train_sample_ratio", type=int, default=1)
    parser.add_argument("--train_aux_sample_ratio", type=int, default=16)
    parser.add_argument("--skip_aux_test", action='store_true', default=False)
    parser.add_argument("--reverse_mask", action='store_true')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    
    arguments.MV_TRAIN_SAMPLE_INTERVAL = args.train_sample_ratio  
    
    if args.train_aux_sample_ratio != 16:
        arguments.MV_AUX_SAMPLE_INTERVAL = args.train_aux_sample_ratio
        print(f"using modifieid aux sampling ratio: {arguments.MV_AUX_SAMPLE_INTERVAL}")
    if args.train_sample_ratio > 1:
        print(f"\n\n\n\n\n\n[WARN!!!!!!!!!!!!!!!!!!!!!!!!!!!!]\n TRAIN SAMPLE RATIO: {args.train_sample_ratio} now\n\n\n")
        
    
    if args.exp_name[:5] == 'debug':
        print("[DEBUG] it's debug mode now!")
        arguments.DEBUG_FAST_LOADING = True
        if arguments.DEBUG_FAST_LOADING:
            print("\n\n\n\n\n\n[WARN!!!!!!!!!!!!!!!!!!!!!!!!!!!!]\n Debugging Option is ON now\n\n\n")
            arguments.MV_TRAIN_SAMPLE_INTERVAL = 64
            arguments.MV_AUX_SAMPLE_INTERVAL = 256
        


    # NEED TO UPDATE BETA from DATASET
    lp_extracted = lp.extract(args)
    # Add human camera paths
    human_camera_paths = args.human_camera_paths
    lp_extracted.human_camera_paths = human_camera_paths
    lp_extracted.human_track_method = args.human_track_method
    lp_extracted.bg_radius = args.bg_radius
    lp_extracted.use_bg_reg = args.use_bg_reg
    lp_extracted.reverse_mask = args.reverse_mask
    

    # Just for realiging up-vectors
    is_panoptic = True
    if args.use_zju_dataset:
        is_panoptic = False


    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp_extracted, op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.fit_scale, args.exp_name, args.camera_paths, test_aux_mv=(not args.skip_aux_test), is_panoptic=is_panoptic)

    # All done
    print("\nTraining complete.")
