
import importlib
numpy = importlib.import_module('numpy')
numpy.float = numpy.float32
numpy.int = numpy.int32
numpy.bool = numpy.bool_
numpy.unicode = numpy.unicode_
numpy.complex = numpy.complex_
numpy.object = numpy.object_
numpy.str = numpy.dtype.str

import torchvision
import os
import json
import sys
import cv2
from typing import List, Union, NamedTuple, Any, Optional, Dict
from random import randint, random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import shutil
import torch
import pandas
from tqdm import tqdm, trange
from omegaconf import OmegaConf


from gtu.renderer.gaussian_renderer import render, render_w_smpl, combined_renderer, diffusion_renderer
from gtu.renderer.renderer_wrapper import render_set, render_traj, project_points_to_cam, render_scene_w_human, render_mv_human_in_scene, render_optim_logs, render_visibility_log, render_human_motions, evaluator
from gtu.renderer.torch3d_renderer import render_w_torch3d


from gtu import arguments
from gtu.dataset.dataloader import load_scene_human
from gtu.arguments import ModelParams, PipelineParams, OptimizationParams, HumanOptimizationParams

from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim, depth_loss_dpt
from utils.image_utils import psnr, gen_videos, tensor2cv, get_crop_img
from gtu.dataset.camera_utils import gen_canon_cam


SMPL_VIEW_DIR_REG = False
if SMPL_VIEW_DIR_REG:
    print("\n\nview-direction regularizer is on now!!! be aware!!!\n\n")

def get_index_in_r(x, y, r=5):
    ind_lists = []

    for i in range(r*2+1):
        for j in range(r*2+1):
            x_ = x - i
            y_ = y - j
            if i**2 + j**2 <= r**2:
                ind_lists.append([x_, y_])
    return ind_lists




def testing(
        dataset, 
        opt, 
        pipe, 
        testing_iterations, 
        saving_iterations, 
        checkpoint_iterations, 
        checkpoint, 
        debug_from,
        fit_scale=False,
        exp_name='default', 
        novel_view_traj=[], 
        test_aux_mv: bool=True, 
        is_panoptic: bool=True, 
        get_indiv_metric: bool=False, 
        render_zoomed_face: bool=False, 
        dump_gaussians_for_webviewer: bool=False
        ):

    # 0. Some test related settings
    do_fitting_scene = False        # Scene is not optimized
    do_smpl_fitting_local = False    # Whether fit SMPL pose during optimization
    do_smpl_fitting_global = False  # Whether fit SMPL global orientation during optimization 
    loaded_fit_smpl = False
    
    # Human train settings (change settings if needed here)
    human_train_opt = HumanOptimizationParams()
    human_train_opt.view_dir_reg = dataset.smpl_view_dir_reg
    human_train_opt.sh_degree = dataset.human_sh_degree
    

    # 1. Load test datasets
    scene, scene_gaussians, people_infos, render_cameras = \
        load_scene_human(
            dataset=dataset,
            pipe=pipe,
            scene_datasets=None,
            iteration=-1,
            exp_name=exp_name,
            human_tracker_name=None, # option for manual comparison 
            novel_view_traj=novel_view_traj,
            use_diffusion_guidance=False,
            is_train=False,
            fit_scale=fit_scale,
            load_aux_mv=test_aux_mv,
            checkpoint=checkpoint,
            human_train_opt=human_train_opt
        )
    
    iteration = people_infos[0].human_scene.loaded_iter

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    white_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") # for testing
    black_bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    eval_bg = black_bg if dataset.eval_with_black_bg else white_bg


    novel_pose_dir = Path(scene.model_path) / "novel_pose"
    novel_pose_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        sorted_fids = [int(cam.colmap_id) for cam in scene.getTrainCameras()]
        sorted_fids = sorted(list(sorted_fids))

        # 0. prepare the canonical rendering camera
        canon_views = gen_canon_cam(res=1024, f=1200)
        cam_back = canon_views[0]
        cam_front = canon_views[len(canon_views)//2]

        #       0.1 Make path to save

        # 1. Render with given pose
        # POSE_FILE_NAME="extradata/animation/aist_demo.npz"
        POSE_FILE_NAME="extradata/animation/inhee_lab.npz"

        a = np.load(POSE_FILE_NAME, allow_pickle=True)['arr_0'][()]["poses"]     # (320, 72)
        demo_poses = torch.from_numpy(a).float().cuda()

        # demo_pose_dir = novel_pose_dir / "aist_demo"
        demo_pose_dir = novel_pose_dir / "inhee_lab"
        demo_pose_dir.mkdir(exist_ok=True)
        video_dirs = []

        
        for person_info in people_infos:
            person_save_path = demo_pose_dir / f"{person_info.human_id}"
            for save_fid, demo_pose in tqdm(enumerate(demo_poses), desc=f"Rendering Demo Pose of {person_info.human_id}"):
                smpl_param = torch.cat([
                    torch.ones(1, dtype=torch.float32).cuda(),
                    torch.zeros(3, dtype=torch.float32).cuda(),
                    demo_pose,
                    person_info.beta.cuda()
                ], dim=-1)
                smpl_param = smpl_param.unsqueeze(0)

                smpl_param[0, 0] = 1.       # Fix scale as 1
                smpl_param[0, 1:4] *= 0     # remove global translation
                # smpl_param[0, 4:7] *= 0     # remove global rotation 
                
                _w_front_canon = render_w_smpl(cam_front, person_info.gaussians, pipe, background, deformer=person_info.smpl_deformer, smpl_param=smpl_param)["render"].detach().cpu()
                _w_back_canon = render_w_smpl(cam_back, person_info.gaussians, pipe, background, deformer=person_info.smpl_deformer, smpl_param=smpl_param)["render"].detach().cpu()

                _w_front_canon[_w_front_canon>1] /= _w_front_canon[_w_front_canon>1]
                _w_back_canon[_w_back_canon>1] /= _w_back_canon[_w_back_canon>1]

                saving = torch.cat([_w_front_canon, _w_back_canon], dim=-1)
                saving = saving.flip(dims=[-2])    # idk why but need to upside-down
                person_save_path.mkdir(exist_ok=True)
                try:
                    torchvision.utils.save_image(saving, os.path.join(str(person_save_path), '{0:05d}'.format(save_fid) + ".jpg"))
                except:
                    print(f"failed to save {os.path.join(str(person_save_path), '{0:05d}'.format(save_fid) + '.jpg')}")

            video_dirs.append(person_save_path)
        gen_videos(video_dirs, is_jpg=True, fps=10, rm_dir=False)


        # 2. Render with original pose
        demo_pose_dir = novel_pose_dir / "original_pose"
        demo_pose_dir.mkdir(exist_ok=True)
        video_dirs = []

        for person_info in people_infos:
            person_save_path = demo_pose_dir / f"{person_info.human_id}"
        
            person_cams = person_info.human_scene.getTrainCameras()
            sorted_pairs = sorted(enumerate(person_info.fids), key=lambda x: x[1])
            # Extracting the indices from the sorted pairs
            sorted_indices = [index for index, value in sorted_pairs]

            for save_fid, idx in tqdm(enumerate(sorted_indices), desc=f"Rendering Train Pose of {person_info.human_id}"):
                smpl_param = torch.cat([
                    torch.ones(1, dtype=torch.float32).cuda(),
                    torch.zeros(3, dtype=torch.float32).cuda(),
                    person_info.smpl_local_poses[idx],
                    person_info.beta.cuda(),
                ], dim=-1)
                smpl_param = smpl_param.unsqueeze(0)

                smpl_param[0, 0] = 1.       # Fix scale as 1
                smpl_param[0, 1:4] *= 0     # remove global translation

                _w_front_canon = render_w_smpl(cam_front, person_info.gaussians, pipe, background, deformer=person_info.smpl_deformer, smpl_param=smpl_param)["render"].detach().cpu()
                _w_back_canon = render_w_smpl(cam_back, person_info.gaussians, pipe, background, deformer=person_info.smpl_deformer, smpl_param=smpl_param)["render"].detach().cpu()

                _w_front_canon[_w_front_canon>1] /= _w_front_canon[_w_front_canon>1]
                _w_back_canon[_w_back_canon>1] /= _w_back_canon[_w_back_canon>1]

                saving = torch.cat([_w_front_canon, _w_back_canon], dim=-1)
                saving = saving.flip(dims=[-2])    # idk why but need to upside-down
                person_save_path.mkdir(exist_ok=True)
                try:
                    torchvision.utils.save_image(saving, os.path.join(str(person_save_path), '{0:05d}'.format(save_fid) + ".jpg"))
                except:
                    print(f"failed to save {os.path.join(str(person_save_path), '{0:05d}'.format(save_fid) + '.jpg')}")

            video_dirs.append(person_save_path)
        gen_videos(video_dirs, is_jpg=True, fps=10, rm_dir=False)



def densify_prune_people_infos(fid, people_infos: List, opt, scene_extent, visibility_filter, radii, viewspace_point_tensor):
    _idx = 0
    p_id = 0
    for pi in people_infos:
        if fid not in pi.fids:
            continue
        
        _gaussians = pi.gaussians
        person_visibility_filter = visibility_filter[_idx:_idx+pi.gaussians.get_n_points]
        person_radii = radii[_idx:_idx+pi.gaussians.get_n_points]
        person_viewspace_point_tensor_grad = viewspace_point_tensor[p_id].grad          # [_idx:_idx+pi.gaussians.get_n_points]


        _gaussians.max_radii2D[person_visibility_filter] = torch.max(_gaussians.max_radii2D[person_visibility_filter], person_radii[person_visibility_filter])
        _gaussians.add_densification_stats(None, person_visibility_filter, person_viewspace_point_tensor_grad)
        
        person_iteration = pi.misc['optimized_step']

        if person_iteration > opt.densify_from_iter and person_iteration % opt.densification_interval == 0:
            size_threshold = 20 if person_iteration > opt.opacity_reset_interval else None
            size_threshold = size_threshold * pi.smpl_scale.item() if size_threshold is not None else None


            ####### In fact, it should be 1 / pi.smpl_scale.item(), since we assume unit scale
            ####### Here, we need to apply smpl_scale on gaussians_scale, but can't so divide in extent, which act as thrs
            _gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene_extent * pi.smpl_scale.item(), size_threshold)
            
            # pi.misc['optimized_step'] = 0
        
        if person_iteration % opt.opacity_reset_interval == 0 or (person_iteration == opt.densify_from_iter):
            _gaussians.reset_opacity()

        _idx += pi.gaussians.get_n_points
        p_id += 1



def visibility_check(cameras, scene_gaussians, people_infos, pipe, save_dir, check_viz=False):
    from utils.jnts_utils import extract_square_bbox
    white_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") # for testing
    black_bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    for pi in people_infos:
        pi.gaussians.reset_visibility()

    if check_viz:
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        save_video_paths = []

    for viewpoint_cam in tqdm(cameras, desc="Checking Visibility though ALL cameras"):
        # reset grad(0)
        scene_gaussians.optimizer.zero_grad(set_to_none = True)
        for pi in people_infos:
            pi.gaussians.optimizer.zero_grad(set_to_none = True)
            if pi.do_trans_grid:
                pi.grid_optimizer.zero_grad()

            if pi.local_pose_optimizer is not None:
                pi.local_pose_optimizer.zero_grad()

            if pi.global_pose_optimizer is not None:
                pi.global_pose_optimizer.zero_grad()


        fid = viewpoint_cam.colmap_id
        HARD_RENDERING = True
        
        render_pkg = combined_renderer(viewpoint_cam, scene_gaussians, people_infos, pipe, black_bg, scaling_modifier = 1.0, override_color = white_bg, hard_rendering=HARD_RENDERING, get_deformed_points=True)
        image, visibility_filter, xyzs, viewspace_point_tensor = render_pkg["render"], render_pkg["visibility_filter"], render_pkg["means3D"], render_pkg["viewspace_points"]

        loss = image.sum()
        loss.backward()

        # Check visibility filter for each person
        _idx = 0
        p_id = 0
        cam_mask_dict = dict()
        cam_vdir_dict = dict()
        input_img_dict = dict()
        for pi in people_infos:
            if fid not in pi.fids:
                continue
            _data_idx = pi.fids.index(fid)
            
            _gaussians = pi.gaussians
            visible_mask = visibility_filter[_idx:_idx+pi.gaussians.get_n_points]

            # Get view-direction in deformed space
            grads = viewspace_point_tensor[p_id].grad   
            grads = grads.sum(-1)
            visible_mask[grads == 0] = False
            if visible_mask.sum() == 0:
                continue


            xyz = xyzs[_idx:_idx+pi.gaussians.get_n_points][visible_mask]
            dir_pp = (xyz - viewpoint_cam.camera_center.repeat(len(xyz), 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)


            # 
            with torch.no_grad():
                beta = pi.beta
                smpl_param = torch.cat([
                    pi.smpl_scale.reshape(-1),
                    pi.smpl_global_poses[_data_idx],
                    pi.smpl_local_poses[_data_idx],
                    beta
                ], dim=-1)
                smpl_param = smpl_param.unsqueeze(0)
            
                
                gt_image = tensor2cv(viewpoint_cam.original_image.detach())
                smpl_output = pi.smpl_deformer.smpl_server(smpl_param)
                smpl_jnts = smpl_output['smpl_jnts'].detach().cpu()
                # print(smpl_jnts.shape)
                pj_jnts = project_points_to_cam(viewpoint_cam, smpl_jnts.squeeze().numpy())
                bbox = extract_square_bbox(pj_jnts, offset_ratio=0.3, get_square=True)
                cropped_img = get_crop_img(gt_image, bbox, rescale=1, resize=512)
            
            

            # Prepare smpl_param
            canon_points = _gaussians.get_xyz
            beta = pi.beta
            if hasattr(pi, 'smpl_params'):
                smpl_param = pi.smpl_params[_data_idx]
                smpl_param[-10:] = beta

                smpl_param = smpl_param.to(canon_points.device).float()
                smpl_param = smpl_param.unsqueeze(0)
            else:
                smpl_param = torch.cat([
                    pi.smpl_scale.reshape(-1),
                    pi.smpl_global_poses[_data_idx],
                    pi.smpl_local_poses[_data_idx],
                    beta
                ], dim=-1)
                smpl_param = smpl_param.unsqueeze(0)

            # Get rotations
            R = pi.smpl_deformer.get_rotations(canon_points, smpl_param, cond=None)     # (N, 3, 3)
            valid_R = R[visible_mask]
            valid_R = valid_R.permute(0, 2, 1)      # inverse the rotation

            # Need to convert back to original space
            canon_dir_pp = torch.einsum('bij,bj->bi', valid_R, dir_pp_normalized)

            # Now finally update visibility 
            canon_dir_pp = canon_dir_pp.detach().cpu()
            pi.gaussians.update_visibility(visible_mask, canon_dir_pp)

            
            cam_mask_dict[pi.human_id] = visible_mask
            cam_vdir_dict[pi.human_id] = canon_dir_pp
            input_img_dict[pi.human_id] = cropped_img

            _idx += pi.gaussians.get_n_points
            p_id += 1
            
        if check_viz and len(cam_mask_dict)>0:
            cam_save_dir = save_dir / f"{int(fid):05}"
            render_visibility_log(cam_save_dir, viewpoint_cam, people_infos, cam_mask_dict, cam_vdir_dict, input_img_dict)
            save_video_paths.append(cam_save_dir)
    
    if check_viz:
        gen_videos(save_video_paths, is_jpg=True, fps=10, rm_dir=True)

    for pi in people_infos:
        pi.gaussians.update_visibility_postfix()




if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--gender', type=str, default='neutral')
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 15_000, 20_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 15_000, 20_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 4_000, 5_000])  
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2_000, 3_000, 4_000, 5_000])  
    parser.add_argument("--camera_paths", nargs="+", type=str, default=[])
    parser.add_argument("--human_camera_paths", nargs="+", type=str, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--fit_scale', action='store_true', default=False)
    parser.add_argument("--exp_name", type=str, default ="debug")
    parser.add_argument("--human_track_method", type=str, default ="phalp")
    parser.add_argument("--use_zju_dataset", action='store_true')
    parser.add_argument("--not_used_all_frame", action='store_true')
    parser.add_argument("--get_indiv_metric", action='store_true')


    parser.add_argument("--render_zoomed_face", action='store_true')
    parser.add_argument("--dump_gaussians_for_webviewer", action='store_true')


    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Testing " + args.model_path)

    args.dump_gaussians_for_webviewer = True        # Let's save it always


    # NEED TO UPDATE BETA from DATASET
    lp_extracted = lp.extract(args)
    # Add human camera paths
    human_camera_paths = args.human_camera_paths
    lp_extracted.human_camera_paths = human_camera_paths
    lp_extracted.human_track_method = args.human_track_method
    lp_extracted.eval = True
    
    is_panoptic = True
    if args.use_zju_dataset:
        is_panoptic = False
    
    if is_panoptic:
        arguments.EVAL_LOAD_ALL_MV = False

    if not args.not_used_all_frame:
        arguments.MV_TRAIN_SAMPLE_INTERVAL = 1
    else:
        arguments.MV_TRAIN_SAMPLE_INTERVAL = 4


    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    testing(
            lp_extracted, 
            op.extract(args), 
            pp.extract(args), 
            args.test_iterations, 
            args.save_iterations, 
            args.checkpoint_iterations, 
            args.start_checkpoint, 
            args.debug_from, 
            args.fit_scale, 
            args.exp_name, 
            args.camera_paths, 
            is_panoptic=is_panoptic,
            get_indiv_metric=args.get_indiv_metric,
            render_zoomed_face=args.render_zoomed_face,
            dump_gaussians_for_webviewer=args.dump_gaussians_for_webviewer,
        )

    # All done
    print("\nTesting complete.")
