
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
from gtu.dataset.camera_utils import get_top_views_from_camera


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

    # 0. Some training related settings
    do_fitting_scene = False        # Scene is not optimized
    do_smpl_fitting_local = False    # Whether fit SMPL pose during optimization
    do_smpl_fitting_global = False  # Whether fit SMPL global orientation during optimization 
    loaded_fit_smpl = False
    
    # Human train settings (change settings if needed here)
    human_train_opt = HumanOptimizationParams()
    human_train_opt.view_dir_reg = dataset.smpl_view_dir_reg
    human_train_opt.sh_degree = dataset.human_sh_degree
    

    # 1. Load train datasets
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

    with torch.no_grad():
        sorted_fids = [int(cam.colmap_id) for cam in scene.getTrainCameras()]
        sorted_fids = sorted(list(sorted_fids))


        # 1.1 Dump points' LBS weighting & Forward Kinematics (B, 4, 4).
        if dump_gaussians_for_webviewer:
            os.makedirs(scene.model_path + "/web_viewer", exist_ok=True)
            web_viwer_data_dir = Path(scene.model_path) / "web_viewer"

            for pi in tqdm(people_infos, desc="saving web-viewer params"):
                ply_fname = web_viwer_data_dir / f"person_{int(pi.human_id):03}.ply"
                tfs_fname = web_viwer_data_dir / f"person_{int(pi.human_id):03}.json"

                # First extract lbs_weights of each points
                person_pc = pi.gaussians
                mean3D = person_pc.get_xyz
                lbs_weights = pi.smpl_deformer.query_weights(mean3D[None])[0]       # [N_points, J]
                
                if lbs_weights.isnan().sum() > 0:
                    print(f"N nan: {lbs_weights.isnan().sum()}")
                    print(f"Nan Indices: {(lbs_weights.isnan().sum(-1).numpy()>0).nonzeros()}")
                    assert(0)

                # Second store lbs_weights of points into PLY file jointly.
                person_pc.save_ply(ply_fname, lbs_weights=lbs_weights)

                # Extract rotations & save them into json files.
                res_dict = dict()
                for fid in sorted_fids:
                    if fid in pi.fids:
                        _idx = pi.fids.index(fid)

                        # Get SMPL parameters
                        smpl_params = torch.cat([
                            pi.smpl_scale.reshape(-1),
                            pi.smpl_global_poses[_idx],
                            pi.smpl_local_poses[_idx],
                            pi.beta
                        ], dim=-1)
                        smpl_params = smpl_params.unsqueeze(0)

                        # smpl_tfs : [J, 4, 4]
                        smpl_tfs = pi.smpl_deformer.smpl_server(smpl_params, absolute=False)['smpl_tfs'][0]    # if absolute : True, assume canon as theta=0
                        smpl_tfs = smpl_tfs.detach().cpu().numpy().tolist()
                        res_dict[fid] = [
                            pi.smpl_scale.detach().cpu().squeeze().item(),
                            pi.smpl_global_poses[_idx].detach().cpu().squeeze().numpy().tolist(),
                            smpl_tfs
                        ]
                
                # save in json
                with open(tfs_fname, 'w') as f:
                    json.dump(res_dict, f)
    
    
        os.makedirs(scene.model_path + "/testing", exist_ok=True)

        if render_zoomed_face:
            from gtu.guidance import DiffusionGuidance
            from gtu.renderer.gaussian_renderer import render_for_diffusion

            dg_log_dir = Path(scene.model_path) / "testing" / "diffusion_guidance"
            indiv_render_dir = Path(scene.model_path) / "testing" / "indiv_render"
            indiv_render_dir.mkdir(exist_ok=True)

            dgm_opt = OmegaConf.load("gtu/guidance/configs/default.yaml")
            DGM = DiffusionGuidance(
                opt=dgm_opt, 
                log_dir=dg_log_dir, 
                textual_inversion_path=None, 
                textual_inversion_expname=None,
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
                do_guid_sched = False,
                sd_version="1.5",
                use_aux_prompt = True,
                use_view_prompt = True,
                cfg_sched = dataset.dgm_cfg_sched,
            )

            for pi in people_infos:
                # load person infos
                _data_idx = 0
                uid = 0

                person_save_dir = indiv_render_dir / f"{pi.human_id}"
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
                        rendered_output, op_cond, _ = render_for_diffusion(
                            mini_cam = cam,
                            pipe = pipe,
                            person_info = pi,
                            smpl_param  = _smpl_param, 
                            uid = uid,
                            bg_color = eval_bg,
                        )

                        rendered_output = rendered_output.detach().cpu().squeeze().permute(1,2,0).numpy()
                        rendered_output = (rendered_output * 255).astype(np.uint8)   
                        rendered_output = rendered_output[..., ::-1]
                        idx += 1
                        cv2.imwrite(str(cam_save_dir / f'render_{idx:03}.jpg'), rendered_output)
                

                gen_videos(vid_dirs, is_jpg=True, fps=10, rm_dir=False, regex_fname="render_*.jpg", save_tag="_rendering")


        render_set(
            scene.model_path + "/testing/test_mv",
            f"train_cam", 
            iteration, 
            scene.getTrainCameras(), 
            scene_gaussians, 
            people_infos, 
            pipe, 
            eval_bg, 
            is_canon=False, 
            calculate_metric=True, 
            top_views=None, 
            turn_off_bg=True, 
            render_indiv=False, 
            get_indiv_metric=False
        ) 
        exit()
        
        if test_aux_mv and len(scene.aux_cam_dict) > 0:
            os.makedirs(scene.model_path + "/testing/test_mv", exist_ok=True)
            print("[INFO] we assume panoptic dataset here!!!!!!")
            if is_panoptic:
                top_view_cameras = get_top_views_from_camera(scene.aux_cam_dict, up_vec=np.array([0, -1, 0]))     
            else:
                top_view_cameras = get_top_views_from_camera(scene.aux_cam_dict, up_vec=np.array([0, 0, -1]))     

            test_cameras = []
            test_cam_names = []
            for cam_name in sorted(list(scene.aux_cam_dict.keys())):
                cams = scene.aux_cam_dict[cam_name]
                test_cameras.append(cams[0])
                test_cam_names.append(cam_name)
                print(cam_name)
                
            # for_viz = render_w_torch3d(
            #     viewpoint_cameras=top_view_cameras,
            #     people_infos = people_infos,
            #     train_camera = scene.getTrainCameras()[0],
            #     test_cameras = test_cameras[:2],
            #     render_camera_position = True,
            #     scene_gaussians = scene_gaussians,
            #     render_bg_as_pc = False, 
            #     for_viz = True,
            #     skip_text = True,
            # )
            # os.makedirs(scene.model_path + "/testing/test_mv/top_views", exist_ok=True)
            # for k, v in for_viz.items():
            #     save_dir = scene.model_path + "/testing/test_mv/top_views/" + f"{k}"
            #     os.makedirs(save_dir, exist_ok=True)
                
            #     for fid in sorted(list(v.keys())):
            #         cv2.imwrite(os.path.join(save_dir, f"{fid:09}.png"), v[fid])

            # top_view_dicts = render_w_torch3d(
            #     viewpoint_cameras = top_view_cameras,
            #     people_infos = people_infos,
            #     train_camera = scene.getTrainCameras()[0],
            #     test_cameras = test_cameras,
            #     render_camera_position = True,
            #     scene_gaussians = scene_gaussians,
            #     render_bg_as_pc = True, 
            #     skip_text = True,
            # )
            
            # for i, aux_cam_name in enumerate(test_cam_names):
            #     aux_cams = scene.aux_cam_dict[aux_cam_name]
            #     top_views = top_view_dicts[i]
            #     render_set(scene.model_path + "/testing/test_mv", f"{aux_cam_name}", iteration, aux_cams, scene_gaussians, people_infos, pipe, eval_bg, is_canon=False, calculate_metric=True, top_views=top_views, turn_off_bg=True, render_indiv=True, get_indiv_metric=get_indiv_metric) 
            
            top_views = None
            render_set(scene.model_path + "/testing/test_mv", f"train_cam", iteration, scene.getTrainCameras(), scene_gaussians, people_infos, pipe, eval_bg, is_canon=False, calculate_metric=True, top_views=top_views, turn_off_bg=True, render_indiv=True, get_indiv_metric=get_indiv_metric) 
            render_scene_w_human(None, None, iteration, pipe, exp_name="test_mv", n_camera=1, use_original_world_time=True, uniform_sampler=True, people_infos=people_infos, scene=scene, save_path=scene.model_path + "/testing")
        else:
            render_scene_w_human(None, None, iteration, pipe, exp_name="test_mv", n_camera=5, use_original_world_time=True, uniform_sampler=True, people_infos=people_infos, scene=scene, save_path=scene.model_path + "/testing")
        
        render_set(scene.model_path + "/testing", "test", iteration, scene.getTrainCameras(), scene_gaussians, people_infos, pipe, eval_bg, is_canon=True)       # turn off is_ablate here

        save_path = scene.model_path + "/testing"
        render_human_motions(iteration, pipe, people_infos, save_path)


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
