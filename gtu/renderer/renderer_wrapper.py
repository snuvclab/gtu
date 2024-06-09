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


import os
import json
import random
from os import makedirs
from argparse import ArgumentParser
from typing import List, Union, NamedTuple, Optional, Dict
from pathlib import Path


import cv2
import pandas
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


from gtu.dataset.scene import Scene
from gtu.dataset.scene.gaussian_model import GaussianModel
from gtu.dataset.scene.dataset_readers import load_human_model
from gtu.dataset.scene.human_loader import HumanScene
from gtu.renderer.gaussian_renderer import render, render_w_smpl, combined_renderer, render_vis_log
from gtu.smpl_deformer.deformer import SMPLDeformer
from gtu.smpl_deformer.smpl_server import SMPLServer
from gtu.arguments import ModelParams, PipelineParams, RenderParams, get_combined_args


from utils.loss_utils import ssim
from utils.general_utils import safe_state
from gtu.dataset.camera_utils import gen_canon_cam, load_ns_traj_from_json, get_top_view, gen_closeup_views, gen_perturbed_camera
from utils.image_utils import gen_videos, tensor2cv, stitch_outputs, img_add_text, depth2rgb, get_colors, padded_resize, get_error_map
from utils.graphics_utils import project_points_to_cam
from utils.image_utils import psnr



VIEW_ABLATE = [
    torch.tensor([0, 0, 1/4]) * torch.pi,
    torch.tensor([0, 0,-1/4]) * torch.pi,
    torch.tensor([1/6, 0, 0]) * torch.pi,
]

figsize = (2, 3)        # (x-dir, y-dir)


class PersonInfos(NamedTuple):
    uids : Union[List, torch.Tensor]        # for trans_grids
    model_path: Path
    smpl_params: torch.Tensor
    beta: torch.Tensor
    smpl_deformer: SMPLDeformer
    gaussians: GaussianModel
    fids: Union[List, torch.Tensor]
    human_id: str

class PersonTrain(NamedTuple):
    uids : Union[List, torch.Tensor]        # for trans_grids
    fids : Union[List, torch.Tensor]        # to check whether it's a scene requiring human
    model_path: Path
    smpl_local_poses: torch.Tensor
    smpl_global_poses: torch.Tensor
    local_pose_optimizer: Optional[torch.optim.Optimizer]
    global_pose_optimizer: Optional[torch.optim.Optimizer]
    smpl_scale_optimizer: Optional[torch.optim.Optimizer]
    beta: torch.Tensor
    smpl_deformer: SMPLDeformer
    gaussians: GaussianModel
    do_trans_grid: bool
    trans_grids: Optional[torch.Tensor]
    grid_optimizer: Optional[torch.optim.Optimizer]
    view_dir_reg: bool
    human_scene: Scene
    human_id: str
    smpl_scale: torch.Tensor
    init_smpl_jnts: torch.Tensor
    cam_centers: torch.Tensor
    cc_smpl_dir: torch.Tensor    
    representative_img: torch.Tensor
    misc: Dict   




def plot_ray_on_3rd_perspective(original_cam, smpl_jnts: np.ndarray, render_cam, rendered_img, colors=None, line_size=2):
    """
    Draw a line of prejocted jnts, into original cameras
    """
    
    cam_origin = original_cam.camera_center.clone().detach().cpu().squeeze().numpy() # (3-Dim)
    jnt_dir = (smpl_jnts - cam_origin[None])         # init_jnts : (J x 3)  
    jnt_dir = jnt_dir / np.sqrt((jnt_dir ** 2).sum(-1)[...,None])


    pj_jnt_in_infinite = project_points_to_cam(render_cam, smpl_jnts)
    pj_cam_origin = project_points_to_cam(render_cam, cam_origin[None])[0]

    po_1 = int(pj_cam_origin[0])
    po_2 = int(pj_cam_origin[1])


    if colors is None:
        colors = get_colors(len(smpl_jnts))

    for p1, c in zip(pj_jnt_in_infinite, colors):
        # extend in view dir
        _dir = (p1 - pj_cam_origin)
        _dir = _dir / np.sqrt((_dir ** 2).sum())
        p1 = p1 + _dir * 9999


        p11 = int(p1[0])
        p12 = int(p1[1])
        cv2.line(rendered_img, (p11, p12), (po_1, po_2), c.tolist(), line_size)

    return rendered_img


def render_traj(save_dir, scene_gaussians, traj_3d: np.ndarray, pipeline, background, scene_views= None, view=None):
    # load camera if not specified
    if view is None:
        view = get_top_view(scene_views, up_vec= np.array([0, -1, 0]), t_scale=1.5, fov=0.6, z_axis = np.array([0,0,-1]))

    # First get background

    scene_rendering = render(view, scene_gaussians, pipeline, background)["render"].detach().cpu()
    scene_rendering = tensor2cv(scene_rendering)

    # Draw trajectory of 3D points
    w2c = view.world_view_transform.clone().detach().cpu().numpy().T
    projection = view.projection_matrix.clone().detach().cpu().numpy().T

    gl2img = np.array([
        [view.image_width/2, 0, 0, view.image_width/2],
        [0, view.image_height/2, 0, view.image_height/2],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    # gl2cv = np.array([
    #     [1, 0, 0],
    #     [0, -1, view.image_height],
    #     [0, 0, 1]
    # ], dtype=np.float32)

    
    if len(traj_3d.shape) == 4:
        progress_plot = True
    elif len(traj_3d.shape) == 3:
        progress_plot = False
        traj_3d = traj_3d[None]
    else:
        raise NotImplementedError()


    # It's case of holding all trajectory
    # traj_3d (N, T, J, 3)
    
    traj_3d = np.concatenate([traj_3d, np.ones_like(traj_3d[...,0:1])], axis=-1)
    traj_3d = np.einsum('ij, btnj -> btni', w2c, traj_3d)
    projected_points = np.einsum('ij, btnj-> btni', projection, traj_3d)

    # convert into image plane
    projected_points = np.einsum('ij, btnj-> btni', gl2img, projected_points)
    # projected_points = np.einsum('ij, btnj-> btni', gl2cv, projected_points)  
    projected_points = projected_points[...,:2] / projected_points[..., -1:]

    
    # now make images
    if progress_plot:
        print("     Render optimization progress")
        os.makedirs(save_dir, exist_ok=True)

        n_traj = projected_points.shape[1]
        colors = get_colors(n_traj)
        j_idx = 0       # Only plot pelvis

        for img_id, traj in enumerate(projected_points):
            # draw trajectory
            img = scene_rendering.copy()
            prev_j2ds = traj[0]
            for i, j2ds in enumerate(traj):
                c = colors[i]
                p11 = int(prev_j2ds[j_idx][0])       
                p12 = int(prev_j2ds[j_idx][1])
                p21 = int(j2ds[j_idx][0])
                p22 = int(j2ds[j_idx][1])
                cv2.line(img, (p11, p12), (p21, p22), c.tolist(), 2)
                prev_j2ds = j2ds

            cv2.imwrite(os.path.join(save_dir, f"{img_id:05}.jpg"), img)
        video_paths = [save_dir]
        gen_videos(video_paths, is_jpg=True, fps=20, rm_dir=False)

    else:
        print("     Render optimized results (trajectory)")
        os.makedirs(save_dir, exist_ok=True)

        n_last = 10
        n_points = projected_points.shape[2]
        colors = get_colors(n_points)
        
        prev_joints = []
        for img_id, jnts in enumerate(projected_points[0]):
            # plot current joints
            img = scene_rendering.copy()

            for j_idx in range(len(jnts)):
                c = colors[j_idx]
                p21 = int(jnts[j_idx][0])
                p22 = int(jnts[j_idx][1])

                cv2.circle(img, (p21, p22), 4, c.tolist())

            # connect prev joints
            cur_jnt = jnts
            for prev_jnt in prev_joints:
                # draw line
                for j_idx in range(len(jnts)):
                    c = colors[j_idx]
                    p11 = int(prev_jnt[j_idx][0])       
                    p12 = int(prev_jnt[j_idx][1])
                    p21 = int(cur_jnt[j_idx][0])
                    p22 = int(cur_jnt[j_idx][1])
                    
                    cv2.line(img, (p11, p12), (p21, p22), c.tolist(), 2)
                cur_jnt = prev_jnt


            # finally append current joint & remove too many values            
            prev_joints.append(jnts)
            if len(prev_joints) > n_last:
                prev_joints.pop(0)

            cv2.imwrite(os.path.join(save_dir, f"{img_id:05}.jpg"), img)
        video_paths = [save_dir]
        gen_videos(video_paths, is_jpg=True, fps=10, rm_dir=True)
    print("[INFO] finished rendering trajectory optimization")


@torch.no_grad()
def render_mv_human_in_scene(model_path, name, iteration, views, scene_gaussians, people_infos, pipeline, background, every_n_camera=10, render_posed_human_together=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "person_mv")
    bg_background = torch.zeros_like(background)
    img_render_res = 512
    makedirs(render_path, exist_ok=True)
    video_paths = []
    
    
    smpl_server = SMPLServer(use_hands=True, use_feet_keypoints=True)
    
    for pi in people_infos:
        person_path = os.path.join(render_path, str(pi.human_id))
        makedirs(person_path, exist_ok=True)
        pi.misc['person_render_path'] = person_path

    cam_id = 0
    for view in views:
        if cam_id % every_n_camera != 0:
            cam_id +=1
            continue
        cam_id +=1
        
        
        fid = view.colmap_id

        
        # select pis in view
        for pi in people_infos:
            if fid in pi.fids:
                person_path = pi.misc['person_render_path']
                person_render_path = os.path.join(person_path, f"{fid:05}")
                makedirs(person_render_path, exist_ok=True)
                person_save_path = os.path.join(person_path, f"{fid:05}" + "_save")
                makedirs(person_save_path, exist_ok=True)
                
                if render_posed_human_together:
                    person_wsmpl_save_path = os.path.join(person_path, f"{fid:05}" + "_save_wsmpl")
                    makedirs(person_wsmpl_save_path, exist_ok=True)
                    person_only_save_path = os.path.join(person_path, f"{fid:05}" + "_save_osmpl")
                    makedirs(person_only_save_path, exist_ok=True)
                    joints_path = os.path.join(person_path, f"{fid:05}" + "_pj_joints")
                    makedirs(joints_path, exist_ok=True)
                    mask_save_path = os.path.join(person_path, f"{fid:05}" + "_mask")
                    makedirs(mask_save_path, exist_ok=True)
                
                
                # get cropped image
                _idx = pi.fids.index(fid)
                p_input_img = tensor2cv(pi.human_scene.getTrainCameras()[_idx].original_image)
                p_input_img = cv2.resize(p_input_img, (img_render_res, img_render_res))
                
                
                # get SMPL center
                smpl_center = pi.smpl_global_poses.clone().detach().cpu().squeeze().numpy()[_idx]
                
                # get SMPL scale
                try:
                    smpl_scale = pi.smpl_scale[_idx]
                except:
                    smpl_scale = pi.smpl_scale
                smpl_scale = smpl_scale.clone().detach().cpu().squeeze().item()
                
                
                # get cameras
                zoomed_views = gen_closeup_views(views, smpl_center, smpl_scale, n_cameras=16, f=400)
                for idx, _view in enumerate(tqdm(zoomed_views, desc="Rendering zoomed views")):
                    smpl_params = torch.cat([
                        pi.smpl_scale.reshape(-1),
                        pi.smpl_global_poses[_idx],
                        pi.smpl_local_poses[_idx],
                        pi.beta
                    ], dim=-1)
                    smpl_params = smpl_params.unsqueeze(0)
                    
                    
                    # set uid / colmap_id
                    _view.colmap_id = view.colmap_id
                    _view.uid = view.uid

                    raw_rendering_output = render(_view, scene_gaussians, pipeline, background)
                    smpl_rendering = render_w_smpl(_view, pi.gaussians, pipeline, bg_background, deformer=pi.smpl_deformer, smpl_param=smpl_params)["render"].detach().cpu()

                    
                    scene_rendering = raw_rendering_output["render"].detach().cpu()    # background rendering
                    combined_rendering = combined_renderer(_view, scene_gaussians, [pi], pipeline, background, scaling_modifier = 1.0, override_color = None, hard_rendering=False)["render"].detach().cpu()
                    
                    if render_posed_human_together:
                        torchvision.utils.save_image(combined_rendering, os.path.join(person_wsmpl_save_path, '{0:05d}'.format(idx) + ".png"))    
                        torchvision.utils.save_image(smpl_rendering, os.path.join(person_only_save_path, '{0:05d}'.format(idx) + ".png")) 

                        # Also render mask (wo using inhee-depth modified renderer)
                        mask_rendering = render_w_smpl(_view, pi.gaussians, pipeline, bg_background, deformer=pi.smpl_deformer, smpl_param=smpl_params, get_viz_mask=True)["render"].detach().cpu()
                        torchvision.utils.save_image(mask_rendering, os.path.join(mask_save_path, '{0:05d}'.format(idx) + ".png")) 
                           
                    torchvision.utils.save_image(scene_rendering, os.path.join(person_save_path, '{0:05d}'.format(idx) + ".png"))
                    
                    smpl_rendering = tensor2cv(smpl_rendering)
                    scene_rendering = tensor2cv(scene_rendering)
                    combined_rendering = tensor2cv(combined_rendering)
                    
                    if 'depth' in raw_rendering_output:
                        depth_rendering = raw_rendering_output["depth"].detach().cpu()
                        depth_rendering = depth2rgb(depth_rendering.squeeze().numpy())
                    else:
                        depth_rendering = None

                    # Check camera working well
                    smpl_output = smpl_server(smpl_params)
                    smpl_jnts = smpl_output['smpl_jnts'].detach().cpu()
                    # print(smpl_jnts.shape)
                    pj_jnts = project_points_to_cam(_view, smpl_jnts.squeeze().numpy())
                    colors = get_colors(len(pj_jnts))
                    for j_idx in range(len(pj_jnts)):
                        c = colors[j_idx]
                        p21 = int(pj_jnts[j_idx][0])
                        p22 = int(pj_jnts[j_idx][1])

                        try:
                            cv2.circle(smpl_rendering, (p21, p22), 4, c.tolist())
                        except:
                            continue
                        
                    if render_posed_human_together:
                        save_jnts = pj_jnts
                        np.save(os.path.join(joints_path, '{0:05d}'.format(idx) + ".npy"), save_jnts)


                    # Draw line of trajectory
                    combined_rendering_wray = np.copy(combined_rendering)
                    plot_ray_on_3rd_perspective(view, smpl_jnts.squeeze().numpy(), _view, smpl_rendering)
                    plot_ray_on_3rd_perspective(view, smpl_jnts.squeeze().numpy(), _view, combined_rendering_wray, line_size=1)

                    if depth_rendering is not None:
                        saving = np.concatenate([
                            np.concatenate([smpl_rendering, scene_rendering], axis=0),
                            np.concatenate([combined_rendering_wray, p_input_img], axis=0),   
                            np.concatenate([combined_rendering, depth_rendering], axis=0)   
                        ], axis=1)
                    else:
                        saving = np.concatenate([
                            np.concatenate([smpl_rendering, scene_rendering], axis=0),
                            np.concatenate([combined_rendering_wray, p_input_img], axis=0)  
                        ], axis=1)
                        
                    
                    cv2.imwrite(os.path.join(person_render_path, '{0:05d}'.format(idx) + ".jpg"), saving)

                video_paths.append(person_render_path)

        gen_videos(video_paths, is_jpg=True, fps=10, rm_dir=False)
        video_paths = []
    

    print("[INFO] Done rendering scenes!")
    for pi in people_infos:
        del pi.misc['person_render_path']
        



from utils.metric_utils import Evaluator
evaluator = Evaluator().cuda()



@torch.no_grad()
def render_set(
                model_path, 
                name, 
                iteration, 
                views, 
                scene_gaussians, 
                people_infos, 
                pipeline, 
                background, 
                train_cams=None,
                is_canon: bool=False, 
                render_canon_combined: bool=True, 
                calculate_metric: bool=False,
                get_indiv_metric: bool=False,
                top_views= None,
                turn_off_bg: bool=False,
                render_indiv: bool=False,
                single_person_opt: bool=False,
    ):
    render_path = os.path.join(model_path, name)
    os.makedirs(render_path, exist_ok=True)
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    os.makedirs(render_path, exist_ok=True)
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    os.makedirs(render_path, exist_ok=True)
    
    canon_render_res = 512
    
    if not (train_cams is None):
        train_cam_dict = dict()
        for train_cam in train_cams:
            train_cam_dict[train_cam.colmap_id] = train_cam

    if single_person_opt:
        assert len(people_infos) == 1, f"Used single_person_opt option but {len(people_infos)} people found"
        human_cam_fdict = dict()
        human_scene = people_infos[0].human_scene
        for human_cam in human_scene.getTrainCameras():
            fid = human_cam.colmap_id
            human_cam_fdict[fid] = human_cam

    if calculate_metric:
        rgb_path = os.path.join(model_path, name, "ours_{}".format(iteration), "rgbs")
        makedirs(rgb_path, exist_ok=True)
        figure_path = os.path.join(model_path, name, "ours_{}".format(iteration), "for_figs")
        makedirs(figure_path, exist_ok=True)
        gt_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gts")
        makedirs(gt_path, exist_ok=True)
        err_path = os.path.join(model_path, name, "ours_{}".format(iteration), "err")
        makedirs(err_path, exist_ok=True)
        
        scene_res_dict = dict()
        people_res_dict = dict()
        for pi in people_infos:
            people_res_dict[pi.human_id] = dict()
            if get_indiv_metric:
                person_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"{pi.human_id}")
                makedirs(person_path, exist_ok=True)

    # First do combined rendering first
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        fid = view.colmap_id
        
        if not (train_cams is None):
            if fid not in train_cam_dict:
                print(f"[INFO] '{fid}' fid is not in train camera sets")
                continue
        if single_person_opt:
            if fid not in human_cam_fdict:
                continue
            
        # Get GT
        gt = view.get_gt_image()[0:3, :, :].detach().cpu()
        
        combined_rendering = combined_renderer(view, scene_gaussians, people_infos, pipeline, background, scaling_modifier = 1.0, override_color = None, render_only_people=turn_off_bg, hard_rendering=False)["render"].detach().cpu()
        combined_rendering[combined_rendering > 1] /= combined_rendering[combined_rendering > 1]
        
        if train_cams is None:
            people_rendering = combined_renderer(view, scene_gaussians, people_infos, pipeline, background, scaling_modifier = 1.0, override_color = None, render_only_people=turn_off_bg)
            if 'mask' in people_rendering:
                is_alpha_mask = True
                people_rendering_mask = people_rendering["mask"].detach().cpu()
            else:
                is_alpha_mask = False
                people_rendering_mask = ((people_rendering["render"] == background.reshape(-1,1,1)).sum(0) != 3).detach().cpu().float()
            people_rendering_rgb = people_rendering["render"].detach().cpu()
            people_rendering = people_rendering_rgb

            people_rendering[people_rendering > 1] /= people_rendering[people_rendering > 1]
            
            # save images
            if calculate_metric:
                _saving = tensor2cv(people_rendering)
                mask = people_rendering_mask.squeeze().unsqueeze(-1).numpy() * 255
                _saving = np.concatenate([_saving, mask], axis=-1)
                cv2.imwrite(os.path.join(rgb_path, '{0:05d}'.format(idx) + ".png"), _saving)
                cv2.imwrite(os.path.join(rgb_path, '{0:05d}'.format(idx) + ".jpg"), _saving)
            
                if background.sum() == 0:
                    print("")
                    # we want white-bg rendering here
                    white_bg = torch.ones_like(background)
                    fig_rendering = combined_renderer(view, scene_gaussians, people_infos, pipeline, white_bg, scaling_modifier = 1.0, override_color = None, render_only_people=turn_off_bg)
                    if 'mask' in fig_rendering:
                        fig_rendering_mask = fig_rendering["mask"].detach().cpu()
                    else:
                        fig_rendering_mask = ((fig_rendering["render"] == white_bg.reshape(-1,1,1)).sum(0) != 3).detach().cpu().float()
                    fig_rendering_mask = fig_rendering_mask * people_rendering_mask
                    fig_rendering = fig_rendering["render"].detach().cpu()
                    fig_rendering[fig_rendering > 1] /= fig_rendering[fig_rendering > 1]
                    
                    fig_saving = tensor2cv(fig_rendering)
                    fig_mask = fig_rendering_mask.squeeze().unsqueeze(-1).numpy() * 255
                    fig_saving = np.concatenate([fig_saving, fig_mask], axis=-1)
                else:
                    fig_saving = _saving
                        
                cv2.imwrite(os.path.join(figure_path, '{0:05d}'.format(idx) + ".png"), fig_saving)
            
        else:
            people_rendering = train_cam_dict[fid].original_image[0:3, :, :].detach().cpu()[None]
            people_rendering = F.interpolate(people_rendering, (view.image_height, view.image_width), mode="bilinear", align_corners=False)
            people_rendering = people_rendering[0]

        if top_views is None or fid not in top_views:
            scene_rendering = render(view, scene_gaussians, pipeline, background)["render"].detach().cpu()     # background rendering
            scene_rendering = tensor2cv(scene_rendering)
        else:
            scene_rendering = top_views[fid]
        
        saving1 = torch.cat([combined_rendering, gt], dim=-1)
        saving1 = tensor2cv(saving1)
        saving2 = np.concatenate([scene_rendering, tensor2cv(people_rendering)], axis=1)
        
        if single_person_opt:
            human_cam = human_cam_fdict[fid]
            
            height = gt.shape[-2]
            width = gt.shape[-1]
            b_x, b_y, b_w, b_h = human_cam.bbox
            sp_mask = torch.zeros((1, height, width), dtype=torch.float32)
                    
            human_mask = human_cam.gt_alpha_mask.cpu()[None]
            human_mask = torch.nn.functional.interpolate(human_mask, (b_h, b_w), mode="bilinear", align_corners=False)
            human_mask = human_mask[0]
            sp_mask[:,b_y:b_y+b_h, b_x:b_x+b_w] = human_mask
            
            bg = 1 if background.sum() > 0 else 0
            sp_masked_gt = gt * sp_mask + (1-sp_mask) * torch.ones_like(gt) * bg

            if not (human_cam.occ_mask is None):
                sp_occ_mask = torch.zeros((1, height, width), dtype=torch.float32)
                 
                human_occ_mask = human_cam.occ_mask.squeeze().cpu()[None, None]
                human_occ_mask = 1 - human_occ_mask  # TODO: I know it's weird, but for individual, loading occluding mask is default while for scene_cam loading unoccluded(flipped) is default
                human_occ_mask = torch.nn.functional.interpolate(human_occ_mask, (b_h, b_w), mode="bilinear", align_corners=False)
                human_occ_mask = human_occ_mask[0]
                sp_occ_mask[:,b_y:b_y+b_h, b_x:b_x+b_w] = human_occ_mask

                sp_combined_rendering = combined_rendering * sp_occ_mask
                sp_masked_gt = sp_masked_gt * sp_occ_mask
            else:
                sp_combined_rendering = combined_rendering
                
            saving3 = torch.cat([sp_combined_rendering, sp_masked_gt], dim=-1)
            saving3 = tensor2cv(saving3)
                    
            saving = np.concatenate([saving1, saving2, saving3], axis=0)
        else:
            saving = np.concatenate([saving1, saving2], axis=0)
        
        cv2.imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + ".jpg"), saving)


        if calculate_metric:
            if iteration == 0:
                # check BG only metric
                estimated_image = combined_rendering.unsqueeze(0).cuda()
            else:
                estimated_image = people_rendering.unsqueeze(0).cuda()
            fid = view.colmap_id

            masks_gts = []
            # calculate individual metrics + stack masks of individuals
            for pi in people_infos:
                human_id = pi.human_id

                if fid not in pi.fids:
                    people_res_dict[human_id][idx] = None
                    print("1")
                    continue
                
                if not 'aux_cam_mask_fname_dict' in pi.misc:
                    people_res_dict[human_id][idx] = None
                    print("2")
                    continue

                if not name in pi.misc['aux_cam_mask_fname_dict']:      # Here name is same as camera name
                    people_res_dict[human_id][idx] = None
                    print("3")
                    continue
                
                frame_mask_dict = pi.misc['aux_cam_mask_fname_dict'][name]
                if fid not in frame_mask_dict:
                    people_res_dict[human_id][idx] = None
                    print("4")
                    continue

                mask = cv2.imread(str(frame_mask_dict[fid]), 0)
                mask = cv2.resize(mask, (view.image_width, view.image_height))
                gt_mask = (torch.from_numpy(mask) > 0).squeeze().float().cuda()
                masks_gts.append(gt_mask)
                
                    
                if get_indiv_metric:
                    gt_image = view.get_gt_image()[0:3, :, :][None].cuda().clone().detach()
                    gt_image[(gt_mask==0)[None].repeat(3,1,1)[None]] = 0
                    _estimated_image = estimated_image.clone().detach()
                    _estimated_image[(gt_mask==0)[None].repeat(3,1,1)[None]] = 0
                    
                    with torch.no_grad():
                        res_dict = evaluator(
                            rgb = _estimated_image, 
                            rgb_gt = gt_image,
                            )
                    
                    person_saving = torch.cat([estimated_image[0].cpu(), gt_image[0].cpu()], dim=-1)
                    torchvision.utils.save_image(person_saving, os.path.join(model_path, name, "ours_{}".format(iteration), f"{human_id}", '{0:05d}'.format(idx) + ".jpg"))
                    
                    people_res_dict[human_id][idx] = dict()
                    for k, v in res_dict.items():
                        people_res_dict[human_id][idx][k] = v.detach().cpu()

            if len(masks_gts) > 0:
                masks_gts = torch.stack(masks_gts, dim=0).sum(0)
                masks_gts = (masks_gts > 0)
                gt_image = view.get_gt_image()[0:3, :, :][None].cuda().clone().detach()
                gt_image[(masks_gts==0)[None].repeat(3,1,1)[None]] = 0 if background.sum() == 0 else 1
                
                torchvision.utils.save_image(gt_image.detach().cpu(), os.path.join(gt_path,'{0:05d}'.format(idx) + ".jpg"))


                # Also save errormap here
                err_map = get_error_map(estimated_image, gt_image)
                cv2.imwrite(os.path.join(err_path, '{0:05d}'.format(idx) + ".jpg"), err_map)
                
                # calculate whole frame metric
                with torch.no_grad():
                    res_dict = evaluator(
                        rgb = estimated_image, 
                        rgb_gt = gt_image,
                        mask = torch.ones_like(gt_mask.squeeze()).cuda().float(),
                        mask_gt = masks_gts.squeeze().cuda().float()
                    )

                    gt_masked_estimate_image = estimated_image.clone()
                    gt_masked_estimate_image[(masks_gts==0)[None].repeat(3,1,1)[None]] = 0
                    res_dict_v2 = evaluator(
                        rgb = gt_masked_estimate_image, 
                        rgb_gt = gt_image,
                    )
                    res_dict['gt_masked_psnr'] = res_dict_v2['psnr']
                    res_dict['gt_masked_ssim'] = res_dict_v2['ssim']
                    res_dict['gt_masked_lpips'] = res_dict_v2['lpips']



            else:
                with torch.no_grad():
                    mask = view.gt_alpha_mask.cuda()
                    masks_gts = 1 - mask
                    masks_gts = (masks_gts > 0).squeeze()
                    gt_image = view.get_gt_image()[0:3, :, :][None].cuda().clone().detach()
                    gt_image[(masks_gts==0)[None].repeat(3,1,1)[None]] = 0 if background.sum() == 0 else 1
                    
                    torchvision.utils.save_image(gt_image.detach().cpu(), os.path.join(gt_path,'{0:05d}'.format(idx) + ".jpg"))
                    
                    # Also save errormap here
                    err_map = get_error_map(estimated_image, gt_image)
                    cv2.imwrite(os.path.join(err_path, '{0:05d}'.format(idx) + ".jpg"), err_map)
                    
                    res_dict = evaluator(
                        rgb = estimated_image, 
                        rgb_gt = gt_image
                    )
                    print(res_dict['psnr'])
           
            scene_res_dict[fid] = dict()
            for k, v in res_dict.items():
                scene_res_dict[fid][k] = v.detach().cpu()


    video_paths = [render_path]
        
    if calculate_metric:
        metric_path = os.path.join(model_path, name, "ours_{}".format(iteration))

        # get avg of individual people
        if get_indiv_metric:
            for pi in people_infos:
                person_avg = dict()
                person_res = people_res_dict[pi.human_id]
                for k, frame_res in person_res.items():
                    if frame_res is None:
                        continue

                    for metric, value in frame_res.items():
                        if not (metric in person_avg):
                            person_avg[metric] = [value]
                        else:
                            person_avg[metric].append(value)


                print(f"[INFO] person_id: {pi.human_id}")
                if len(person_avg) > 0:
                    for metric, value in person_avg.items():
                        person_avg[metric] = torch.tensor(value).mean()
                        print(str(metric), ": {:>12.7f}".format(person_avg[metric], ".5"))
                    print(f" ")
                    
                    # dump results
                    evaluator.dump_as_json(os.path.join(metric_path, f'{pi.human_id}_per_view.json'), person_res)
                    evaluator.dump_as_json(os.path.join(metric_path, f'{pi.human_id}_results.json'), person_avg)


                    person_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"{pi.human_id}")
                    video_paths.append(person_path)

        
        # get avg of whole scene
        global_avg = dict()
        for k, frame_res in scene_res_dict.items():
            for metric, value in frame_res.items():
                if not (metric in global_avg):
                    global_avg[metric] = [value]
                else:
                    global_avg[metric].append(value)


        print(f"[INFO] SCENE OVERALL METRIC")
        for metric, value in global_avg.items():
            global_avg[metric] = torch.tensor(value).mean()
            print(str(metric), ": {:>12.7f}".format(global_avg[metric], ".5"))
        print(f" ")
        
            
        # dump results
        evaluator.dump_as_json(os.path.join(metric_path, 'overall_per_view.json'), scene_res_dict)
        evaluator.dump_as_json(os.path.join(metric_path, 'overall_results.json'), global_avg)

    
    if render_indiv:
        indiv_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "indiv")
        makedirs(indiv_render_path, exist_ok=True)
        
        indiv_render_path_dict = dict()
        indiv_mask_fname_dict = dict()
        for pi in people_infos:
            indiv_render_path_dict[int(pi.human_id)] = os.path.join(indiv_render_path, f"{int(pi.human_id):05}")
            makedirs(indiv_render_path_dict[int(pi.human_id)], exist_ok=True)

            # Get GT with applying indiv mask
            human_mask_fname_dict = dict()
            for _cam in pi.human_scene.getTrainCameras():
                fid = _cam.colmap_id
                mask_fname = _cam.mask_fname
                if mask_fname is None:
                    continue
                human_mask_fname_dict[fid] = mask_fname
            indiv_mask_fname_dict[pi.human_id] = human_mask_fname_dict
    
        
        for idx, view in enumerate(tqdm(views, desc="Rendering individuals")):
            if idx % 5 != 0:
                continue
            fid = view.colmap_id
            gt_image = view.get_gt_image()[0:3, :, :][None].cuda()


            for pi in people_infos:
                if fid in pi.fids:
                    # Render individuals
                    rendering = combined_renderer(view, scene_gaussians, [pi], pipeline, torch.zeros_like(background), scaling_modifier = 1.0, override_color = None, render_only_people=True, hard_rendering=False)["render"].detach().cpu()
                    torchvision.utils.save_image(rendering, os.path.join(indiv_render_path_dict[int(pi.human_id)], '{0:05d}'.format(fid) + ".jpg"))

                    if False:
                        ##### This part is for evaluating withimages rendered with  "ONLY"  target human
                        indiv_mask_fname = indiv_mask_fname_dict[pi.human_id][fid]
                        indiv_mask = cv2.imread(str(indiv_mask_fname))
                        if len(indiv_mask.shape) == 3:
                            indiv_mask = indiv_mask.sum(-1) / indiv_mask.shape[-1]
                        indiv_mask = indiv_mask / 255.
                        indiv_mask = indiv_mask > 0.5
                        indiv_mask = indiv_mask[None][None]

                        # apply masking
                        rendering[rendering > 1] /= rendering[rendering > 1]

                        indiv_est_img = rendering.squeeze().unsqueeze(0)
                        indiv_est_img[indiv_mask==0] = 0
                        indiv_gt_img = gt_image.clone().detach()
                        indiv_gt_img[indiv_mask==0] = 0

                        res_dict = evaluator(
                            rgb = indiv_est_img, 
                            rgb_gt = indiv_gt_img
                        )

    if is_canon:
        canon_views = gen_canon_cam(res=canon_render_res)

    if render_canon_combined:
        rendered_views = {}

    if is_canon:
        # check people results
        people_path = os.path.join(model_path, name, "ours_{}".format(iteration), "people")
        makedirs(people_path, exist_ok=True)

        if render_canon_combined:
            canon_stitched_path = os.path.join(model_path, name, "ours_{}".format(iteration), "canon_overview")
            makedirs(canon_stitched_path, exist_ok=True)
            canon_stitched_path = Path(canon_stitched_path)

        for pi in people_infos:
            # make person_paths
            person_path = os.path.join(people_path, str(pi.human_id))
            makedirs(person_path, exist_ok=True)


            # Make a person
            representative_img = tensor2cv(pi.representative_img)
            representative_img = cv2.resize(representative_img, (canon_render_res, canon_render_res))

        
            views = pi.human_scene.getTrainCameras()

            person_canon_path = os.path.join(person_path, "canons")
            makedirs(person_canon_path, exist_ok=True)
            person_canon_trans_path = os.path.join(person_path, "canon_trans_grid")
            makedirs(person_canon_trans_path, exist_ok=True)


            _pipe_view_dir_reg = pipeline.view_dir_reg
            pipeline.view_dir_reg = pi.view_dir_reg
            
            white_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") 
            black_bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

            canon_rgbs = []
            for idx, view in enumerate(tqdm(canon_views)):
                # w_smpl rendering == rendering after trans-grids
                w_canon_rendering = render(view, pi.gaussians, pipeline, white_bg)["render"].detach().cpu()            
                b_canon_rendering = render(view, pi.gaussians, pipeline, black_bg)["render"].detach().cpu()  
                
                saving = torch.cat([w_canon_rendering, b_canon_rendering], dim=-1)    
                saving = tensor2cv(saving)
                saving = np.concatenate([saving, representative_img], axis=1)
                canon_rgbs.append(saving)
                
                cv2.imwrite(os.path.join(person_canon_path, '{0:05d}'.format(idx) + ".jpg"), saving)
            video_paths.append(person_canon_path)
            
            # extract two cameras
            cam_back = canon_views[0]
            cam_front = canon_views[len(canon_views)//2]
            for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
                cam_back.uid = view.uid
                cam_front.uid = view.uid

                # add corresponding views' deformation 
                deformer_cond = dict()
                if view.colmap_id in pi.fids:
                    _data_idx = pi.fids.index(view.colmap_id)
                    thetas = pi.smpl_local_poses[_data_idx].squeeze()[3:][None]
                    deformer_cond['thetas'] = thetas


                w_front_deformed = render_w_smpl(cam_front, pi.gaussians, pipeline, white_bg, deformer=pi.smpl_deformer, deformer_cond=deformer_cond)["render"].detach().cpu()
                w_back_deformed = render_w_smpl(cam_back, pi.gaussians, pipeline, white_bg, deformer=pi.smpl_deformer, deformer_cond=deformer_cond)["render"].detach().cpu()
                w_saving = torch.cat([w_front_deformed, w_back_deformed], dim=-1)
                
                b_front_deformed = render_w_smpl(cam_front, pi.gaussians, pipeline, black_bg, deformer=pi.smpl_deformer, deformer_cond=deformer_cond)["render"].detach().cpu()
                b_back_deformed = render_w_smpl(cam_back, pi.gaussians, pipeline, black_bg, deformer=pi.smpl_deformer, deformer_cond=deformer_cond)["render"].detach().cpu()
                b_saving = torch.cat([b_front_deformed, b_back_deformed], dim=-1)
                
                saving = torch.cat([w_saving, b_saving], dim=-1)
                
                try:
                    torchvision.utils.save_image(saving, os.path.join(person_canon_trans_path, '{0:05d}'.format(idx) + ".jpg"))
                except:
                    continue

            video_paths.append(person_canon_trans_path)
            pipeline.view_dir_reg = _pipe_view_dir_reg


            # render overview
            if render_canon_combined:
                rendered_views[str(pi.human_id)] = canon_rgbs 
                if len(rendered_views) == figsize[0] * figsize[1]:
                    idx = 0
                    render_list = []
                    for k,v in rendered_views.items():
                        if idx % figsize[1] == 0:
                            render_list.append(dict())
                        render_list[idx//figsize[1]][k] = v
                        idx += 1
                    pids = list(rendered_views.keys())

                    render_imgs = stitch_outputs(render_list)
                    merged_path = canon_stitched_path / f"every_{figsize[0] * figsize[1]}"
                    merged_path.mkdir(exist_ok=True)

                    # 7. save the images
                    save_path = merged_path / str(pids[0])
                    save_path.mkdir(exist_ok=True)
                    for i, r_img in enumerate(render_imgs):
                        # write names on them
                        r_img = np.concatenate([np.ones((100, r_img.shape[1], 3))*255, r_img], axis=0)
                        r_img = img_add_text(r_img, f"iter_{iteration}")
                        save_name = str(save_path / f"{i:05}.jpg")
                        cv2.imwrite(save_name, r_img)

                    video_paths.append(str(save_path))
                    rendered_views = {}

        # Make video of remainings
        if len(rendered_views) > 0:
            pids = list(rendered_views.keys())
            last_img = rendered_views[pids[0]][0]
            _j = 0
            while(len(rendered_views)<figsize[0] * figsize[1]):
                rendered_views[f"__{_j:02}"] = np.ones_like(last_img) 
                _j += 1

            idx = 0
            render_list = []
            for k,v in rendered_views.items():
                if idx % figsize[1] == 0:
                    render_list.append(dict())
                render_list[idx//figsize[1]][k] = v
                idx += 1
            pids = list(rendered_views.keys())

            render_imgs = stitch_outputs(render_list)
            merged_path = canon_stitched_path / f"every_{figsize[0] * figsize[1]}"
            merged_path.mkdir(exist_ok=True)

            # 7. save the images
            save_path = merged_path / str(pids[0])
            save_path.mkdir(exist_ok=True)
            for i, r_img in enumerate(render_imgs):
                # write names on them
                r_img = np.concatenate([np.ones((100, r_img.shape[1], 3))*255, r_img], axis=0)
                r_img = img_add_text(r_img, f"iter_{iteration}")
                save_name = str(save_path / f"{i:05}.jpg")
                cv2.imwrite(save_name, r_img)
            video_paths.append(str(save_path))

    if len(views) > 0:
        gen_videos(video_paths, is_jpg=True, fps=10, rm_dir=True)
    

    print("[INFO] Done rendering scenes!")



@torch.no_grad()
def render_optim_logs(save_idx, iteration, pipe, people_infos, scene_gaussians, save_path, n_camera=4, n_person=5, n_frame=5):
    # select target to render
    # Here we want "frequently shown people"
    pids = list(range(len(people_infos)))
    n_frames = [len(v.fids) for v in people_infos]

    sorted_pids = list(zip(n_frames, pids))
    sorted_pids.sort()
    
    if n_person > len(sorted_pids):
        n_person = len(sorted_pids)
    top_n_pids = [sorted_pids[-i-1][1] for i in range(n_person)]

    bg_color = [1,1,1]      # Here I enforce white color, for better visualization
    background = torch.tensor(bg_color, dtype=torch.float32).to(scene_gaussians.get_xyz.device)


    canon_views = gen_canon_cam(res=512)
    cam_back = canon_views[0]
    cam_front = canon_views[len(canon_views)//2]



    save_path = Path(str(save_path)) / "optim_log"
    save_path.mkdir(exist_ok=True)

    video_dirs = []

    for pid in tqdm(top_n_pids):
        person_save_path = save_path / str(pid)
        person_save_path.mkdir(exist_ok=True)


        # Render front-view
        # [TODO] Here I assumed human camera is quite accurate
        person_info = people_infos[pid]
        person_cams = person_info.human_scene.getTrainCameras()

        
        # Select n_fids
        fid_interval = len(person_info.fids) // n_frame
        if fid_interval == 0:
            fid_interval = 1
            
        for i in range(n_frame):
            idx = i*fid_interval
            
            if idx > (n_frame-1):
                break
                
            cam_save_path = person_save_path / f"{person_info.fids[idx]}"
            cam_save_path.mkdir(exist_ok=True)

            raw_data_save_path = cam_save_path / "_raw"
            raw_data_save_path.mkdir(exist_ok=True)

            # get camera & render front
            render_cam = person_cams[idx]
            only_person = combined_renderer(render_cam, scene_gaussians, [person_info], pipe, background, scaling_modifier = 1.0, override_color = None, hard_rendering=False, render_only_people=True)["render"].detach().cpu()
            combined_render = combined_renderer(render_cam, scene_gaussians, people_infos, pipe, background, scaling_modifier = 1.0, override_color = None, hard_rendering=False, render_only_people=False)["render"].detach().cpu()

            only_person[only_person>1] /= only_person[only_person>1]
            combined_render[combined_render>1] /= combined_render[combined_render>1]

            # Render front & back 

            v_dir_reg = pipe.view_dir_reg
            pipe.view_dir_reg = False
            _w_front_canon = render_w_smpl(cam_front, person_info.gaussians, pipe, background, deformer=person_info.smpl_deformer)["render"].detach().cpu()
            _w_back_canon = render_w_smpl(cam_back, person_info.gaussians, pipe, background, deformer=person_info.smpl_deformer)["render"].detach().cpu()
            pipe.view_dir_reg = v_dir_reg

            _w_front_canon[_w_front_canon>1] /= _w_front_canon[_w_front_canon>1]
            _w_back_canon[_w_back_canon>1] /= _w_back_canon[_w_back_canon>1]
            
            only_person = padded_resize(tensor2cv(only_person), (512,512))
            combined_render = padded_resize(tensor2cv(combined_render), (512,512))  
            gt_image = padded_resize(tensor2cv(render_cam.original_image.detach()), (512, 512))
            w_front_canon = padded_resize(tensor2cv(_w_front_canon), (512,512))
            w_back_canon = padded_resize(tensor2cv(_w_back_canon), (512,512))  

            render_target = [
                dict(gt=gt_image, front=combined_render),
                dict(canon_front=w_front_canon, front=only_person),
                dict(canon_back=w_back_canon)
            ]

            # Make perturbed_camera
            if False:
                perturbed_cams = gen_perturbed_camera(render_cam, n_camera, radius=0.2)     # r time larger than before

                for _j, p_cam in enumerate(perturbed_cams):
                    only_person = combined_renderer(p_cam, scene_gaussians, [person_info], pipe, background, scaling_modifier = 1.0, override_color = None, hard_rendering=False, render_only_people=True)["render"].detach().cpu()
                    combined_render = combined_renderer(p_cam, scene_gaussians, people_infos, pipe, background, scaling_modifier = 1.0, override_color = None, hard_rendering=False, render_only_people=False)["render"].detach().cpu()

                    
                    only_person = padded_resize(tensor2cv(only_person), (512,512))
                    combined_render = padded_resize(tensor2cv(combined_render), (512,512))  

                    if _j < 2:
                        render_target[0][f"perturb_{_j:02}"] = only_person
                        render_target[1][f"perturb_{_j:02}"] = combined_render
                    else:
                        render_target[2][f"perturb_{_j:02}"] = only_person
                        render_target[3][f"perturb_{_j:02}"] = combined_render

            # Finally make stitched output
            render_imgs = stitch_outputs(render_target)
            render_img = render_imgs[0]


            render_img = np.concatenate([np.ones((100, render_img.shape[1], 3))*255, render_img], axis=0)
            render_img = img_add_text(render_img, f"pid: {person_info.human_id} | f ({idx}/{len(person_info.fids)}) | iter: {iteration:06} | n_gauss: {person_info.gaussians.get_n_points}")
            save_name = str(cam_save_path / f"{save_idx:05}.jpg")
            cv2.imwrite(save_name, render_img)


            # Additionally save renderings
            cv2.imwrite(str(raw_data_save_path / f"front_{save_idx:05}.jpg"), tensor2cv(_w_front_canon))
            cv2.imwrite(str(raw_data_save_path / f"back_{save_idx:05}.jpg"), tensor2cv(_w_back_canon))

            video_dirs.append(cam_save_path)

    return video_dirs






@torch.no_grad()
def render_visibility_log(save_dir: Path, frame_camera, people_infos, cam_mask_dict=None, cam_vdir_dict=None, input_img_dict=dict(), n_cameras: int=32):
    save_dir.mkdir(exist_ok=True)

    _canon_views = gen_canon_cam(n_cameras=n_cameras, res=512)
    
    canon_views = []
    for cam in _canon_views:
        # set same fid as rendered camera here
        cam.colmap_id = frame_camera.colmap_id
        canon_views.append(cam)
        

    res_img_dict = dict()
    for pi in people_infos:
        if not pi.human_id in cam_mask_dict:
            print(f"human {pi.human_id} isn't visible here")
            continue
        

        visible_mask = cam_mask_dict[pi.human_id]
        canon_dir_pp = cam_vdir_dict[pi.human_id]
        input_img = input_img_dict[pi.human_id] if pi.human_id in input_img_dict else np.zeros((512,512,3))
        input_img = cv2.resize(input_img, (512,512))
        input_img = img_add_text(input_img, f"pid: {int(pi.human_id):03}")

        for idx, view in enumerate(tqdm(canon_views)):
            # w_smpl rendering == rendering after trans-grids

            hard_mask, soft_mask = render_vis_log(view, pi, visible_mask, canon_dir_pp, use_canon_pose=True)

            hard_mask = tensor2cv(hard_mask)
            hard_mask = img_add_text(hard_mask, "hard mask")
            soft_mask = tensor2cv(soft_mask)
            soft_mask = img_add_text(soft_mask, "soft mask")
            saving = np.concatenate([input_img, hard_mask, soft_mask], axis=1)

            if not (idx in res_img_dict):
                res_img_dict[idx] = [saving]
            else:
                res_img_dict[idx].append(saving)

    for idx, res_imgs in res_img_dict.items():
        res_img = np.concatenate(res_imgs, axis=0)
        res_img = np.concatenate([np.ones((100, res_img.shape[1], 3))*255, res_img], axis=0)
        res_img = img_add_text(res_img, f"cam_id: {int(frame_camera.colmap_id):05}")
        cv2.imwrite(os.path.join(str(save_dir), '{0:05d}'.format(idx) + ".jpg"), res_img)
    
    return save_dir
        




@torch.no_grad()
def render_human_motions(iteration, pipe, people_infos, save_path):
    # select target to render
    # Here we want "frequently shown people"
    bg_color = [1,1,1]      # Here I enforce white color, for better visualization
    background = torch.tensor(bg_color, dtype=torch.float32).cuda()


    canon_views = gen_canon_cam(n_cameras=4, res=1024, f=800)[:4]

    save_path = Path(str(save_path)) / "mv_human_motion"
    save_path.mkdir(exist_ok=True)

    save_path = save_path / f"{iteration:06}"
    save_path.mkdir(exist_ok=True)


    video_dirs = []

    for p_idx, person_info in tqdm(enumerate(people_infos)):
        pid = person_info.human_id
        person_save_path = save_path / str(pid)
        person_save_path.mkdir(exist_ok=True)

        beta = person_info.beta
        # Render front-view
        # [TODO] Here I assumed human camera is quite accurate
        person_cams = person_info.human_scene.getTrainCameras()


        # As the fids are not sorted, we need to get sorted order here
        # Using enumerate to pair each element with its index, then sorting by the element
        sorted_pairs = sorted(enumerate(person_info.fids), key=lambda x: x[1])
        # Extracting the indices from the sorted pairs
        sorted_indices = [index for index, value in sorted_pairs]
            
        for idx in sorted_indices:
            # get camera & render front
            render_cam = person_cams[idx]

            smpl_param = torch.cat([
                person_info.smpl_scale.reshape(-1),
                person_info.smpl_global_poses[idx],
                person_info.smpl_local_poses[idx],
                beta
            ], dim=-1)
            smpl_param = smpl_param.unsqueeze(0)

            smpl_param[0, 0] = 1.       # Fix scale as 1
            smpl_param[0, 1:4] *= 0     # remove global translation
            # smpl_param[0, 4:7] *= 0     # remove global rotation 


            v_dir_reg = pipe.view_dir_reg
            pipe.view_dir_reg = person_info.view_dir_reg
            savings = []
            for _cam in canon_views:
                _cam.uid = idx
                rendered_rgb = render_w_smpl(_cam, person_info.gaussians, pipe, background, deformer=person_info.smpl_deformer, smpl_param=smpl_param)["render"].detach().cpu()
                savings.append(rendered_rgb)
            pipe.view_dir_reg = v_dir_reg
            
            
            saving1 = torch.cat([savings[0], savings[1]], dim=-2)
            saving2 = torch.cat([savings[2], savings[3]], dim=-2)
            saving = torch.cat([saving1, saving2], dim=-1)
            person_save_path.mkdir(exist_ok=True)
            try:
                torchvision.utils.save_image(saving, os.path.join(str(person_save_path), '{0:05d}'.format(idx) + ".jpg"))
            except:
                print(f"failed to save {os.path.join(str(person_save_path), '{0:05d}'.format(idx) + '.jpg')}")

        video_dirs.append(person_save_path)
    gen_videos(video_dirs, is_jpg=True, fps=10, rm_dir=False)


def render_scene_w_traj_w_people(dataset : ModelParams, scene_datasets: RenderParams, iteration : int, pipeline : PipelineParams, exp_name: str="debug",  human_tracker_name: str='phalp'):
    # locate humans in proper positions.
    # human location -> represented with smpl params (86-dim) [1, 3, 72, 10]
    # Here, we also consider scales!. Please be aware that global coordinate is defined properly
    
    # define paths to render
    print(f"[INFO] loading scene data from {dataset.source_path}")
    
    render_path = os.path.join(dataset.model_path, exp_name, "renders")
    scenes_path = os.path.join(dataset.model_path, exp_name, "scene")
    # canon_path = os.path.join(dataset.model_path, name, "ours_{}".format(iteration), "canon")
    # canon_trans_path = os.path.join(dataset.model_path, name, "ours_{}".format(iteration), "canon_trans_grid")
    
    makedirs(render_path, exist_ok=True)
    makedirs(scenes_path, exist_ok=True)

    traj_transform_path = Path(str(dataset.source_path)) / 'dataparser_transforms.json'
    print(traj_transform_path)
    traj_transform_path = traj_transform_path if traj_transform_path.exists() else None
    print(traj_transform_path is None)
    
    
    
    # set some default settings datas
    gender = 'neutral'
    scale = 1.
    human_sh_degree = 0
    device = torch.device("cuda:0")
    
    bg_color = [1,1,1]      # Here I enforce white color, for better visualization
    background = torch.tensor(bg_color, dtype=torch.float32).to(device)

    
    cam_path = Path(scene_datasets.camera_paths) if isinstance(scene_datasets.camera_paths, str) else scene_datasets.camera_paths
    # render_cameras = load_ns_traj_from_json(cam_path, torch.device('cpu'))
    
    with torch.no_grad():
        # Load main scene first
        # scene_datasets.camera_paths = ""
        use_fast_loader = (scene_datasets.camera_paths != "")
        scene_gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, scene_gaussians, load_iteration=iteration, shuffle=False, fast_loader=use_fast_loader)
    
        # 1. load scene datasets
        if scene_datasets.camera_paths == "":
            print("[INFO] rendering camera path isn't specified here. loading train camera for rendering.")
            render_cameras = scene.getTrainCameras()
        else:
            print(f"[INFO] load rendering cameras from {str(scene_datasets.camera_paths)}")
            cam_path = Path(scene_datasets.camera_paths) if isinstance(scene_datasets.camera_paths, str) else scene_datasets.camera_paths
            render_cameras = load_ns_traj_from_json(cam_path, torch.device('cpu'), traj_transform_path)
            
            
        N_camera = len(render_cameras)
        # 2. load human datasets
        
        # First, automatically load trained human data
        human_model_path = Path(dataset.model_path) / 'humans'
        human_data_path = Path(dataset.source_path) / 'segmentations' / human_tracker_name / 'indiv'
        
        
        people_infos = []
        if human_model_path.exists():
            # load human models in dictionary format.
            human_chkpts = human_model_path.glob("**/point_cloud")
            for human_chkpt in human_chkpts:
                human_id = human_chkpt.parent.name
                model_path = human_chkpt / ("iteration_" + str(scene.loaded_iter))
                
                person_gaussians = GaussianModel(human_sh_degree)
                person_gaussians.load_ply(str(model_path / "point_cloud.ply"))
                
                # we need human_data path (to load poses)
                # data_path = human_data_path / human_id
                mean_beta = torch.tensor(np.load(model_path / 'mean_shape.npy')).squeeze()
                smpl_poses = pandas.read_pickle(model_path / "smpl_params.pkl")
                
                if smpl_poses.shape[-1] == 75:
                    # Missed scale!
                    # smpl_scale = pandas.read_pickle(model_path / "smpl_scale.pkl")
                    smpl_scale = torch.tensor([[0.287]], dtype=torch.float32)
                    smpl_poses = torch.cat([
                        smpl_scale.repeat(len(smpl_poses), 1),
                        smpl_poses
                    ], dim=-1)
                    
                    
                # load smpl_defomer
                smpl_deformer = SMPLDeformer(gender=gender, beta=mean_beta, smpl_scale=scale)
                smpl_deformer.load_trans_grid(path=model_path, load_iteration=False)
                
                poses = torch.cat([
                    smpl_poses,
                    mean_beta.reshape(1, -1).repeat(len(smpl_poses), 1)
                ], dim=-1)
                
                uids = [0 for _ in range(len(poses))]
                
                # repeat poses if there're not enough poses
                while (len(poses) < N_camera+5):      # +1 is enough, but for safety
                    uids = uids + uids[::-1]
                    poses = torch.cat([
                        poses,
                        poses.flip(0)
                    ], dim=0 
                    )
                
                people_infos.append(
                    PersonInfos(
                        uids = uids,
                        model_path = model_path,
                        smpl_params = poses,
                        beta = mean_beta,
                        smpl_deformer = smpl_deformer,
                        gaussians = person_gaussians,
                        fids = None,
                        human_id = None
                    )
                )
                
        # 3. render scenes
        # (Init) First, simply we render scene without consdiering bbox of human (visibility)
        # (If OOM occurs so frequently, consider different approach)
        for idx, view in enumerate(tqdm(render_cameras, desc="Rendering progress")):
             # Also render scene wo human
            only_scene = render(view, scene_gaussians, pipeline, background)["render"].detach().cpu()
            
            # render w/ human if exists
            if len(people_infos) > 0:
                render_output = combined_renderer(view, scene_gaussians, people_infos, pipeline, background, scaling_modifier = 1.0, override_color = None)
                render_output = render_output["render"].detach().cpu()
            
                # saving = render_output
                saving = torch.cat([render_output, only_scene], dim=-2)
                torchvision.utils.save_image(saving, os.path.join(render_path, '{0:05d}'.format(idx) + ".jpg"))
            torchvision.utils.save_image(only_scene, os.path.join(scenes_path, '{0:05d}'.format(idx) + ".jpg"))
        
        
    # make videos
    video_paths = [scenes_path]
    if len(people_infos) > 0:
        video_paths.append(render_path)
        
    if len(render_cameras) > 0:
        gen_videos(video_paths, is_jpg=True, fps=10, rm_dir=False)
    


def render_scene_w_human(dataset : ModelParams, scene_datasets: RenderParams, iteration : int, pipeline : PipelineParams, exp_name: str="debug", n_camera=5, use_original_world_time=False, uniform_sampler=False, people_infos=None, scene=None, save_path=None,  human_tracker_name: str='phalp'):
     # locate humans in proper positions.
    # human location -> represented with smpl params (86-dim) [1, 3, 72, 10]
    # Here, we also consider scales!. Please be aware that global coordinate is defined properly
    
    
    print(f"[INFO] Choosing random cameras to render scene")
    
    # set some default settings datas
    gender = 'neutral'
    scale = 1.
    human_sh_degree = 0
    device = torch.device("cuda:0")
    
    bg_color = [1,1,1]      # Here I enforce white color, for better visualization
    background = torch.tensor(bg_color, dtype=torch.float32).to(device)
    
    # 1. Select random path
    with torch.no_grad():
        if scene is None:
            use_fast_loader = False
            scene_gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, scene_gaussians, load_iteration=iteration, shuffle=False, fast_loader=use_fast_loader)
        else:
            scene_gaussians = scene.gaussians
        train_cameras = scene.getTrainCameras()
        scene_image_shape = (scene.getTrainCameras()[0].image_width, scene.getTrainCameras()[0].image_height)
        
        max_fid = max([cam.colmap_id for cam in train_cameras])
        
        if not uniform_sampler:
            random_cam_indices = random.sample(range(len(train_cameras)), n_camera)
        else:
            _step_size = len(train_cameras) // n_camera
            random_cam_indices = [int(i*_step_size) for i in range(n_camera)]


        # 2. load human datasets
        # First, automatically load trained human data
        if people_infos is None:
            human_model_path = Path(dataset.model_path) / 'humans'
            people_infos = []
            if human_model_path.exists():
                # load human models in dictionary format.
                human_chkpts = human_model_path.glob("**/point_cloud")
                for human_chkpt in human_chkpts:
                    human_id = human_chkpt.parent.name
                    model_path = human_chkpt / ("iteration_" + str(scene.loaded_iter))
                    
                    person_gaussians = GaussianModel(human_sh_degree)
                    person_gaussians.load_ply(str(model_path / "point_cloud.ply"))
                    
                    # we need human_data path (to load poses)
                    # data_path = human_data_path / human_id
                    mean_beta = torch.tensor(np.load(model_path / 'mean_shape.npy')).squeeze()
                    smpl_poses = pandas.read_pickle(model_path / "smpl_params.pkl")
                    
                    if smpl_poses.shape[-1] == 75:
                        # Missed scale!
                        # smpl_scale = pandas.read_pickle(model_path / "smpl_scale.pkl")
                        smpl_scale = torch.tensor([[0.287]], dtype=torch.float32)
                        smpl_poses = torch.cat([
                            smpl_scale.repeat(len(smpl_poses), 1),
                            smpl_poses
                        ], dim=-1)
                        
                        
                    # load smpl_defomer
                    smpl_deformer = SMPLDeformer(gender=gender, beta=mean_beta, smpl_scale=scale)
                    smpl_deformer.load_trans_grid(path=model_path, load_iteration=False)
                    
                    poses = torch.cat([
                        smpl_poses,
                        mean_beta.reshape(1, -1).repeat(len(smpl_poses), 1)
                    ], dim=-1)
                    
                    uids = [0 for _ in range(len(poses))]
                    
                    if use_original_world_time:
                        # Need to know original camera ids

                        human_camera_path = Path(dataset.source_path) / 'segmentations' / human_tracker_name / 'indiv'
                        human_scene = HumanScene(human_camera_path, human_model_path, scene_img_shape=scene_image_shape, eval=False, view_dir_reg=False, sh_degree=human_sh_degree)
                        cameras = human_scene.getTrainCameras()

                        original_frame_ids = [cam.colmap_id for cam in cameras]

                    else:
                        # repeat poses if there're not enough poses
                        while (len(poses) < max_fid+5):      # +1 is enough, but for safety
                            uids = uids + uids[::-1]
                            poses = torch.cat([
                                    poses,
                                    poses.flip(0)
                                ], dim=0)
                        original_frame_ids = None
                    
                    _pi_ = PersonInfos(
                        uids = uids,
                        model_path = model_path,
                        smpl_params = poses,
                        beta = mean_beta,
                        smpl_deformer = smpl_deformer,
                        gaussians = person_gaussians,
                        fids = original_frame_ids,
                        human_id = human_id
                    )
                    people_infos.append(_pi_)

        # 3. Think about 
    
    # start rendering
    for cam_index in tqdm(random_cam_indices):
        render_camera = train_cameras[cam_index]
        
        if save_path is None:
            render_path = os.path.join(dataset.model_path, exp_name, f"{cam_index:03}_renders")
            rgbs_path = os.path.join(dataset.model_path, exp_name, f"{cam_index:03}_rgbs")
        else:
            render_path = os.path.join(save_path, exp_name, "ours_{}".format(iteration), f"{cam_index:03}_renders")
            rgbs_path = os.path.join(save_path, exp_name, "ours_{}".format(iteration), f"{cam_index:03}_rgbs")

        # canon_path = os.path.join(dataset.model_path, name, "ours_{}".format(iteration), "canon")
        # canon_trans_path = os.path.join(dataset.model_path, name, "ours_{}".format(iteration), "canon_trans_grid")
        
        makedirs(render_path, exist_ok=True)
        makedirs(rgbs_path, exist_ok=True)
        
        only_scene = render(render_camera, scene_gaussians, pipeline, background)["render"].detach().cpu()


        original_colmap_id = render_camera.colmap_id
        # 3. render scenes
        # (Init) First, simply we render scene without consdiering bbox of human (visibility)
        # (If OOM occurs so frequently, consider different approach)
        for idx, view in enumerate(tqdm(train_cameras, desc="Rendering progress")):
            # Also render scene wo human   
            render_camera.colmap_id = view.colmap_id
            render_camera.uid = view.uid
  
            render_output = combined_renderer(render_camera, scene_gaussians, people_infos, pipeline, background, scaling_modifier = 1.0, override_color = None)
            render_output = render_output["render"].detach().cpu()
            
            # saving = render_output
            saving = torch.cat([render_output, only_scene], dim=-1)
            torchvision.utils.save_image(saving, os.path.join(render_path, '{0:05d}'.format(idx) + ".jpg"))
            torchvision.utils.save_image(render_output, os.path.join(rgbs_path, '{0:05d}'.format(idx) + ".jpg"))
        
        render_camera.colmap_id = original_colmap_id
        # make videos
        video_paths = [render_path, rgbs_path]    
        gen_videos(video_paths, is_jpg=True, fps=10, rm_dir=False)
        


def render_scene_w_traj(dataset : ModelParams, scene_datasets: RenderParams, iteration : int, pipeline : PipelineParams, exp_name: str="debug", human_tracker_name: str='phalp'):
    # locate humans in proper positions.
    # human location -> represented with smpl params (86-dim) [1, 3, 72, 10]
    # Here, we also consider scales!. Please be aware that global coordinate is defined properly
    
    # define paths to render
    print(f"[INFO] loading scene data from {dataset.source_path}")
    
    render_path = os.path.join(dataset.model_path, exp_name, "renders")
    scenes_path = os.path.join(dataset.model_path, exp_name, "scene")
    # canon_path = os.path.join(dataset.model_path, name, "ours_{}".format(iteration), "canon")
    # canon_trans_path = os.path.join(dataset.model_path, name, "ours_{}".format(iteration), "canon_trans_grid")
    
    makedirs(render_path, exist_ok=True)
    makedirs(scenes_path, exist_ok=True)


    traj_transform_path = Path(str(dataset.source_path)) / 'dataparser_transforms.json'
    print(traj_transform_path)
    traj_transform_path = traj_transform_path if traj_transform_path.exists() else None
    print(traj_transform_path is None)

    

    
    # set some default settings datas
    gender = 'neutral'
    scale = 1.
    human_sh_degree = 0
    device = torch.device("cuda:0")
    
    bg_color = [1,1,1]      # Here I enforce white color, for better visualization
    background = torch.tensor(bg_color, dtype=torch.float32).to(device)

    
    cam_path = Path(scene_datasets.camera_paths) if isinstance(scene_datasets.camera_paths, str) else scene_datasets.camera_paths
    # render_cameras = load_ns_traj_from_json(cam_path, torch.device('cpu'))
    
    with torch.no_grad():
        # Load main scene first
        use_fast_loader = (scene_datasets.camera_paths != "")
        scene_gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, scene_gaussians, load_iteration=iteration, shuffle=False, fast_loader=use_fast_loader)
    
        # 1. load scene datasets
        if scene_datasets.camera_paths == "":
            print("[INFO] rendering camera path isn't specified here. loading train camera for rendering.")
            render_cameras = scene.getTrainCameras()
        else:
            print(f"[INFO] load rendering cameras from {str(scene_datasets.camera_paths)}")
            cam_path = Path(scene_datasets.camera_paths) if isinstance(scene_datasets.camera_paths, str) else scene_datasets.camera_paths
            render_cameras = load_ns_traj_from_json(cam_path, torch.device('cpu'), traj_transform_path)
            
            
        # 2. load human datasets
        people_infos = []
        human_paths = []
        if len(scene_datasets.human_model_paths) == 0:
            print("[INFO] there's no human in scene. Simply render scene only")
        if len(scene_datasets.traj_paths) == 0:
            print("[INFO] there's no valid human trajectory. Simply render scene only")
        else:
            if len(scene_datasets.human_model_paths) == 1 and len(scene_datasets.traj_paths) > 1:
                human_paths = [scene_datasets.human_model_paths[0] for _ in range(len(scene_datasets.traj_paths))]
            else:
                human_paths = scene_datasets.human_model_paths
                
            
            # load datas
            for human_path, pose_path in zip(human_paths, scene_datasets.traj_paths):
                
                gaussians, poses, beta, human_path = load_human_model(Path(human_path), Path(pose_path), human_sh_degree)
                
                uids = [0 for _ in range(len(poses))]
                smpl_deformer = SMPLDeformer(gender=gender, beta=beta, smpl_scale=scale)
                smpl_deformer.load_trans_grid(path=human_path, load_iteration=False)
        
                
                #### TODO Need to combine "different" Spherical Harmonics Dimensions in single space
                
                people_infos.append(
                    PersonInfos(
                        uids = uids,
                        model_path = human_path,
                        smpl_params = poses,
                        beta = beta,
                        smpl_deformer = smpl_deformer,
                        gaussians = gaussians,
                        fids = None,
                        human_id = None
                    )
                )

                
        # 3. render scenes
        # (Init) First, simply we render scene without consdiering bbox of human (visibility)
        # (If OOM occurs so frequently, consider different approach)
        for idx, view in enumerate(tqdm(render_cameras, desc="Rendering progress")):
             # Also render scene wo human
            only_scene = render(view, scene_gaussians, pipeline, background)["render"].detach().cpu()
            
            # render w/ human if exists
            if len(people_infos) > 0:
                render_output = combined_renderer(view, scene_gaussians, people_infos, pipeline, background, scaling_modifier = 1.0, override_color = None)
                render_output = render_output["render"].detach().cpu()
            
                # saving = render_output
                saving = torch.cat([render_output, only_scene], dim=-2)
                torchvision.utils.save_image(saving, os.path.join(render_path, '{0:05d}'.format(idx) + ".jpg"))
            torchvision.utils.save_image(only_scene, os.path.join(scenes_path, '{0:05d}'.format(idx) + ".jpg"))
        
        
    # make videos
    video_paths = [scenes_path]
    if len(people_infos) > 0:
        video_paths.append(render_path)
        
    if len(render_cameras) > 0:
        gen_videos(video_paths, is_jpg=True, fps=10, rm_dir=False)
    





def render_mv_human(dataset : ModelParams, scene_datasets: RenderParams, iteration : int, pipeline : PipelineParams, exp_name: str="debug", n_camera=5, uniform_sampler=False, people_infos=None, scene=None, save_path=None,  human_tracker_name: str='phalp'):
     # locate humans in proper positions.
    # human location -> represented with smpl params (86-dim) [1, 3, 72, 10]
    # Here, we also consider scales!. Please be aware that global coordinate is defined properly
    
    
    print(f"[INFO] Choosing random cameras to render scene")
    
    # set some default settings datas
    gender = 'neutral'
    scale = 1.
    human_sh_degree = 0
    device = torch.device("cuda:0")
    do_trans_grid = dataset.use_trans_grid
    human_view_dir_reg = False
    
    bg_color = [1,1,1]      # Here I enforce white color, for better visualization
    background = torch.tensor(bg_color, dtype=torch.float32).to(device)
    
    # 1. Select random path
    with torch.no_grad():
        if scene is None:
            use_fast_loader = False
            scene_gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, scene_gaussians, load_iteration=iteration, shuffle=False, fast_loader=use_fast_loader)
        else:
            scene_gaussians = scene.gaussians
        train_cameras = scene.getTrainCameras()
        scene_image_shape = (scene.getTrainCameras()[0].image_width, scene.getTrainCameras()[0].image_height)
        
        max_fid = max([cam.colmap_id for cam in train_cameras])
        
        if not uniform_sampler:
            random_cam_indices = random.sample(range(len(train_cameras)), n_camera)
        else:
            _step_size = len(train_cameras) // n_camera
            random_cam_indices = [int(i*_step_size) for i in range(n_camera)]
        

        # 2. load human datasets
        # First, automatically load trained human data
        if people_infos is None:
            human_model_path = Path(dataset.model_path) / 'humans'
            people_infos = []
            if human_model_path.exists():
                # load human models in dictionary format.
                human_chkpts = human_model_path.glob("**/point_cloud")
                for human_chkpt in human_chkpts:
                    human_id = human_chkpt.parent.name
                    human_camera_path = Path(dataset.source_path) / 'segmentations' / human_tracker_name / 'indiv' / human_id
                    if not human_camera_path.exists():
                        continue
                    
                    model_path = human_chkpt / ("iteration_" + str(scene.loaded_iter))
                    
                    person_gaussians = GaussianModel(human_sh_degree)
                    person_gaussians.load_ply(str(model_path / "point_cloud.ply"))
                    
                    # we need human_data path (to load poses)
                    # data_path = human_data_path / human_id
                    mean_beta = torch.tensor(np.load(model_path / 'mean_shape.npy')).squeeze()
                    smpl_poses = pandas.read_pickle(model_path / "smpl_params.pkl")
                    
                    if smpl_poses.shape[-1] == 75:
                        # Missed scale!
                        # smpl_scale = pandas.read_pickle(model_path / "smpl_scale.pkl")
                        smpl_scale = torch.tensor([[0.287]], dtype=torch.float32)
                        smpl_poses = torch.cat([
                            smpl_scale.repeat(len(smpl_poses), 1),
                            smpl_poses
                        ], dim=-1)
                        
                        
                    # load smpl_defomer
                    smpl_deformer = SMPLDeformer(gender=gender, beta=mean_beta, smpl_scale=scale)
                    smpl_deformer.load_trans_grid(path=model_path, load_iteration=False)
                    
                    poses = torch.cat([
                        smpl_poses,
                        mean_beta.reshape(1, -1).repeat(len(smpl_poses), 1)
                    ], dim=-1)
                    
                    uids = [0 for _ in range(len(poses))]
                    
                    human_scene = HumanScene(human_camera_path, human_model_path, scene_img_shape=scene_image_shape, eval=False, view_dir_reg=False, sh_degree=human_sh_degree)
                    cameras = human_scene.getTrainCameras()
                    original_frame_ids = [cam.colmap_id for cam in cameras]
                    
                    
                    uids = [cam.uid for cam in cameras]
                    fids = [cam.colmap_id for cam in cameras]
                    beta = mean_beta.float().cuda()
                    smpl_poses = smpl_poses.cuda()
                    
                    largest_res = -1
                    largest_img = None
                    for cam in cameras:         # It's detached, no loss flow anymore.
                        if max(cam.original_image.shape) > largest_res:
                            largest_img = cam.original_image
                            largest_res = max(cam.original_image.shape)
                            
                    people_infos.append(
                        PersonTrain(
                            uids = uids,
                            fids = fids,
                            smpl_local_poses = smpl_poses[:, 4:],
                            smpl_scale = smpl_poses[0,0],
                            smpl_global_poses = smpl_poses[:, 1:4],
                            local_pose_optimizer = None,
                            global_pose_optimizer = None,
                            smpl_scale_optimizer = None,
                            model_path = human_model_path,
                            beta = beta,
                            smpl_deformer = smpl_deformer,
                            gaussians = person_gaussians,
                            do_trans_grid = do_trans_grid,
                            trans_grids = None,
                            grid_optimizer = None,
                            view_dir_reg = human_view_dir_reg,
                            human_scene = human_scene,
                            human_id = human_id,
                            init_smpl_jnts = None,
                            cam_centers = None,
                            cc_smpl_dir = None,
                            representative_img = largest_img,
                        misc = dict()
                        )
                    )


        # 3. Think about 
    
    # Render human in scene
    os.makedirs(str(scene.model_path) + f"/{exp_name}", exist_ok=True)
    os.makedirs(str(scene.model_path) + f"/{exp_name}/render_human", exist_ok=True)
    render_mv_human_in_scene(str(scene.model_path) + f"/{exp_name}/render_human", "init", 0, scene.getTrainCameras(), scene_gaussians, people_infos, pipeline, background, every_n_camera=5, render_posed_human_together=True)


   
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    scene_datasets = RenderParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--exp_name", type=str, default="debug_default")
    parser.add_argument("--traj_paths", nargs="+", type=str, default=[])
    parser.add_argument("--human_model_paths", nargs="+", type=str, default=[])
    parser.add_argument("--port", default=-1, type=int)
    parser.add_argument("--scene_human", action="store_true")
    parser.add_argument("--render_humans", action="store_true")
    parser.add_argument("--render_mv_humans", action="store_true")
    parser.add_argument("--use_original_world_time", action="store_true")
    parser.add_argument("--n_camera", default=5, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--human_track_method", type=str, default ="phalp")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    scene_datasets.traj_paths = args.traj_paths
    scene_datasets.human_model_paths = args.human_model_paths
    
    if args.human_track_method not in ["phalp", "alphapose"]:
        raise AssertionError("Wrong human_tracker method name")
    
    if args.human_track_method == "phalp":
        model = model.extract(args)
        data_path = Path(model.source_path) / "segmentations" / "phalp" / "indiv"
        if not data_path.exists():
            human_track_method = "phalp_v2"

    if args.scene_human:
        render_scene_w_traj_w_people(model.extract(args), scene_datasets.extract(args), args.iteration, pipeline.extract(args), args.exp_name, human_tracker_name=args.human_track_method)    

    elif args.render_humans:
        render_scene_w_human(model.extract(args), scene_datasets.extract(args), args.iteration, pipeline.extract(args), args.exp_name, n_camera=args.n_camera, use_original_world_time=args.use_original_world_time, human_tracker_name=args.human_track_method)    

    elif args.render_mv_humans:
        render_mv_human(model.extract(args), scene_datasets.extract(args), args.iteration, pipeline.extract(args), args.exp_name, n_camera=args.n_camera, human_tracker_name=args.human_track_method)    

    else:
        render_scene_w_traj(model.extract(args), scene_datasets.extract(args), args.iteration, pipeline.extract(args), args.exp_name, human_tracker_name=args.human_track_method)