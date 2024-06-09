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

import torch
import numpy as np
import math
import random
from typing import Optional, List
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from gtu.smpl_deformer.deformer import SMPLDeformer
from gtu.dataset.scene.gaussian_model import GaussianModel

from utils.sh_utils import eval_sh
from utils.general_utils import rot_weighting, build_rotation, unstrip_symmetric
from utils.graphics_utils import project_points_to_cam
from utils.jnts_utils import extract_square_bbox, filter_invisible_joints
from utils.image_utils import tensor2cv, get_crop_img
from utils.loss_utils import get_cd_loss, denisty_reg_loss
from utils.draw_op_jnts import smpl_joints2op_joints, draw_op_img

from gtu.guidance.joint_utils import filter_invisible_face_joints_w_prompts, get_view_prompt_of_body, combine_prompts


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, opacity_thrs=-1, render_normal: bool=False, get_viz_mask: bool=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation


    # Part for view-dir regularization
    if pipe.view_dir_reg:
        rotations = pc.get_rotation
        dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        
        occ_weights = rot_weighting(build_rotation(rotations), dir_pp_normalized)
        opacity = opacity * occ_weights

    
    # Part for normal-rendering
    if render_normal and pipe.view_dir_reg:
        _, rot_vectors = rot_weighting(build_rotation(rotations), dir_pp_normalized, return_rot_vector=True)
        override_color = rot_vectors / 2 + 0.5


    # Part of rendering visible mask
    if get_viz_mask and pipe.view_dir_reg:
        bg_color = torch.zeros_like(bg_color)
        override_color = torch.ones_like(means3D)
        opacity[occ_weights>0] = 1 

        # To set black on invisible parts
        override_color[occ_weights==0] = 0


    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
        
        
    


    if opacity_thrs > 0:
        opacity[opacity < opacity_thrs] = opacity[opacity < opacity_thrs] * 0
        # opacity[opacity > 0] = 1.
        

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    ras_outputs = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    rendered_image = ras_outputs[0]
    radii = ras_outputs[1]
    
    _return = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

    
    if len(ras_outputs) == 3:
        # _return['depth'] = ras_outputs[2]
        _return['mask'] = ras_outputs[2]
        
    if len(ras_outputs) > 3:
        print("[WARNING] Using old rasterizers now!")
        _return['mask'] = ras_outputs[3]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return _return


def render_w_smpl(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, deformer: Optional[SMPLDeformer]=None, smpl_param=None, get_viz_mask: bool=False, deformer_cond=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    if get_viz_mask and pipe.view_dir_reg:
        bg_color = torch.ones_like(bg_color)    # Should allow inpainting of BG here
        

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python or True:
        ### WE MUST CALCULATE HERE!!!
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        
    if pipe.view_dir_reg:
        _rotations = build_rotation(pc.get_rotation)
    else:
        _rotations = None
        
    # transform points according to SMPL
    smpl_rots = None
    if not deformer is None:
        if smpl_param is None:
            smpl_param = viewpoint_camera.smpl_param
            # if we render canonical, smppl_param can be None
            if smpl_param is not None:
                smpl_param = smpl_param.to(means3D.device).float()
        cond = dict(
            img_idx=viewpoint_camera.uid
        )
        if deformer_cond is not None:
            for k, v in deformer_cond.items():
                if k not in cond:
                    cond[k] = v

        means3D, cov3D_precomp, smpl_rots = deformer.deform_gp(means3D, cov3D_precomp, smpl_param, cond=cond)
        if _rotations is not None and smpl_rots is not None:
            _rotations = torch.bmm(smpl_rots, _rotations)
            
    
    # Part of view-direction regularizer
    if pipe.view_dir_reg:
        dir_pp = (means3D.detach() - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        
         
        occ_weights = rot_weighting(_rotations, dir_pp_normalized)
        
        if not get_viz_mask:
            opacity = opacity * occ_weights
        

    # Part of rendering visible mask
    if get_viz_mask and pipe.view_dir_reg:
        override_color = torch.ones_like(means3D)
        override_color[occ_weights.squeeze()>0] *= 0 

        # To set black on invisible parts
        # override_color[occ_weights.squeeze()==0] *= 0
        # opacity[occ_weights.squeeze()==0] = torch.ones_like(opacity[occ_weights.squeeze()==0])
        


    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            if smpl_rots is not None:
                # Rotate in canonical space 
                dir_pp_normalized = torch.einsum('bij,bj->bi', smpl_rots.transpose(1,2), dir_pp_normalized)
                dir_pp_normalized = dir_pp_normalized/dir_pp_normalized.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
        
        
    

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    ras_outputs = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    rendered_image = ras_outputs[0]
    radii = ras_outputs[1]
    
    _return = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

    
    if len(ras_outputs) > 2:
        # _return['depth'] = ras_outputs[2]
        _return['mask'] = ras_outputs[2]
        
    if len(ras_outputs) > 3:
        _return['mask'] = ras_outputs[3]
        

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return _return


@ torch.no_grad()
def render_top_view(viewpoint_camera, up_vec, basis_point, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    (But remove all points below basis_point)
    """
    # get mask of points
    xyz = pc.get_xyz
    basis_point = basis_point.reshape(1, 3)
    v_dir = ((basis_point - xyz) * up_vec.reshape(1, 3)).sum(-1)
    filtering_mask = (v_dir > 0)        # set occupancy as zero for those points


    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
        
        
    if pipe.view_dir_reg:
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        
        occ_weights = rot_weighting(build_rotation(rotations), dir_pp_normalized)
        opacity = opacity * occ_weights
        

    # do filtering
    opacity[filtering_mask] *= 0


    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    ras_outputs = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    rendered_image = ras_outputs[0]
    radii = ras_outputs[1]
    
    _return = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

    
    if len(ras_outputs) > 2:
        _return['mask'] = ras_outputs[2]
        

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return _return


def combined_renderer(viewpoint_camera, scene_pc : GaussianModel, people_infos: List, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, hard_rendering=False, render_only_people=False, offsets=None, offset_id=0, render_normal=False, get_deformed_points=False):
    data_idx = viewpoint_camera.colmap_id       # it's same as fid
    
    means3D_people = []
    means2D_people = []
    shs_people = []
    colors_precomp_people = []
    opacities_people = []
    scales_people = []
    rotations_people = []
    cov3D_precomp_people = []
    screenspace_points_people = []
    
    if offsets is not None:
        offset_people = []
    
    
    for person_idx, person_info in enumerate(people_infos):
        person_pc = person_info.gaussians
        
        if hasattr(person_info, 'fids') and person_info.fids is not None:
            if data_idx not in person_info.fids:
                continue
            _data_idx = person_info.fids.index(data_idx)
            beta = person_info.beta

            # load smpl_param
            if hasattr(person_info, 'smpl_params'):
                smpl_param = person_info.smpl_params[_data_idx]
                smpl_param[-10:] = beta

                smpl_param = smpl_param.to(means3D.device).float()
                smpl_param = smpl_param.unsqueeze(0)
            else:
                smpl_param = torch.cat([
                    person_info.smpl_scale.reshape(-1),
                    person_info.smpl_global_poses[_data_idx],
                    person_info.smpl_local_poses[_data_idx],
                    beta
                ], dim=-1)
                smpl_param = smpl_param.unsqueeze(0)
                
            
            uid = person_info.uids[_data_idx]
        else:
            smpl_param = person_info.smpl_params[data_idx].unsqueeze(0).cuda()
            uid = person_info.uids[data_idx]
            
            
        smpl_deformer = person_info.smpl_deformer
        
        if offsets is not None and person_idx == offset_id:
            off = offsets(person_pc.get_xyz).clone().detach()
            off.requires_grad = True
            offset_people.append(off)
            
            off_xyz = person_pc.get_xyz
            xyz = off_xyz + off
        else:
            xyz = person_pc.get_xyz
        screenspace_points = torch.zeros_like(xyz, dtype=person_pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        means3D = xyz
        means2D = screenspace_points
        opacity = person_pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = person_pc.get_covariance(scaling_modifier)
        
        if getattr(person_info, 'view_dir_reg', False):
            _rotations = build_rotation(person_pc.get_rotation)
        else:
            _rotations = None
        
            
        # transform points according to SMPL
        cond = dict(
            img_idx=uid
        )
        means3D, cov3D_precomp, smpl_rots = smpl_deformer.deform_gp(means3D, cov3D_precomp, smpl_param, cond=cond, rotations=_rotations)
        
        if _rotations is not None and smpl_rots is not None:
            _rotations = torch.bmm(smpl_rots, _rotations)
    
        if getattr(person_info, 'view_dir_reg', False):
            dir_pp = (means3D.detach() - viewpoint_camera.camera_center.repeat(person_pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            
            occ_weights = rot_weighting(_rotations, dir_pp_normalized)
            opacity = opacity * occ_weights


        # Part for normal-rendering
        if render_normal and getattr(person_info, 'view_dir_reg', False):
            _, rot_vectors = rot_weighting(_rotations, dir_pp_normalized, return_rot_vector=True)
            override_color = rot_vectors / 2 + 0.5

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            # safe 
            if pipe.convert_SHs_python or True:
                shs_view = person_pc.get_features.transpose(1, 2).view(-1, 3, (person_pc.max_sh_degree+1)**2)
                dir_pp = (means3D - viewpoint_camera.camera_center.repeat(person_pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                # Rotate in canonical space 
                dir_pp_normalized = torch.einsum('bij,bj->bi', smpl_rots.transpose(1,2), dir_pp_normalized)
                dir_pp_normalized = dir_pp_normalized/dir_pp_normalized.norm(dim=1, keepdim=True)

                sh2rgb = eval_sh(person_pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = person_pc.get_features
        else:
            if len(override_color) != len(means3D):
                colors_precomp = override_color.reshape(-1, 3).repeat(len(means3D), 1)
            else:
                colors_precomp = override_color
                
        
            

        if hard_rendering:
            opacity[opacity > 0.1] /= opacity[opacity > 0.1].detach()
            opacity[opacity <= 0.1] *= 0
            # opacity = torch.ones_like(opacity).float().to(opacity.device)
            
        means3D_people.append(means3D)
        means2D_people.append(means2D)
        shs_people.append(shs)
        colors_precomp_people.append(colors_precomp)
        opacities_people.append(opacity)
        scales_people.append(scales)
        rotations_people.append(rotations)
        cov3D_precomp_people.append(cov3D_precomp)
        screenspace_points_people.append(screenspace_points)

      
      
    # 2. Add scene points
    if render_only_people:
        pass
    else:
        if offsets is not None and person_idx==-1:
            off = offsets(scene_pc.get_xyz).clone().detach()
            off.requires_grad = True
            offset_people.append(off)
            
            off_xyz = person_pc.get_xyz
            xyz = off_xyz + off
        else:
            xyz = scene_pc.get_xyz
            
        screenspace_points = torch.zeros_like(xyz, dtype=scene_pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
    
        means3D = xyz
        means2D = screenspace_points
        opacity = scene_pc.get_opacity 

        if render_only_people:
            opacity = opacity * 0

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = scene_pc.get_covariance(scaling_modifier)


        if pipe.view_dir_reg:
            dir_pp = (xyz - viewpoint_camera.camera_center.repeat(scene_pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            
            occ_weights = rot_weighting(build_rotation(scene_pc.get_rotation), dir_pp_normalized)
            opacity = opacity * occ_weights
            

        # Part for normal-rendering
        if render_normal and pipe.view_dir_reg:
            _, rot_vectors = rot_weighting(build_rotation(scene_pc.get_rotation), dir_pp_normalized, return_rot_vector=True)
            override_color = rot_vectors / 2 + 0.5

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python or True:
                shs_view = scene_pc.get_features.transpose(1, 2).view(-1, 3, (scene_pc.max_sh_degree+1)**2)
                dir_pp = (xyz - viewpoint_camera.camera_center.repeat(scene_pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(scene_pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = scene_pc.get_features
        else:
            if render_normal and not pipe.view_dir_reg:
                colors_precomp = bg_color.clone().reshape(-1,3).repeat(len(means3D), 1)
            elif len(override_color.reshape(-1)) == 3:
                colors_precomp = override_color.reshape(-1, 3).repeat(len(means3D), 1)
            else:
                colors_precomp = override_color
            
        means3D_people.append(means3D)
        means2D_people.append(means2D)
        shs_people.append(shs)
        colors_precomp_people.append(colors_precomp)
        opacities_people.append(opacity)
        scales_people.append(scales)
        rotations_people.append(rotations)
        cov3D_precomp_people.append(cov3D_precomp)
        screenspace_points_people.append(screenspace_points)
    
    # concate the tensors
    if means3D is not None:
        means3D = torch.cat(means3D_people, dim=0).contiguous()
    else:
        means3D = None
        
    if means2D is not None:
        means2D = torch.cat(means2D_people, dim=0).contiguous()
    else:
        means2D = None
        
    if shs is not None and shs_people[0] is not None:
        shs = torch.cat(shs_people, dim=0).contiguous()
    else:
        shs = None
        
    if colors_precomp is not None:
        colors_precomp = torch.cat(colors_precomp_people, dim=0).contiguous()
    else:
        colors_precomp = None
    
    if opacity is not None:
        opacity = torch.cat(opacities_people, dim=0).contiguous()
    else:
        opacity = None
        
    if scales is not None:
        scales = torch.cat(scales_people, dim=0).contiguous()
    else:
        scales = None
        
    if rotations is not None:
        rotations = torch.cat(rotations_people, dim=0).contiguous()
    else:
        rotations = None
            
    if cov3D_precomp is not None:
        cov3D_precomp = torch.cat(cov3D_precomp_people, dim=0).contiguous()
    else:
        cov3D_precomp = None
        
    if offsets is not None:
        offsets = torch.cat(offset_people, dim=0).contiguous()
        
        
    

    
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=scene_pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    ras_outputs = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    rendered_image = ras_outputs[0]
    radii = ras_outputs[1]
    
    _return = {"render": rendered_image,
            "viewspace_points": screenspace_points_people,        # it's same as screen_space_points
            "visibility_filter" : radii > 0,
            "radii": radii}

    
    if len(ras_outputs) > 2:
        # _return['depth'] = ras_outputs[2]
        # _return['mask'] = (ras_outputs[2] > 0).float()
        _return['mask'] = ras_outputs[2]
        
    if len(ras_outputs) > 3:
        _return['mask'] = ras_outputs[3]

    if offsets is not None:
        _return['points'] = off_xyz
        _return['offsets'] = off


    if get_deformed_points:
        _return['means3D'] = means3D.detach()
        

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return _return


def render_for_diffusion(
                        mini_cam, 
                        pipe, 
                        person_info, 
                        smpl_param, 
                        uid, 
                        bg_color, 
                        scaling_modifier=1., 
                        override_color=None,
                        normal_rendering=False
                        ):
    smpl_deformer = person_info.smpl_deformer
    person_pc = person_info.gaussians

    xyz = person_pc.get_xyz
    screenspace_points = torch.zeros_like(xyz, dtype=person_pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    means3D = xyz
    means2D = screenspace_points
    opacity = person_pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = person_pc.get_covariance(scaling_modifier)

    if getattr(person_info, 'view_dir_reg', False):
        _rotations = build_rotation(person_pc.get_rotation)
    else:
        _rotations = None

        
    # transform points according to SMPL
    cond = dict(
        img_idx=uid
    )

    means3D, cov3D_precomp, smpl_rots = smpl_deformer.deform_gp(means3D, cov3D_precomp, smpl_param, cond=cond)
        
    if _rotations is not None and smpl_rots is not None:
        _rotations = torch.bmm(smpl_rots, _rotations)
    

    if getattr(person_info, 'view_dir_reg', False):
        dir_pp = (means3D.detach() - mini_cam.camera_center.repeat(person_pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        
        occ_weights = rot_weighting(_rotations, dir_pp_normalized)
        opacity = opacity * occ_weights

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        # safe 
        if pipe.convert_SHs_python or True:
            if normal_rendering:
                normals = person_pc.init_normal
                normals = normals / 2.
                colors_precomp = torch.clamp_min(normals + 0.5, 0.0)
            else:
                shs_view = person_pc.get_features.transpose(1, 2).view(-1, 3, (person_pc.max_sh_degree+1)**2)
                dir_pp = (means3D - mini_cam.camera_center.repeat(person_pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                # Rotate in canonical space 
                dir_pp_normalized = torch.einsum('bij,bj->bi', smpl_rots.transpose(1,2), dir_pp_normalized)
                dir_pp_normalized = dir_pp_normalized/dir_pp_normalized.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(person_pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = person_pc.get_features
    else:
        colors_precomp = override_color
    

    if normal_rendering:
        opacity = torch.ones_like(opacity).float().to(opacity.device)


    # Get rendering
    tanfovx = math.tan(mini_cam.FoVx * 0.5)
    tanfovy = math.tan(mini_cam.FoVy * 0.5)


    raster_settings = GaussianRasterizationSettings(
        image_height=int(mini_cam.image_height),
        image_width=int(mini_cam.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=mini_cam.world_view_transform,
        projmatrix=mini_cam.full_proj_transform,
        sh_degree=person_pc.active_sh_degree,                   # though we precalculate radiance here, we use active_sh_degree of person here.
        campos=mini_cam.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    ras_outputs = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rendered_image = ras_outputs[0]
    radii = ras_outputs[1]
    rendered_image = rendered_image.unsqueeze(0)

    dgm_cond = []
    with torch.no_grad():
        smpl_output = person_info.smpl_deformer.smpl_server(smpl_param)
        smpl_jnts = smpl_output['smpl_jnts'].detach().cpu()
        # print(smpl_jnts.shape)
        pj_jnts = project_points_to_cam(mini_cam, smpl_jnts.squeeze().numpy(), image_res=(512,512))
        
        op_joints = smpl_joints2op_joints(pj_jnts)
        op_3d_jnt = smpl_joints2op_joints(smpl_jnts.squeeze().numpy())
        
        
        # New filtering with visibility
        lower_body_prompt = get_view_prompt_of_body(op_3d_jnt, mini_cam, is_lower_body=True)
        upper_body_prompt = get_view_prompt_of_body(op_3d_jnt, mini_cam, is_lower_body=False)
        filtered_op_3d_jnts, head_prompt = filter_invisible_face_joints_w_prompts(op_3d_jnt, mini_cam)
        image_res = (mini_cam.image_height, mini_cam.image_width)
        new_prompt = combine_prompts(head_prompt, upper_body_prompt, lower_body_prompt, op_joints, image_res)

        # # filter with visibility
        # op_joints = filter_invisible_joints(op_joints)
        for idx, _op_3d_jnt in enumerate(filtered_op_3d_jnts):
                if _op_3d_jnt is None:
                    op_joints[idx] = None
                    # op_joints[idx][0] = -1
                    # op_joints[idx][1] = -1
                        
        
        op_cond_img = draw_op_img(op_joints, 512)
        dgm_cond.append(op_cond_img)
        

    return rendered_image, dgm_cond, new_prompt


def diffusion_renderer(
    DGM, 
    viewpoint_camera, 
    scene_pc : GaussianModel, 
    people_infos: List, 
    pipe, 
    bg_color : torch.Tensor, 
    scaling_modifier = 1.0, 
    override_color = None, 
    hard_rendering=False, 
    offsets=None, 
    offset_id=0, 
    iteration: int=-1,
    render_normal=False, 
    do_optim: bool=False, 
    dgm_loss_weight: float=0.1, 
    cd_loss_weight: float=1,
    canon_loss_mode: Optional[str] = None,
    non_directional_visibility: bool=False,
    ddim_num_step_inferences: int=10,
    ddim_fixed_max_time_step: bool=True,
    minimum_mask_thrs: float=0.02,
    masking_optimizer: bool=False,
    cfg_rescale_weight: float=0.8,
    density_reg_loss_weight: float=0.0,
    grid_trans_reg_loss_weight: float=0.0,
    only_density_reg_loss: bool=False,
    ):
    
    data_idx = viewpoint_camera.colmap_id       # it's same as fid

    use_canon_pose = False
    get_single_fixed_camera = False
    if not (canon_loss_mode is None):
        use_canon_pose = True
        get_single_fixed_camera = canon_loss_mode.endswith("fixed")
        if canon_loss_mode in ['all', 'all_fixed']:
            # apply canon loss on ALL people
            data_idx = -1
        elif canon_loss_mode == ['viz', 'viz_fixed']:
            # apply canon loss on VISIBLE people
            pass
        else:
            raise TypeError(f"canon loss mode {canon_loss_mode} is invalid!")
        
    
    losses = 0
    log_dict = dict()
    for person_idx, person_info in enumerate(people_infos):
        person_pc = person_info.gaussians

        if not (person_info.human_id in DGM.train_pids):
            # skip unavailable pids
            continue
        
        if hasattr(person_info, 'fids') and person_info.fids is not None:
            if (data_idx >= 0) and (data_idx not in person_info.fids):
                
                print("person not in frame!")
                print("Optimize anyway.")
                _data_idx = random.randint(0, len(person_info.fids)-1) 
                person_info.misc['optimized_step'] += 1
                
            elif data_idx == -1:
                _data_idx = random.randint(0, len(person_info.fids)-1)
            else:
                _data_idx = person_info.fids.index(data_idx)
            beta = person_info.beta

            # load smpl_param
            if hasattr(person_info, 'smpl_params'):
                smpl_param = person_info.smpl_params[_data_idx]
                smpl_param[-10:] = beta

                smpl_param = smpl_param.to('cuda').float()
                smpl_param = smpl_param.unsqueeze(0)
            else:
                smpl_param = torch.cat([
                    person_info.smpl_scale.reshape(-1),
                    person_info.smpl_global_poses[_data_idx],
                    person_info.smpl_local_poses[_data_idx],
                    beta
                ], dim=-1)
                smpl_param = smpl_param.unsqueeze(0)
                
            
            uid = person_info.uids[_data_idx]
        else:
            smpl_param = person_info.smpl_params[data_idx].unsqueeze(0).cuda()
            uid = person_info.uids[data_idx]
            
        
        smpl_deformer = person_info.smpl_deformer
        
        if offsets is not None and person_idx == offset_id:
            off = offsets(person_pc.get_xyz).clone().detach()
            off.requires_grad = True
            
            off_xyz = person_pc.get_xyz
            xyz = off_xyz + off
        else:
            xyz = person_pc.get_xyz
        screenspace_points = torch.zeros_like(xyz, dtype=person_pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        means3D = xyz
        means2D = screenspace_points
        opacity = person_pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = person_pc.get_covariance(scaling_modifier)
        
        if getattr(person_info, 'view_dir_reg', False):
            _rotations = build_rotation(person_pc.get_rotation)
        else:
            _rotations = None
        
            
        # transform points according to SMPL
        cond = dict(
            img_idx=uid
        )
        smpl_param[0, 0] = 1.       # Fix scale as 1
        smpl_param[0, 1:4] *= 0     # remove global translation
        smpl_param[0, 2] = 0.3     # remove global translation (transl + 0.3 on y direction)
        smpl_param[0, 4:7] *= 0     # remove global rotation
        
        if use_canon_pose:
            smpl_param[0, 7:-10] *= 0 # remove local rotation
            # set joint angle as canonical shape
            smpl_param[0, 9] = torch.tensor((torch.pi / 6)).to(smpl_param.device).float()
            smpl_param[0, 12] = (-torch.tensor((torch.pi / 6))).to(smpl_param.device).float()
        smpl_scale = smpl_param[0, 0]


        # Now select camera from camera sampler
        pid = person_info.human_id
        mini_cam, invert_bg_color, mini_cam_vers, mini_cam_hors, mini_cam_radii, smpl_param, camera_type_name = DGM.get_render_camera(
                                        pid, 
                                        smpl_scale.item(), 
                                        get_single_fixed_camera=get_single_fixed_camera,
                                        smpl_param = smpl_param,
                                        smpl_deformer=smpl_deformer,
                                        )


        means3D, cov3D_precomp, _rotations = smpl_deformer.deform_gp(means3D, cov3D_precomp, smpl_param, cond=cond, rotations=_rotations)
        
        if getattr(person_info, 'view_dir_reg', False):
            dir_pp = (means3D.detach() - mini_cam.camera_center.repeat(person_pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            
            occ_weights = rot_weighting(_rotations, dir_pp_normalized)
            opacity = opacity * occ_weights


        # Part for normal-rendering
        if render_normal and getattr(person_info, 'view_dir_reg', False):
            _, rot_vectors = rot_weighting(_rotations, dir_pp_normalized, return_rot_vector=True)
            override_color = rot_vectors / 2 + 0.5

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            # safe 
            if pipe.convert_SHs_python or True:
                shs_view = person_pc.get_features.transpose(1, 2).view(-1, 3, (person_pc.max_sh_degree+1)**2)
                dir_pp = (means3D - mini_cam.camera_center.repeat(person_pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(person_pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = person_pc.get_features
        else:
            colors_precomp = override_color
            

        if hard_rendering:
            opacity = torch.ones_like(opacity).float().to(opacity.device)


        # Get rendering
        tanfovx = math.tan(mini_cam.FoVx * 0.5)
        tanfovy = math.tan(mini_cam.FoVy * 0.5)
        render_bg_color = 1 - bg_color if invert_bg_color else bg_color


        raster_settings = GaussianRasterizationSettings(
            image_height=int(mini_cam.image_height),
            image_width=int(mini_cam.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=render_bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=mini_cam.world_view_transform,
            projmatrix=mini_cam.full_proj_transform,
            sh_degree=scene_pc.active_sh_degree,
            campos=mini_cam.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        ras_outputs = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        rendered_image = ras_outputs[0]
        radii = ras_outputs[1]
        rendered_image = rendered_image.unsqueeze(0)
        
        # clip the highest value
        rendered_image[rendered_image > 1] /= rendered_image[rendered_image > 1]

        if density_reg_loss_weight > 0:
            assert (len(ras_outputs) > 2), "Invalid rasterizer (don't return alpha)"
            alpha = ras_outputs[-1]
            density_reg_loss = denisty_reg_loss(alpha) * density_reg_loss_weight
        else:
            density_reg_loss = 0

        grid_trans_reg_loss = 0
        if person_info.do_trans_grid:
            grid_trans_reg_loss += (person_info.smpl_deformer.last_trans[-1] ** 2).mean()
            grid_trans_reg_loss = grid_trans_reg_loss * grid_trans_reg_loss_weight

        

        if only_density_reg_loss:
            ### When skipping DGM loss calculating
            log_dict[person_info.human_id] = dict()
            log_dict[person_info.human_id]['density_reg_loss'] = (density_reg_loss.detach().cpu() if isinstance(density_reg_loss, torch.Tensor) else 0.0)
            log_dict[person_info.human_id]['trans_reg_loss_novel_view'] = (grid_trans_reg_loss.detach().cpu() if isinstance(grid_trans_reg_loss, torch.Tensor) else 0.0)
            _loss = density_reg_loss + grid_trans_reg_loss
        else:
            if masking_optimizer:
                print(f"[INFO] we mask optimizers instead of rendering visibility")
                assert do_optim, f"masking optimizer is only valid with OPT_INDIV options"
                mask, gaussian_visibility = render_visibility_mask(
                                mini_cam, 
                                smpl_deformer, 
                                raster_settings, 
                                person_pc, 
                                smpl_param, means3D, means2D, opacity, scales, rotations, cov3D_precomp,
                                non_directional=non_directional_visibility,
                                masking_optimizer=masking_optimizer)
                
                gaussian_visibility += minimum_mask_thrs
                gaussian_visibility = torch.clamp(gaussian_visibility, 0, 1)
            elif DGM.is_inpaint:
                mask = render_visibility_mask(
                                    mini_cam, 
                                    smpl_deformer, 
                                    raster_settings, 
                                    person_pc, 
                                    smpl_param, means3D, means2D, opacity, scales, rotations, cov3D_precomp,
                                    non_directional=non_directional_visibility)
            else:
                mask = None


            dgm_cond = []
            new_prompt = None

            if DGM.enable_controlnet:
                with torch.no_grad():
                    smpl_output = person_info.smpl_deformer.smpl_server(smpl_param)
                    smpl_jnts = smpl_output['smpl_jnts'].detach().cpu()
                    # print(smpl_jnts.shape)
                    image_res = (512, 512)
                    pj_jnts = project_points_to_cam(mini_cam, smpl_jnts.squeeze().numpy(), image_res=image_res)
                    op_joints = smpl_joints2op_joints(pj_jnts)
                    op_3d_jnt = smpl_joints2op_joints(smpl_jnts.squeeze().numpy())
                    
                    
                    # New filtering with visibility
                    lower_body_prompt = get_view_prompt_of_body(op_3d_jnt, mini_cam, is_lower_body=True)
                    upper_body_prompt = get_view_prompt_of_body(op_3d_jnt, mini_cam, is_lower_body=False)
                    filtered_op_3d_jnts, head_prompt = filter_invisible_face_joints_w_prompts(op_3d_jnt, mini_cam, image_res=image_res)
                    new_prompt = combine_prompts(head_prompt, upper_body_prompt, lower_body_prompt, op_joints, image_res)

                    # # filter with visibility
                    # op_joints = filter_invisible_joints(op_joints)
                    for idx, _op_3d_jnt in enumerate(filtered_op_3d_jnts):
                        if _op_3d_jnt is None:
                            op_joints[idx] = None
                
                    op_cond_img = draw_op_img(op_joints, 512)
                    dgm_cond.append(op_cond_img)
                        
            if cd_loss_weight == 0:
                # Turn of cd_loss
                DGM.color_correction = False
            else:
                DGM.color_correction = True
                
            dg_loss, step_ratio, guid_loss_dict = DGM.get_loss(
                                                rendered_image, 
                                                pid, 
                                                vers=mini_cam_vers, 
                                                hors=mini_cam_hors, 
                                                radii=mini_cam_radii, 
                                                iteration=iteration, 
                                                cond_image=dgm_cond, 
                                                additional_prompt=new_prompt,
                                                mask=mask,
                                                ddim_num_step_inferences=ddim_num_step_inferences,
                                                ddim_fixed_max_time_step=ddim_fixed_max_time_step,
                                                save_intermediate=True,
                                                img_description=camera_type_name,
                                                minimum_mask_thrs=minimum_mask_thrs,
                                                cfg_rescale_weight=cfg_rescale_weight
                                                )
            _loss = dg_loss * dgm_loss_weight + density_reg_loss + grid_trans_reg_loss
            # dg_loss scale: initially, around 129

            if (dg_loss.isnan().sum() + dg_loss.isinf().sum()) > 0:
                print("something goes wrong")

            log_dict[person_info.human_id] = dict()
            log_dict[person_info.human_id]['dg_loss'] = _loss.detach().cpu()
            log_dict[person_info.human_id]['density_reg_loss'] = (density_reg_loss.detach().cpu() if isinstance(density_reg_loss, torch.Tensor) else 0.0)
            log_dict[person_info.human_id]['trans_reg_loss_novel_view'] = (grid_trans_reg_loss.detach().cpu() if isinstance(grid_trans_reg_loss, torch.Tensor) else 0.0)
            log_dict[person_info.human_id]['noise_ratio'] = step_ratio
            log_dict[person_info.human_id]['render_height'] = int(mini_cam.image_height)

            if len(guid_loss_dict) > 0:
                for k, v in guid_loss_dict.items():
                    log_dict[person_info.human_id][k] = v

            
            
        
        # Color consistency Loss
        if cd_loss_weight > 0 and False:
            assert not DGM.is_inpaint, "Not checked yet with masking"
            lambda_cd = DGM.get_lambda_cd(pid)
            lambda_cd = lambda_cd * cd_loss_weight
            if lambda_cd == 0:
                continue

            # First Get GT pixel sets
            gt_pixel_lists = person_info.misc['color_distribution']
            
            # Second, Get valid pixel sets
            # if len(ras_outputs) == 3:
            #     mask = (ras_outputs[2] > 0).float()  
            # elif len(ras_outputs) > 3:
            #     mask = ras_outputs[3]
            # else:
            #     raise AssertionError("[ERROR] Color Consistency Loss is invalid without mask-rasterizer")
            
            rendered_image = rendered_image.squeeze()
            # mask = mask.squeeze().unsqueeze(0).repeat(3,1,1)
            # rendered_image[mask == 0] = -1
            rendered_pixel_lists = rendered_image.reshape(3, -1).T
            rendered_pixel_lists = rendered_pixel_lists[rendered_pixel_lists.sum(-1) > 0]   # remove black bg
            rendered_pixel_lists = rendered_pixel_lists[rendered_pixel_lists.sum(-1) < 3]   # remove white  bg
            
            cd_loss = get_cd_loss(gt_pixel_lists, rendered_pixel_lists)
            cd_loss = cd_loss * cd_loss_weight
            log_dict[person_info.human_id]['cd_loss'] = cd_loss.detach().cpu()
            _loss += cd_loss

        # Get loss or not
        
        if do_optim:
            _loss.backward()
            
            if masking_optimizer:
                # remove gradient with mask
                # thrs = (step_ratio / DGM.guidance_controlnet.num_train_timesteps) 
                thrs = 0.5          # naive on off      
                mask = gaussian_visibility < thrs           # higher thrs -> large change -> (less optimized points) (here visibility is highest if it's 0)
                gaussian_viz_mask = mask.squeeze()
                person_info.gaussians.prune_gradients(gaussian_viz_mask)
                
                print("Do pruning ")
                
            
            # Here, DGM.opt is equal to trainer.opt -> so it's fine to use same variable name here
            if person_info.misc['optimized_step'] >= DGM.opt.density_start_iter and person_info.misc['optimized_step'] <= DGM.opt.density_end_iter:
                viewspace_point_tensor, visibility_filter, radii = screenspace_points, (radii > 0), radii
                
                # if masking_optimizer:
                #     visibility_filter[gaussian_viz_mask] = False
                    
                person_info.gaussians.max_radii2D[visibility_filter] = torch.max(person_info.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                person_info.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                
                # Densification / resetting will be done in main.py (as we don't count Diffusion loss separatedly)
                # if person_info.misc['optimized_step'] % DGM.opt.densification_interval == 0:
                #     # size_threshold = 20 if self.step > self.opt.opacity_reset_interval else None
                #     person_info.gaussians.densify_and_prune(DGM.opt.densify_grad_threshold, min_opacity=0.01, extent=0.5, max_screen_size=1)
                
                # if person_info.misc['optimized_step'] % DGM.opt.opacity_reset_interval == 0:
                #     person_info.gaussians.reset_opacity()
                
            if (person_info.misc['optimized_step'] > DGM.opt.iter_smpl_densify) and (person_info.misc['optimized_step'] <= DGM.opt.iter_prune_smpl_until):
                if person_info.misc['optimized_step'] % DGM.opt.densification_interval == 0:
                    person_info.gaussians.prune_gaussians(0.005, 5, None)
                
            # set optimizer as zero
            person_info.gaussians.optimizer.step()
            person_info.gaussians.optimizer.zero_grad(set_to_none = True)

            if person_info.do_trans_grid:
                person_info.grid_optimizer.step()
                person_info.grid_optimizer.zero_grad()
                person_info.smpl_deformer.last_trans = []
                
                
            n_nan = person_info.gaussians.prune_infnan_points()
            if n_nan > 0:
                log_dict[person_info.human_id]['n_nan'] = n_nan

                
            _loss = _loss.detach()

        losses += _loss

    return losses, log_dict


@torch.no_grad()
def render_visibility_mask(view_camera, smpl_deformer, raster_settings, person_gaussian, smpl_param, means3D, means2D, opacity, scales, rotations, cov3D_precomp, is_hard_rendering=True, non_directional=False, visibility_mask=None, masking_optimizer=False):
    if visibility_mask is None:
        # get view-direction
        dir_pp = (means3D.detach() - view_camera.camera_center)
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)

        # convert it to back to canonical space
        R = smpl_deformer.get_rotations(person_gaussian.get_xyz, smpl_param, cond=None)     # (N, 3, 3)
        R = R.permute(0, 2, 1)      # inverse the rotation

        # Need to convert back to original space
        canon_dir_pp = torch.einsum('bij,bj->bi', R, dir_pp_normalized)

        # Get weighted mask
        # 1: visible, 0: invisible, 0~1 partially visible.
        visibility_mask = person_gaussian.get_viz_mask(canon_dir_pp, non_directional_vizmask=non_directional)                # Currently, we are using cosine as visibility weight
        
    if masking_optimizer:
        return None, visibility_mask.reshape(-1, 1)

    
    # set color as black for visible points (especially, the inner product)
    colors_precomp = torch.ones_like(means3D) * visibility_mask.reshape(-1,1)   # (N. 3)    


    # Set color as white_bg
    white_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") # for testing
    raster_settings = raster_settings._replace(bg=white_bg)

        
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if is_hard_rendering:
        opacity[opacity > 0.2] = 1
        opacity[opacity <= 0.2] = 0

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    ras_outputs = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    rendered_image = ras_outputs[0]
    mask = rendered_image.squeeze().mean(0)

    return mask


def query_point_density_color(query_points, scene_pc, viewpoint_camera, people_infos=None, scaling_modifier = 1.0, scene_view_dir_reg: bool=False, render_only_people: bool=False):
    data_idx = viewpoint_camera.colmap_id       # it's same as fid
    
    means3D_people = []
    colors_precomp_people = []
    opacities_people = []
    rotations_people = []
    cov3D_precomp_people = []
    
    if people_infos is None:
        people_infos = []

    for person_info in people_infos:
        person_pc = person_info.gaussians
        
        if hasattr(person_info, 'fids') and person_info.fids is not None:
            if data_idx not in person_info.fids:
                continue
            _data_idx = person_info.fids.index(data_idx)
            beta = person_info.beta

            # load smpl_param
            if hasattr(person_info, 'smpl_params'):
                smpl_param = person_info.smpl_params[_data_idx]
                smpl_param[-10:] = beta

                smpl_param = smpl_param.to(means3D.device).float()
                smpl_param = smpl_param.unsqueeze(0)
            else:
                smpl_param = torch.cat([
                    person_info.smpl_scale.reshape(-1),
                    person_info.smpl_global_poses[_data_idx],
                    person_info.smpl_local_poses[_data_idx],
                    beta
                ], dim=-1)
                smpl_param = smpl_param.unsqueeze(0)
                
            
            uid = person_info.uids[_data_idx]
        else:
            smpl_param = person_info.smpl_params[data_idx].unsqueeze(0).cuda()
            uid = person_info.uids[data_idx]
            
            
        smpl_deformer = person_info.smpl_deformer
        means3D = person_pc.get_xyz
        opacity = person_pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = person_pc.get_covariance(scaling_modifier)
        
        if getattr(person_info, 'view_dir_reg', False):
            _rotations = build_rotation(person_pc.get_rotation)
        else:
            _rotations = None
        
            
        # transform points according to SMPL
        cond = dict(
            img_idx=uid
        )
        means3D, cov3D_precomp, smpl_rots = smpl_deformer.deform_gp(means3D, cov3D_precomp, smpl_param, cond=cond)        
        if _rotations is not None and smpl_rots is not None:
            _rotations = torch.bmm(smpl_rots, _rotations)  
        

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        
        shs_view = person_pc.get_features.transpose(1, 2).view(-1, 3, (person_pc.max_sh_degree+1)**2)
        dir_pp = (person_pc.get_xyz - viewpoint_camera.camera_center.repeat(person_pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        # Rotate in canonical space 
        dir_pp_normalized = torch.einsum('bij,bj->bi', smpl_rots.transpose(1,2), dir_pp_normalized)
        dir_pp_normalized = dir_pp_normalized/dir_pp_normalized.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(person_pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

            
        if getattr(person_info, 'view_dir_reg', False):
            dir_pp = (person_pc.get_xyz - viewpoint_camera.camera_center.repeat(person_pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            
            occ_weights = rot_weighting(_rotations, dir_pp_normalized)
            opacity = opacity * occ_weights
            

        rotations = _rotations
        means3D_people.append(means3D)
        colors_precomp_people.append(colors_precomp)
        opacities_people.append(opacity)
        cov3D_precomp_people.append(cov3D_precomp)
        rotations_people.append(_rotations)

      
      
    # 2. Add scene points
    means3D = scene_pc.get_xyz
    opacity = scene_pc.get_opacity 

    if render_only_people:
        opacity = opacity * 0

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = scene_pc.get_covariance(scaling_modifier)
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    
    shs_view = scene_pc.get_features.transpose(1, 2).view(-1, 3, (scene_pc.max_sh_degree+1)**2)
    dir_pp = (scene_pc.get_xyz - viewpoint_camera.camera_center.repeat(scene_pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(scene_pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
 

    if scene_view_dir_reg:
        dir_pp = (scene_pc.get_xyz - viewpoint_camera.camera_center.repeat(scene_pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        
        occ_weights = rot_weighting(build_rotation(scene_pc.get_rotation), dir_pp_normalized)
        opacity = opacity * occ_weights


        rotations = build_rotation(person_pc.get_rotation)
        
            
    means3D_people.append(means3D)
    colors_precomp_people.append(colors_precomp)
    opacities_people.append(opacity)
    rotations_people.append(rotations)
    cov3D_precomp_people.append(cov3D_precomp)
    
    # concate the tensors
    if means3D is not None:
        means3D = torch.cat(means3D_people, dim=0).contiguous()
    else:
        means3D = None
        
    if colors_precomp is not None:
        colors_precomp = torch.cat(colors_precomp_people, dim=0).contiguous()
    else:
        colors_precomp = None
    
    if opacity is not None:
        opacity = torch.cat(opacities_people, dim=0).contiguous()
    else:
        opacity = None
        
    if rotations is not None:
        rotations = torch.cat(rotations_people, dim=0).contiguous()
    else:
        rotations = None
            
    if cov3D_precomp is not None:
        cov3D_precomp = torch.cat(cov3D_precomp_people, dim=0).contiguous()
    else:
        cov3D_precomp = None
        
    if rotations is not None:
        _, rot_vectors = rot_weighting(rotations, torch.empty_like(cov3D_precomp), return_rot_vector=True)
    else:
        rot_vectors = torch.zeros_like(means3D)


    if query_points is None:
        return opacity, means3D, colors_precomp, rot_vectors

    # From now, Let's calculate affecting points
    query_points = query_points.reshape(-1, 3)  # reshape into 3-D

    # Get distance of queried points <-> mean3D
    dists = (query_points.reshape(-1, 1, 3) - means3D.reshape(1,-1,3))  # (Q, N, 3)     Q: num of queried points

    cov3D_precomp = unstrip_symmetric(cov3D_precomp)
    power = -0.5 * torch.einsum("qnd,ndk,qnk->qn", dists, cov3D_precomp, dists) # (Q. N)


    alpha = (opacity.squeeze().unsqueeze(0) * torch.exp(power))
    alpha[alpha < 1/255] *= 0.
    alpha[alpha > 1] = 1.
    rgb = alpha.unsqueeze(-1) * colors_precomp.squeeze()
    normal = alpha.unsqueeze(-1) *  rot_vectors.squeeze()


    return alpha.sum(-1), rgb.sum(1), normal.sum(1)



def render_vis_log(
    viewpoint_camera, 
    person_info,
    visible_mask=None,
    mask_canon_dir_pp=None,
    is_hard_rendering=True,
    use_canon_pose=True,
    scaling_modifier = 1.0, 
    override_color = None, 
    ):
    
    data_idx = viewpoint_camera.colmap_id       # it's same as fid
    person_pc = person_info.gaussians

    render_bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") 


    if hasattr(person_info, 'fids') and person_info.fids is not None:
        if data_idx == -1:
            _data_idx = random.randint(0, len(person_info.fids)-1)
        elif data_idx in person_info.fids:
            _data_idx = person_info.fids.index(data_idx)
        else:
            _data_idx = 0
        beta = person_info.beta

        # load smpl_param
        if hasattr(person_info, 'smpl_params'):
            smpl_param = person_info.smpl_params[_data_idx]
            smpl_param[-10:] = beta

            smpl_param = smpl_param.to('cuda').float()
            smpl_param = smpl_param.unsqueeze(0)
        else:
            smpl_param = torch.cat([
                person_info.smpl_scale.reshape(-1),
                person_info.smpl_global_poses[_data_idx],
                person_info.smpl_local_poses[_data_idx],
                beta
            ], dim=-1)
            smpl_param = smpl_param.unsqueeze(0)
            
        
        uid = person_info.uids[_data_idx]
    else:
        smpl_param = person_info.smpl_params[data_idx].unsqueeze(0).cuda()
        uid = person_info.uids[data_idx]
        
    
    smpl_deformer = person_info.smpl_deformer
    xyz = person_pc.get_xyz
    screenspace_points = torch.zeros_like(xyz, dtype=person_pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    means3D = xyz
    means2D = screenspace_points
    opacity = person_pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = person_pc.get_covariance(scaling_modifier)
    _rotations = build_rotation(person_pc.get_rotation)

    
        
    # transform points according to SMPL
    cond = dict(
        img_idx=uid
    )
    smpl_param[0, 0] = 1.       # Fix scale as 1
    smpl_param[0, 1:4] *= 0     # remove global translation
    smpl_param[0, 4:7] *= 0     # remove global rotation
    
    if use_canon_pose:
        smpl_param[0, 7:-10] *= 0 # remove local rotation
        # set joint angle as canonical shape
        smpl_param[0, 9] = torch.tensor((torch.pi / 6)).to(smpl_param.device).float()
        smpl_param[0, 12] = (-torch.tensor((torch.pi / 6))).to(smpl_param.device).float()
    
    means3D, cov3D_precomp, smpl_rots = smpl_deformer.deform_gp(means3D, cov3D_precomp, smpl_param, cond=cond)
    if _rotations is not None and smpl_rots is not None:
        _rotations = torch.bmm(smpl_rots, _rotations)

    dir_pp = (means3D.detach() - viewpoint_camera.camera_center.repeat(person_pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)

    ########### converting visibility back to canonical space
    R = smpl_deformer.get_rotations(person_pc.get_xyz, smpl_param, cond=None)     # (N, 3, 3)
    R = R.permute(0, 2, 1)      # inverse the rotation
    canon_dir_pp = torch.einsum('bij,bj->bi', R, dir_pp_normalized)

    
        

    if getattr(person_info, 'view_dir_reg', False): 
        occ_weights = rot_weighting(_rotations, dir_pp_normalized)
        opacity = opacity * occ_weights

    if is_hard_rendering:
        opacity[opacity>0.2] = 1
        opacity[opacity<=0.2] = 0

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        # safe 
        if True:
            shs_view = person_pc.get_features.transpose(1, 2).view(-1, 3, (person_pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(person_pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # Rotate in canonical space 
            dir_pp_normalized = torch.einsum('bij,bj->bi', smpl_rots.transpose(1,2), dir_pp_normalized)
            dir_pp_normalized = dir_pp_normalized/dir_pp_normalized.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(person_pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = person_pc.get_features
    else:
        colors_precomp = override_color
        

    # Get rendering
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)


    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=render_bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)


    # get hard mask
    if visible_mask is None:
        visible_mask = person_pc.get_viz_mask(canon_dir_pp, non_directional_vizmask=True) 
    hard_viz_mask = 1 - visible_mask.float()
    hard_mask = render_visibility_mask(
                            viewpoint_camera, 
                            smpl_deformer, 
                            raster_settings, 
                            person_pc, 
                            smpl_param, means3D, means2D, opacity, scales, rotations, cov3D_precomp,
                            visibility_mask=hard_viz_mask)
    # get soft mask
    if mask_canon_dir_pp is None:
        soft_viz_mask = person_pc.get_viz_mask(canon_dir_pp, non_directional_vizmask=False) 
    else:
        nearest_angle = torch.einsum('bi,bi->b', mask_canon_dir_pp.cuda(), canon_dir_pp[visible_mask])
        # nearest_angle = (mask_canon_dir_pp.cuda() @ canon_dir_pp[visible_mask].T)
        nearest_angle[nearest_angle>1] = 1
        nearest_angle[nearest_angle<0] = 0
        
        # Smoothing.
        nearest_angle = torch.sin(nearest_angle * torch.pi / 2)
        soft_viz_mask = torch.zeros_like(visible_mask).float()
        soft_viz_mask[visible_mask] = nearest_angle
        soft_viz_mask = 1 - soft_viz_mask

    soft_mask = render_visibility_mask(
                            viewpoint_camera, 
                            smpl_deformer, 
                            raster_settings, 
                            person_pc, 
                            smpl_param, means3D, means2D, opacity, scales, rotations, cov3D_precomp,
                            visibility_mask=soft_viz_mask)
    
    return hard_mask, soft_mask




