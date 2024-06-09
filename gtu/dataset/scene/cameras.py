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
from torch import nn
from PIL import Image

from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch



class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, cx, cy, image, gt_alpha_mask, 
                 image_name, uid, depth=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 smpl_param=None,
                 zfar = 100.0, znear=0.01, image_width=None, image_height=None,
                 img_fname=None, mask_fname=None,
                 occ_mask=None, occ_mask_fname=None,
                 bbox=None,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id      # aka. fid
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.smpl_param = smpl_param
        
        self.img_fname = img_fname
        self.mask_fname = mask_fname
        self.occ_mask_fname = occ_mask_fname
        self.bbox = bbox

        self.cx = cx
        self.cy = cy

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if image is not None:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        else:
            self.original_image = None
            
        self.image_width = self.original_image.shape[2] if image_width is None else image_width
        self.image_height = self.original_image.shape[1] if image_height is None else image_height

        # No need to have masked image as gt_alpha, since we have mask!! 
        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            # self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None

        # No need to have masked image as gt_alpha, since we have mask!! 
        if occ_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.occ_mask = occ_mask.to(self.data_device)
        else:
            # self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.occ_mask = None


        if depth is not None:
            self.estimated_depth = torch.Tensor(depth).to(self.data_device)
        else:
            self.estimated_depth = depth
            
        if smpl_param is not None:
            self.smpl_param = torch.from_numpy(self.smpl_param).to(self.data_device)
            self.smpl_param = self.smpl_param.squeeze().reshape(1, -1)  # change in batched form

        self.zfar = zfar
        self.znear = znear

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=cx, cy=cy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        
    def get_gt_image(self):
        if not (self.original_image is None):
            gt_image = self.original_image
        else:
            # read image
            resolution = (self.image_width, self.image_height)
            gt_image = Image.open(self.img_fname)
            gt_image = PILtoTorch(gt_image, resolution)[:3, ...]
            gt_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        
        return gt_image
            

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

