# conda activate 4d_humans_tracker
import os
import sys
DEPTH_ANYTHING_PATH="/home/inhee/VCL/repos_2024/Depth-Anything"
sys.path.append(DEPTH_ANYTHING_PATH)
from pathlib import Path

from depth_anything.dpt import DPT_DINOv2
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

import argparse
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Compose
from tqdm import tqdm, trange

from math import floor
import math
from utils.io_utils import write_pickle


def load_metric_depth_anything(is_indoor=False):
    METRIC_DEPTH_ANYTHING_PATH="/home/inhee/VCL/repos_2024/Depth-Anything/metric_depth"
    sys.path.append(METRIC_DEPTH_ANYTHING_PATH)
    from zoedepth.utils.easydict import EasyDict as edict
    from zoedepth.data.data_mono import DepthDataLoader
    from zoedepth.models.builder import build_model
    from zoedepth.utils.arg_utils import parse_unknown
    from zoedepth.utils.config import change_dataset, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
    from zoedepth.utils.misc import (RunningAverageDict, colors, compute_metrics, count_parameters)

    if is_indoor:
        dataset_name = 'nyu'
        pretrained_data_path = "local::/home/inhee/VCL/repos_2024/Depth-Anything/checkpoints/depth_anything_metric_depth_indoor.pt"
    else:
        dataset_name = 'kitti'
        pretrained_data_path = "local::/home/inhee/VCL/repos_2024/Depth-Anything/checkpoints/depth_anything_metric_depth_outdoor.pt"

    config = get_config('zoedepth', "eval", dataset_name, **{'pretrained_resource': pretrained_data_path})
    model = build_model(config)
    model = model.cuda()
    model = model.eval()


    return model


@torch.no_grad()
def metric_depth_infer(model, images, **kwargs):
    """Inference with flip augmentation"""
    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    pred1 = model(images, **kwargs)
    pred1 = get_depth_from_prediction(pred1)

    pred2 = model(torch.flip(images, [3]), **kwargs)
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2, [3])

    mean_pred = 0.5 * (pred1 + pred2)

    return mean_pred



def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


margin_width = 50
caption_height = 60

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2


@torch.no_grad()
def render_pc_w_depth(xyzs, p3d_ids, cam_dict):
    from pytorch3d.renderer import (
        PerspectiveCameras,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        HardPhongShader,
        PointsRasterizationSettings,
        PointsRasterizer,
        PointsRenderer,
        AlphaCompositor
    )
    from pytorch3d.structures.meshes import join_meshes_as_scene
    from pytorch3d.structures import Meshes, Pointclouds
    from pytorch3d.renderer.mesh import Textures

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fids = sorted(list(cam_dict.keys()))
    img_size = [cam_dict[fids[0]]['height'], cam_dict[fids[0]]['width']]
    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    pcd_raster_settings = PointsRasterizationSettings(
        image_size=img_size, 
        radius = 0.05,
        points_per_pixel = 10
    )

    if len(p3d_ids.shape) == 2:
        p3d_ids = p3d_ids[...,0]

    scene_pcds = torch.from_numpy(xyzs).squeeze().float().to(device)

    res_dict = dict()
    for fid, cam in tqdm(cam_dict.items()):
        fx = cam['focal_length_x']
        fy = cam['focal_length_y']

        cx = cam['cx']
        cy = cam['cy']

        valid_p3d_ids = cam['point_3d_ids']
        valid_indices = [np.where(p3d_ids == query)[0][0] for query in valid_p3d_ids if query in p3d_ids]
        valid_pcds = scene_pcds[valid_indices]
        print(f"queried n indices: {len(valid_p3d_ids)}, existing 3D points: {len(valid_pcds)}")

        focal_length = torch.tensor([fx, fy]).unsqueeze(0).to(device).float()
        principal_point = torch.tensor([cx, cy]).unsqueeze(0).to(device).float()

        cam_R = torch.diag(torch.tensor([1, 1, 1]))[None].float()
        cam_T = torch.zeros(3)[None].float() 
        cam_R[:, :2, :] *= -1.0
        cam_T[:, :1] *= -1.0
        cam_T[:, :2] *= -1.0


        R = torch.from_numpy(cam['R']).float().to(device)
        T = torch.from_numpy(cam['T']).float().to(device)

        # Define Point-clouds in camera-space
        pcds = valid_pcds.clone()
        pcds = torch.einsum('ij, bj -> bi', R, pcds) + T.unsqueeze(0)
        rgbs = torch.tensor([0.7, 0.7, 0.7]).to(device).unsqueeze(0).repeat(len(pcds), 1).float()
        pcds = Pointclouds(points=[pcds], features=[rgbs])

        # Define rendering systems
        cameras = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point, R=cam_R, T=cam_T, device=device, in_ndc=False, image_size=[img_size])
        pc_rasterizer = PointsRasterizer(cameras=cameras, raster_settings=pcd_raster_settings)
        pc_renderer = PointsRenderer(
            rasterizer=pc_rasterizer,
            compositor=AlphaCompositor()
        )

        # Render Point Clouds
        pc_frags = pc_rasterizer(pcds)
        zbufs = pc_frags.zbuf[0].detach()
        depth = zbufs[..., 0]

        images_pc = pc_renderer(pcds)[0]

        rendered_pc = images_pc.detach().cpu().squeeze().numpy()
        rendered_pc[rendered_pc>1] = 1
        rendered_pc = (rendered_pc * 255).astype(np.uint8)[...,::-1]
        
        res_dict[fid] = dict(
            depth=depth.detach().cpu(),
            rendering=rendered_pc   # OpenCV format
        )
    return res_dict


def depth_to_opencv(depth, render_max_depth=None):
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()

    # if not is_disparity:
    #     zero_depth = (depth < 1e-8)
    #     depth = 1 / (depth + 1e-8)
    #     depth[zero_depth] = 0

    min_depth = 0               # depth.min()
    max_depth = depth.max()

    if not (render_max_depth is None):
        print(f'fixed max depth as {render_max_depth}')
        depth[depth>render_max_depth] = render_max_depth
        depth = (depth - min_depth) / render_max_depth * 255.0
    else:
        depth = (depth - min_depth) / (depth.max() - min_depth) * 255.0
    depth = depth.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

    return depth_color, max_depth



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/mnt/hdd2/230907_set1")
    parser.add_argument('--encoder', type=str, default='vitl')
    parser.add_argument('--localhub', dest='localhub', action='store_true', default=False)
    args = parser.parse_args()
    

    ##### Depth-Anything settings
    assert args.encoder in ['vits', 'vitb', 'vitl']
    if args.encoder == 'vits':
        depth_anything = DPT_DINOv2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], localhub=args.localhub).cuda()
    elif args.encoder == 'vitb':
        depth_anything = DPT_DINOv2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768], localhub=args.localhub).cuda()
    else:
        depth_anything = DPT_DINOv2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], localhub=args.localhub).cuda()
    
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    

    depth_anything.load_state_dict(torch.load(Path(DEPTH_ANYTHING_PATH) / 'checkpoints/depth_anything_vitl14.pth'))
    # depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'), strict=True)
    depth_anything.eval()

    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])



    # Now list up directories.
    input_dir = Path(args.data_path)
    image_dir = input_dir / 'images'

    # Saving directories
    depth_viz_dir = input_dir / 'mono_depth_viz'
    depth_viz_dir.mkdir(exist_ok=True)

    ## make img fname dict
    img_fname_dict = dict()
    for img_fname in sorted(list(image_dir.glob("*.png"))):
        fid = int(img_fname.name.split(".")[0])
        img_fname_dict[fid] = str(img_fname)


    depth_res_dict = dict()
    for fid in tqdm(sorted(list(img_fname_dict.keys())), desc="DepthAnything"):
        raw_image = cv2.imread(img_fname_dict[fid])
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        ## Estimate depth 
        h, w = image.shape[:2]
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).cuda()
        
        with torch.no_grad():
            depth = depth_anything(image)
            
            # in fact, it's disparity. (so near has high, far has low value)
            depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
            depth_res_dict[fid] = depth.clone().detach().cpu().numpy()
            
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.cpu().numpy().astype(np.uint8)
            depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            depth_mono = np.stack([depth, depth, depth], axis=-1)
            
            save_img = np.concatenate([raw_image, depth_color, depth_mono], axis=1)
            cv2.imwrite(str(depth_viz_dir / f"{fid:09}.jpg"), save_img)
    
    
    # Save in depth
    write_pickle(input_dir/'mono_depth.pkl', depth_res_dict)