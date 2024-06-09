
import os
import random
import torch
import cv2
import numpy as np
from tqdm import tqdm

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

from utils.camera_utils import fov2focal
from utils.render_utils import draw_cams, get_opencv_cam, get_color


def camdict_to_torch3d(camdict, device, zoom_scale=1.):
    img_size = [int(camdict['H']), int(camdict['W'])]

    if 'f' in camdict:
        fx = camdict['f'] * zoom_scale
        fy = camdict['f'] * zoom_scale
    else:
        fx = camdict['fx'] * zoom_scale
        fy = camdict['fy'] * zoom_scale

    cx = camdict['cx']
    cy = camdict['cy']


    focal_length = torch.tensor([fx, fy]).unsqueeze(0).to(device).float()
    principal_point = torch.tensor([cx, cy]).unsqueeze(0).to(device).float()


    cam_R = torch.diag(torch.tensor([1, 1, 1]))[None].float()
    cam_T = torch.zeros(3)[None].float() 
    cam_R[:, :2, :] *= -1.0
    cam_T[:, :1] *= -1.0
    cam_T[:, :2] *= -1.0
    # self.cam_R = torch.transpose(self.cam_R,1,2)
    
    cameras = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point, R=cam_R, T=cam_T, device=device, in_ndc=False, image_size=[img_size])

    return cameras


@torch.no_grad()
def render_w_torch3d(
                    render_camdicts, 
                    people_dict, 
                    smpl_server,
                    capture_camera_dicts = dict(), 
                    render_camera_position: bool=True, 
                    bg_xyzs: torch.Tensor=None, 
                    pcd_color: torch.Tensor=None,
                    render_bg_as_pc: bool=False,
                    get_smpl_alpha: bool=False,
                    zoom_scale=1.,
                    for_viz: bool=False,
                    skip_text: bool=False
                    ):
    """
    For demo sequence renderer. (to show underlying structure of people)
    """
    _first_camera = list(render_camdicts.values())[0]
    img_size = [_first_camera['H'], _first_camera['W']]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # prepare basic rendering settings
    smpl_faces = smpl_server.smpl.faces
    smpl_faces = torch.tensor(smpl_faces.astype(np.int64)).to(device)
    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    if render_bg_as_pc:
        point_radius = 5.0
        pcd_raster_settings = PointsRasterizationSettings(
            image_size=img_size, 
            radius = 0.003,
            points_per_pixel = 10
        )
        scene_pcds = bg_xyzs.detach().cpu().to(device)
        if pcd_color is None:
            pcd_color = get_color(idx=3, theme='blue_grey')
            pcd_color = torch.tensor(pcd_color).to(device).unsqueeze(0).repeat(len(scene_pcds), 1).float() / 255.
        else:
            pcd_color = pcd_color.detach().cpu().to(device).float()
            if pcd_color.max() > 1:
                pcd_color /= 255.
        
        
    # Get color_dict
    smpl_colors = dict()
    _, color_lists = get_color(idx=0, interval=1, get_color_lists=True, theme=['green', 'amber', 'indigo'])
    
    while (len(color_lists) < len(people_dict)):
        color_lists = color_lists + color_lists
    indices = random.sample(range(len(color_lists)), len(people_dict))
    
    
    if False:
        # This part is for custom color setting 
        if len(people_dict) == 6:
            print("Assuming Panoptic, use predefined colors sets")

            # color_lists = []
            color_lists = [
                [253, 205, 75],
                [150, 124, 116],
                [134, 154, 164],
                [90, 47, 50],
                [106, 104, 94],
                [170, 148, 213]
            ]
            color_lists =[ c[::-1] for c in color_lists]
            indices = list(range(6))
        elif False:
            ### rendering for demo
            print("using predefined colors")
            color_lists = [
                # [26, 132, 193],
                # [132, 178, 193]
                [95, 107, 186],
                [72, 146, 243]
            ]
            color_lists =[ c[::-1] for c in color_lists]
            indices = [0, 1]

    for i, pid in zip(indices, people_dict.keys()):
        smpl_colors[pid] = torch.tensor(color_lists[i], dtype=torch.float32).to(device) / 255.
    
    # Get camera colors
    smpl_alphas = dict()
    result_imgs = dict()
    if render_camera_position:
        train_cam_color = get_color(idx=6, theme='red')
        test_main_color = get_color(idx=9, theme='deep_purple')
        test_others_color = get_color(idx=4, theme='lime')
        if for_viz:
            test_others_color = get_color(idx=4, theme='green')
            
            smpl_colors = dict()
            _, color_lists = get_color(idx=0, interval=1, get_color_lists=True, theme=['brown', 'grey', 'blue_grey'])
            indices = random.sample(range(len(color_lists)), len(people_dict))
            for i, pi in zip(indices, people_infos):
                smpl_colors[pi.human_id] = torch.tensor(color_lists[i], dtype=torch.float32).to(device) / 255.
                    

        all_cams = []
        test_cam_dict = dict()
        for i, cap_cam_fid in enumerate(list(capture_camera_dicts.keys())):
            test_cam_dict[cap_cam_fid] = get_opencv_cam(torch.from_numpy(capture_camera_dicts[cap_cam_fid]['w2c']).inverse().to(device))
            all_cams.append(test_cam_dict[cap_cam_fid])

        all_cams = torch.stack(all_cams)
        all_cams = all_cams.reshape(-1, 4)[:, :3]                # (B, 5, 3+1)


    # Now start rendering!!!
    for i, render_fid in enumerate(tqdm(sorted(list(render_camdicts.keys())), desc="Torch3D render for visualize")):
        # 1. Define Renderer
        camdict = render_camdicts[render_fid]
        torch3d_camera = camdict_to_torch3d(camdict, device, zoom_scale)

        mesh_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=torch3d_camera, raster_settings=raster_settings),
            shader=HardPhongShader(device=device, cameras=torch3d_camera)  # Use the default shader
        )
        if render_bg_as_pc:
            pc_renderer = PointsRenderer(
                rasterizer=PointsRasterizer(cameras=torch3d_camera, raster_settings=pcd_raster_settings),
                compositor=AlphaCompositor(background_color=torch.ones(3).float().to(device))
            )

        # 2. Set SMPL vertices
        smpl_verts_dict = dict()
        fid = camdict['fid']
        for pid, person_dict in people_dict.items():
            if fid not in person_dict:
                continue

            if person_dict[fid]['smpl_param'] is None:
                continue

            smpl_param = torch.from_numpy(person_dict[fid]['smpl_param']).squeeze().float()
            smpl_param = smpl_param.unsqueeze(0).to(device)

            smpl_output = smpl_server(smpl_param)
            smpl_verts = smpl_output['smpl_verts'].detach()
            smpl_verts_dict[pid] = smpl_verts.squeeze()
        
        # Here simply transform Meshes
        w2c = torch.from_numpy(camdict['w2c']).to(device).float()

        if len(smpl_verts_dict) == 0:
            smpl_images = torch.zeros(img_size).float().cuda()[None].unsqueeze(-1).repeat(1,1,1,4)
        else:
            render_faces = []
            render_verts = []
            render_rgbs = []
            for pid, v in smpl_verts_dict.items():
                v = torch.cat([v, torch.ones_like(v[...,0:1])], axis=-1)
                v = torch.einsum('ij, bj -> bi', w2c, v)
                v = v[...,:3] / (v[..., 3:] + 1e-9)

                render_faces.append(smpl_faces)
                render_verts.append(v)
                render_rgbs.append(smpl_colors[pid].reshape(1, -1).repeat(len(v), 1))

            mesh = Meshes(verts=render_verts, faces=render_faces, textures=Textures(verts_rgb=render_rgbs))
            mesh = join_meshes_as_scene(mesh, True)
            smpl_images = mesh_renderer(meshes_world=mesh)

        if render_bg_as_pc:
            pcds = scene_pcds.clone().detach()
            pcds = torch.cat([pcds, torch.ones_like(pcds[...,0:1])], axis=-1)
            pcds = torch.einsum('ij, bj -> bi', w2c, pcds)
            pcds = pcds[...,:3] / (pcds[..., 3:] + 1e-9)

            rgbs = pcd_color
            pcds = Pointclouds(points=[pcds], features=[rgbs])
            images_pc = pc_renderer(pcds, point_size=point_radius)

            # place SMPL in front of mesh
            final_image = images_pc * (1-smpl_images[0][...,3:]) + smpl_images[0][...,:3] * smpl_images[0][...,3:]
        else:
            final_image = smpl_images[0][...,:3] * smpl_images[0][...,3:] + (1-smpl_images[0][...,3:]) * torch.ones_like(smpl_images[0][...,:3])
            smpl_alphas[fid] = (smpl_images[0][...,3:] * 255).squeeze().detach().cpu().numpy().astype(np.uint8)


        final_image = final_image.detach().cpu().squeeze().numpy()
        final_image[final_image>1] = 1
        final_image = (final_image * 255).astype(np.uint8)
        final_image[...,[2,1,0]]

        # Now render cameras 
        if render_camera_position:
            # Project to image space
            font_colors = (train_cam_color, test_main_color, test_others_color)
            all_cams = all_cams.detach().cpu().numpy().reshape(-1, 3)
            all_cams = np.concatenate([all_cams, np.ones_like(all_cams[:, :1])], axis=-1)
            pj_jnts = np.einsum('ij,bj->bi', camdict['projection'], all_cams) 
            pj_jnts = pj_jnts[:, :2] / (pj_jnts[:, 2:3] + 1e-9)
            pj_jnts = pj_jnts.reshape(-1, 5, 2)

            
            colors = [train_cam_color] + [test_others_color for _ in range(len(test_cameras))]
            thickness = 3
            thicknesses = [thickness for _ in colors]
            thicknesses[i+1] = thickness*2 
            colors[i+1] = test_main_color
            result_imgs[fid] = draw_cams(final_image, pj_jnts, colors, font_colors, thicknesses=thicknesses, skip_text=skip_text)
        else:
            result_imgs[fid] = final_image

    if get_smpl_alpha:
        return result_imgs, smpl_alphas
    
    return result_imgs







