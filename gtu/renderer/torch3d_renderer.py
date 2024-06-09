
import os
import random
import torch
import cv2
import numpy as np
from tqdm import tqdm

from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    PointLights,
    AlphaCompositor,
    HardPhongShader,
)
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer.mesh import Textures

from utils.camera_utils import fov2focal
from utils.graphics_utils import project_points_to_cam
from utils.image_utils import img_add_text
from utils.render_utils import get_color


def gscam_to_torch3d(cam, device, zoom_scale=1.):
    
    img_size = [int(cam.image_height), int(cam.image_width)]

    fx = fov2focal(cam.FoVx, cam.image_width) * zoom_scale
    fy = fov2focal(cam.FoVy, cam.image_height) * zoom_scale

    cx = cam.cx * cam.image_width + 0.5 * img_size[1]
    cy = cam.cy * cam.image_height + 0.5 * img_size[0]


    focal_length = torch.tensor([fx, fy]).unsqueeze(0).to(device)
    principal_point = torch.tensor([cx, cy]).unsqueeze(0).to(device)


    cam_R = torch.diag(torch.tensor([1, 1, 1]))[None].float()
    cam_T = torch.zeros(3)[None].float() 
    cam_R[:, :2, :] *= -1.0
    cam_T[:, :1] *= -1.0
    cam_T[:, :2] *= -1.0
    # self.cam_R = torch.transpose(self.cam_R,1,2)
    
    cameras = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point, R=cam_R, T=cam_T, device=device, in_ndc=False, image_size=[img_size])

    return cameras


def get_opencv_cam(c2w, scale=0.4, z=1.5):
    p0 = torch.tensor([0,0,0], dtype=torch.float32).to(c2w.device)
    p1 = torch.tensor([1,1,z], dtype=torch.float32).to(c2w.device) * scale
    p2 = torch.tensor([1,-1,z], dtype=torch.float32).to(c2w.device) * scale
    p3 = torch.tensor([-1,-1,z], dtype=torch.float32).to(c2w.device) * scale
    p4 = torch.tensor([-1,1,z], dtype=torch.float32).to(c2w.device) * scale

    points = torch.stack([p0, p1, p2, p3, p4], dim=0)   # (b,3)
    points = torch.cat([points, torch.ones(5,1).float().to(points.device)], dim=-1)   # (b,4)
    world_points = torch.einsum('ij,bj->bi', c2w, points)

    return world_points



    
def draw_cams(image, cam_points, colors, font_colors, thicknesses = 2, skip_text=False):
    """Here we copy image"""
    image = image.copy()

    edges = [
        (0,1),
        (0,2),
        (0,3),
        (0,4),
        (1,2),
        (2,3),
        (3,4),
        (4,1),
    ]

    thicknesses = [thicknesses for _ in colors] if isinstance(thicknesses, int) else thicknesses

    for cam_point, color, thickness in zip(cam_points, colors, thicknesses):
        color = (color[2], color[1], color[0])  # RGB to BGR
        # draw lines
        for edge in edges:
            start_point = (int(cam_point[edge[0]][0]), int(cam_point[edge[0]][1]))
            end_point = (int(cam_point[edge[1]][0]), int(cam_point[edge[1]][1]))
            cv2.line(image, start_point, end_point, color, thickness)

    texts = ["train view", "current test view", "other test view"]
    _font_colors = []
    for color in font_colors:
        _font_colors.append((color[2], color[1], color[0]))

    image = image.copy()
    if not skip_text:
        img_add_text(image, texts, 1.5, _font_colors)

    return image






@torch.no_grad()
def render_w_torch3d(
                    viewpoint_cameras, 
                    people_infos, 
                    train_camera, 
                    test_cameras, 
                    render_camera_position: bool=True, 
                    scene_gaussians=None, 
                    render_bg_as_pc: bool=False,
                    get_smpl_alpha: bool=False,
                    zoom_scale=1.,
                    for_viz: bool=False,
                    skip_text: bool=False
                    ):
    """
    For demo sequence renderer. (to show underlying structure of people)
    """
    img_size = [viewpoint_cameras[0].image_height, viewpoint_cameras[0].image_width]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # prepare basic rendering settings
    smpl_faces = people_infos[0].smpl_deformer.smpl_server.smpl.faces
    smpl_faces = torch.tensor(smpl_faces.astype(np.int64)).to(device)
    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    if render_bg_as_pc:
        pcd_raster_settings = PointsRasterizationSettings(
            image_size=img_size, 
            radius = 0.003,
            points_per_pixel = 10
        )
        scene_pcds = scene_gaussians.get_xyz.detach().cpu().to(device)
        pcd_color = get_color(idx=3, theme='blue_grey')
        point_radius = 5.0
        
        
    # Get color_dict
    smpl_colors = dict()
    _, color_lists = get_color(idx=0, interval=1, get_color_lists=True, theme=['green', 'amber', 'indigo'])
    indices = random.sample(range(len(color_lists)), len(people_infos))

    if len(people_infos) == 6:
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
    elif True:
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

    for i, pi in zip(indices, people_infos):
        smpl_colors[pi.human_id] = torch.tensor(color_lists[i], dtype=torch.float32).to(device) / 255.
    
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
            indices = random.sample(range(len(color_lists)), len(people_infos))
            for i, pi in zip(indices, people_infos):
                smpl_colors[pi.human_id] = torch.tensor(color_lists[i], dtype=torch.float32).to(device) / 255.
                    


        train_cam = get_opencv_cam(train_camera.world_view_transform.detach().T.inverse().to(device))
        all_cams = [train_cam]
        test_cam_dict = dict()
        for i in range(len(test_cameras)):
            test_cam_dict[i] = get_opencv_cam(test_cameras[i].world_view_transform.detach().T.inverse().to(device))
            all_cams.append(test_cam_dict[i])
            result_imgs[i] = dict()
        all_cams = torch.stack(all_cams)
        all_cams = all_cams.reshape(-1, 4)[:, :3]                # (B, 5, 3+1)


    # Now start rendering!!!
    for i, cam in tqdm(enumerate(viewpoint_cameras), desc="Torch3D render for visualize"):
        # 1. Define Renderer
        torch3d_camera = gscam_to_torch3d(cam, device, zoom_scale)

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
        data_idx = cam.colmap_id
        for person_info in people_infos:
            if data_idx not in person_info.fids:
                continue
            _data_idx = person_info.fids.index(data_idx)
            beta = person_info.beta

            smpl_param = torch.cat([
                person_info.smpl_scale.reshape(-1),
                person_info.smpl_global_poses[_data_idx],
                person_info.smpl_local_poses[_data_idx],
                beta
            ], dim=-1)
            smpl_param = smpl_param.unsqueeze(0)

            smpl_output = person_info.smpl_deformer.smpl_server(smpl_param)
            smpl_verts = smpl_output['smpl_verts'].detach()
            smpl_verts_dict[person_info.human_id] = smpl_verts.squeeze()
        
        # Here simply transform Meshes
        w2c = cam.world_view_transform.clone().detach().T.to(device)

        if len(smpl_verts_dict) == 0:
            print(f"{data_idx} fid not has person")
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
            pcds = scene_pcds.detach()
            pcds = torch.cat([pcds, torch.ones_like(pcds[...,0:1])], axis=-1)
            pcds = torch.einsum('ij, bj -> bi', w2c, pcds)
            pcds = pcds[...,:3] / (pcds[..., 3:] + 1e-9)

            rgbs = torch.tensor(pcd_color).to(device).unsqueeze(0).repeat(len(pcds), 1).float() / 255.
            pcds = Pointclouds(points=[pcds], features=[rgbs])
            images_pc = pc_renderer(pcds, point_size=point_radius)

            # place SMPL in front of mesh
            final_image = images_pc * (1-smpl_images[0][...,3:]) + smpl_images[0][...,:3] * smpl_images[0][...,3:]
        else:
            final_image = smpl_images[0][...,:3] * smpl_images[0][...,3:] + (1-smpl_images[0][...,3:]) * torch.ones_like(smpl_images[0][...,:3])
            smpl_alphas[data_idx] = (smpl_images[0][...,3:] * 255).squeeze().detach().cpu().numpy().astype(np.uint8)


        final_image = final_image.detach().cpu().squeeze().numpy()
        final_image[final_image>1] = 1
        final_image = (final_image * 255).astype(np.uint8)
        final_image[...,[2,1,0]]

        # Now render cameras 
        if render_camera_position:
            # Project to image space
            font_colors = (train_cam_color, test_main_color, test_others_color)
            pj_jnts = project_points_to_cam(cam, all_cams.detach().cpu().numpy())
            pj_jnts = pj_jnts.reshape(-1, 5, 2)

            for i in range(len(test_cameras)):
                colors = [train_cam_color] + [test_others_color for _ in range(len(test_cameras))]
                thickness = 3
                thicknesses = [thickness for _ in colors]
                thicknesses[i+1] = thickness*2 
                colors[i+1] = test_main_color
                result_imgs[i][data_idx] = draw_cams(final_image, pj_jnts, colors, font_colors, thicknesses=thicknesses, skip_text=skip_text)
        else:
            result_imgs[data_idx] = final_image

    if get_smpl_alpha:
        return result_imgs, smpl_alphas
    
    return result_imgs









@torch.no_grad()
def render_emf_on_smpl(person_info, save_dir, n_frames=20):
    img_size = [1024, 1024]
    device = person_info.beta.device
    save_dir.mkdir(exist_ok=True)

    # 0. get EMFs & conver to RGBs
    emfs = person_info.gaussians.smpl_emfs   
    emfs_numpy = emfs.clone().detach().cpu().numpy()


    color_for_0 = np.array([255, 0, 0])  # Red
    color_for_pi = np.array([0, 0, 255])  # Blue
    grey_color = np.array([127, 127, 127])  # Grey 
    
        # Initialize an array for RGB colors, defaulting to grey for NaNs initially
    rgb_colors_custom = np.full((len(emfs), 3), grey_color)

        # Linearly interpolate colors for values between 0 and pi (excluding NaNs)
    for i, val in enumerate(emfs_numpy):
        if val >= 0:
            fraction = val / np.pi
            rgb_colors_custom[i] = (1 - fraction) * color_for_0 + fraction * color_for_pi

        # Convert BGR to RGB
    render_rgbs = rgb_colors_custom[:, [2, 1, 0]]
    render_rgbs = torch.from_numpy(render_rgbs).float().to(device) / 255.
    render_rgbs = render_rgbs.unsqueeze(0)

    # 1. prepare SMPL mesh
    beta = person_info.beta
    smpl_param = torch.zeros(1,86).to(device)
    smpl_param[0,-10:] = beta
    smpl_param[0,0] = 1
    smpl_param[0,2] = 0.3
    smpl_param[0, 9] = torch.pi / 6
    smpl_param[0, 12] = -torch.pi / 6
    smpl_param[0, 9 + 3*12] = -torch.pi / 6
    smpl_param[0, 9 + 3*13] = torch.pi / 6

    smpl_output = person_info.smpl_deformer.smpl_server(smpl_param)
    smpl_verts = smpl_output['smpl_verts'].detach()
    smpl_faces = person_info.smpl_deformer.smpl_server.smpl.faces
    smpl_faces = torch.tensor(smpl_faces.astype(np.int64)).to(device).squeeze()[None]

    mesh = Meshes(verts=smpl_verts, faces=smpl_faces, textures=Textures(verts_rgb=render_rgbs))
    lights = PointLights(location=[[0.0, 0.0, 3.0]], device=device)
    cameras = OpenGLPerspectiveCameras(device=device)
    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    mesh_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras)  # Use the default shader
    )



    # Start rendering
    distance = 2.0
    elevation = 0
    viewpoints = []
    for i in range(n_frames):
        angle = i * 360 / n_frames
        azimuth = angle - 360 if angle > 180 else angle


        # Generate the view transform.
        R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth, device=device)
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
        
        # Update the renderer's camera
        mesh_renderer.rasterizer.cameras = cameras
        
        # Render the mesh from the current viewpoint.
        image = mesh_renderer(mesh)
        
        # Convert the image to CPU and numpy for visualization or saving.
        image = image.cpu().numpy()[0] * 255

        # Save
        save_fname = str(save_dir / f"{i:09}.png")
        cv2.imwrite(save_fname, image)

