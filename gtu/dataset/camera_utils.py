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


from typing import Dict, Any, Optional

import json
import torch
import numpy as np
import math
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import List
from utils.general_utils import PILtoTorch, quaternion_rotation_y, quaternion_rotation_x, quaternion_rotation_z
from utils.graphics_utils import fov2focal, focal2fov, rotation_matrix_from_vectors
from gtu.dataset.scene.colmap_loader import qvec2rotmat
from gtu.dataset.scene.cameras import Camera

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, data_device, _resolution, is_aux=False, preload_masks=False, dilate_human_occ_masks_ratio: float=0.0):
    if _resolution is None:
        _resolution = args.resolution
    
    if data_device is None:
        data_device = args.data_device
    
    orig_w, orig_h = cam_info.width, cam_info.height

    if _resolution in [1, 2, 4, 8]:
        scale = (resolution_scale * _resolution)
        resolution = round(orig_w/scale), round(orig_h/scale)
    else:  # should be a type that converts to float
        if _resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / _resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    bbox = cam_info.bbox

    # load image
    if cam_info.image is None:
        gt_image = None
        resized_image_rgb = None
        loaded_mask = None
        occ_mask = None
    else:
        if not (bbox is None):   
            bb = bbox
            _img = cam_info.image.crop((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))
        else:
            _img = cam_info.image
        resized_image_rgb = PILtoTorch(_img, resolution)      # C, H, W: 0~1 value range
        gt_image = resized_image_rgb[:3, ...]
        loaded_mask = None
        occ_mask = None

    # load mask
    if (not resized_image_rgb is None) and resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
        
    elif not cam_info.mask is None:
        if not (bbox is None):   
            bb = bbox
            _mask = cam_info.mask.crop((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))
        else:
            _mask = cam_info.mask
        loaded_mask = PILtoTorch(_mask, resolution)
        if loaded_mask.shape[0] == 3:
            loaded_mask = (loaded_mask.sum(0) > 0).float()
        
    elif (not cam_info.mask_path is None) and preload_masks:
        loaded_mask = Image.open(cam_info.mask_path)
        if not (cam_info.bbox is None):   
            bb = bbox
            _mask = loaded_mask.crop((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))
        else:
            _mask = loaded_mask
        loaded_mask = PILtoTorch(_mask, resolution)

    if getattr(cam_info, 'flip_mask', False) and (not loaded_mask is None):
        loaded_mask = 1 - loaded_mask

    # load occ_mask
    if not (cam_info.occ_mask is None):
        occ_mask = cam_info.occ_mask
    elif (not cam_info.occ_mask_path is None) and preload_masks:
        occ_mask = Image.open(cam_info.occ_mask_path)
        
    if not (occ_mask is None):
        if dilate_human_occ_masks_ratio > 0:
            from utils.mask_utils import dilate_mask 
            occ_mask = np.asarray(occ_mask)
            dilate_kernel_size = int((resolution[0] + resolution[1]) * dilate_human_occ_masks_ratio)
            occ_mask = dilate_mask(occ_mask, kernel_size = dilate_kernel_size)
            
            print(f"dilate_kernel_size: {dilate_kernel_size}")
            occ_mask = Image.fromarray(occ_mask)

        if not (bbox is None):   
            bb = bbox
            occ_mask = occ_mask.crop((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))
        else:
            occ_mask = occ_mask
        occ_mask = PILtoTorch(occ_mask, resolution)


    loaded_depth = None
    if cam_info.depth is not None:  
        loaded_depth = torch.Tensor(cam_info.depth).float()


    smpl_param = None
    if not cam_info.smpl_param is None:
        smpl_param = cam_info.smpl_param

    # As also GT image is loaded with resized shape, we need resize bbox
    if not (bbox is None):   
        bbox = [int(bb / scale) for bb in bbox]
    else:
        bbox = None
        
    if preload_masks and (loaded_mask is None):
        print(f"[WARNING] mask of frame {int(cam_info.fid):08} is not loaded! check the data file in {str(cam_info.mask_path)}")


    return Camera(colmap_id=cam_info.fid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  cx=cam_info.cx, cy=cam_info.cy,
                  image=gt_image, gt_alpha_mask=loaded_mask, depth=loaded_depth,
                  image_name=cam_info.image_name, uid=cam_info.uid, data_device=data_device, smpl_param=smpl_param, 
                  image_width=resolution[0], image_height=resolution[1], img_fname=cam_info.image_path, mask_fname=cam_info.mask_path,
                  occ_mask=occ_mask, occ_mask_fname=cam_info.occ_mask_path, bbox=bbox)

def cameraList_from_camInfos(
        cam_infos, 
        resolution_scale, 
        args=None, 
        data_device=None, 
        resolution=None, 
        is_aux=False, 
        preload_masks=False,
        dilate_human_occ_masks_ratio: float=0.0):
    camera_list = []

    for id, c in enumerate(tqdm(cam_infos)):
        camera_list.append(loadCam(
                                    args, 
                                    id, 
                                    c, 
                                    resolution_scale, 
                                    data_device, 
                                    resolution, 
                                    is_aux=is_aux, 
                                    preload_masks=preload_masks,
                                    dilate_human_occ_masks_ratio=dilate_human_occ_masks_ratio
                                    )
                           )

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry



def gen_canon_cam(n_cameras: int=36, t=3, f=700, res=512, device=torch.device("cuda"), rot_x=False, rot_z=False):
    fov = focal2fov(f, res)

    # Generate quaternion rotations
    angles_deg = np.arange(0, 361, 360//n_cameras)

    if rot_x:
        quaternions = [quaternion_rotation_x(angle) for angle in angles_deg]
    elif rot_z:
        quaternions = [quaternion_rotation_z(angle) for angle in angles_deg]
    else:
        quaternions = [quaternion_rotation_y(angle) for angle in angles_deg]


    org_image = torch.zeros(3, res, res)

    # Print the generated quaternions
    uid = 0
    cameras = []
    for angle, quat in zip(angles_deg, quaternions):
        # print(f"Angle: {angle} degrees, Quaternion: {quat}")
        R = qvec2rotmat(quat)

        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = np.array([0, 0.3, t])      # I shift 0.3 up in depth direction
        c2w = np.linalg.inv(w2c)

        c2w[0:3, 0] *= -1
        c2w[0:3, 1] *= -1    
        
        ext = np.linalg.inv(c2w)
        T = ext[:3, 3]
        R = ext[:3, :3].T           ##### It should be transposed!


        cam = Camera(colmap_id=uid, R=R, T=T, 
                  FoVx=fov, FoVy=fov, 
                  cx=0., cy=0.,
                  image=org_image, gt_alpha_mask=None, depth=None,
                  image_name=None, uid=0, data_device=device, smpl_param=None)

        uid += 1
        cameras.append(cam)

    return cameras



def gen_perturbed_camera(view, n_cameras: int=36, radius: float=0.01):
    angles_deg = np.arange(0, 361, 360//n_cameras)

    # get c2w
    c2w_R = view.R
    w2c_T = view.T

    # get cam_center
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = c2w_R.transpose()
    Rt[:3, 3] = w2c_T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    uid = view.uid
    colmap_id = view.colmap_id
    R = C2W[:3,:3]
    fovx = view.FoVx
    fovy = view.FoVy
    cx = view.cx
    cy = view.cy
    device = torch.device
    org_image = view.original_image
    data_device = view.data_device


    perturb_cams = []
    for angle in angles_deg:
        x = np.cos(angle*np.pi/180) * radius
        y = np.sin(angle*np.pi/180) * radius

        # New camera center in camer space
        new_center = np.array([x, y, 0])
        new_c2w = C2W.copy()
        new_c2w[:3, 3] += C2W[:3,:3] @ new_center

        T = np.linalg.inv(new_c2w)[:3, 3]
        
        
        cam = Camera(colmap_id=colmap_id, R=R, T=T, 
                  FoVx=fovx, FoVy=fovy, 
                  cx=cx, cy=cy,
                  image=org_image, gt_alpha_mask=None, depth=None,
                  image_name=None, uid=uid, data_device=data_device, smpl_param=None)

        perturb_cams.append(cam)
        
    return perturb_cams

        



### Part for rendering novel views (render trajectory)

def three_js_perspective_camera_focal_length(fov: float, image_height: int):
    """Returns the focal length of a three.js perspective camera.

    Args:
        fov: the field of view of the camera in degrees.
        image_height: the height of the image in pixels.
    """
    if fov is None:
        print("Warning: fov is None, using default value")
        return 50
    pp_h = image_height / 2.0
    focal_length = pp_h / np.tan(fov * (np.pi / 180.0) / 2.0)
    return focal_length

def load_ns_traj_from_json(traj_path: Path, device=torch.device("cuda"), transform_path: Optional[Path]=None):
    """Takes a camera path dictionary and returns a trajectory as a Camera instance.

    Args:
        camera_path: A dictionary of the camera path information coming from the viewer.

    Returns:
        A Cameras instance with the camera path.
    """
    
    with open(traj_path, "r", encoding="utf-8") as f:
        camera_path = json.load(f)
        
        
    # get applied transform
    if transform_path is not None:
        print(f"Load transform from {str(transform_path)}")
        with open(transform_path, "r", encoding="utf-8") as f:
            applied_transform = json.load(f)['transform']
            applied_transform = np.array(applied_transform)

    elif (traj_path.parents[1] / "dataparser_transforms.json").exists():
        with open(traj_path.parents[1] / "dataparser_transforms.json", "r", encoding="utf-8") as f:
            applied_transform = json.load(f)['transform']
            applied_transform = np.array(applied_transform)
    else:
        applied_transform = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0]], dtype=np.float32)
    applied_transform = np.concatenate([applied_transform, np.array([[0, 0, 0, 1]])], axis=0, dtype=np.float32)
    
    image_height = camera_path["render_height"]
    image_width = camera_path["render_width"]

    if "camera_type" not in camera_path:
        camera_type = "perspective"
    elif camera_path["camera_type"] != "perspective":
        raise NotImplementedError(f"{camera_type}: is not prepared yet")
        
    w2cs = []
    fovs = []
    for camera in camera_path["camera_path"]:
        # pose
        c2w = torch.tensor(camera["camera_to_world"]).view(4, 4).numpy()
        
        # First cancel out applied transfomrs
        c2w = np.linalg.inv(applied_transform) @ c2w
        c2w[0:3, 1:3] *= -1
        
        # 1. it's OpenGL camera coordinates
        # camtoworlds_opengl = c2w
        # camtoworlds_opencv = camtoworlds_opengl @ np.diag([1, -1, -1, 1])
        # camtoworlds_opencv = camtoworlds_opencv[[1,2,0,3]]
        # c2w = camtoworlds_opencv
        
        # c2w[0:2, :] *= -1
        # c2w = c2w[np.array([1, 0, 2, 3]), :]    # first convert x-y order
        # c2w = c2w[np.array([0, 2, 1, 3]), :]    # first convert y-z (idk why)
        # c2w[0:3, 0] *= -1
        # c2w[0:3, 1] *= -1
        
        # c2w[0:3, 1:3] *= -1
        
        print(c2w[:3, 3])
        
        # 2. invert back to w2c coordinates.
        w2c = np.linalg.inv(c2w)
        w2cs.append(w2c)
        
        # field of view
        fov = camera["fov"] * math.pi / 180
        fovs.append(fov)
    
    org_image = torch.zeros(3, image_height, image_width)
    
    # change it to cameras
    cameras = []
    for i, w2c in enumerate(w2cs):
        fov = fovs[i]
        
        focal = fov2focal(fov, image_height)
        fovy = focal2fov(focal, image_height)
        fovx = focal2fov(focal, image_width)
        
        R = w2c[:3, :3].T
        T = w2c[:3, 3]
    
        cam = Camera(colmap_id=i, R=R, T=T, 
                    FoVx=fovx, FoVy=fovy, 
                    cx=0, cy=0, 
                    image=org_image, gt_alpha_mask=None,
                    image_name=None, uid=i, data_device=device, smpl_param=None)
        
        cameras.append(cam)
    
    return cameras
    

def get_up_vec(cameras, up_vec=np.array([0, -1, 0])):
    world_up_vecs = []
    cam_centers = []
    for cam in cameras:
        c2w_R = cam.R
        w2c_T = cam.T

        # get cam_center
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = c2w_R.transpose()
        Rt[:3, 3] = w2c_T
        Rt[3, 3] = 1.0
        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]


        world_up_vecs.append(c2w_R @ up_vec)
        cam_centers.append(cam_center)

    # get top-view camera matrices
    up_vec = np.array(world_up_vecs).mean(0)
    center = np.array(cam_centers).mean(0)
    up_vec = up_vec / np.sqrt(((up_vec)**2).sum())
    
    return up_vec


# def get_yaw_rot_angle(rot_vec):
#     angle = np.linalg.norm(rot_vec + 1e-8)
#     rot_dir = rot_vec[0] / angle

#     cos = np.cos(angle)
#     sin = np.sin(angle)

#     # Bx1 arrays
#     rx, ry, rz = rot_dir[0], rot_dir[1], rot_dir[2]
#     K = np.array([0, -rz, ry, rz, 0, -rx, -ry, rx, 0]).reshape(3, 3)

#     ident = np.eye(3)
#     rot_mat = ident + sin * K + (1 - cos) * (K @ K)
#     front_axis = np.array([0, 0, 1]) @ rot_mat.T
#     yaw = np.arctan2(front_axis[0], front_axis[2])



### Part that extracting top-view vectors

def get_top_view(cameras: List, res=1024, up_vec=np.array([0, -1, 0]), z_axis = np.array([0,0,-1]), t_scale=1.5, fov=0.05, device=torch.device("cuda")):
    cam_centers = []
    for cam in cameras:
        cam_center = cam.camera_center.clone().detach().float().squeeze()
        cam_centers.append(cam_center)
    center = np.array(cam_centers).mean(0)
    

    up_vec = get_up_vec(cameras, up_vec)
    cam_position = center + up_vec * t_scale
    fov = math.pi * fov


    # it's openGL coordinate, so up-vec should be z-axis.
    w2c_R = rotation_matrix_from_vectors(up_vec, z_axis)
    c2w_T = cam_position + up_vec * 100     

    R = w2c_R.T
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = w2c_R.T
    Rt[:3, 3] = c2w_T
    Rt[3, 3] = 1.0
    W2C = np.linalg.inv(Rt)
    T = W2C[:3, 3]
    

    org_image = torch.zeros(3, res, res)
    top_cam = Camera(
        colmap_id=0, R=R, T=T, 
        FoVx=fov, FoVy=fov, 
        cx=0, cy=0, 
        image=org_image, gt_alpha_mask=None,
        image_name=None, uid=0, data_device=device, smpl_param=None,
        zfar=200, znear=100,
        )
    

    return top_cam #, torch.from_numpy(up_vec), torch.from_numpy(cam_position)





def get_top_views_from_camera(camera_dicts, up_vec: np.ndarray, fov=0.05):
    cam_centers = []
    for cam in camera_dicts.values():
        cam = cam[0]
        cam_center = cam.camera_center.clone().detach().float().squeeze().cpu().numpy()
        cam_centers.append(cam_center)
    cam_centers = np.array(cam_centers)
    center = cam_centers.mean(0)

    max_dist = np.sqrt(((cam_centers - center[None]) ** 2).sum(-1)).max()
    fov = math.pi * fov
    focal = fov2focal(fov, cam.image_height)
    max_dist = (focal / cam.image_height) * 2 * 1.2 * max_dist

    fovy = fov
    fovx = focal2fov(focal, cam.image_width)

    cam_position = center + up_vec * max_dist

    z_axis = np.array([0,0,-1])
    c2w_R = rotation_matrix_from_vectors(z_axis, up_vec)
    c2w_T = cam_position   

    R = c2w_R
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = c2w_R
    Rt[:3, 3] = c2w_T
    Rt[3, 3] = 1.0
    W2C = np.linalg.inv(Rt)
    T = W2C[:3, 3]


    org_image = torch.zeros(3, cam.image_height, cam.image_width)
    
    
    top_cams = []
    for cam in list(camera_dicts.values())[0]:
        top_cam = Camera(
            colmap_id=cam.colmap_id, R=R, T=T, 
            FoVx=fovx, FoVy=fovy, 
            cx=0, cy=0, 
            image=org_image, gt_alpha_mask=None,
            image_name=None, uid=0, data_device=torch.device("cuda"), smpl_param=None,
            zfar=max_dist*3, znear=max_dist*0.2,
            )
        top_cams.append(top_cam)
    
    return top_cams




### Get canon cameras
def gen_closeup_views(cameras, center, scale:float=1., up_vec=np.array([0, -1, 0]), n_cameras: int=8, f=500, res=512, device=torch.device("cuda")):
    # Get transformation, normalizing center of camera position
    world_up_vec = get_up_vec(cameras, up_vec)
    R_smpl = rotation_matrix_from_vectors(world_up_vec, -up_vec)
    T_smpl = -center
    

    fov = focal2fov(f, res)

    # Generate quaternion rotations
    angles_deg = np.arange(0, 361, 360//n_cameras)
    quaternions = [quaternion_rotation_y(angle) for angle in angles_deg]


    org_image = torch.zeros(3, res, res)

    # Print the generated quaternions
    T = scale * 2.
    T = np.array([0, 0, T])
    uid = 0
    cameras = []
    for angle, quat in zip(angles_deg, quaternions):
        # print(f"Angle: {angle} degrees, Quaternion: {quat}")
        R = qvec2rotmat(quat)
    
        Rp = R @ R_smpl
        Tp = T + R @ R_smpl @ T_smpl

        w2c = np.eye(4)
        w2c[:3, :3] = Rp
        w2c[:3, 3] = Tp
        c2w = np.linalg.inv(w2c)

        c2w[0:3, 0] *= -1
        c2w[0:3, 1] *= -1    
        
        ext = np.linalg.inv(c2w)
        T_final = ext[:3, 3]
        R_final = ext[:3, :3].T


        cam = Camera(colmap_id=uid, R=R_final, T=T_final, 
                  FoVx=fov, FoVy=fov, 
                  cx=0., cy=0.,
                  image=org_image, gt_alpha_mask=None, depth=None,
                  image_name=None, uid=uid, data_device=device, smpl_param=None)

        uid += 1
        cameras.append(cam)

    return cameras