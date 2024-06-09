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
import sys
import glob
import json
import pickle5
from pathlib import Path
from typing import NamedTuple, Optional
from plyfile import PlyData, PlyElement


import cv2
import pandas
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from gtu.dataset.scene.gaussian_model import BasicPointCloud, GaussianModel
from gtu.dataset.scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from gtu.dataset.system_utils import searchForMaxIteration
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.sh_utils import SH2RGB


from gtu import arguments


class CameraInfo(NamedTuple):
    uid: int
    fid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    mask: Optional[np.array] 
    mask_path: Optional[str] 
    mask_name: Optional[str]
    occ_mask: Optional[np.array] 
    occ_mask_path: Optional[str] 
    width: int
    height: int
    smpl_param: Optional[np.array]
    cx: float
    cy: float
    depth_path: Optional[str]
    depth: Optional[np.array]
    flip_mask: bool
    bbox: Optional[list]

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    

    # single camera case
    # radius determine the "scene scale" -> important when densify / pruning
    if radius < 1e-3: 
        radius = 1
    print(f"[INFO] scene radius: {radius}")

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, mask_folder=None, depth_folder=None):
    cam_infos = []


    depth_dict = dict()
    if depth_folder is not None:
        depth_files = Path(depth_folder).glob("*.npy")
        for df in depth_files:
            depth_dict[int(df.name.split(".")[0])] = str(df)

    img_dict = dict()
    img_files = Path(images_folder).glob("*.png")
    for img_file in img_files:
        img_dict[int(img_file.name.split(".")[0])] = str(img_file)


    cc_dict = dict()
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        
        cx = 0. # width / 2
        cy = 0. #height / 2

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        if not os.path.exists(image_path):
            image_name = image_name.split("_")[-1]
            fid = int(image_name)
            image_path = img_dict[fid]
            image_name = os.path.basename(image_path).split(".")[0]
        else:
            fid = int(image_name)
            
        image = Image.open(image_path)
 

        if fid in depth_dict:
            depth_path = depth_dict[fid]
            depth = np.load(depth_path)
        else:
            depth_path = None
            depth = None

        if mask_folder is not None:
            mask_path = os.path.join(mask_folder, image_name + '.png')+".png"
            mask_name = os.path.basename(mask_path).split(".")[0]
            mask = Image.open(mask_path)
        else:
            mask = None
            mask_name = None
            mask_path = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, fid=fid,
                            image_path=image_path, image_name=image_name, width=width, height=height, 
                            mask=mask, mask_name=mask_name, mask_path=mask_path, 
                            occ_mask=None, occ_mask_path=None,
                            depth_path=depth_path, depth=depth, smpl_param=None, cx=cx, cy=cy, flip_mask=False, bbox=None)
        cam_infos.append(cam_info)


        # calculate cam_center
        c2w = np.eye(4)
        c2w[:3,:3] = R.T
        c2w[:3,3] = T
        w2c = np.linalg.inv(c2w)
        cc = w2c[:3, 3]

        cc_dict[key] = cc

    sys.stdout.write('\n')
    return cam_infos, cc_dict

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb, normals=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    if normals is None:
        normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted, cc_dict = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # if eval:
    #     train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    #     test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    # else:
    train_cam_infos = cam_infos
    test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/points3D.ply")
    bin_path = os.path.join(path, "sparse/points3D.bin")
    txt_path = os.path.join(path, "sparse/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _, normal = read_points3D_text(txt_path, cc_dict=cc_dict, get_normal=True)
        except:
            xyz, rgb, _ = read_points3D_binary(bin_path)
            normal = None
        storePly(ply_path, xyz, rgb, normals=normal)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)


            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            fid = int(image_name)
            
            

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, fid=fid, depth=None, depth_path=None,
                            occ_mask=None, occ_mask_path=None,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], cx=0., cy=0., flip_mask=False, bbox=None))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readColmapMaskSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images

    mask_folder = os.path.join(path,"segmentations","masks")
    depth_folder = os.path.join(path, 'depths')
    cam_infos_unsorted, cc_dict = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), mask_folder=mask_folder, depth_folder=depth_folder)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # if eval:
    #     train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    #     test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    # else:
    train_cam_infos = cam_infos
    test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/points3D.ply")
    bin_path = os.path.join(path, "sparse/points3D.bin")
    txt_path = os.path.join(path, "sparse/points3D.txt")
    if not os.path.exists(ply_path) or True:
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        # xyz, rgb, _, normal = read_points3D_text(txt_path, cc_dict=cc_dict, get_normal=True)
        try:
            xyz, rgb, _, normal = read_points3D_text(txt_path, cc_dict=cc_dict, get_normal=True)
        except Exception as e:
            print('Error while reading files: '+ str(e))
            xyz, rgb, _, normal = read_points3D_binary(bin_path, cc_dict=cc_dict, get_normal=True)
        storePly(ply_path, xyz, rgb, normals=normal)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info



def readHumanSceneInfo(path, images, eval, llffhold=1, scene_img_shape=None, flip_mask=False, manual_smpl_scale=-1):
    # load points
    ply_path = os.path.join(path, "points3D.ply")
    txt_path = os.path.join(path, "points3D.txt") 
    txt_path = txt_path if Path(txt_path).exists() else os.path.join(path, "background_points.txt")
    optimized_smpl_path = os.path.join(path, "optimized_smpl.pkl")
    
    xyz, rgb, _, normal = read_points3D_text(txt_path, get_normal=True)
    
    
    storePly(ply_path, xyz, rgb, normals=normal)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # define some of paths
    reading_dir = 'images' if images == None else images
    reading_dir = os.path.join(path, reading_dir)
    mask_folder = os.path.join(path, 'masks')
    if not os.path.exists(mask_folder):
        mask_folder = os.path.join(path, 'segmentations', 'masks')
    if not os.path.exists(mask_folder):
        mask_folder = os.path.join(path, 'raw_mask')
    if not os.path.exists(mask_folder):
        mask_folder = os.path.join(path, 'people_segmentation')
        
    mask_folder = mask_folder if os.path.exists(mask_folder) else None

    img_list = glob.glob(os.path.join(reading_dir, "*.png"))
    img_list_jpg = glob.glob(os.path.join(reading_dir, "*.jpg"))
    img_list = sorted(list(img_list)+list(img_list_jpg))
    print(f"{os.path.join(reading_dir,'*.png')}, N images: {len(img_list)}")
    img_dict = dict()
    for img_fname in img_list:
        fid = int(os.path.basename(img_fname).split(".")[0])
        # print(img_fname)
        img_dict[fid] = img_fname

    
    # load cameras
    cam_file = os.path.join(path, "cameras.pkl")
    if Path(cam_file).exists():
        try:
            cam_data = pandas.read_pickle(cam_file)
        except:
            import pickle5
            with open(cam_file, 'rb') as f:
                cam_data = pickle5.load(f)
    else:
        # if not camera exists, load from human reults
        cam_file_list = sorted(list(Path(path).glob('results_p*.pkl')))
        cam_file = str(cam_file_list[0])
        try:
            cam_data = pandas.read_pickle(cam_file)
        except:
            import pickle5
            with open(cam_file, 'rb') as f:
                cam_data = pickle5.load(f)


    # load optimized smpl
    if os.path.exists(optimized_smpl_path):
        optimized_smpls = pandas.read_pickle(optimized_smpl_path)
        optimized_smpls = optimized_smpls.numpy()

        assert len(cam_data) == len(optimized_smpls), "camera size is different!"
    else:
        optimized_smpls = None


    if manual_smpl_scale < -1:
        print(f"\n\n\n Assuming fixed position and fitting smpl scales on it!!!! {manual_smpl_scale}\n\n\n")
        smpl_scale = torch.ones(1).cuda().float()
        smpl_scale.requires_grad = True
        
        opt_params = [{'params': smpl_scale, 'lr': 1e-2}]
        optimizer = torch.optim.Adam(opt_params, lr=2e-3, betas=(0.9, 0.999))
        loop = tqdm(range(1000))
    
        
        
        cam_Ts = []
        smpl_Ts = []
        cam_Rs = []
        inv_cam_Rs = []
        for idx, fid in enumerate(sorted(list(cam_data.keys()))):
            v = cam_data[fid]
            R = torch.from_numpy(v['camera']['rotation']).reshape(3,3)
        
            if 'trans' in v['camera']:
                T = v['camera']['trans']
            else:
                T = v['camera']['translation']
            T = torch.tensor(T).reshape(3)
            
            smpl_param = v['smpl_param']
            smpl_Ts.append(torch.from_numpy(smpl_param[0, 1:4]).cuda().float())
            cam_Rs.append(R.cuda().float())
            inv_cam_Rs.append(R.cuda().float().T)
            cam_Ts.append(T.cuda().float())
            
        cam_Ts = torch.stack(cam_Ts)
        cam_Rs = torch.stack(cam_Rs)
        inv_cam_Rs = torch.stack(inv_cam_Rs)
        smpl_Ts = torch.stack(smpl_Ts)
        
        
        for it in loop:
            optimizer.zero_grad()
            rel_transl = torch.einsum('bij,bj->bi', cam_Rs, smpl_Ts) + cam_Ts
            new_t = rel_transl * smpl_scale - cam_Ts
            new_t = torch.einsum('bij,bj->bi', inv_cam_Rs, new_t)
            
            loss = ((new_t - new_t.mean(0)[None]) ** 2).mean()
            loss.backward()
            optimizer.step()
            l_str = '%05d | Iter: %d' % (0, it)
            l_str += ', %s: %0.4f, scale %0.5f' % ("loss", loss.item(), smpl_scale.item())
            loop.set_description(l_str)
            
        
        smpl_global_scale = smpl_scale.detach().cpu().numpy()
        print(f"-----\n\n\nsmpl_global_scale: {smpl_global_scale}\n\n\n-----")
    else:
        # print("assuming New jeans")
        # smpl_global_scale = 0.2 

        if manual_smpl_scale == -1:
            smpl_global_scale = 1.
        else:
            smpl_global_scale = manual_smpl_scale
            print(f"\n\n\nUsing given SMPL scale!!!! {manual_smpl_scale}\n\n\n")
        
            
        
    cam_infos_unsorted = []
    for idx, fid in enumerate(sorted(list(cam_data.keys()))):
        v = cam_data[fid]
        
        image_path = img_dict[int(fid)]
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # i don't know why.
        # but, authors get c2w Rotation, and w2c translation
        # easily, R^T and T. 
        # However, it transpose again, when getting world2cam transformation....

        R = np.transpose(v['camera']['rotation'])
        
        if 'trans' in v['camera']:
            T = v['camera']['trans']
        else:
            T = v['camera']['translation']
        focal_length_x = v['camera']['intrinsic'][0,0]
        focal_length_y = v['camera']['intrinsic'][1,1]

        if 'width' in v['camera']:
            width = v['camera']['width']
            height = v['camera']['height']
        else:
            width = image.size[0]
            height = image.size[1]


        if scene_img_shape is not None and False:
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            # FovY = focal2fov(focal_length_y, scene_img_shape[1])
            # FovX = focal2fov(focal_length_x, scene_img_shape[0])
            cx = (v['camera']['intrinsic'][0,2] / (scene_img_shape[0] / 2)) - 1.
            cy = (v['camera']['intrinsic'][1,2] / (scene_img_shape[1] / 2)) - 1.
        else:
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            cx = (v['camera']['intrinsic'][0,2] / (width / 2)) - 1.
            cy = (v['camera']['intrinsic'][1,2] / (height / 2)) - 1.

        smpl_param = v['smpl_param']
        if optimized_smpls is not None:
            optimized_smpl = optimized_smpls[idx]
            smpl_param[:,4:76] = smpl_param[:,4:76] * 0 
            smpl_param[:,4:76] = smpl_param[:,4:76] + optimized_smpl[np.newaxis]
        
        
        if True:
            smpl_param[0, 0] = smpl_global_scale
            # smpl_param[:, 0] = 1/5 # For newjeans
            
            rel_transl = R.T @ smpl_param[0, 1:4] + T
            new_t = rel_transl * smpl_param[0, 0] - T
            new_t = R @ new_t
            smpl_param[0, 1:4] = new_t

            # print(f"\n\n\n\n\nsmpl param {smpl_param[0, 0]}\n\n\n\n")

        if mask_folder is not None:
            mask_path = os.path.join(mask_folder, os.path.basename(image_path)) ## No additional ".png" as we skipped colmap here
            if not os.path.exists(mask_path):
                mask_path = mask_path + ".png"
            
            mask_name = os.path.basename(mask_path).split(".")[0]
            mask = Image.open(mask_path)

        else:
            mask_path = None
            mask_name = None
            mask = None
            
            
        # print(f"mask_name: ", mask_name)
            
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, fid=fid,
                                image_path=image_path, image_name=image_name, width=width, height=height,
                                mask=mask, mask_name=mask_name, mask_path=mask_path, depth=None, depth_path=None,
                                occ_mask=None, occ_mask_path=None,
                                smpl_param=smpl_param, cx=cx, cy=cy, flip_mask=flip_mask, bbox=None)

        cam_infos_unsorted.append(cam_info)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # if eval:
    #     train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    #     test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    # else:
    #     train_cam_infos = cam_infos
    #     test_cam_infos = []

    train_cam_infos = cam_infos
    test_cam_infos = []



    nerf_normalization = getNerfppNorm(train_cam_infos)

   
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def load_human_model(human_path: Path, pose_path: Path, sh_degree:int = 0):
    # get model path
    if human_path.name.split("_")[0] == "iteration":
        pc_path = human_path / "point_cloud.ply"
    else:
        loaded_iter = searchForMaxIteration(os.path.join(str(human_path), "point_cloud"))
        pc_path = human_path / "point_cloud" / ("iteration_" + str(loaded_iter)) / "point_cloud.ply"
        
    print(f"     Loading Human model from {str(pc_path)}")
    print(f"     Loading pose path from {str(pose_path)}")
    
    # load gaussians (PCs)
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(str(pc_path))
    
    # get mean_beta
    if (human_path / 'mean_shape.npy').exists():
        beta = np.load(human_path / 'mean_shape.npy').astype(np.float32)
    else:
        beta = np.zeros(10)

    
    # load poses
    if pose_path.name.split(".")[-1] != "npy":
        raise NotImplementedError(f"[ERROR] For human pose, only NPY format is supported!")
    
    poses = np.load(pose_path)
    poses = poses.astype(np.float32)
    poses = torch.from_numpy(poses)
    beta = torch.from_numpy(beta)
    
    assert poses.shape[-1] == 86, "[ERROR] pose dimension should be 82"
    
    return gaussians, poses, beta, human_path
     
    




def readMVSceneInfo(
        path, images, eval, llffhold=8, scene_img_shape=None, main_cam_id=None, load_aux_mv: bool=False, 
        load_all_mv: bool=False, train_sample_interval: int=-1, aux_sample_interval: int=-1, fast_pcd_load: bool=False, camera_sampler=""):
    
    if train_sample_interval < 0:
        train_sample_interval = arguments.MV_TRAIN_SAMPLE_INTERVAL
    if aux_sample_interval < 0:
        aux_sample_interval = arguments.MV_AUX_SAMPLE_INTERVAL

    if eval:
        train_sample_interval = 1
        aux_sample_interval = 1
        load_all_mv = arguments.EVAL_LOAD_ALL_MV
        # train_sample_interval = 4
        # aux_sample_interval = 4

    
    # load points
    ply_path = os.path.join(path, "points3D.ply")
    txt_path = os.path.join(path, "background_points.txt")
    if not fast_pcd_load:
        xyz, rgb, _, normal = read_points3D_text(txt_path, get_normal=True)
        storePly(ply_path, xyz, rgb, normals=normal)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
    else:
        pcd = None

    # get camera dicts
    _cam_list = sorted(list(Path(path).iterdir()))
    cam_list = []
    cam_name_dict = dict()
    for _cam in _cam_list:
        if _cam.is_dir() and (_cam / 'images').exists():
            cam_name_dict[int(_cam.name)] = _cam.name
            cam_list.append(_cam.name)


    if main_cam_id is not None:
        print(f"[INFO] main camera from argument: {main_cam_id} ,   ignore 'selected_mono_cam.txt'")
        if main_cam_id in cam_name_dict:
            cam_name = cam_name_dict[main_cam_id]
        else:
            raise AssertionError(f"[ERROR] given main_camera: {main_cam_id} is not valid!")
    else:
        try:
            with open(os.path.join(path, 'selected_mono_cam.txt'), 'r') as f:
                data = int(f.readline())
            cam_name = cam_name_dict[data]
        except:
            print(f"[INFO] error while reading camera name from{os.path.join(path, 'selected_mono_cam.txt')}")
            print(f"[INFO]     -> Use {cam_list[0]} instead here")
            cam_name = cam_list[0]


    # read data from camera directory

    reading_dir = os.path.join(path, cam_name, 'images')
    assert os.path.exists(reading_dir)
    mask_folder = os.path.join(path, cam_name, 'segmentations', 'masks')
    if not os.path.exists(mask_folder):
        mask_folder = os.path.join(path, cam_name, 'people_segmentation')
    mask_folder = mask_folder if os.path.exists(mask_folder) else None

    mask_fdict = dict()
    if not (mask_folder is None):
        for mask_fname in (list(Path(mask_folder).glob("*.png")) + list(Path(mask_folder).glob("*.jpg"))):
            fid = int(mask_fname.name.split(".")[0])
            mask_fdict[fid] = str(mask_fname)


    img_list_png = list(glob.glob(os.path.join(reading_dir, "*.png")))
    img_list_jpg = list(glob.glob(os.path.join(reading_dir, "*.jpg")))
    img_list = img_list_png + img_list_jpg
    img_list = sorted(img_list)
        
    print(f"        {os.path.join(reading_dir,'*.png')}, N images: {len(img_list)}")
    img_dict = dict()
    for img_fname in img_list:
        fid = int(os.path.basename(img_fname).split(".")[0])
        # print(img_fname)
        img_dict[fid] = img_fname

    
    # load cameras
    if (Path(path, cam_name) / 'cameras.pkl').exists():
        cam_file = str((Path(path, cam_name) / 'cameras.pkl'))
    else:
        # To support Legacy
        print("[WARNING] we use first people pickle file to load frames. (it's for supporting legacy)")
        cam_file_list = sorted(list(Path(path, cam_name).glob('results_p*.pkl')))
        cam_file = str(cam_file_list[0])

    # read camera datas
    try:
        cam_data = pandas.read_pickle(cam_file)
    except:
        import pickle5
        with open(cam_file, 'rb') as f:
            cam_data = pickle5.load(f)


    cam_infos_unsorted = []
    for idx, fid in enumerate(sorted(list(cam_data.keys()))):
        if idx % train_sample_interval != 0:
            continue
        v = cam_data[fid]
        
        image_path = img_dict[int(fid)]
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # i don't know why.
        # but, authors get c2w Rotation, and w2c translation
        # easily, R^T and T. 
        # However, it transpose again, when getting world2cam transformation....
        R = np.transpose(v['camera']['rotation'])
        
        if 'trans' in v['camera']:
            T = v['camera']['trans'].reshape(-1)
        else:
            T = v['camera']['translation'].reshape(-1)
        focal_length_x = v['camera']['intrinsic'][0,0]
        focal_length_y = v['camera']['intrinsic'][1,1]

        if 'width' in v['camera']:
            width = v['camera']['width']
            height = v['camera']['height']
        else:
            width = image.size[0]
            height = image.size[1]


        if scene_img_shape is not None and False:
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            cx = (v['camera']['intrinsic'][0,2] / (scene_img_shape[0] / 2)) - 1.
            cy = (v['camera']['intrinsic'][1,2] / (scene_img_shape[1] / 2)) - 1.
        else:
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            cx = (v['camera']['intrinsic'][0,2] / (width / 2)) - 1.
            cy = (v['camera']['intrinsic'][1,2] / (height / 2)) - 1.
        
        # No need to hold smpl_param in scene cameras
        smpl_param = np.zeros((1, 86))
        

        if mask_folder is not None:
            mask_path = mask_fdict[int(fid)] 
            mask_name = os.path.basename(mask_path).split(".")[0]
            mask = Image.open(mask_path)
        else:
            mask_path = None
            mask_name = None
            mask = None
            
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, fid=fid,
                                image_path=image_path, image_name=image_name, width=width, height=height,
                                mask=mask, mask_name=mask_name, mask_path=mask_path, depth=None, depth_path=None,
                                occ_mask=None, occ_mask_path=None,
                                smpl_param=smpl_param, cx=cx, cy=cy, flip_mask=True, bbox=None)

        cam_infos_unsorted.append(cam_info)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # if False:
    #     train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    #     test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    if camera_sampler != "":
        if camera_sampler == "first20":
            n_camera = len(cam_infos)
            first_20 = n_camera // 5
            train_cam_infos = cam_infos[:first_20]
        elif camera_sampler == "first10":
            n_camera = len(cam_infos)
            first_10 = n_camera // 10
            train_cam_infos = cam_infos[:first_10]
        else:
            raise NotImplementedError(f"'{camera_sampler}' is invalid sampling method")
        test_cam_infos = []
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    print(f"n_cam: {len(train_cam_infos)}")
    nerf_normalization = getNerfppNorm(train_cam_infos)

   
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    # load additional cameras
    aux_cam_dict = dict()
    main_cam_name = cam_name
    if load_aux_mv:
        # Load aux-cams
        if load_all_mv:
            aux_cam_ids = []
            for aux_cam_id in cam_list:
                if int(main_cam_name) == int(aux_cam_id):
                    continue
                else:
                    aux_cam_ids.append(int(aux_cam_id))

        elif os.path.exists(os.path.join(path, 'selected_aux_cam.npy')):
            _aux_cam_ids = np.load(os.path.join(path, 'selected_aux_cam.npy'))
            aux_cam_ids = []
            # check whether duplicated exists
            for aux_cam_id in _aux_cam_ids:
                if int(main_cam_name) == int(aux_cam_id):
                    continue
                else:
                    aux_cam_ids.append(int(aux_cam_id))
                
        else:
            aux_cam_ids = []
            interval = len(cam_list) // 6
            for _aux_cam_id in range(6):
                aux_cam_id = _aux_cam_id * interval
                aux_cam_id = int(cam_list[aux_cam_id])
                if int(main_cam_name) == int(aux_cam_id):
                    continue
                elif aux_cam_id not in cam_name_dict:
                    continue # 1-based cameras
                else:
                    aux_cam_ids.append(int(aux_cam_id))
            aux_cam_ids = aux_cam_ids[:5]



        for aux_cam in tqdm(aux_cam_ids, desc="Loading auxiliary cams"):
            cam_name = cam_name_dict[aux_cam]

            cam_file_list = sorted(list(Path(path, cam_name).glob('results_p*.pkl')))
            cam_file = str(cam_file_list[0])

            # read camera datas
            try:
                cam_data = pandas.read_pickle(cam_file)
            except:
                import pickle5
                with open(cam_file, 'rb') as f:
                    cam_data = pickle5.load(f)
            
            reading_dir = os.path.join(path, cam_name, 'images')
            assert os.path.exists(reading_dir)
            mask_folder = os.path.join(path, cam_name, 'segmentations', 'masks')
            mask_folder = mask_folder if os.path.exists(mask_folder) else None

            img_list_png = list(glob.glob(os.path.join(reading_dir, "*.png")))
            img_list_jpg = list(glob.glob(os.path.join(reading_dir, "*.jpg")))
            img_list = img_list_png + img_list_jpg
            img_list = sorted(img_list)

            img_dict = dict()
            for img_fname in img_list:
                fid = int(os.path.basename(img_fname).split(".")[0])
                # print(img_fname)
                img_dict[fid] = img_fname

            cam_infos_unsorted = []
            for idx, fid in enumerate(sorted(list(cam_data.keys()))):
                if idx % aux_sample_interval != 0:
                    continue

                v = cam_data[fid]

                image_path = img_dict[int(fid)]
                image_name = os.path.basename(image_path).split(".")[0]
                if idx == 0:
                    sample_image = Image.open(image_path)
                    image = None
                else:
                    image = None

                # i don't know why.
                # but, authors get c2w Rotation, and w2c translation
                # easily, R^T and T. 
                # However, it transpose again, when getting world2cam transformation....

                R = np.transpose(v['camera']['rotation'])
                
                if 'trans' in v['camera']:
                    T = v['camera']['trans'].reshape(-1)
                else:
                    T = v['camera']['translation'].reshape(-1)
                focal_length_x = v['camera']['intrinsic'][0,0]
                focal_length_y = v['camera']['intrinsic'][1,1]

                if 'width' in v['camera']:
                    width = v['camera']['width']
                    height = v['camera']['height']
                else:
                    width = sample_image.size[0]
                    height = sample_image.size[1]


                if scene_img_shape is not None and False:
                    FovY = focal2fov(focal_length_y, height)
                    FovX = focal2fov(focal_length_x, width)
                    # FovY = focal2fov(focal_length_y, scene_img_shape[1])
                    # FovX = focal2fov(focal_length_x, scene_img_shape[0])
                    cx = (v['camera']['intrinsic'][0,2] / (scene_img_shape[0] / 2)) - 1.
                    cy = (v['camera']['intrinsic'][1,2] / (scene_img_shape[1] / 2)) - 1.
                else:
                    FovY = focal2fov(focal_length_y, height)
                    FovX = focal2fov(focal_length_x, width)
                    cx = (v['camera']['intrinsic'][0,2] / (width / 2)) - 1.
                    cy = (v['camera']['intrinsic'][1,2] / (height / 2)) - 1.

                smpl_param = v['smpl_param']
                

                if mask_folder is not None:
                    mask_path = os.path.join(mask_folder, os.path.basename(image_path)) ## No additional ".png" as we skipped colmap here
                    if not os.path.exists(mask_path):
                        mask_path = mask_path + ".png"
                    
                    mask_name = os.path.basename(mask_path).split(".")[0]
                    # mask = Image.open(mask_path)
                    mask = None

                else:
                    mask_path = None
                    mask_name = None
                    mask = None
                    
                cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, fid=fid,
                                        image_path=image_path, image_name=image_name, width=width, height=height,
                                        mask=mask, mask_name=mask_name, mask_path=mask_path, depth=None, depth_path=None,
                                        occ_mask=None, occ_mask_path=None,
                                        smpl_param=smpl_param, cx=cx, cy=cy, flip_mask=True, bbox=None)

                cam_infos_unsorted.append(cam_info)
            cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)


            if camera_sampler != "":
                if camera_sampler == "first20":
                    n_camera = len(cam_infos)
                    first_20 = n_camera // 5
                    cam_infos = cam_infos[:first_20]
                elif camera_sampler == "first10":
                    n_camera = len(cam_infos)
                    first_10 = n_camera // 10
                    cam_infos = cam_infos[:first_10]
                else:
                    raise NotImplementedError(f"'{camera_sampler}' is invalid sampling method")

            aux_cam_dict[cam_name] = cam_infos

            if arguments.DEBUG_FAST_LOADING:
                print("[INFO] load just single view for fast load")
                break

            if (not eval) and len(aux_cam_dict) == 10:
                print("[INFO] due to OOM, break with two eval sets")
                break
    
    return scene_info, main_cam_name, aux_cam_dict



# deprecated
def readMV_bg_SceneInfo(path, images, eval, llffhold=8, scene_img_shape=None):
    # load points
    ply_path = os.path.join(path, "points3D.ply")
    txt_path = os.path.join(path, "background_points.txt")
    xyz, rgb, _, normal = read_points3D_text(txt_path, get_normal=True)
    storePly(ply_path, xyz, rgb, normals=normal)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # choose which camera to use
    cam_name = 'bg_mv_cams'

    # read data from camera directory
    reading_dir = os.path.join(path, cam_name, 'mv_images')
    assert os.path.exists(reading_dir)
    mask_folder = os.path.join(path, cam_name, 'segmentations', 'masks')
    mask_folder = mask_folder if os.path.exists(mask_folder) else None

    img_list = glob.glob(os.path.join(reading_dir, "*.png"))
    img_list = sorted(list(img_list))
    print(f"        {os.path.join(reading_dir,'*.png')}, N images: {len(img_list)}")
    img_dict = dict()
    for img_fname in img_list:
        fid = int(os.path.basename(img_fname).split(".")[0])
        # print(img_fname)
        img_dict[fid] = img_fname

    
    # load cameras
    cam_file_list = sorted(list(Path(path, cam_name).glob('results_p*.pkl')))
    cam_file = str(cam_file_list[0])

    # read camera datas
    try:
        cam_data = pandas.read_pickle(cam_file)
    except:
        import pickle5
        with open(cam_file, 'rb') as f:
            cam_data = pickle5.load(f)


    cam_infos_unsorted = []
    for idx, fid in enumerate(sorted(list(cam_data.keys()))):
        v = cam_data[fid]
        
        image_path = img_dict[int(fid)]
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # i don't know why.
        # but, authors get c2w Rotation, and w2c translation
        # easily, R^T and T. 
        # However, it transpose again, when getting world2cam transformation....

        R = np.transpose(v['camera']['rotation'])
        
        if 'trans' in v['camera']:
            T = v['camera']['trans'].reshape(-1)
        else:
            T = v['camera']['translation'].reshape(-1)
        focal_length_x = v['camera']['intrinsic'][0,0]
        focal_length_y = v['camera']['intrinsic'][1,1]

        if 'width' in v['camera']:
            width = v['camera']['width']
            height = v['camera']['height']
        else:
            width = image.size[0]
            height = image.size[1]


        if scene_img_shape is not None and False:
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            # FovY = focal2fov(focal_length_y, scene_img_shape[1])
            # FovX = focal2fov(focal_length_x, scene_img_shape[0])
            cx = (v['camera']['intrinsic'][0,2] / (scene_img_shape[0] / 2)) - 1.
            cy = (v['camera']['intrinsic'][1,2] / (scene_img_shape[1] / 2)) - 1.
        else:
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            cx = (v['camera']['intrinsic'][0,2] / (width / 2)) - 1.
            cy = (v['camera']['intrinsic'][1,2] / (height / 2)) - 1.

        smpl_param = v['smpl_param']
        

        if mask_folder is not None:
            mask_path = os.path.join(mask_folder, os.path.basename(image_path)) ## No additional ".png" as we skipped colmap here
            if not os.path.exists(mask_path):
                mask_path = mask_path + ".png"
            
            mask_name = os.path.basename(mask_path).split(".")[0]
            mask = Image.open(mask_path)

        else:
            mask_path = None
            mask_name = None
            mask = None
            
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, fid=fid,
                                image_path=image_path, image_name=image_name, width=width, height=height,
                                mask=mask, mask_name=mask_name, mask_path=mask_path, depth=None, depth_path=None,
                                occ_mask=None, occ_mask_path=None,
                                smpl_param=smpl_param, cx=cx, cy=cy, flip_mask=False, bbox=None)

        cam_infos_unsorted.append(cam_info)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # if eval:
    #     train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    #     test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    # else:
    train_cam_infos = cam_infos
    test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

   
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readHumanSceneInfo_MVdata(
    human_path, 
    eval, 
    llffhold=1, 
    scene_img_shape=None, 
    scene_fids=[], 
    person_mask_path: str="", 
    occ_mask_path: str="",
    is_bg=False, 
    manual_smpl_scale: float=-1.,
    use_skip_fids_for_indiv_loading: bool=False,
    ):
    # load points

    human_id = str(int(human_path.name[:-4].split("results_p")[-1]))

    ply_path = str(human_path.parents[1] / f"human_points_{human_id}.ply")
    txt_path = str(human_path.parents[1] / f"human_points_{human_id}.txt")

    if not Path(txt_path).exists():
        print("[INFO] human points per each camera -> SLAHMR processed dataset")
        print(f"human_id: {human_id}")
        ply_path = str(human_path.parent / f"human_points_{human_id}.ply")
        txt_path = str(human_path.parent / f"human_points_{human_id}.txt")


    path = str(human_path.parent / human_id)
    optimized_smpl_path = os.path.join(path, "optimized_smpl.pkl")
    
    xyz, rgb, _, normal = read_points3D_text(txt_path, get_normal=True)
    storePly(ply_path, xyz, rgb, normals=normal)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    
    # load mask if exists
    if person_mask_path != "":
        mask_list = glob.glob(os.path.join(person_mask_path, "*.png"))
        mask_list_jpg = glob.glob(os.path.join(person_mask_path, "*.jpg"))
        mask_list = sorted(list(mask_list)+list(mask_list_jpg))
        print(f"        {os.path.join(person_mask_path,'*.png/jpg')}, N masks: {len(mask_list)}")
        mask_dict = dict()
        for mask_fname in mask_list:
            fid = int(os.path.basename(mask_fname).split(".")[0])
            mask_dict[fid] = mask_fname
    
    occ_mask_dict = dict()
    if occ_mask_path != "" and occ_mask_path != "none":
        occ_mask_list = glob.glob(os.path.join(occ_mask_path, "*.png"))
        occ_mask_list_jpg = glob.glob(os.path.join(occ_mask_path, "*.jpg"))
        occ_mask_list = sorted(list(occ_mask_list)+list(occ_mask_list_jpg))
        print(f"        {os.path.join(occ_mask_path,'*.png/jpg')}, N Occlusion masks: {len(occ_mask_list)}")
        
        for mask_fname in occ_mask_list:
            fid = int(os.path.basename(mask_fname).split(".")[0])
            occ_mask_dict[fid] = mask_fname

    # (Optional) Load invalid fid lists 
    skip_fids = []
    if use_skip_fids_for_indiv_loading:
        skip_fid_fname = Path(str(person_mask_path)).parent / "skip_fids.txt"
        if skip_fid_fname.exists():
            with open(skip_fid_fname, 'r') as f:
                for line in f:
                    # Strip whitespace and check if the line is not empty
                    stripped_line = line.strip()
                    if stripped_line:
                        try:
                            # Convert to integer and append to list if successful
                            skip_fids.append(int(stripped_line))
                        except ValueError:
                            # Handle the case where conversion to integer fails
                            print(f"Warning: Skipping invalid skip fid '{stripped_line}'")
        else:
            print(f"[WARNING] You used to use skip_fids.txt but {str(skip_fid_fname)} not exists")
                
    
    # load cameras
    try:
        cam_data = pandas.read_pickle(human_path)
    except:
        import pickle5
        with open(human_path, 'rb') as f:
            cam_data = pickle5.load(f)


    # load optimized smpl
    if os.path.exists(optimized_smpl_path):
        optimized_smpls = pandas.read_pickle(optimized_smpl_path)
        optimized_smpls = optimized_smpls.numpy()

        assert len(cam_data) == len(optimized_smpls), "camera size is different!"
    else:
        optimized_smpls = None

    default_image = np.zeros((512, 512, 3))
    cam_infos_unsorted = []
    cam_uid_idx = 0
    for idx, fid in enumerate(sorted(list(cam_data.keys()))):
        if len(scene_fids) > 0:
            if fid not in scene_fids:
                print(f"{fid} not in scene_fids: {scene_fids}")
                continue
        
        if len(skip_fids) > 0:
            if fid in skip_fids:
                print(f"{fid} in skip_lists: {scene_fids}")
                continue
            
        v = cam_data[fid]
       

        # i don't know why.
        # but, authors get c2w Rotation, and w2c translation
        # easily, R^T and T. 
        # However, it transpose again, when getting world2cam transformation....

        R = np.transpose(v['camera']['rotation'])
        
        if 'trans' in v['camera']:
            T = v['camera']['trans'].reshape(-1)
        else:
            T = v['camera']['translation'].reshape(-1)
        focal_length_x = v['camera']['intrinsic'][0,0]
        focal_length_y = v['camera']['intrinsic'][1,1]

        if 'width' in v['camera']:
            width = v['camera']['width']
            height = v['camera']['height']
        else:
            width = default_image.shape[1]
            height = default_image.shape[0]


        if 'gt_bbox' not in v:
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            cx = (v['camera']['intrinsic'][0,2] / (width / 2)) - 1.
            cy = (v['camera']['intrinsic'][1,2] / (height / 2)) - 1.
            bbox = None
        else:
            bbox_dilation = 1.1

            gt_bbox = v['gt_bbox']
            b_x, b_y, b_w, b_h = gt_bbox
            b_x = int(b_x - b_w * (bbox_dilation-1)/2)
            b_y = int(b_y - b_h * (bbox_dilation-1)/2)
            b_w = int(b_w * bbox_dilation)
            b_h = int(b_h * bbox_dilation)
            

            b_x = 0 if b_x < 0 else b_x
            b_y = 0 if b_y < 0 else b_y
            b_w = width-b_x-1 if b_w + b_x >= width else b_w
            b_h = height-b_y-1 if b_h + b_y >= height else b_h

            new_cx = -(b_x) + v['camera']['intrinsic'][0,2]
            new_cy = -(b_y) + v['camera']['intrinsic'][1,2]

            # change intrinsic
            FovY = focal2fov(focal_length_y, b_h)
            FovX = focal2fov(focal_length_x, b_w)
            cx = (new_cx / (b_w / 2)) - 1.
            cy = (new_cy / (b_h / 2)) - 1.
            bbox = [b_x, b_y, b_w, b_h]

            # Finally change height and widht
            width = b_w
            height = b_h


        smpl_param = v['smpl_param']
        
     
        image_path = str(fid)
        image_name = str(fid)

        # to save memory load dummy image
        image = np.zeros((16, 16, 3))
        image = Image.fromarray(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))

        flip_mask = False
        if person_mask_path != "":
            if fid not in mask_dict:
                # it's occluded case. (we don't need it here)
                mask_path = None
                mask_name = None
                mask = None
            # assert fid in mask_dict, f"frame {fid} is not in {person_mask_path}"
            else:
                if is_bg:
                    mask_path = mask_dict[fid]
                    mask_name = os.path.basename(mask_path).split(".")[0]
                    mask = Image.open(mask_path)
                    flip_mask=True

                else:
                    mask_name = None
                    mask_path = mask_dict[fid]
                    mask = None
                
                if cam_uid_idx == 0:
                    print(f"[INFO] We skip loading person masks here! {str(mask_path)} (omitted remaining logs)")
        else:
            mask_path = None
            mask_name = None
            mask = None
            
        # Load occ_mask
        if occ_mask_path != "" and occ_mask_path != "none":
            if fid not in occ_mask_dict:
                # it's occluded case. (we don't need it here)
                occ_mask_path = None
                occ_mask = None
            # assert fid in mask_dict, f"frame {fid} is not in {person_mask_path}"
            else:
                if is_bg:
                    occ_mask_path = occ_mask_dict[fid]
                    occ_mask = None

                else:
                    occ_mask_path = occ_mask_dict[fid]
                    occ_mask = None
                
                if cam_uid_idx == 0:
                    print(f"[INFO] We skip loading person masks here! {str(mask_path)} (omitted remaining logs)")
        else:
            occ_mask_path = None
            occ_mask = None
        
        cam_info = CameraInfo(uid=cam_uid_idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, fid=fid,
                                image_path=image_path, image_name=image_name, width=width, height=height,
                                mask=mask, mask_name=mask_name, mask_path=mask_path, depth=None, depth_path=None, 
                                occ_mask=occ_mask, occ_mask_path=occ_mask_path,
                                smpl_param=smpl_param, cx=cx, cy=cy, flip_mask=flip_mask, bbox=bbox)   # set not to flip mask


        cam_infos_unsorted.append(cam_info)
        cam_uid_idx += 1
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = cam_infos
        test_cam_infos = []
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

   
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info, cam_data



sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Colmap_mask" : readColmapMaskSceneInfo,
    "HumanScene": readHumanSceneInfo,
    "MVdataset": readMVSceneInfo,
    "MVBGdataset": readMV_bg_SceneInfo,
}