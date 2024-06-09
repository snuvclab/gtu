import argparse
import os
from pathlib import Path
from tqdm import tqdm

import cv2
import torch
import numpy as np
import trimesh
import shutil

from utils.io_utils import storePly, read_pickle, write_pickle
from gtu.smpl_deformer.smpl_server import SMPLServer



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None, help='overall data dir')
    parser.add_argument('--output_dir', type=str, default=None, help='overall data dir')
    parser.add_argument('--train_start', type=int)
    parser.add_argument('--train_end', type=int)
    parser.add_argument('--test_start', type=int)
    parser.add_argument('--test_end', type=int)
    parser.add_argument('--train_cam_id', type=int)
    parser.add_argument('--test_cam_id', type=int)
    parser.add_argument('--genders', type=str)
    parser.add_argument('--skip', type=int)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    train_output_dir = output_dir / f"{args.train_cam_id}"
    train_output_dir.mkdir(exist_ok=True)
    test_output_dir = output_dir / f"{args.test_cam_id}"
    test_output_dir.mkdir(exist_ok=True)
    
    mean_beta = np.load(data_dir / "mean_shape.npy").reshape(-1)
    poses = np.load(data_dir / 'poses.npz')
    camera = np.load(data_dir / 'cameras.npz')
    
    optimized_train_pose = np.load(data_dir / 'poses' / 'anim_nerf_train.npz')
    optimized_test_pose = np.load(data_dir / 'poses' / 'anim_nerf_test.npz')
    
    
    device = torch.device("cuda:0")
    smpl_server = SMPLServer(gender=args.genders)
    
    

    img_dict = dict()
    for img_fname in (data_dir / "images").glob("image_*"):
        fid = int(img_fname.name.split("_")[1][:-4])
        img_dict[fid] = img_fname
    
    mask_dict = dict()
    for mask_fname in (data_dir / "masks").glob("mask_*"):
        fid = int(mask_fname.name.split("_")[1][:-4])
        mask_dict[fid] = mask_fname
        
    
    
    smpl_server = SMPLServer(use_feet_keypoints=True)
    
    # Let's save
    train_inds = list(range(args.train_start, args.train_end, args.skip))
    
    #   Save Scene Cameras (cameras.pkl)
    cam_save_dict = dict()
    for fid in train_inds:
        extrinsic = camera['extrinsic']
        intrinsic = camera['intrinsic']
        height = camera['height']
        width = camera['width']
        
        w2c_R = extrinsic[:3,:3]
        w2c_T = extrinsic[:3, 3]
        
        cam_save_dict[fid] = dict(
            camera = dict(
                    width = width,
                    height = height,
                    rotation = w2c_R,
                    translation = w2c_T,
                    intrinsic = intrinsic
                ),
        )
        
        # also copy images
        img_save_dir = train_output_dir / "images"
        img_save_dir.mkdir(exist_ok=True)
        shutil.copy(img_dict[fid], img_save_dir / f"{fid:09}.png")
        
        # Copy masks
        mask_save_dir = train_output_dir / "people_segmentation"
        mask_save_dir.mkdir(exist_ok=True)
        mask = np.load(mask_dict[fid])
        mask = (mask * 255).astype(np.uint8)
        cv2.imwrite(str(mask_save_dir / f"{fid:09}.png"), mask)
        

    write_pickle(train_output_dir / f'cameras.pkl', cam_save_dict)
    
    #   Save individual
    save_dict = dict()

    for idx, fid in enumerate(train_inds):
        extrinsic = camera['extrinsic']
        intrinsic = camera['intrinsic']
        height = camera['height']
        width = camera['width']
        
        w2c_R = extrinsic[:3,:3]
        w2c_T = extrinsic[:3, 3]
        
        body_rot = optimized_train_pose['global_orient'][idx]
        body_pose = optimized_train_pose['body_pose'][idx]
        body_transl = optimized_train_pose['transl'][idx]
        smpl_param = np.concatenate(
            [
                np.ones((1), dtype=np.float32),
                body_transl,
                body_rot,
                body_pose,
                mean_beta 
            ], axis=0
        )
        
        save_dict[fid] = dict(
            smpl_param = np.expand_dims(smpl_param, axis=0),      # (1, 86)
            camera = dict(
                    width = width,
                    height = height,
                    rotation = w2c_R,
                    translation = w2c_T,
                    intrinsic = intrinsic
                ),
        )
    write_pickle(train_output_dir / f'results_p0.pkl', save_dict)

    # Also save person initial SMPL 3D GSs (canonical SMPL model)
    param_canon = np.concatenate([
                            np.ones( (1,1)), 
                            np.zeros( (1,3)),
                            np.zeros( (1,72)),
                            mean_beta.reshape(1,-1)], axis=1)
    param_canon[0, 9] = np.pi / 6
    param_canon[0, 12] = -np.pi / 6
    smpl_params = torch.from_numpy(param_canon).to(smpl_server.smpl.faces_tensor.device).float()
    smpl_output = smpl_server(smpl_params)

    canon_smpl_verts = smpl_output['smpl_verts'].data.cpu().numpy().squeeze() 
    smpl_faces = smpl_server.smpl.faces.astype(np.int64)

    # set normal & rgb of init mesh
    trimesh_mesh = trimesh.Trimesh(vertices=canon_smpl_verts, faces=smpl_faces)
    normals = trimesh_mesh.vertex_normals
    rgbs = np.ones_like(canon_smpl_verts, dtype=np.uint8) * np.array([[128, 128, 128]])   # mean gray color

    # save points
    with open(train_output_dir / f"human_points_0.txt", 'w') as f:
        for i, p in enumerate(canon_smpl_verts):
            # first 0: id
            # 1:4 = xyz
            # 4:7 = rgb (uint8)
            # 8 = error
            # 9:11 = normal 
            xyz = p.tolist()
            rgb = rgbs[i].tolist()
            normal = normals[i].tolist()
            error = 0
            f.write(f"{i} {xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]} {error} {normal[0]} {normal[1]} {normal[2]}\n")


    
    ########## save test
    
    test_inds = list(range(args.test_start, args.test_end, args.skip))
    
    #   Save Scene Cameras (cameras.pkl)
    cam_save_dict = dict()
    for fid in test_inds:
        extrinsic = camera['extrinsic']
        intrinsic = camera['intrinsic']
        height = camera['height']
        width = camera['width']
        
        w2c_R = extrinsic[:3,:3]
        w2c_T = extrinsic[:3, 3]
        
        cam_save_dict[fid] = dict(
            camera = dict(
                    width = width,
                    height = height,
                    rotation = w2c_R,
                    translation = w2c_T,
                    intrinsic = intrinsic
                ),
        )
        
        # also copy images
        img_save_dir = test_output_dir / "images"
        img_save_dir.mkdir(exist_ok=True)
        shutil.copy(img_dict[fid], img_save_dir / f"{fid:09}.png")
        
        # Copy masks
        mask_save_dir = test_output_dir / "segmentations"
        mask_save_dir.mkdir(exist_ok=True)
        mask_save_dir = test_output_dir / "segmentations" / "masks"
        mask_save_dir.mkdir(exist_ok=True)
        mask = np.load(mask_dict[fid])
        mask = (mask * 255).astype(np.uint8)
        cv2.imwrite(str(mask_save_dir / f"{fid:09}.png"), mask)
        
        # print(str(img_save_dir))
        # print(str(mask_save_dir))
        

        
    write_pickle(test_output_dir / f'cameras.pkl', cam_save_dict)
    
    #   Save individual
    save_dict = dict()

    for idx, fid in enumerate(test_inds):
        extrinsic = camera['extrinsic']
        intrinsic = camera['intrinsic']
        height = camera['height']
        width = camera['width']
        
        w2c_R = extrinsic[:3,:3]
        w2c_T = extrinsic[:3, 3]
        
        mean_beta = optimized_test_pose['betas'][0]
        body_rot = optimized_test_pose['global_orient'][idx]
        body_pose = optimized_test_pose['body_pose'][idx]
        body_transl = optimized_test_pose['transl'][idx]
        smpl_param = np.concatenate(
            [
                np.ones((1), dtype=np.float32),
                body_transl,
                body_rot,
                body_pose,
                mean_beta 
            ], axis=0
        )
        
        save_dict[fid] = dict(
            smpl_param = np.expand_dims(smpl_param, axis=0),      # (1, 86)
            camera = dict(
                    width = width,
                    height = height,
                    rotation = w2c_R,
                    translation = w2c_T,
                    intrinsic = intrinsic
                ),
        )
    write_pickle(test_output_dir / f'results_p0.pkl', save_dict)
    
    np.save(test_output_dir/'mean_shape.npy', mean_beta)

    # Also save person initial SMPL 3D GSs (canonical SMPL model)
    param_canon = np.concatenate([
                            np.ones( (1,1)) * 1, 
                            np.zeros( (1,3)),
                            np.zeros( (1,72)),
                            mean_beta.reshape(1,-1)], axis=1)
    param_canon[0, 9] = np.pi / 6
    param_canon[0, 12] = -np.pi / 6
    smpl_params = torch.from_numpy(param_canon).to(smpl_server.smpl.faces_tensor.device).float()
    smpl_output = smpl_server(smpl_params)

    canon_smpl_verts = smpl_output['smpl_verts'].data.cpu().numpy().squeeze() 
    smpl_faces = smpl_server.smpl.faces.astype(np.int64)

    # set normal & rgb of init mesh
    trimesh_mesh = trimesh.Trimesh(vertices=canon_smpl_verts, faces=smpl_faces)
    normals = trimesh_mesh.vertex_normals
    rgbs = np.ones_like(canon_smpl_verts, dtype=np.uint8) * np.array([[128, 128, 128]])   # mean gray color

    # save points
    with open(test_output_dir / f"human_points_0.txt", 'w') as f:
        for i, p in enumerate(canon_smpl_verts):
            # first 0: id
            # 1:4 = xyz
            # 4:7 = rgb (uint8)
            # 8 = error
            # 9:11 = normal 
            xyz = p.tolist()
            rgb = rgbs[i].tolist()
            normal = normals[i].tolist()
            error = 0
            f.write(f"{i} {xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]} {error} {normal[0]} {normal[1]} {normal[2]}\n")

    print("Done!")
    
    
    # Save smpl gender
    smpl_gender = [args.genders]
    np.save(output_dir / 'smpl_gender.npy', smpl_gender)
    print("saved smpl gender!")