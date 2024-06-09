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
import random
import json
import pandas
import torch

from typing import List, Optional
from pathlib import Path
from gtu.dataset.system_utils import searchForMaxIteration
from gtu.dataset.scene.dataset_readers import sceneLoadTypeCallbacks
from gtu.dataset.scene.gaussian_model import GaussianModel
from gtu.dataset.camera_utils import cameraList_from_camInfos, camera_to_JSON
from gtu.arguments import ModelParams

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=False, resolution_scales=[1.0], fast_loader=False, view_dir_reg=False, init_opactiy=None, load_aux_mv: bool=True, load_mv_bg: bool=False, exp_name=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.init_opactiy = 0.1 if init_opactiy is None else init_opactiy
        self.init_w_normal = view_dir_reg
        self.model_path = args.model_path

        if not (exp_name is None):
            self.model_path = os.path.join(self.model_path, exp_name)
        self.loaded_iter = None
        self.gaussians = gaussians
        self.init_opacity = 0.1 if init_opactiy is None else init_opactiy

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.aux_cam_dict = {}
        self.cam_name = None



        if load_mv_bg:
            scene_info = sceneLoadTypeCallbacks["MVBGdataset"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "cameras.pkl")):
            scene_info = sceneLoadTypeCallbacks["HumanScene"](args.source_path, args.images, args.eval, manual_smpl_scale=args.manual_smpl_global_scale)
        elif os.path.exists(os.path.join(args.source_path, "sparse")) and os.path.exists(os.path.join(args.source_path, "segmentations")):
            scene_info = sceneLoadTypeCallbacks["Colmap_mask"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "background_points.txt")):
            # DEFAULT loader
            print("Found Multi-view train dataset. Start processing")
            main_cam_id = args.main_camera if args.main_camera > 0 else None
                
            scene_info, cam_name, aux_cam_dict = sceneLoadTypeCallbacks["MVdataset"](
                                                                                    args.source_path, 
                                                                                    args.white_background, 
                                                                                    args.eval, 
                                                                                    main_cam_id=main_cam_id, 
                                                                                    load_aux_mv=load_aux_mv, 
                                                                                    fast_pcd_load=(self.loaded_iter),
                                                                                    camera_sampler=args.frame_sample_ratio
                                                                                    )
            self.cam_name = cam_name
            if len(aux_cam_dict)> 0:
                for aux_cam_id, aux_cam in aux_cam_dict.items():
                    self.aux_cam_dict[aux_cam_id] = cameraList_from_camInfos(aux_cam, resolution_scales[0], args, is_aux=True)

        else:
            main_cam_id = args.main_camera

            # choose which camera to use
            _cam_list = sorted(list(Path(args.source_path).iterdir()))
            cam_list = []
            for _cam in _cam_list:
                if _cam.is_dir() and (_cam / 'images').exists():
                    cam_list.append(_cam.name)
            # cam_list = [c_dir.parent.name for c_dir in cam_list]
            cam_name_dict = dict()
            print(f"cam list: {cam_list}")
            print(f"src path: {args.source_path}")
            for cam_name in cam_list:
                cam_name_dict[int(cam_name)] = cam_name
                print(int(cam_name))
                
            main_cam_name = cam_name_dict[main_cam_id]
            self.cam_name = main_cam_name

            if os.path.exists(os.path.join(args.source_path, main_cam_name, "background_points.txt")):
                print("[INFO] loading SLAMHR format dataset")
                scene_info = sceneLoadTypeCallbacks["HumanScene"](
                                                                os.path.join(args.source_path, main_cam_name), 
                                                                args.images, 
                                                                args.eval, 
                                                                flip_mask=args.flip_bg_mask, 
                                                                manual_smpl_scale=args.manual_smpl_global_scale
                                                                )
            else:
                assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
                
            # with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            #     json.dump(json_cams, file)

        # I turned off shuffle, as randomness is already prepared in train()
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            
            if fast_loader:
                self.train_cameras = scene_info.train_cameras
            else:
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            print("Loading Optimized Point Clouds")
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            if getattr(args, 'background_mode', False):
                print("making Inital Background Mode")
                self.gaussians.create_from_camera(self.train_cameras[resolution_scales[0]][0], self.cameras_extent, self.init_w_normal, self.init_opacity)

            else:
                print("Loading Initial Point Clouds")
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.init_w_normal, self.init_opacity)

    def save(self, iteration, smpl_params=None, deformer=None, people_infos: Optional[List]=None):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

        if smpl_params is not None:
            smpl_params = smpl_params.clone().detach().cpu()
            pandas.to_pickle(smpl_params, os.path.join(point_cloud_path, "smpl_params.pkl"))

        if deformer is not None:
            deformer.dump_trans_grid(Path(point_cloud_path))

        # save person infos
        if people_infos is not None:
            for pi in people_infos:
                # Though it's dimension could be different, we should handle that during data-loading time.
                smpl_poses = torch.cat([
                    pi.smpl_scale.reshape(1,1).repeat(len(pi.smpl_global_poses), 1),
                    pi.smpl_global_poses,
                    pi.smpl_local_poses
                ], dim=-1)
                pi.human_scene.save(iteration, smpl_params=smpl_poses, deformer=pi.smpl_deformer)  
                
            


    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    
        
    