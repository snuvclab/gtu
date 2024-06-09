import os
import json
from pathlib import Path

import pandas
import numpy as np

from gtu.dataset.scene.dataset_readers import readHumanSceneInfo, readHumanSceneInfo_MVdata
from gtu.dataset.system_utils import searchForMaxIteration
from gtu.dataset.scene import Scene, GaussianModel
from gtu.dataset.camera_utils import cameraList_from_camInfos, camera_to_JSON



class HumanScene(Scene):
    def __init__(
        self, 
        human_path: Path, 
        model_path: Path, 
        mask_path: str= "", 
        occ_mask_path: str="",
        load_iteration=None, 
        resolution_scales=[1.0], 
        scene_img_shape=None, 
        eval: bool=False, 
        view_dir_reg: bool=False, 
        sh_degree: int=0, 
        init_opacity: float=0.9, 
        scene_fids=[], 
        manual_smpl_scale=-1, 
        preload_masks: bool=False,
        dilate_human_occ_masks_ratio: float=0.,
        use_skip_fids_for_indiv_loading: bool=False,
        data_resolution=1,
        ):

        self.init_w_normal = True
        self.gaussians = GaussianModel(sh_degree)
        self.loaded_iter = None
        self.model_path = model_path
        self.preload_masks = preload_masks

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(str(self.model_path), "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        if human_path.name[-4:] == '.pkl':
            print("[INFO] loading MV dataset (This is standard approach in future)")
            if mask_path != "":
                print(f"[INFO] loading Mask for evaluation from {mask_path}")

            human_id = int(human_path.name[:-4].split("results_p")[-1])
            # Make dir of person, to save optimized SMPL parameters
            self.human_path = human_path.parent / str(human_id)
            self.human_path.mkdir(exist_ok=True)


            scene_info, human_data = readHumanSceneInfo_MVdata(
                human_path, 
                eval, 
                scene_img_shape=scene_img_shape, 
                scene_fids=scene_fids, 
                person_mask_path=mask_path,
                occ_mask_path=occ_mask_path,
                use_skip_fids_for_indiv_loading=use_skip_fids_for_indiv_loading,
                )
            self.beta = list(human_data.values())[0]['smpl_param'][0, -10:].astype(np.float32)

        else:
            print(f"{manual_smpl_scale}\n")
            
            self.human_path = Path(human_path) if isinstance(human_path, str) else human_path
            scene_info = readHumanSceneInfo(human_path, 'image', eval, scene_img_shape=scene_img_shape, manual_smpl_scale=manual_smpl_scale)

            # load mean_beta
            if (Path(self.human_path) / 'mean_shape.npy').exists():
                self.beta = np.load((Path(self.human_path)/ 'mean_shape.npy')).astype(np.float32)
            else:
                self.beta = np.zeros(10)



        # load iterations
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(str(self.model_path), "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            # with open(os.path.join(str(self.model_path), "cameras.json"), 'w') as file:
            #     json.dump(json_cams, file)

        # setting for define gaussian scale (which is critical for gaussin radius)        
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # make cameras
        data_device = "cpu"     ## for faster train, u can set it as 'cuda', but would be super heavy
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            
            if False:       # For fast-loading options (not for optimizations, option for debugging)
                self.train_cameras = scene_info.train_cameras
            else:
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                                    scene_info.train_cameras, 
                                    resolution_scale, 
                                    None,
                                    data_device, 
                                    data_resolution, 
                                    preload_masks=preload_masks,
                                    dilate_human_occ_masks_ratio=dilate_human_occ_masks_ratio
                                    )
            
            # print("Loading Test Cameras")
            # self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, None, data_device, data_resolution)

            if self.preload_masks:
                new_cam_list = []
                for _cam in self.train_cameras[resolution_scale]:
                    if _cam.gt_alpha_mask is None:
                        with open(os.path.join(str(self.model_path), 'log.txt'), 'a') as f:
                            f.write(f"[INFO] fid:{int(_cam.colmap_id):09} doesn't have gt_alpha mask. {str(human_path)}")
                    else:
                        new_cam_list.append(_cam)
                self.train_cameras[resolution_scale] = new_cam_list


        if self.loaded_iter:
            print("Loading Optimized Point Clouds")
            self.gaussians.load_ply(os.path.join(str(self.model_path),
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "point_cloud.ply"))
        else:
            print("Loading Initial Point Clouds")
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.init_w_normal, init_opacity)



    
     
    

