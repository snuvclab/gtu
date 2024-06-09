import os
import time
from pathlib import Path
from typing import List, Union, NamedTuple, Any, Optional, Dict

import torch
import pandas
import numpy as np
import cv2
from tqdm import tqdm, trange

from gtu.dataset.scene import Scene, GaussianModel
from gtu.dataset.scene.human_loader import HumanScene
from gtu.arguments import ModelParams, PipelineParams, RenderParams
from gtu.smpl_deformer.deformer import SMPLDeformer

from gtu.dataset.camera_utils import load_ns_traj_from_json
from utils.log_utils import print_cli
from utils.draw_op_jnts import smpl_joints2op_joints
from utils.graphics_utils import project_points_to_cam

from gtu.guidance.joint_utils import filter_invisible_face_joints_w_prompts, get_view_prompt_of_body, combine_prompts



class PersonTrain(NamedTuple):
    uids : Union[List, torch.Tensor]        # for trans_grids
    fids : Union[List, torch.Tensor]        # to check whether it's a scene requiring human
    model_path: Path
    smpl_local_poses: torch.Tensor
    smpl_global_poses: torch.Tensor
    local_pose_optimizer: Optional[torch.optim.Optimizer]
    global_pose_optimizer: Optional[torch.optim.Optimizer]
    smpl_scale_optimizer: Optional[torch.optim.Optimizer]
    beta: torch.Tensor
    smpl_deformer: SMPLDeformer
    gaussians: GaussianModel
    do_trans_grid: bool
    trans_grids: Optional[torch.Tensor]
    grid_optimizer: Optional[torch.optim.Optimizer]
    view_dir_reg: bool
    human_scene: Scene
    human_id: str
    smpl_scale: torch.Tensor
    init_smpl_jnts: torch.Tensor
    cam_centers: torch.Tensor
    cc_smpl_dir: torch.Tensor    
    representative_img: torch.Tensor
    misc: Dict   


def load_scene_human(
    dataset : ModelParams, 
    pipe : PipelineParams, 
    scene_datasets: Optional[RenderParams]=None, 
    iteration : int=-1, 
    exp_name: str="debug", 
    human_tracker_name: Optional[str]=None,        # Option for render-time version
    novel_view_traj: List=[],
    device = torch.device("cuda:0"),
    is_train: bool=False,
    for_ti: bool=False,
    fit_scale: bool=False,
    load_aux_mv: bool=True,
    human_train_opt = None,
    checkpoint = -1,
    skip_loading_bg: bool=False,
    **kwargs
    ):

    # prepare datasets to train
    scene_gaussians = GaussianModel(dataset.sh_degree)

    if for_ti:
        print_cli("TI loading mode", "info")
        scene = Scene(dataset, scene_gaussians, load_iteration=False, view_dir_reg=pipe.view_dir_reg, load_aux_mv=load_aux_mv, exp_name=None)

    elif is_train:
        scene_iteration = iteration
        if dataset.optimize_bg_from_zero or skip_loading_bg:
            print("\n[INFO]Optimizing background from zero here\n")
            scene_iteration = False

        scene = Scene(dataset, scene_gaussians, load_iteration=scene_iteration, view_dir_reg=pipe.view_dir_reg, load_aux_mv=load_aux_mv, exp_name=None)      # load from checkpoint
        scene.model_path = str(os.path.join(str(scene.model_path), exp_name))
        os.makedirs(scene.model_path, exist_ok=True)
    else:
        scene = Scene(dataset, scene_gaussians, load_iteration=iteration, view_dir_reg=pipe.view_dir_reg, load_aux_mv=load_aux_mv, exp_name=exp_name)      # load from checkpoint


    # load human datasets
    smpl_canon_scale = 1.0      # It's fixed value! (But we need to consider, applying different weighting on pruning / dividing)
    human_sh_degree = human_train_opt.sh_degree     # here I hard-coded human sh-degree
    human_view_dir_reg = human_train_opt.view_dir_reg
    people_infos = []

    load_fitted_smpl = False

    scene_cams = scene.getTrainCameras()
    scene_image_shape = (scene_cams[0].image_width, scene_cams[0].image_height)
    scene_cam_dict = dict()
    for cam in scene_cams:
        fid = cam.colmap_id
        scene_cam_dict[fid] = cam
    scene_fids = sorted(list(scene_cam_dict.keys()))


    # Load occlusion masks
    if dataset.occlusion_mask_path != "" and dataset.occlusion_mask_path != "none":
        print("try to load front-occlusion masks")
        if scene.cam_name is None:
            raise NotImplementedError(f"[ERROR] mask_path isn't supported for non-mv dataset, yet")

        occ_mask_path = os.path.join(dataset.occlusion_mask_path, scene.cam_name)
        occ_cam_dict = dict()
        occ_mask_path = Path(occ_mask_path)

        assert occ_mask_path.exists(), f"{str(occ_mask_path)} does not exist here"

        for mask_fname in sorted(list(occ_mask_path.glob("*.png"))+list(occ_mask_path.glob("*.jpg"))):
            fid = int(mask_fname.name.split(".")[0])
            occ_cam_dict[fid] = torch.from_numpy(cv2.imread(str(mask_fname), 0)>1).float().squeeze()

        scene.occ_cam_dict = occ_cam_dict
    else:
        scene.occ_cam_dict = None



    if len(novel_view_traj) > 0:
        traj_transform_path = Path(str(dataset.source_path)) / 'dataparser_transforms.json'
        print(traj_transform_path)
        traj_transform_path = traj_transform_path if traj_transform_path.exists() else None
        print("[INFO] traj_transform exists: ",  traj_transform_path is not None)
        render_cameras = load_ns_traj_from_json(novel_view_traj[0], torch.device('cpu'), traj_transform_path)
    else:
        render_cameras = scene_cams

    if human_tracker_name is not None:
        dataset.human_track_method = human_tracker_name


    # Check human_camera paths
    if len(dataset.human_camera_paths) == 0:
        human_camera_paths = []
        human_camera_paths_dict = dict()
        # Let's find human camera path
        if dataset.human_track_method == "alphapose":
            data_path = Path(dataset.source_path) / "segmentations" / "alphapose" / "indiv"
            human_candidates = data_path.glob("**/romp")
            
            for hc in human_candidates:
                n_human = len(list(hc.glob("*.npz")))
                
                if n_human > 5 and (hc.parent / "points3D.txt").exists():
                    human_camera_paths.append(str(hc.parent))
                    human_id = os.path.basename(str(hc.parent))
                    human_camera_paths_dict[human_id] = hc.parent

        elif dataset.human_track_method == "phalp":
            data_path = Path(dataset.source_path) / "segmentations" / "phalp" / "indiv"
            if not data_path.exists():
                data_path = Path(dataset.source_path) / "segmentations" / "phalp_v2" / "indiv"

            human_candidates = data_path.glob("**/points3D.txt")
            
            print(data_path)
            for hc in human_candidates:
                human_camera_paths.append(str(hc.parent))
                human_id = os.path.basename(str(hc.parent))
                human_camera_paths_dict[human_id] = hc.parent

        elif dataset.human_track_method == "multiview":
            data_path = Path(dataset.source_path) / scene.cam_name
            data_lists = sorted(list(data_path.glob("results_*.pkl")))
            for data_file in data_lists:
                human_camera_paths.append(str(data_file))
                human_id = data_file.name[:-4].split("results_p")[-1]
                human_camera_paths_dict[human_id] = data_file

        else:
            raise AssertionError("Wrong human_tracker method name")
    else:
        human_camera_paths = dataset.human_camera_paths

        for hc in human_camera_paths:
            human_id = os.path.basename(str(hc))
            human_camera_paths_dict[human_id] = Path(str(hc))
            
            
    # Check SMPL Gender files exists
    if (data_path / 'smpl_gender.npy').exists():
        smpl_genders = np.load(data_path / 'smpl_gender.npy')
    elif (data_path.parent / 'smpl_gender.npy').exists():
        smpl_genders = np.load(data_path.parent / 'smpl_gender.npy')
    else:
        smpl_genders = ['neutral' for _ in range(len(human_camera_paths_dict))]
        
    # Filter out pid if we want to optimize single person
    preload_masks = dataset.preload_human_masks 
    if not (dataset.target_pid < 0):
        raw_pids = sorted(list(human_camera_paths_dict.keys()))

        del_ids = []
        for human_id in human_camera_paths_dict.keys():
            if int(human_id) == int(dataset.target_pid):
                continue
            else:
                del_ids.append(human_id)

        for human_id in del_ids:
            del human_camera_paths_dict[human_id]
                
        assert len(human_camera_paths_dict) == 1, f"{dataset.target_pid} is not in {raw_pids}"
        
        print(f"[INFO] We only use {dataset.target_pid} for optimization here")
        
        if not preload_masks:
            print(f"[WARNGING] We set 'preload_masks = True' as it's single-person optimization case")

            
        
        

    # Start loading human datas
    if for_ti:
        # Load human datas

        for _idx, human_id in enumerate(sorted(list(human_camera_paths_dict.keys()))):
            human_camera_path = human_camera_paths_dict[human_id]
            manual_smpl_scale = dataset.manual_smpl_global_scale if not isinstance(dataset.manual_smpl_global_scale, list) else dataset.manual_smpl_global_scale[_idx]
            
            human_model_path = Path(scene.model_path) / 'humans' / human_id
            human_model_path.mkdir(exist_ok=True, parents=True)

            aux_cam_mask_fname_dict = dict()
            if dataset.mask_path != "":
                if scene.cam_name is not None:
                #     raise NotImplementedError(f"[ERROR] mask_path isn't supported for non-mv dataset, yet")
                    person_mask_path = os.path.join(dataset.mask_path, scene.cam_name, human_id, 'masks')
                    if not Path(person_mask_path).exists():
                        person_mask_path = os.path.join(dataset.mask_path, scene.cam_name, f"{int(human_id):03}", 'masks')
                    if not Path(person_mask_path).exists():                    
                        person_mask_path = os.path.join(dataset.mask_path, scene.cam_name, 'segmentations', 'masks')

                # case loading from phalp
                else:                  
                    person_mask_path = os.path.join(dataset.mask_path, f"{int(human_id):04}", 'raw_mask')
                    if not Path(person_mask_path).exists():     
                        print(f"[WARN WARN WARN] failed to load person mask.")
                        person_mask_path = None

            else:
                person_mask_path = ""
            human_scene = HumanScene(
                                    human_camera_path, 
                                    human_model_path, 
                                    scene_img_shape=scene_image_shape, 
                                    eval=False, 
                                    view_dir_reg=human_view_dir_reg, 
                                    sh_degree=human_sh_degree, 
                                    scene_fids=scene_fids, 
                                    mask_path=person_mask_path,
                                    manual_smpl_scale=manual_smpl_scale
                                    )
            
            
            # we need human_data path (to load poses)
            # data_path = human_data_path / human_id
            mean_beta = torch.from_numpy(human_scene.beta).float().cuda()
            cameras = human_scene.getTrainCameras()

            person_gaussians = human_scene.gaussians
            original_frame_ids = [cam.colmap_id for cam in cameras]

            print(f"[INFO] Loading human in {str(human_model_path)}\n")
            cameras = human_scene.getTrainCameras()
            smpl_local_poses = []
            smpl_global_poses = []
            smpl_params = []
            cam_centers = []
            
            largest_res = -1
            largest_img = None
            for cam in cameras:
                smpl_local_poses.append(cam.smpl_param[:,4:76])
                smpl_global_poses.append(cam.smpl_param[:,1:4])
                smpl_params.append(cam.smpl_param.clone().detach().float())

                fid = cam.colmap_id
                cam_center = scene_cam_dict[fid].camera_center.clone().detach().float().squeeze()
                cam_centers.append(cam_center)          # It's detached, no loss flow anymore.

                if max(cam.original_image.shape) > largest_res:
                    largest_img = cam.original_image
                    largest_res = max(cam.original_image.shape)
                
            smpl_local_poses = torch.cat(smpl_local_poses, dim=0)
            smpl_local_poses = smpl_local_poses.float().cuda()

            smpl_global_poses = torch.cat(smpl_global_poses, dim=0)
            smpl_global_poses = smpl_global_poses.float().cuda()

            init_smpl_params = torch.cat(smpl_params, dim=0).cuda()
            smpl_scale = init_smpl_params[:, 0].mean().detach()


            # define defome deformer
            uids = [cam.uid for cam in cameras]
            fids = [cam.colmap_id for cam in cameras]
            beta = torch.from_numpy(human_scene.beta).float().cuda()
            smpl_gender = smpl_genders[_idx]
            smpl_deformer = SMPLDeformer(gender=smpl_gender, beta=beta, smpl_scale=smpl_canon_scale)

            # extract initial joints
            smpl_output = smpl_deformer.smpl_server(init_smpl_params)
            smpl_jnts = smpl_output['smpl_jnts'].detach().cpu()
            smpl_verts = smpl_output['smpl_verts'].detach().cpu()

            
            
            uids = [cam.uid for cam in cameras]
            fids = [cam.colmap_id for cam in cameras]
            beta = mean_beta.float().cuda()
            # smpl_poses = smpl_poses.cuda()
            
            largest_res = -1
            largest_img = None
            for cam in cameras:         # It's detached, no loss flow anymore.
                if max(cam.original_image.shape) > largest_res:
                    largest_img = cam.original_image
                    largest_res = max(cam.original_image.shape)
                    
            people_infos.append(
                PersonTrain(
                    uids = uids,
                    fids = fids,
                    smpl_local_poses = smpl_local_poses,
                    smpl_scale = smpl_scale,
                    smpl_global_poses = smpl_global_poses,
                    local_pose_optimizer = None,
                    global_pose_optimizer = None,
                    smpl_scale_optimizer = None,
                    model_path = human_model_path,
                    beta = beta,
                    smpl_deformer = smpl_deformer,
                    gaussians = person_gaussians,
                    do_trans_grid = False,
                    trans_grids = None,
                    grid_optimizer = None,
                    view_dir_reg = human_view_dir_reg,
                    human_scene = human_scene,
                    human_id = human_id,
                    init_smpl_jnts = smpl_jnts,
                    cam_centers = None,
                    cc_smpl_dir = None,
                    representative_img = largest_img,
                    misc = dict()
                )
            )

            project_op_jnts = []
            op_3d_jnts = []
            new_prompts = []
            for idx, fid in enumerate(fids):
                scene_cam = scene_cam_dict[fid]
                smpl_jnt = smpl_jnts[idx].clone().detach().cpu()
                pj_jnts = project_points_to_cam(scene_cam, smpl_jnt.squeeze().numpy(), image_res=None)
                op_joints = smpl_joints2op_joints(pj_jnts)
                
                op_3d_jnt = smpl_joints2op_joints(smpl_jnt.squeeze().numpy())
                lower_body_prompt = get_view_prompt_of_body(op_3d_jnt, scene_cam, is_lower_body=True)
                upper_body_prompt = get_view_prompt_of_body(op_3d_jnt, scene_cam, is_lower_body=False)
                filtered_op_3d_jnt, head_prompt = filter_invisible_face_joints_w_prompts(op_3d_jnt, scene_cam)
                image_res = (scene_cam.image_height, scene_cam.image_width)
                new_prompt = combine_prompts(head_prompt, upper_body_prompt, lower_body_prompt, op_joints, image_res)
                
                for idx, _op_3d_jnt in enumerate(filtered_op_3d_jnt):
                    if _op_3d_jnt is None:
                        op_joints[idx] = None
                        # op_joints[idx][0] = -1
                        # op_joints[idx][1] = -1

                
                project_op_jnts.append(op_joints)
                op_3d_jnts.append(op_3d_jnt)
                new_prompts.append(new_prompt)
            people_infos[-1].misc['projected_op_jnts'] = project_op_jnts
            people_infos[-1].misc['3d_op_jnts'] = op_3d_jnts
            people_infos[-1].misc['body_prompts'] = new_prompts
            
            
            


        return scene, people_infos
    


    elif is_train:
        # load datas
        for _idx, human_id in enumerate(sorted(list(human_camera_paths_dict.keys()))):
            human_camera_path = human_camera_paths_dict[human_id]
            manual_smpl_scale = dataset.manual_smpl_global_scale if not isinstance(dataset.manual_smpl_global_scale, list) else dataset.manual_smpl_global_scale[_idx]
            
            human_model_path = Path(scene.model_path) / 'humans' / human_id
            human_model_path.mkdir(exist_ok=True, parents=True)

            aux_cam_mask_fname_dict = dict()
            if dataset.mask_path != "":
                if scene.cam_name is None:
                    raise NotImplementedError(f"[ERROR] mask_path isn't supported for non-mv dataset, yet")
                person_mask_path = os.path.join(dataset.mask_path, scene.cam_name, human_id, 'masks')
                if not Path(person_mask_path).exists():
                    person_mask_path = os.path.join(dataset.mask_path, scene.cam_name, f"{int(human_id):03}", 'masks')
                if not Path(person_mask_path).exists():
                    person_mask_path = os.path.join(dataset.mask_path, scene.cam_name, 'segmentations', 'masks')

                occ_mask_path = os.path.join(dataset.mask_path, scene.cam_name, human_id, 'occ_masks')
                if not Path(occ_mask_path).exists():
                    occ_mask_path = os.path.join(dataset.mask_path, scene.cam_name, f"{int(human_id):03}", 'occ_masks')
                if not Path(occ_mask_path).exists():
                    occ_mask_path = ""

                # load mask for unseen cameras, too (just file name)
                for aux_cam_name, aux_cams in scene.aux_cam_dict.items():
                    aux_cam_mask_fname_dict[aux_cam_name] = dict()

                    _mask_path = Path(os.path.join(dataset.mask_path, aux_cam_name, human_id, 'masks'))
                    if not _mask_path.exists():
                        _mask_path = Path(os.path.join(dataset.mask_path, aux_cam_name, 'segmentations', 'masks'))

                    for mask_fname in sorted(list(_mask_path.glob("*.png"))+list(_mask_path.glob("*.jpg"))):
                        fid = int(mask_fname.name.split(".")[0])
                        aux_cam_mask_fname_dict[aux_cam_name][fid] = mask_fname

            else:
                person_mask_path = ""
                occ_mask_path = ""

            human_scene = HumanScene(
                                    human_camera_path, 
                                    human_model_path, 
                                    scene_img_shape=scene_image_shape, 
                                    eval=False, 
                                    view_dir_reg=human_view_dir_reg, 
                                    sh_degree=human_sh_degree, 
                                    scene_fids=scene_fids, 
                                    mask_path=person_mask_path,
                                    occ_mask_path=occ_mask_path,
                                    manual_smpl_scale=manual_smpl_scale,
                                    preload_masks=preload_masks,
                                    dilate_human_occ_masks_ratio=dataset.dilate_human_occ_masks_ratio,
                                    use_skip_fids_for_indiv_loading=dataset.use_skip_fids_for_indiv_loading,
                                    data_resolution=dataset.resolution
                                    )

            # extract smpl poses
            print(f"[INFO] Loading human in {str(human_model_path)}\n")
            cameras = human_scene.getTrainCameras()
            smpl_local_poses = []
            smpl_global_poses = []
            smpl_params = []
            cam_centers = []
            
            largest_res = -1
            largest_img = None
            for cam in cameras:
                smpl_local_poses.append(cam.smpl_param[:,4:76])
                smpl_global_poses.append(cam.smpl_param[:,1:4])
                smpl_params.append(cam.smpl_param.clone().detach().float())

                fid = cam.colmap_id
                cam_center = scene_cam_dict[fid].camera_center.clone().detach().float().squeeze()
                cam_centers.append(cam_center)          # It's detached, no loss flow anymore.

                if max(cam.original_image.shape) > largest_res:
                    largest_img = cam.original_image
                    largest_res = max(cam.original_image.shape)
                
            smpl_local_poses = torch.cat(smpl_local_poses, dim=0)
            smpl_local_poses = smpl_local_poses.float().cuda()
            smpl_local_poses = smpl_local_poses.requires_grad_()

            smpl_global_poses = torch.cat(smpl_global_poses, dim=0)
            smpl_global_poses = smpl_global_poses.float().cuda()
            smpl_global_poses = smpl_global_poses.requires_grad_()


            # calculate cam -> smpl center direction
            cam_centers = torch.stack(cam_centers, dim=0)
            cc_smpl_dir = smpl_global_poses.clone().detach() - cam_centers
            # cc_smpl_dir = cc_smpl_dir / cc_smpl_dir.norm(dim=-1, keepdim=True)        # as scale == 1 for al initial, we can use distance-weighted direction simply instead.


            init_smpl_params = torch.cat(smpl_params, dim=0).cuda()
            smpl_scale = init_smpl_params[:, 0].mean().detach()
            smpl_scale = smpl_scale.requires_grad_()

            local_pose_optimizer = torch.optim.Adam([smpl_local_poses], lr=1e-4)
            global_pose_optimizer = torch.optim.Adam([smpl_global_poses], lr=1e-4)
            smpl_scale_optimizer = torch.optim.Adam([smpl_scale], lr=1e-4)
            smpl_scale = smpl_scale.reshape(-1)



            # define defome deformer
            uids = [cam.uid for cam in cameras]
            fids = [cam.colmap_id for cam in cameras]
            beta = torch.from_numpy(human_scene.beta).float().cuda()
            
            smpl_gender = smpl_genders[_idx]
            smpl_deformer = SMPLDeformer(gender=smpl_gender, beta=beta, smpl_scale=smpl_canon_scale)

            # extract initial joints
            smpl_output = smpl_deformer.smpl_server(init_smpl_params)
            smpl_jnts = smpl_output['smpl_jnts'].detach().cpu()
            smpl_verts = smpl_output['smpl_verts'].detach().cpu()

            
            # Also initialize transform grida, if needed
            do_trans_grid = dataset.use_trans_grid
            if do_trans_grid:
                n_frames = len(human_scene.getTrainCameras())
                trans_grids, grid_optimizer = smpl_deformer.activate_trans_grid(n_frames=n_frames)

                if trans_grids is not None:
                    print(f"[INFO] trans grids activated, n_frames: {n_frames}")
                else:
                    print(f"[INFO] trans grids failed to be initalized. invalid n_frames: {n_frames}")
                    do_trans_grid = False
            else:
                trans_grids = None
                grid_optimizer = None



            # set training settings
            person_gaussian = human_scene.gaussians
            person_gaussian.training_setup(human_train_opt)

            
            if checkpoint and False:
                # it's option loading human model from specific directory (which is not capable now)
                person_checkpoint = checkpoint
                (model_params, first_iter) = torch.load(checkpoint)
                person_gaussian.restore(model_params, opt)


            if (human_scene.human_path / "init_fitted_smpl.pkl").exists() and not fit_scale:
                save_dict = pandas.read_pickle(human_scene.human_path / "init_fitted_smpl.pkl")

                smpl_scale = save_dict['smpl_scale'].to(smpl_scale.device)
                smpl_global_poses = save_dict['smpl_global_pose'].float().cuda()
                smpl_global_poses = smpl_global_poses.requires_grad_()
                global_pose_optimizer = torch.optim.Adam([smpl_global_poses], lr=1e-4)
                    
                load_fitted_smpl = True
            
                
    
            #### TODO Need to combine "different" Spherical Harmonics Dimensions in single space
            people_infos.append(
                PersonTrain(
                    uids = uids,
                    fids = fids,
                    smpl_local_poses = smpl_local_poses,
                    smpl_scale = smpl_scale,
                    smpl_global_poses = smpl_global_poses,
                    local_pose_optimizer = local_pose_optimizer,
                    global_pose_optimizer = global_pose_optimizer,
                    smpl_scale_optimizer = smpl_scale_optimizer,
                    model_path = human_model_path,
                    beta = beta,
                    smpl_deformer = smpl_deformer,
                    gaussians = person_gaussian,
                    do_trans_grid = do_trans_grid,
                    trans_grids = trans_grids,
                    grid_optimizer = grid_optimizer,
                    view_dir_reg = human_view_dir_reg,
                    human_scene = human_scene,
                    human_id = human_id,
                    init_smpl_jnts = smpl_jnts,
                    cam_centers = cam_centers,
                    cc_smpl_dir = cc_smpl_dir,
                    representative_img = largest_img,
                    misc = dict(
                        optimized_step=0
                    )
                )
            )

            # Set initial vertices
            people_infos[-1].misc['smpl_verts'] = smpl_verts
            
            # Set masks of aux-camera (for future validation)
            if len(aux_cam_mask_fname_dict) > 0:
                people_infos[-1].misc['aux_cam_mask_fname_dict'] = aux_cam_mask_fname_dict
            
            # get pixel distribution
            if person_mask_path != "":
                pixel_lists = []
                print("[INFO] start loading color distribution")
                
                if False:
                    pixel_list_cache_dir = os.path.join(person_mask_path, "..", "pixel_list.pt")
                    if os.path.exists(pixel_list_cache_dir) and True:
                        print(f"\n[INFO] Loading pixel list from cached dir {pixel_list_cache_dir}\n")
                        pixel_list = torch.load(pixel_list_cache_dir)
                    else:
                        print(f"[INFO] Calculating pixel list...")
                        start = time.time()
                        for cam_id, cam in scene_cam_dict.items():
                            fid = cam.colmap_id
                            if not (fid in people_infos[-1].fids):
                                continue
                            _data_idx = people_infos[-1].fids.index(fid)
                                
                            gt_image = cam.original_image.clone().detach()
                            person_cam = people_infos[-1].human_scene.getTrainCameras()[_data_idx]
                            if person_cam.gt_alpha_mask is None:
                                continue
                            
                            # remove invalid pixels
                            gt_image[person_cam.gt_alpha_mask.repeat(3,1,1) == 0] = -1
                            
                            # extract pixels
                            pixels = gt_image.reshape(3,-1).T   # (N_pixel, 3)
                            pixel_lists.append(pixels)
                        
                        # remove duplicated pixels
                        if len(pixel_lists) > 0:
                            pixel_list = torch.cat(pixel_lists, dim=0)
                            pixel_list = torch.unique(pixel_list, dim=0)
                            pixel_list = pixel_list[pixel_list.sum(-1) >= 0]        # remove invalid pixels
                            print(f"[INFO] loaded pixel distribution of person {human_id}: ({time.time() - start} seconds)")
                            torch.save(pixel_list, pixel_list_cache_dir)
                        else:
                            pixel_list = None
                            torch.save(None, pixel_list_cache_dir)
                        
                    people_infos[-1].misc['color_distribution'] = pixel_list
            
    else:
        print("[INFO] loading data for testing")
        # 1. Select random path
        if True:
            scene_image_shape = (scene_cams[0].image_width, scene_cams[0].image_height)
            max_fid = max([cam.colmap_id for cam in scene_cams])
            

            # 2. load human datasets
            # First, automatically load trained human data
            human_model_path = Path(scene.model_path) / 'humans'
            people_infos = []
            if human_model_path.exists():
                # load human models in dictionary format.
                human_chkpts = human_model_path.glob("**/point_cloud")
                human_chkpts = sorted(list(human_chkpts))
                for _idx, human_chkpt in enumerate(human_chkpts):
                    human_id = human_chkpt.parent.name
                    human_camera_path = human_camera_paths_dict[human_id]
                    if not human_camera_path.exists():
                        continue
                    
                    manual_smpl_scale = dataset.manual_smpl_global_scale if not isinstance(dataset.manual_smpl_global_scale, list) else dataset.manual_smpl_global_scale[_idx]
            
                    aux_cam_mask_fname_dict = dict()
                    if dataset.mask_path != "":
                        if scene.cam_name is None:
                            raise NotImplementedError(f"[ERROR] mask_path isn't supported for non-mv dataset, yet")

                        person_mask_path = os.path.join(dataset.mask_path, scene.cam_name, human_id, 'masks')
                        if not Path(person_mask_path).exists():
                            person_mask_path = os.path.join(dataset.mask_path, scene.cam_name, f"{int(human_id):03}", 'masks')
                        if not Path(person_mask_path).exists():
                            person_mask_path = os.path.join(dataset.mask_path, scene.cam_name, 'segmentations', 'masks')

                        # load mask for unseen cameras, too (just file name)
                        for aux_cam_name, aux_cams in scene.aux_cam_dict.items():
                            aux_cam_mask_fname_dict[aux_cam_name] = dict()

                            _mask_path = Path(os.path.join(dataset.mask_path, aux_cam_name, human_id, 'masks'))
                            if not _mask_path.exists():
                                _mask_path = Path(os.path.join(dataset.mask_path, aux_cam_name, 'segmentations', 'masks'))
                            if not _mask_path.exists():
                                _mask_path = Path(os.path.join(dataset.mask_path, aux_cam_name, f"{int(human_id):03}", 'masks'))

                            for mask_fname in sorted(list(_mask_path.glob("*.png"))+list(_mask_path.glob("*.jpg"))):
                                fid = int(mask_fname.name.split(".")[0])
                                aux_cam_mask_fname_dict[aux_cam_name][fid] = mask_fname
                            print(f"{len(aux_cam_mask_fname_dict[aux_cam_name])} for hid: {human_id}, cam_id {aux_cam_name}")

                    else:
                        person_mask_path = ""

                    human_scene = HumanScene(
                                            human_camera_path, 
                                            human_model_path / human_id, 
                                            scene_img_shape=scene_image_shape, 
                                            eval=True, 
                                            view_dir_reg=human_view_dir_reg, 
                                            sh_degree=human_sh_degree, 
                                            scene_fids=scene_fids, 
                                            mask_path=person_mask_path,
                                            load_iteration=iteration,
                                            manual_smpl_scale=manual_smpl_scale
                                            )

                    model_path = human_chkpt / ("iteration_" + str(human_scene.loaded_iter))
                    
                    
                    
                    # we need human_data path (to load poses)
                    # data_path = human_data_path / human_id
                    if dataset.use_data_beta_for_test:
                        first_cam = human_scene.getTrainCameras()[0]
                        first_beta = first_cam.smpl_param[:,-10:].clone().detach().cpu().squeeze()
                        mean_beta = first_beta
                    else:
                        mean_beta = torch.tensor(np.load(model_path / 'mean_shape.npy')).squeeze()
                        
                    
                    try:
                        smpl_poses = pandas.read_pickle(model_path / "smpl_params.pkl")
                    except:
                        import pickle5
                        with open(model_path / "smpl_params.pkl", 'rb') as f:
                            smpl_poses = pickle5.load(f)
                    cameras = human_scene.getTrainCameras()
                         
                        
                    # load smpl_defomer
                    do_trans_grid = dataset.use_trans_grid
                    do_trans_grid = False if len(cameras) != len(smpl_poses) else do_trans_grid
                    smpl_gender = smpl_genders[_idx]
                    smpl_deformer = SMPLDeformer(gender=smpl_gender, beta=mean_beta, smpl_scale=smpl_canon_scale)
                    if do_trans_grid:
                        smpl_deformer.load_trans_grid(path=model_path, load_iteration=False)
                    

                    person_gaussians = human_scene.gaussians
                    original_frame_ids = [cam.colmap_id for cam in cameras]

                    print(f"[INFO] Loading human in {str(human_model_path)}\n")
                    cameras = human_scene.getTrainCameras()
                    smpl_local_poses = []
                    smpl_global_poses = []
                    smpl_params = []
                    cam_centers = []
                    
                    largest_res = -1
                    largest_img = None
                    for cam in cameras:
                        smpl_local_poses.append(cam.smpl_param[:,4:76])
                        smpl_global_poses.append(cam.smpl_param[:,1:4])
                        smpl_params.append(cam.smpl_param.clone().detach().float())

                        fid = cam.colmap_id
                        cam_center = scene_cam_dict[fid].camera_center.clone().detach().float().squeeze()
                        cam_centers.append(cam_center)          # It's detached, no loss flow anymore.

                        if max(cam.original_image.shape) > largest_res:
                            largest_img = cam.original_image
                            largest_res = max(cam.original_image.shape)
                        
                    smpl_local_poses = torch.cat(smpl_local_poses, dim=0)
                    smpl_local_poses = smpl_local_poses.float().cuda()

                    smpl_global_poses = torch.cat(smpl_global_poses, dim=0)
                    smpl_global_poses = smpl_global_poses.float().cuda()

                    init_smpl_params = torch.cat(smpl_params, dim=0).cuda()
                    smpl_scale = init_smpl_params[:, 0].mean().detach()

                    
                    
                    uids = [cam.uid for cam in cameras]
                    fids = [cam.colmap_id for cam in cameras]
                    beta = mean_beta.float().cuda()
                    # smpl_poses = smpl_poses.cuda()
                    
                    largest_res = -1
                    largest_img = None
                    for cam in cameras:         # It's detached, no loss flow anymore.
                        if max(cam.original_image.shape) > largest_res:
                            largest_img = cam.original_image
                            largest_res = max(cam.original_image.shape)
                            
                    people_infos.append(
                        PersonTrain(
                            uids = uids,
                            fids = fids,
                            smpl_local_poses = smpl_local_poses,
                            smpl_scale = smpl_scale,
                            smpl_global_poses = smpl_global_poses,
                            local_pose_optimizer = None,
                            global_pose_optimizer = None,
                            smpl_scale_optimizer = None,
                            model_path = human_model_path,
                            beta = beta,
                            smpl_deformer = smpl_deformer,
                            gaussians = person_gaussians,
                            do_trans_grid = do_trans_grid,
                            trans_grids = None,
                            grid_optimizer = None,
                            view_dir_reg = human_view_dir_reg,
                            human_scene = human_scene,
                            human_id = human_id,
                            init_smpl_jnts = None,
                            cam_centers = None,
                            cc_smpl_dir = None,
                            representative_img = largest_img,
                            misc = dict()
                        )
                    )

                    # Set masks of aux-camera (for future validation)
                    if len(aux_cam_mask_fname_dict) > 0:
                        people_infos[-1].misc['aux_cam_mask_fname_dict'] = aux_cam_mask_fname_dict
                    
            if len(people_infos) == 0:
                print(f"[INFO] no human chkpt path {str(human_model_path)}, skip loading people")

    if len(scene.aux_cam_dict) > 0:
        # it means it's mv dataset. No need of fitting scale / transls
        load_fitted_smpl = True


    if is_train:
        return scene, scene_gaussians, people_infos, render_cameras, load_fitted_smpl
    else:
        return scene, scene_gaussians, people_infos, render_cameras




def load_init_person_info(dataset, scene):

    scene_cams = scene.getTrainCameras()
    scene_image_shape = (scene_cams[0].image_width, scene_cams[0].image_height)
    scene_cam_dict = dict()
    for cam in scene_cams:
        fid = cam.colmap_id
        scene_cam_dict[fid] = cam
    scene_fids = sorted(list(scene_cam_dict.keys()))

    # Check human_camera paths
    if len(dataset.human_camera_paths) == 0:
        human_camera_paths = []
        human_camera_paths_dict = dict()
        # Let's find human camera path
        if dataset.human_track_method == "alphapose":
            data_path = Path(dataset.source_path) / "segmentations" / "alphapose" / "indiv"
            human_candidates = data_path.glob("**/romp")
            
            for hc in human_candidates:
                n_human = len(list(hc.glob("*.npz")))
                
                if n_human > 5 and (hc.parent / "points3D.txt").exists():
                    human_camera_paths.append(str(hc.parent))
                    human_id = os.path.basename(str(hc.parent))
                    human_camera_paths_dict[human_id] = hc.parent

        elif dataset.human_track_method == "phalp":
            data_path = Path(dataset.source_path) / "segmentations" / "phalp" / "indiv"
            if not data_path.exists():
                data_path = Path(dataset.source_path) / "segmentations" / "phalp_v2" / "indiv"

            human_candidates = data_path.glob("**/points3D.txt")
            
            print(data_path)
            for hc in human_candidates:
                human_camera_paths.append(str(hc.parent))
                human_id = os.path.basename(str(hc.parent))
                human_camera_paths_dict[human_id] = hc.parent

        elif dataset.human_track_method == "multiview":
            data_path = Path(dataset.source_path) / scene.cam_name
            data_lists = sorted(list(data_path.glob("results_*.pkl")))
            for data_file in data_lists:
                human_camera_paths.append(str(data_file))
                human_id = data_file.name[:-4].split("results_p")[-1]
                human_camera_paths_dict[human_id] = data_file

        else:
            raise AssertionError("Wrong human_tracker method name")
    else:
        human_camera_paths = dataset.human_camera_paths

        for hc in human_camera_paths:
            human_id = os.path.basename(str(hc))
            human_camera_paths_dict[human_id] = Path(str(hc))


    human_model_path = Path(scene.model_path) / 'humans' 
    
    human_chkpts = sorted(list(human_model_path.glob("**/point_cloud")))
    human_id = human_chkpts[0].parent.name
    human_camera_path = human_camera_paths_dict[human_id]


    human_scene = HumanScene(
                            human_camera_path, 
                            human_model_path / human_id, 
                            scene_img_shape=scene_image_shape, 
                            eval=True, 
                            view_dir_reg=True, 
                            sh_degree=0, 
                            scene_fids=scene_fids, 
                            mask_path="",
                            load_iteration=False
                            )

    # set training settings
    person_gaussian = human_scene.gaussians
    cameras = human_scene.getTrainCameras()
    smpl_local_poses = []
    smpl_global_poses = []
    smpl_params = []
    cam_centers = []
    
    largest_res = -1
    largest_img = None
    for cam in cameras:
        smpl_local_poses.append(cam.smpl_param[:,4:76])
        smpl_global_poses.append(cam.smpl_param[:,1:4])
        smpl_params.append(cam.smpl_param.clone().detach().float())

        fid = cam.colmap_id
        cam_center = scene_cam_dict[fid].camera_center.clone().detach().float().squeeze()
        cam_centers.append(cam_center)          # It's detached, no loss flow anymore.

        if max(cam.original_image.shape) > largest_res:
            largest_img = cam.original_image
            largest_res = max(cam.original_image.shape)
        
    smpl_local_poses = torch.cat(smpl_local_poses, dim=0)
    smpl_local_poses = smpl_local_poses.float().cuda()

    smpl_global_poses = torch.cat(smpl_global_poses, dim=0)
    smpl_global_poses = smpl_global_poses.float().cuda()


    # cc_smpl_dir = cc_smpl_dir / cc_smpl_dir.norm(dim=-1, keepdim=True)        # as scale == 1 for al initial, we can use distance-weighted direction simply instead.


    init_smpl_params = torch.cat(smpl_params, dim=0).cuda()
    smpl_scale = init_smpl_params[:, 0].mean().detach()
    smpl_scale = smpl_scale.reshape(-1)



    # define defome deformer
    uids = [cam.uid for cam in cameras]
    fids = [cam.colmap_id for cam in cameras]
    beta = torch.from_numpy(human_scene.beta).float().cuda()
    smpl_deformer = SMPLDeformer(gender='neutral', beta=beta, smpl_scale=1.)
    
    
    person_info = PersonTrain(
        uids = uids,
        fids = fids,
        smpl_local_poses = smpl_local_poses,
        smpl_scale = smpl_scale,
        smpl_global_poses = smpl_global_poses,
        local_pose_optimizer = None,
        global_pose_optimizer = None,
        smpl_scale_optimizer = None,
        model_path = human_model_path,
        beta = beta,
        smpl_deformer = smpl_deformer,
        gaussians = person_gaussian,
        do_trans_grid = False,
        trans_grids = None,
        grid_optimizer = None,
        view_dir_reg = False,
        human_scene = human_scene,
        human_id = human_id,
        init_smpl_jnts = None,
        cam_centers = None,
        cc_smpl_dir = None,
        representative_img = largest_img,
        misc = dict(
            optimized_step=0
        )
    )

    return person_info
