# This part required when we call SMPLServer
import importlib
numpy = importlib.import_module('numpy')
numpy.float = numpy.float32
numpy.int = numpy.int32
numpy.bool = numpy.bool_
numpy.unicode = numpy.unicode_
numpy.complex = numpy.complex_
numpy.object = numpy.object_
numpy.str = numpy.dtype.str


import argparse
import os
from pathlib import Path
from tqdm import tqdm

import cv2
import torch
import numpy as np
import trimesh

# Camera Loading Utils
from preprocess.humans4d.dataset import load_default_camdicts
from utils.colmap_loader import load_colmap_cameras, read_points3D_text, read_points3D_binary
from utils.render_smpl import render_w_torch3d
from utils.image_utils import gen_videos, load_img_dict, img_add_text, draw_bbox
from utils.camera_utils import euler_angles_to_rotation_matrix, rotation_matrix_to_euler_angles, estimate_translation_cv2
from utils.io_utils import storePly, read_pickle, write_pickle
from utils.draw_op_jnts import draw_op_img, op25_to_op18, draw_bodypose_with_color, draw_handpose, draw_facepose

# SMPL rendering settings
from gtu.smpl_deformer.smpl_server import SMPLServer
from preprocess.smpl_fitting_loss import single_step, BatchedPerspectiveCamera

BG_SPHERE_RADIUS_SCALE_FROM_FARTHEST_CAM = 1.5
BG_SPHERE_RESOLUTION = 150

##### OPTIMIZATION PARAMETERS #####
STAGE1_ITERATIONS = 150
STAGE2_ITERATIONS = 500             # Default: 500

MINIMUM_JNT_CONF = 0.4
HAND_MINIMUM_JNT_CONF = 0.4
MINIMUM_NUM_OF_JNTS = 5

PSUEDO_GT_CONF = 0.3


def create_mesh(center, radius, resolution=100):
    theta = np.linspace(0, 2*np.pi, resolution)
    phi = np.linspace(0, np.pi, resolution)
    theta, phi = np.meshgrid(theta, phi)

    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)

    return x, y, z

def normalize_cameras_and_smpls(smpl_param, w2c_R, w2c_T, scale, transl):
    """
        Here we assume ALL inputs are numpy array
    """
    # org_smpl_transl = smpl_param[1:4]
    # smpl_param[0] /= scale
    # smpl_param[1:4] -= transl
    # new_w2c_T = w2c_R@transl + w2c_T/scale + (1-1/scale)*(w2c_R@org_smpl_transl)
    new_w2c_T = w2c_T/scale + w2c_R@transl
    if smpl_param is None:
        return None, new_w2c_T

    org_smpl_transl = smpl_param[1:4]
    smpl_param[0] /= scale
    smpl_param[1:4] -= transl
    smpl_param[1:4] /= scale
    return smpl_param, new_w2c_T

def interpolate_smpls(person_dict):
    fids = sorted(list(person_dict.keys()))
    smpl_params = [] 
    for i, fid in enumerate(fids):
        smpl_params.append(person_dict[fid]['smpl_param'])
    filled_smpls = interpolate_none(smpl_params, fids)

    for fid, filled_smpl in zip(fids, filled_smpls):
        if person_dict[fid]['smpl_param'] is None:
            person_dict[fid]['smpl_param'] = filled_smpl
        else:
            assert (person_dict[fid]['smpl_param'] - filled_smpl).sum() < 1e-9, "Value shouldn't be changed has been altered"

    return person_dict

    
def interpolate_none(values, positions):
    """Interpolate None values in a list using distances from a positions list."""
    if not values or not positions or len(values) != len(positions):
        return None  # Basic validation

    # Function to interpolate between two points
    def interpolate(v0, v1, p0, p1, px):
        """Interpolate using inverse distance weighting."""
        # Handle edge cases where values might be the same
        if p0 == p1:
            return (v0 + v1) / 2
        # Inverse distance weighting
        weight0 = 1 / abs(px - p0)
        weight1 = 1 / abs(px - p1)
        return (v0 * weight0 + v1 * weight1) / (weight0 + weight1)

    # Main loop to fill None values
    for i, (val, pos) in enumerate(zip(values, positions)):
        if val is None:
            # Find previous non-None value
            prev_val, prev_pos = None, None
            for j in range(i - 1, -1, -1):
                if values[j] is not None:
                    prev_val, prev_pos = values[j], positions[j]
                    break

            # Find next non-None value
            next_val, next_pos = None, None
            for j in range(i + 1, len(values)):
                if values[j] is not None:
                    next_val, next_pos = values[j], positions[j]
                    break

            # Interpolate if both neighbors are found
            if prev_val is not None and next_val is not None:
                values[i] = interpolate(prev_val, next_val, prev_pos, next_pos, pos)
            elif next_val is None:
                values[i] = prev_val
            elif prev_val is None:
                values[i] = next_val
            else:
                raise ValueError("It means list is fully composed of None values")

    return values






def fit_single_person(
                    cam_dicts, 
                    person_dict, 
                    smpl_server, 
                    device, 
                    use_depth: bool=False, 
                    skip_beta_optim: bool=False, 
                    use_init_joints_as_pseudo_guide: bool=False,
                    fit_smpl_hand: bool=False
                    ):
    # prepare batched cameras
    Rs = []
    Ts = []
    centers = []
    fxs = []
    fys = []

    for fid in person_dict.keys():
        cam_dict = cam_dicts[fid]
        R = torch.from_numpy(cam_dict['w2c'][:3,:3]).float()
        T = torch.from_numpy(cam_dict['w2c'][:3,-1]).float()
        center = torch.tensor([cam_dict['cx'], cam_dict['cy']]).float()
        fx = torch.tensor(cam_dict['fx'], dtype=torch.float32)
        fy = torch.tensor(cam_dict['fy'], dtype=torch.float32)

        Rs.append(R)
        Ts.append(T)
        centers.append(center)
        fxs.append(fx)
        fys.append(fy)
        
    rotation = torch.stack(Rs).to(device)
    translation = torch.stack(Ts).to(device)
    fxs = torch.stack(fxs).to(device)
    fys = torch.stack(fys).to(device)
    centers = torch.stack(centers).to(device)

    bcam = BatchedPerspectiveCamera(
        rotation = rotation,
        translation = translation,
        focal_length_x= fxs,
        focal_length_y= fys,
        center = centers
    )

    # Prepare Parameters to optimize
    person_dict = interpolate_smpls(person_dict)

    mean_shape = []
    smpl_poses = []
    smpl_transls = []
    op_jnts_stack = []
    op_confs_stack = []
    bbox_scale_stack = []
    hand_jnts_stack = []
    hand_confs_stack = []
    
    if use_depth:
        op_depths_stack = []

    valid_fids = sorted(list(person_dict.keys()))
    for fid in valid_fids:
        frame_info = person_dict[fid]
        
        # Collect OpenPose GTs
        if frame_info['body_pose'] is None:
            openpose_j2d = np.zeros((25, 2))
            openpose_conf = np.zeros((25))
        else:
            body_pose = frame_info['body_pose']
            openpose_j2d = []
            openpose_conf = []
            n_valid_jnt = 0
            for jnt in body_pose:
                if jnt is None:
                    openpose_j2d.append(np.zeros(2, dtype=np.float32))
                    openpose_conf.append(0)
                else:
                    openpose_j2d.append(jnt[:2])
                    conf = 0 if jnt[2] < MINIMUM_JNT_CONF else jnt[2]
                    openpose_conf.append(conf)
                    if conf >= MINIMUM_JNT_CONF:
                        n_valid_jnt += 1

            openpose_j2d = np.stack(openpose_j2d, axis=0)
            openpose_conf = np.array(openpose_conf)
            if n_valid_jnt < MINIMUM_NUM_OF_JNTS:
                print(f"skipping {fid} due to possible inaccurate joints (#: {n_valid_jnt})")
                openpose_conf *= 0
                
        
        # Collect projected SMPL joints
        if use_init_joints_as_pseudo_guide:
            if 'phalp_j2ds' in frame_info:
                smpl_2d_jnts = frame_info['phalp_j2ds']
                
                for idx in range(len(openpose_j2d)):
                    op_conf = openpose_conf[idx]
                    if op_conf == 0:
                        openpose_j2d[idx] = smpl_2d_jnts[idx][:2]
                        openpose_conf[idx] = PSUEDO_GT_CONF
                        
            else:
                print(f"[WARNING] phalp_j2ds not included in fid:{fid}")
            
        
        # Collect hand joints
        if fit_smpl_hand:
            if 'hand_pose' in frame_info:
                left_hand = frame_info['hand_pose']['left_hand']
                right_hand = frame_info['hand_pose']['right_hand']
                
                # thumb - index - middle - ring - little
                finger_jnt_inds = [2, 5, 9, 13, 17]
                hand_j2d = []
                hand_conf = []
                
                # left joints
                for f_idx in finger_jnt_inds:
                    jnt = left_hand[f_idx]
                    if jnt is None:
                        hand_j2d.append(np.zeros(2, dtype=np.float32))
                        hand_conf.append(0)
                    else:
                        hand_j2d.append(jnt[:2])
                        conf = 0 if jnt[2] < HAND_MINIMUM_JNT_CONF else jnt[2]
                        hand_conf.append(conf)
                
                # right joints
                for f_idx in finger_jnt_inds:
                    jnt = right_hand[f_idx]
                    if jnt is None:
                        hand_j2d.append(np.zeros(2, dtype=np.float32))
                        hand_conf.append(0)
                        
                        np.zeros(2, dtype=np.float32)
                    else:
                        hand_j2d.append(jnt[:2])
                        conf = 0 if jnt[2] < HAND_MINIMUM_JNT_CONF else jnt[2]
                        hand_conf.append(conf)  
                
                hand_j2d = np.stack(hand_j2d, axis=0)
                hand_conf = np.array(hand_conf)
            else: 
                hand_j2d = np.zeros((10, 2))
                hand_conf = np.zeros((10)) 
                
            hand_jnts_stack.append(hand_j2d)
            hand_confs_stack.append(hand_conf)
                
                
            
        bbox_size = (frame_info['bbox'][2] + frame_info['bbox'][3]) / 2.
        mean_shape.append(frame_info['smpl_param'][-10:])
        smpl_poses.append(frame_info['smpl_param'][4:-10])
        smpl_transls.append(frame_info['smpl_param'][1:4])
        op_jnts_stack.append(openpose_j2d)
        op_confs_stack.append(openpose_conf)
        bbox_scale_stack.append(bbox_size)

        if use_depth:
            raise NotImplementedError()
            op_jnts_depth = None
            op_depths_stack.append(op_jnts_depth)

    mean_shape = torch.from_numpy(np.stack(mean_shape, axis=0).mean(0)).float().to(device)
    smpl_poses = torch.from_numpy(np.stack(smpl_poses, axis=0)).float().to(device)              # (B, 72)
    smpl_transls = torch.from_numpy(np.stack(smpl_transls, axis=0)).float().to(device)              # (B, 3)
    op_jnts_stack = torch.from_numpy(np.stack(op_jnts_stack, axis=0)).float().to(device)              # (B, 25, 2)
    op_confs_stack = torch.from_numpy(np.stack(op_confs_stack, axis=0)).float().to(device)              # (B, 25)
    bbox_scale_stack = torch.from_numpy(np.stack(bbox_scale_stack, axis=0)).float().squeeze().to(device)              # (B)
    if use_depth:
        op_depths_stack = torch.from_numpy(np.stack(op_depths_stack, axis=0)).float().to(device)              # (B, 25)

    if fit_smpl_hand:
        hand_jnts_stack = torch.from_numpy(np.stack(hand_jnts_stack, axis=0)).float().to(device)              # (B, 10, 2)
        hand_confs_stack = torch.from_numpy(np.stack(hand_confs_stack, axis=0)).float().to(device)              # (B, 10)

    # Optimization Targets
    opt_scale = torch.ones(1, dtype=torch.float32, requires_grad=True, device=device)
    opt_beta = torch.tensor(mean_shape[None].cpu().numpy(), dtype=torch.float32, requires_grad=True, device=device)
    opt_global_rot = torch.tensor(smpl_poses[:, :3].cpu().numpy(), dtype=torch.float32, requires_grad=True, device=device)
    opt_pose = torch.tensor(smpl_poses[:, 3:].cpu().numpy(), dtype=torch.float32, requires_grad=True, device=device)
    opt_trans = torch.tensor(smpl_transls.cpu().numpy(), dtype=torch.float32, requires_grad=True, device=device)


    # Temporal loss: 3D SMPL joints distance.
    ###### [Stage 1] fit scale - translation - global oritent
    opt_params = [
                {'params': opt_scale, 'lr': 1e-2},
                {'params': opt_trans, 'lr': 1e-2},
                {'params': opt_global_rot, 'lr': 1e-2}]
    optimizer = torch.optim.Adam(opt_params, lr=1e-2, betas=(0.9, 0.999))

    loop = tqdm(range(STAGE1_ITERATIONS))
    weight_dict = {
        "reprojection_loss" : lambda cst, it: cst,
        "temporal_loss":  lambda cst, it: cst,
        "depth_loss":  lambda cst, it: cst,
    }

    for it in loop:
        optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        if True:
            loss_dict = single_step(
                smpl_server=smpl_server,
                opt_scale = opt_scale,
                opt_beta = opt_beta,
                opt_pose = torch.cat([opt_global_rot, opt_pose], dim=-1),
                opt_trans = opt_trans,
                fids = valid_fids,
                batched_cameras=bcam,
                op_jnts=op_jnts_stack,
                op_confs=op_confs_stack,
                reproj_loss_scaler=bbox_scale_stack,
                op_depths=op_depths_stack if use_depth else None,
            )

            tot_loss = 0
            w_loss = dict()
            for k in weight_dict:
                if k not in loss_dict:
                    continue
                w_loss[k] = weight_dict[k](loss_dict[k], it)
                tot_loss += w_loss[k].sum()

            if tot_loss.isnan().sum() > 0:
                raise AssertionError()
            tot_loss.backward()
            optimizer.step()

        l_str = '[Scale-Transl fit] %05d | Iter: %d' % (fid, it)
        for k in w_loss:
            l_str += ', %s: %0.4f' % (k, w_loss[k].sum().item())
            loop.set_description(l_str)


    ###### [Stage 2] fit reprojection error
    if skip_beta_optim:
        opt_params = [
                    {'params': opt_pose, 'lr': 1e-2},
                    {'params': opt_global_rot, 'lr': 1e-2}]
    else:
        opt_params = [
                    {'params': opt_beta, 'lr': 1e-2},
                    {'params': opt_pose, 'lr': 1e-2},
                    {'params': opt_global_rot, 'lr': 1e-2}]
    optimizer = torch.optim.Adam(opt_params, lr=1e-2, betas=(0.9, 0.999))

    loop = tqdm(range(STAGE2_ITERATIONS))
    weight_dict = {
        "reprojection_loss" : lambda cst, it: 5 * cst,
        "temporal_loss":  lambda cst, it: 0 * cst,
        "pose_prior_loss":  lambda cst, it: cst,
        "angle_prior_loss":  lambda cst, it: cst,
        "shape_prior_loss":  lambda cst, it: cst,
        "hand_reprojection_loss" : lambda cst, it: 0.05 * cst,
    }

    for it in loop:
        optimizer.zero_grad()
        loss_dict = single_step(
            smpl_server=smpl_server,
            opt_scale = opt_scale,
            opt_beta = opt_beta,
            opt_pose = torch.cat([opt_global_rot, opt_pose], dim=-1),
            opt_trans = opt_trans,
            fids = valid_fids,
            batched_cameras=bcam,
            op_jnts=op_jnts_stack,
            op_confs=op_confs_stack,
            reproj_loss_scaler=bbox_scale_stack,
            op_depths=op_depths_stack if use_depth else None,
            hand_jnts_stack = hand_jnts_stack if fit_smpl_hand else None,
            hand_confs_stack = hand_confs_stack if fit_smpl_hand else None,
        )

        tot_loss = 0
        w_loss = dict()
        for k in weight_dict:
            if k not in loss_dict:
                continue
            w_loss[k] = weight_dict[k](loss_dict[k], it)
            tot_loss += w_loss[k].sum()

        tot_loss.backward()
        optimizer.step()

        l_str = '[Skeletion fit] %05d | Iter: %d' % (fid, it)
        for k in w_loss:
            l_str += ', %s: %0.4f' % (k, w_loss[k].sum().item())
            loop.set_description(l_str)
        
                
    # Check Visualization
    n_frames = len(valid_fids)
    fitted_smpl_params = torch.cat([
        opt_scale.detach().cpu().reshape(1, 1).repeat(n_frames, 1),
        opt_trans.detach().cpu(),
        opt_global_rot.detach().cpu(),
        opt_pose.detach().cpu(),
        opt_beta.detach().cpu().reshape(1, -1).repeat(n_frames, 1)
    ], dim=-1)
    fitted_smpl_params = fitted_smpl_params.detach().cpu().numpy()

    for i, fid in enumerate(valid_fids):
        person_dict[fid]['smpl_param'] = fitted_smpl_params[i]

    return person_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_dir', type=str, default=None, help='SfM camera dir')
    parser.add_argument('--data_dir', type=str, default=None, help='overall data dir')
    parser.add_argument('--is_static_camera', action='store_true', help='If true, use posetrack tracklet as GT for tracking')
    parser.add_argument('--no-normalize_human_position', action='store_false', dest='normalize_human_position', help='If turn off normalization, we do not re-centerize and rescale the world with human position')
    parser.add_argument('--render_smpl', action='store_true', help='If true, render fitted SMPLs')
    parser.add_argument('--debug', action='store_true', help='If true, render SMPLs using updated cameras (colmap cameras)')
    parser.add_argument('--skip_beta_optim', action='store_true', help='If true, do not optimize beta and use mean-beta instead')
    parser.add_argument('--skip_fitting', action='store_true', help='If true, Just use given SMPL and change camera to corresponding instead. (only valid with --is_static_camera option)')
    parser.add_argument('--use_init_joints_as_pseudo_guide', action='store_true', help='If true, for joints which are not visible, we use SMPL predicted joint as pseudo GT)')
    parser.add_argument('--fit_smpl_hand', action='store_true', help='If true, we fit smpl hands on estimation results to optmize hand joint angle')
    args = parser.parse_args()

    # Load default cam data
    data_dir = Path(args.data_dir)
    h4d_camdicts = load_default_camdicts(data_dir / "phalp_v2" / "results" / "demo_images_jpg.pkl")

    # Load tracking results
    tracking_res_file = data_dir / "op_phalp_merged.pkl"
    assert tracking_res_file.exists(), f"Tracking result file not found. | {str(tracking_res_file)}"
    tracking_res = read_pickle(tracking_res_file)
    
    print(str(tracking_res_file))
    print(tracking_res.keys())
    # assert(0)

    # Prepare SMPL server
    device = torch.device("cuda:0")
    smpl_server = SMPLServer(use_feet_keypoints=True)

    # Prepare Img Dicts
    img_dir = data_dir / 'images'
    img_dict = dict()
    for img_fname in (list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))):
        img = cv2.imread(str(img_fname))
        fid = int(img_fname.name.split(".")[0])
        img_dict[fid] = img
        


    save_dict = dict()
    if args.is_static_camera:
        sfm_camdicts = h4d_camdicts

        del_keys = []
        for fid in h4d_camdicts.keys():
            if fid not in img_dict:
                del_keys.append(fid)
        for _key in del_keys:
            del h4d_camdicts[_key]

    else:
        # Load colmap cameras
        sfm_dir = Path(args.sfm_dir)
        assert sfm_dir.exists(), f"SfM file does not exist {str(sfm_dir)}"
        sfm_camdicts, sfm_cc_dict = load_colmap_cameras(sfm_dir)


        # Need to find Global R|t that fits to NEW camera! (apply on raw SMPL estimations)
        for fid, cam_dict in sfm_camdicts.items():
            h4d_camdict = h4d_camdicts[fid]
            new_intrinsic = cam_dict['intrinsic'][:3,:3]

            for pid, person_dict in tracking_res.items():
                if fid in person_dict and (person_dict[fid]['smpl_param'] is not None):
                    if person_dict[fid]['smpl_param'][0] != 1:
                        raise NotImplementedError("We currently do not consider the initial SMPL scale != 1")

                    # Get smpl joints in initial world
                    smpl_param = torch.from_numpy(person_dict[fid]['smpl_param']).to(device).float().squeeze().reshape(1, -1)
                    smpl_output = smpl_server(smpl_param)
                    smpl_jnts = smpl_output['smpl_jnts'].squeeze().detach().cpu()       # (J, 3)
                    smpl_jnts = smpl_jnts[:25]                                          # Only consider SMPL-25 skeletons here
                    smpl_jnts_homo = torch.cat([smpl_jnts, torch.ones_like(smpl_jnts[:, :1])], dim=-1)
                    smpl_jnts_homo = smpl_jnts_homo.numpy()
                    original_pj_jnts = np.einsum('ij,bj->bi', h4d_camdict['projection'], smpl_jnts_homo) 
                    original_pj_jnts = original_pj_jnts[:, :2] / (original_pj_jnts[:, 2:3] + 1e-9)


                    # Change ONLY intrinsic here
                    pnp_transl, r_pred = estimate_translation_cv2(smpl_jnts.numpy(), original_pj_jnts, proj_mat=new_intrinsic)
                    pnp_transl = pnp_transl.reshape(-1).astype(np.float32)
                    pnp_rot = cv2.Rodrigues(r_pred)[0].astype(np.float32)

                    # Get pelvis translations
                    neutral_param = torch.zeros(1, 86).float().to(device)
                    neutral_param[0,0] = 1
                    neutral_param[0,-10:] = smpl_param.squeeze()[-10:]    # copy beta
                    smpl_output = smpl_server(neutral_param)
                    t_pelvis = smpl_output['smpl_jnts'].squeeze().detach().cpu()[0].numpy()


                    # 1. convert regressed SMPL global parameters to corresponding world space
                    smpl_pose = person_dict[fid]['smpl_param'][4:-10]      #72-dim
                    reg_smpl_rot = cv2.Rodrigues(smpl_pose[:3])[0].astype(np.float32)
                    reg_smpl_transl = person_dict[fid]['smpl_param'][1:4].astype(np.float32)
                    # reg_smpl_transl = np.zeros(3).astype(np.float32)

                    cam_smpl_rot = pnp_rot @ reg_smpl_rot
                    cam_smpl_transl = pnp_rot @ (t_pelvis + reg_smpl_transl) + pnp_transl - t_pelvis

                    # 2. convert camera-space SMPL global parameters to world-space
                    R = cam_dict['w2c'][:3, :3]
                    T = cam_dict['w2c'][:3, -1]
                    w2c_rot = R.astype(np.float32)
                    w2c_transl = T.reshape(-1).astype(np.float32)

                    world_smpl_rot = w2c_rot.T @ cam_smpl_rot
                    world_smpl_transl = w2c_rot.T @ (t_pelvis + cam_smpl_transl - w2c_transl) - t_pelvis

                    # 3. convert to parameter shape
                    world_smpl_rot_vec = cv2.Rodrigues(world_smpl_rot)[0]
                    smpl_pose[:3] = world_smpl_rot_vec.reshape(-1)
                    smpl_trans = world_smpl_transl


                    # 4. collect new SMPL parameters
                    new_smpl_param = person_dict[fid]['smpl_param']
                    new_smpl_param[1:4] = smpl_trans
                    new_smpl_param[4:-10] = smpl_pose
                    tracking_res[pid][fid]['smpl_param'] = new_smpl_param

        # Check camera conversion done properly
        if args.debug:
            debug_render_save_dir = data_dir / 'debug_camera_smpl_rendering' 
            debug_render_save_dir.mkdir(exist_ok=True)

            smpl_rendered_dicts, smpl_rendering_alpha_dicts = render_w_torch3d(
                render_camdicts=sfm_camdicts,
                people_dict=tracking_res,
                smpl_server=smpl_server,
                render_camera_position=False,
                render_bg_as_pc=False,
                get_smpl_alpha=True,
                for_viz=False,
                skip_text=True
            ) 

            for fid, smpl_rendering in smpl_rendered_dicts.items():
                alpha_rendering = smpl_rendering_alpha_dicts[fid] / 255.
                alpha_rendering = alpha_rendering[..., None]
                img = img_dict[fid]
                img = img * (1 - alpha_rendering) + smpl_rendering * alpha_rendering
                img = img_add_text(img, f"{fid:04}")
                img_dict[fid] = img

                img = img.copy()
                for pid, person_dict in tracking_res.items():
                    if fid not in person_dict:
                        continue
                    frame_res = person_dict[fid]
                    if frame_res['bbox'] is not None:
                        img = draw_bbox(img, frame_res['bbox'], (255, 0, 0) , pid=pid)
                    if frame_res['body_pose'] is not None:
                        _body_jnts = []
                        for jnt in frame_res['body_pose']:
                            if jnt is None:
                                _body_jnts.append(None)
                            elif jnt[-1] < MINIMUM_JNT_CONF:
                                _body_jnts.append(None)
                            else:
                                _body_jnts.append(jnt)
                        img = draw_bodypose_with_color(img, _body_jnts, img.shape[:2], (0, 255, 0), mode='op25')

                cv2.imwrite(str(debug_render_save_dir / f"{fid:07}.jpg"), img)
            gen_videos([debug_render_save_dir], is_jpg=True, fps=10, rm_dir=True)  


    if args.render_smpl:
        smpl_render_save_dir = data_dir / 'smpl_fitting_results' 
        smpl_render_save_dir.mkdir(exist_ok=True)
        img_dict = load_img_dict(data_dir / 'images')


    smpl_avg_transl = np.zeros(3)
    smpl_avg_scale = 1.
    smpl_transls = []
    smpl_scales = []

    # Now Start Fitting of SMPL parameters
    # We save fids which is valid in tracking_res.items() ONLY.
    if args.skip_fitting and args.is_static_camera:
        for pid, person_dict in tracking_res.items():
            fids = sorted(list(person_dict.keys()))

            # Simply use average to calculate SMPL beta
            betas = []
            valid_fids = []
            for i, fid in enumerate(fids):
                smpl_param = person_dict[fid]['smpl_param']
                if smpl_param is None:
                    continue
                beta = smpl_param[-10:]
                betas.append(beta)
                valid_fids.append(fid)
            mean_beta = np.stack(betas, axis=0).mean(0)

            # Update the tracking_res
            for i, fid in enumerate(fids):
                if fid in valid_fids:
                    person_dict[fid]['smpl_param'][-10:] = mean_beta
                else:
                    # Discard None cases.
                    del person_dict[fid]
            
            tracking_res[pid] = person_dict

    else:
        for pid, person_dict in tracking_res.items():
            person_dict = fit_single_person(
                                            sfm_camdicts, 
                                            person_dict, 
                                            smpl_server, 
                                            device, 
                                            skip_beta_optim=args.skip_beta_optim, 
                                            use_init_joints_as_pseudo_guide=args.use_init_joints_as_pseudo_guide,
                                            fit_smpl_hand=args.fit_smpl_hand
                                            )

            # collect individuals' position and scale for furthur processing
            for fid in person_dict:
                smpl_transls.append(person_dict[fid]['smpl_param'][1:4])
            smpl_scales.append(list(person_dict.values())[0]['smpl_param'][0])
            tracking_res[pid] = person_dict


    if args.render_smpl:
        combined_img_dict = dict()
        for fid, img in img_dict.items():
            combined_img_dict[fid] = img.copy()
        
        for pid, person_dict in tracking_res.items():
            p_render_dir = smpl_render_save_dir / f"{pid:05}"
            p_render_dir.mkdir(exist_ok=True)
            smpl_rendered_dicts, smpl_rendering_alpha_dicts = render_w_torch3d(
                render_camdicts=sfm_camdicts,
                people_dict={pid: person_dict},
                smpl_server=smpl_server,
                render_camera_position=False,
                render_bg_as_pc=False,
                get_smpl_alpha=True,
                for_viz=False,
                skip_text=True
            ) 
            
            for fid, smpl_rendering in smpl_rendered_dicts.items():
                alpha_rendering = smpl_rendering_alpha_dicts[fid] / 255.
                alpha_rendering = alpha_rendering[..., None]
                alpha_rendering = alpha_rendering * 0.7                         # We want to overlay it.
                img = img_dict[fid].copy()
                img = img * (1 - alpha_rendering) + smpl_rendering * alpha_rendering

                combined_img = combined_img_dict[fid]
                combined_img = combined_img * (1 - alpha_rendering) + smpl_rendering * alpha_rendering
                
                
                # visualize frames.
                if fid in person_dict:
                    frame_res = person_dict[fid]
                    if frame_res['bbox'] is not None:
                        img = draw_bbox(img, frame_res['bbox'], (255, 0, 0) , pid=pid)
                        combined_img = draw_bbox(combined_img, frame_res['bbox'], (255, 0, 0) , pid=pid)
                    if frame_res['body_pose'] is not None:
                        _body_jnts = []
                        for jnt in frame_res['body_pose']:
                            if jnt is None:
                                _body_jnts.append(None)
                            elif jnt[-1] < MINIMUM_JNT_CONF:
                                _body_jnts.append(None)
                            else:
                                _body_jnts.append(jnt)
                        img = draw_bodypose_with_color(img, _body_jnts, img.shape[:2], (0, 255, 0), mode='op25')
                        combined_img = draw_bodypose_with_color(combined_img, _body_jnts, img.shape[:2], (0, 255, 0), mode='op25')
                    if frame_res['hand_pose'] is not None:
                        hands = []
                        if frame_res['hand_pose']['left_hand'] is not None and len(frame_res['hand_pose']['left_hand'])==21:
                            hands.append(frame_res['hand_pose']['left_hand'])
                        if frame_res['hand_pose']['right_hand'] is not None and len(frame_res['hand_pose']['right_hand'])==21:
                            hands.append(frame_res['hand_pose']['right_hand'])
                        if len(hands) > 0:
                            img = draw_handpose(img, hands)
                            combined_img = draw_handpose(combined_img, hands)
                    if frame_res['face'] is not None:
                        img = draw_facepose(img, [frame_res['face']])
                        combined_img = draw_facepose(combined_img, [frame_res['face']])
                else:
                    img = img_add_text(img, "\n No person in frame")
                
                # save img
                cv2.imwrite(str(p_render_dir / f"{fid:07}.jpg"), img)
                combined_img_dict[fid] = combined_img
                
            gen_videos([p_render_dir], is_jpg=True, fps=10, rm_dir=True) 
            
        
        combined_save_dir = smpl_render_save_dir / "_combined"
        combined_save_dir.mkdir(exist_ok=True)
        for fid, img in combined_img_dict.items():
            cv2.imwrite(str(combined_save_dir / f"{fid:07}.jpg"), img)
        gen_videos([combined_save_dir], is_jpg=True, fps=10, rm_dir=True) 
        

    
    if args.normalize_human_position and len(smpl_scales) >= 1:
        # get global camera movement
        smpl_avg_transl = np.stack(smpl_transls, axis=0).mean(axis=0)
        smpl_avg_scale = np.array(smpl_scales).mean()
        print(f"[INFO] Average SMPL scale: {smpl_avg_scale}")
        print(f"[INFO] Average SMPL transl: {smpl_avg_transl}")
        print(f"[INFO] Average SMPL scales: {smpl_scales}")

    
    # Now save in GTU formats
    # print("\n\n\n[WARNING] As we don't use single fixed scale parameter whole all human, Here we don't normalize the world with AVG SMPL SCALE parameter.!!!!\n\n\n")


    # Save Scene Cameras (cameras.pkl)
    save_dict = dict()
    for fid, camdict in sfm_camdicts.items():
        w2c_R = camdict['w2c'][:3,:3]
        w2c_T = camdict['w2c'][:3, 3]
        if args.normalize_human_position:
            _, w2c_T = normalize_cameras_and_smpls(None, w2c_R, w2c_T, smpl_avg_scale, smpl_avg_transl)
        
        save_dict[fid] = dict(
            camera = dict(
                    width = camdict['W'],
                    height = camdict['H'],
                    rotation = w2c_R,
                    translation = w2c_T,
                    intrinsic = camdict['intrinsic'][:3,:3]
                ),
        )
    write_pickle(data_dir / f'cameras.pkl', save_dict)


    # Save Individuals
    for i, pid in enumerate(sorted(list(tracking_res.keys()))):
        person_dict = tracking_res[pid]
        save_dict = dict()

        for fid, frame_info in person_dict.items():
            camdict = sfm_camdicts[fid]
            smpl_param = frame_info['smpl_param']
            w2c_R = camdict['w2c'][:3,:3]
            w2c_T = camdict['w2c'][:3, 3]
            if args.normalize_human_position:
                smpl_param, w2c_T = normalize_cameras_and_smpls(smpl_param, w2c_R, w2c_T, smpl_avg_scale, smpl_avg_transl)
            
            save_dict[fid] = dict(
                gt_body_pose = frame_info['body_pose'],
                gt_bbox = frame_info['bbox'],
                gt_hands = frame_info['hand_pose'],
                gt_face = frame_info['face'],
                smpl_param = np.expand_dims(smpl_param, axis=0),      # (1, 86)
                camera = dict(
                        width = camdict['W'],
                        height = camdict['H'],
                        rotation = w2c_R,
                        translation = w2c_T,
                        intrinsic = camdict['intrinsic'][:3,:3]
                    ),
            )
        write_pickle(data_dir / f'results_p{i}.pkl', save_dict)

        # Also save person initial SMPL 3D GSs (canonical SMPL model)
        smpl_shape = list(save_dict.values())[0]['smpl_param'][:, -10:]
        param_canon = np.concatenate([
                                np.ones( (1,1)) * 1, 
                                np.zeros( (1,3)),
                                np.zeros( (1,72)),
                                smpl_shape], axis=1)
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
        with open(data_dir / f"human_points_{i}.txt", 'w') as f:
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


    
    # Save backgrounds 
    if args.is_static_camera:
        # Let's collect ALL camera centers
        cam_centers = []
        smpl_centers = []
        for cam_fid, camdict in sfm_camdicts.items():
            w2c_R = camdict['w2c'][:3,:3]
            w2c_T = camdict['w2c'][:3, 3]
            if args.normalize_human_position:
                for i, pid in enumerate(sorted(list(tracking_res.keys()))):
                    person_dict = tracking_res[pid]

                    if cam_fid in person_dict:
                        smpl_param = person_dict[cam_fid]['smpl_param']
                        smpl_param, w2c_T = normalize_cameras_and_smpls(smpl_param, w2c_R, w2c_T, smpl_avg_scale, smpl_avg_transl)
                        smpl_centers.append(smpl_param[1:4])

            R = w2c_R
            T = w2c_T
            cam_center = - R.T @ T
            cam_centers.append(cam_center.reshape(-1))

            
        # Define a Scene sphere, center on mean of cam_centers, including ALL points inside cam
        cam_centers = np.array(cam_centers)
        smpl_centers = np.array(smpl_centers)
        
        sphere_center = np.mean(smpl_centers, axis=0)
        dists = np.sqrt(((cam_centers - sphere_center[None])**2).sum(-1))
        R = dists.max() * BG_SPHERE_RADIUS_SCALE_FROM_FARTHEST_CAM
        resolution = BG_SPHERE_RESOLUTION
        x, y, z = create_mesh(sphere_center, R, resolution)

        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)

        xyzs = np.stack([x,y,z], axis=-1)
        rgbs = np.ones_like(xyzs, dtype=np.uint8) * np.array([[128, 128, 128]])   # mean gray color

        norm_dir = sphere_center[None] - xyzs 
        print(sphere_center, R, resolution)
        normal = (norm_dir / np.sqrt((norm_dir**2).sum(-1, keepdims=True))).tolist()

        # Save in ply format
        save_fname = data_dir / "points3D.ply"
        storePly(str(save_fname), xyzs, rgbs, normals=normal)

    else:
        # Let's load SfM background points. (just convert from SfM files, so that reduce complexity of code structure)
        save_fname = data_dir / "points3D.ply"
        sfm_pc_bin_file = sfm_dir / "points3D.bin"
        sfm_pc_txt_file = sfm_dir / "points3D.txt"
        if sfm_pc_txt_file.exists():
            xyz, rgb, _, normal = read_points3D_text(sfm_pc_txt_file, cc_dict=sfm_cc_dict, get_normal=True)
        else:
            xyz, rgb, _, normal = read_points3D_binary(sfm_pc_bin_file, cc_dict=sfm_cc_dict, get_normal=True)
            
        if args.normalize_human_position:
            xyz = xyz / smpl_avg_scale + smpl_avg_transl

        # Save in ply format
        storePly(str(save_fname), xyz, rgb, normals=normal)