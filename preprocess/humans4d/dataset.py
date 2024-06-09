# Dataloader of Human4D results


# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import sys

import json
import time
import cv2
import time
import torch
import argparse
import joblib
import numpy as np

from pycocotools import mask as mask_utils
from pathlib import Path
from tqdm import tqdm
from math import floor, sqrt
from utils.camera_utils import rotation_matrix_to_axis_angle




def load_default_camdicts(phalp_res_path: Path):
    """
    camdict is dictionary composed of
    - fid
    - w2c
    - projection
    - f
    - cx
    - cy
    - H
    - W
    which all python native variable or numpy array
    """
    phalp_res = joblib.load(phalp_res_path)

    cam_dicts = dict()
    for k in sorted(list(phalp_res.keys())):
        v = phalp_res[k]
        if len(v['size']) == 0:
            print(f"skipping detction due to non valid detection")
            continue

        H = v['size'][0][0]
        W = v['size'][0][1]
        cam_dict = dict()
        f = 5000 / 256 * max(H, W)                                                # PHALP default settings
        cx = W / 2
        cy = H / 2
        w2c = np.eye(4, dtype=np.float32)
        intrinsic = np.array([
            [f, 0., cx, 0.],
            [0., f, cy, 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])
        fid = int(Path(k).name.split(".")[0])
        
        cam_dict['fid'] = fid
        cam_dict['H'] = H
        cam_dict['W'] = W
        cam_dict['f'] = f
        cam_dict['fx'] = f
        cam_dict['fy'] = f
        cam_dict['cx'] = cx
        cam_dict['cy'] = cy
        cam_dict['w2c'] = w2c
        cam_dict['intrinsic'] = intrinsic
        cam_dict['projection'] = intrinsic @ w2c
        cam_dicts[fid] = cam_dict
    
    return cam_dicts



def phalp_smpl2op(smpl_joints):
    j_inds = np.arange(25)
    j_inds[19] = 22
    j_inds[20] = 23
    j_inds[21] = 24
    j_inds[22] = 19
    j_inds[23] = 20
    j_inds[24] = 21


    op_joints = []
    for j_ind in j_inds:
        if j_ind >= len(smpl_joints):
            op_joints.append(None)
        else:
            op_joints.append(smpl_joints[j_ind])

    return op_joints
            


def load_results(phalp_res_path: Path):
    # load phalp
    start_time = time.time()
    print("[INFO] loading PHALP output (it takes a while)")
    phalp_res = joblib.load(phalp_res_path)
    print("[INFO] loaded PHALP output")
    print("[DEBUG] PHALP result loading: --- %s seconds ---" % (time.time() - start_time))
    
    # Get all trakced_ids
    tracked_ids = []
    for v in phalp_res.values():
        tracked_ids.extend(v['tracked_ids'])
    tracked_ids = list(set(tracked_ids))

    # make mpt style results
    track_res = dict()
    for pid in tracked_ids:
        track_res[pid] = dict()
    
    for k in sorted(list(phalp_res.keys())):
        v = phalp_res[k]
        # print(Path(k).name)
        frame_id = int(Path(k).name[:-4])
        for _, pid in enumerate(v['tracked_ids']):
            i = v['tid'].index(pid)
            img_size = max([v['size'][i][1], v['size'][i][0]])
            j2d = v['2d_joints'][i].reshape(-1, 2)
            j2d = j2d * img_size
            j2d = j2d + np.array([[v['size'][i][1] - img_size, v['size'][i][0]-img_size]]) / 2
            # j2d = j2d * np.array([v['bbox'][i][3], v['bbox'][i][2]]) + np.array([v['bbox'][i][1], v['bbox'][i][0]])
            j2d = j2d[:25]
            j2d = phalp_smpl2op(j2d)            # load in openpose format

            # extract mask
            mask = mask_utils.decode(v['mask'][i][0])

            # extract SMPL estimation
            smpl_estimation=v['smpl'][i]
            global_orient = rotation_matrix_to_axis_angle(smpl_estimation['global_orient']).reshape(-1)
            body_pose = rotation_matrix_to_axis_angle(smpl_estimation['body_pose']).reshape(-1)
            beta = smpl_estimation['betas']
            transl = v['camera'][i]                         # PHALP assumed F=5000 here. We need to align it in future

            smpl_param = np.concatenate([
                np.ones(1, dtype=np.float32),
                transl,
                global_orient,
                body_pose,
                beta
            ])

            track_res[pid][frame_id] = dict(
                bbox=v['bbox'][i],
                smpl_param=smpl_param,
                phalp_mask=mask,
                phalp_j3ds=v['3d_joints'][i],
                phalp_j2ds=j2d
            )

    return track_res