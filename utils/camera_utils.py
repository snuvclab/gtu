
import cv2
import torch
import math
import numpy as np
from typing import NamedTuple

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))



### Not checked it's perfectly working yet
def rotation_matrix_to_axis_angle(R):
    trace = R[:,0,0] + R[:,1,1] + R[:,2,2]
    angle = np.arccos((trace - 1) / 2.0)

    
    axis = np.array([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]])
    axis = axis.T
    axis /= np.sqrt((axis ** 2).sum(-1, keepdims=True))
    
    axis[angle < 1e-6] = axis[angle < 1e-6]*0 + np.array([1.0, 0.0, 0.0])[None]
    
    # Finally convert in SMPL format
    # Where, norm is angle, axis is just direction
    axis = axis * angle[..., None]
    
    return axis
    

def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    
    singular = sy < 1e-6

    # First calculate assuming that all not singular region

    roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    pitch = np.arctan2(-R[:, 2, 0], sy)
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])

    # Second, calculate assuming singular
    roll[singular] = np.arctan2(-R[singular, 1, 2], R[singular, 1, 1])
    pitch[singular] = np.arctan2(-R[singular, 2, 0], sy[singular])
    yaw[singular] = 0.0

    res = np.stack([roll, pitch, yaw], axis=-1)
    
    return res



############################################# It's same as above two functions. (legacy)

def euler_angles_to_rotation_matrix(roll, pitch, yaw):
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    
    return roll, pitch, yaw




def estimate_translation_cv2(joints_3d, joints_2d, focal_length=600, img_size=np.array([512.,512.]), proj_mat=None, cam_dist=None):
    if proj_mat is None:
        camK = np.eye(3)
        camK[0,0], camK[1,1] = focal_length, focal_length
        camK[:2,2] = img_size//2
    else:
        camK = proj_mat
    _, rvec, tvec,inliers = cv2.solvePnPRansac(joints_3d, joints_2d, camK, cam_dist,\
                                            flags=cv2.SOLVEPNP_EPNP,reprojectionError=30,iterationsCount=2000)

    print(len(inliers), len(joints_2d))
    if inliers is None:
        INVALID_TRANS=np.ones(3)*-1
        return INVALID_TRANS
    else:
        r_pred = rvec[:, 0]
        tra_pred = tvec[:,0]            
        return tra_pred, r_pred
