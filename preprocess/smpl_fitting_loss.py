
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.draw_op_jnts import smpl2op


# Loss weights
pose_prior_weight=4.78
shape_prior_weight=5
angle_prior_weight=15.2 

num_joints = 25
joints_to_ign = [1,9,12]        # (neck, lr hip (due to estimator ambiguity))
joint_weights = torch.ones(num_joints)   
joint_weights[joints_to_ign] = 0
joint_weights = joint_weights.reshape((-1,1)).cuda()


smpl2op_mapping = torch.tensor([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 32, 33, 34, 29, 30, 31], dtype=torch.long).cuda()

smplverts2hand_mapping = torch.tensor(
    [
        1986, 2135, 2214, 2173, 2062,
        5748, 5595, 5675, 5636, 5525
    ]
    , dtype=torch.long).cuda()

DEFAULT_DTYPE = torch.float32


    


def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


    
class BatchedPerspectiveCamera(nn.Module):
    def __init__(self, rotation, translation,
                 focal_length_x, focal_length_y,
                 center, dtype=torch.float32):
        super(BatchedPerspectiveCamera, self).__init__()
        
        self.dtype = dtype
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix

        self.register_buffer('focal_length_x', focal_length_x)
        self.register_buffer('focal_length_y', focal_length_y)
        if isinstance(center, torch.Tensor):
            self.center = center.to(dtype=self.dtype).to(rotation.device)
        else:
            self.center = torch.tensor(center, dtype=self.dtype).to(rotation.device)

        n_cams = len(rotation)
        self.camera_mat = torch.zeros([n_cams, 2, 2], dtype=self.dtype, device=rotation.device)

        if isinstance(self.focal_length_x, torch.Tensor):
            self.camera_mat[:, 0, 0] = self.focal_length_x.to(dtype=self.dtype).to(rotation.device)
        else:
            self.camera_mat[:, 0, 0] = torch.tensor(self.focal_length_x, dtype=self.dtype).to(rotation.device)

        if isinstance(self.focal_length_y, torch.Tensor):
            self.camera_mat[:, 1, 1] = self.focal_length_y.to(dtype=self.dtype).to(rotation.device)
        else:
            self.camera_mat[:, 1, 1] = torch.tensor(self.focal_length_y, dtype=self.dtype).to(rotation.device)


        rotation = nn.Parameter(rotation, requires_grad=False)
        self.register_parameter('rotation', rotation)
        translation = nn.Parameter(translation, requires_grad=False)
        self.register_parameter('translation', translation)

        self.camera_transform = transform_mat(self.rotation,
                                            self.translation.unsqueeze(dim=-1))

        c2w_rot = torch.transpose(rotation, 1, 2)
        self.global_camera_position = torch.einsum('bij,bj->bi', c2w_rot, translation)


    def forward(self, points):
        
        device = points.device
        homog_coord = torch.ones(list(points.shape)[:-1] + [1],
                                 dtype=points.dtype,
                                 device=device)
        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points, homog_coord], dim=-1)

        

        projected_points = torch.einsum('bki,bji->bjk',
                                        [self.camera_transform, points_h])

        img_points = torch.div(projected_points[:, :, :2],
                               projected_points[:, :, 2].unsqueeze(dim=-1))
        img_points = torch.einsum('bki,bji->bjk', [self.camera_mat, img_points]) \
            + self.center.unsqueeze(dim=1)
        return img_points

def single_step(
                smpl_server, 
                opt_scale, 
                opt_beta, 
                opt_pose, 
                opt_trans, 
                fids, 
                batched_cameras, 
                op_jnts, 
                op_confs, 
                reproj_loss_scaler, 
                op_depths=None,
                hand_jnts_stack=None,
                hand_confs_stack=None
                ):
    batch_size = len(fids)
    smpl_param = torch.cat([
        opt_scale.reshape(1, 1).repeat(batch_size, 1),
        opt_trans,
        opt_pose,
        opt_beta.repeat(batch_size, 1)
    ], dim=-1)
    smpl_output = smpl_server(smpl_param)
    smpl_jnts = smpl_output['smpl_jnts']
    smpl_verts = smpl_output['smpl_verts']

    global smpl2op_mapping
    world_op_j3d = torch.index_select(smpl_jnts, 1, smpl2op_mapping)
    cam_smpl_op_j2d = batched_cameras(world_op_j3d)

    loss = dict()
    # 1. calculate reprojection error
    loss['reprojection_loss'] = joints_2d_loss(op_jnts, cam_smpl_op_j2d, op_confs, reproj_loss_scaler)     # already included GMoF robustifier # pixel L2-distance

    # 2. Pose prior loss
    loss['pose_prior_loss'] = (pose_prior_weight ** 2) * pose_prior(opt_pose[:,3:], opt_beta)

    # 3. Angle prior for knees and elbows
    loss['angle_prior_loss'] = (angle_prior_weight ** 2) * angle_prior(opt_pose[:,3:]).sum(dim=-1)

    # 4. beta Regularizer to prevent betas from taking large values
    loss['shape_prior_loss'] = (shape_prior_weight ** 2) * (opt_beta ** 2).sum(dim=-1)

    # 5. temporal smoothing loss on nearby frames.
    loss['temporal_loss'] = joints_temporal_loss(world_op_j3d, fids, opt_scale)        # World L2-distance mean, fixed-scale in SMPL 

    # 6. depth loss
    loss['depth_loss'] = torch.zeros(1, dtype=torch.float32).to(opt_beta.device)    # World L2-distance mean (w/ robustifier), fixed_scale in SMPL
    if not (op_depths is None):
        loss['depth_loss'] = smpl_depth_loss(world_smpl_j3d, batched_cameras, op_depths, op_confs, opt_scale)
    
    # 7. Hand reprojection loss
    loss['hand_reprojection_loss'] = torch.zeros(1, dtype=torch.float32).to(opt_beta.device)    # World L2-distance mean (w/ robustifier), fixed_scale in SMPL
    if not (hand_jnts_stack is None):
        loss['hand_reprojection_loss'] = hand_joints_loss(smpl_verts, batched_cameras, hand_jnts_stack, hand_confs_stack, reproj_loss_scaler)

    return loss









class MaxMixturePrior(nn.Module):
    def __init__(self, prior_folder='prior',
                 num_gaussians=6, dtype=DEFAULT_DTYPE, epsilon=1e-16,
                 use_merged=True,
                 **kwargs):
        super(MaxMixturePrior, self).__init__()

        if dtype == DEFAULT_DTYPE:
            np_dtype = np.float32
        elif dtype == torch.float64:
            np_dtype = np.float64
        else:
            print('Unknown float type {}, exiting!'.format(dtype))
            sys.exit(-1)

        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged
        gmm_fn = 'gmm_{:02d}.pkl'.format(num_gaussians)

        full_gmm_fn = os.path.join(prior_folder, gmm_fn)
        if not os.path.exists(full_gmm_fn):
            print('The path to the mixture prior "{}"'.format(full_gmm_fn) +
                  ' does not exist, exiting!')
            sys.exit(-1)

        with open(full_gmm_fn, 'rb') as f:
            import pickle5 as pickle
            gmm = pickle.load(f, encoding='latin1')

        if type(gmm) == dict:
            means = gmm['means'].astype(np_dtype)
            covs = gmm['covars'].astype(np_dtype)
            weights = gmm['weights'].astype(np_dtype)
        elif 'sklearn.mixture.gmm.GMM' in str(type(gmm)):
            means = gmm.means_.astype(np_dtype)
            covs = gmm.covars_.astype(np_dtype)
            weights = gmm.weights_.astype(np_dtype)
        else:
            print('Unknown type for the prior: {}, exiting!'.format(type(gmm)))
            sys.exit(-1)

        self.register_buffer('means', torch.tensor(means, dtype=dtype))

        self.register_buffer('covs', torch.tensor(covs, dtype=dtype))

        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions).astype(np_dtype)

        self.register_buffer('precisions',
                             torch.tensor(precisions, dtype=dtype))

        # The constant term:
        sqrdets = np.array([(np.sqrt(np.linalg.det(c)))
                            for c in gmm['covars']])
        const = (2 * np.pi)**(69 / 2.)

        nll_weights = np.asarray(gmm['weights'] / (const *
                                                   (sqrdets / sqrdets.min())))
        nll_weights = torch.tensor(nll_weights, dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('nll_weights', nll_weights)

        weights = torch.tensor(gmm['weights'], dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('weights', weights)

        self.register_buffer('pi_term',
                             torch.log(torch.tensor(2 * np.pi, dtype=dtype)))

        cov_dets = [np.log(np.linalg.det(cov.astype(np_dtype)) + epsilon)
                    for cov in covs]
        self.register_buffer('cov_dets',
                             torch.tensor(cov_dets, dtype=dtype))

        # The dimensionality of the random variable
        self.random_var_dim = self.means.shape[1]

    def get_mean(self):
        ''' Returns the mean of the mixture '''
        mean_pose = torch.matmul(self.weights, self.means)
        return mean_pose

    def merged_log_likelihood(self, pose, betas):
        diff_from_mean = pose.unsqueeze(dim=1) - self.means

        prec_diff_prod = torch.einsum('mij,bmj->bmi',
                                      [self.precisions, diff_from_mean])
        diff_prec_quadratic = (prec_diff_prod * diff_from_mean).sum(dim=-1)

        curr_loglikelihood = 0.5 * diff_prec_quadratic - \
            torch.log(self.nll_weights)
        #  curr_loglikelihood = 0.5 * (self.cov_dets.unsqueeze(dim=0) +
        #  self.random_var_dim * self.pi_term +
        #  diff_prec_quadratic
        #  ) - torch.log(self.weights)

        min_likelihood, _ = torch.min(curr_loglikelihood, dim=1)
        return min_likelihood

    def log_likelihood(self, pose, betas, *args, **kwargs):
        ''' Create graph operation for negative log-likelihood calculation
        '''
        likelihoods = []

        for idx in range(self.num_gaussians):
            mean = self.means[idx]
            prec = self.precisions[idx]
            cov = self.covs[idx]
            diff_from_mean = pose - mean

            curr_loglikelihood = torch.einsum('bj,ji->bi',
                                              [diff_from_mean, prec])
            curr_loglikelihood = torch.einsum('bi,bi->b',
                                              [curr_loglikelihood,
                                               diff_from_mean])
            cov_term = torch.log(torch.det(cov) + self.epsilon)
            curr_loglikelihood += 0.5 * (cov_term +
                                         self.random_var_dim *
                                         self.pi_term)
            likelihoods.append(curr_loglikelihood)

        log_likelihoods = torch.stack(likelihoods, dim=1)
        min_idx = torch.argmin(log_likelihoods, dim=1)
        weight_component = self.nll_weights[:, min_idx]
        weight_component = -torch.log(weight_component)

        return weight_component + log_likelihoods[:, min_idx]

    def forward(self, pose, betas):
        if self.use_merged:
            return self.merged_log_likelihood(pose, betas)
        else:
            return self.log_likelihood(pose, betas)



def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared =  x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


pose_prior = MaxMixturePrior(prior_folder='./extradata/spin', num_gaussians=8, dtype=torch.float32).cuda()

def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
    """
    # We subtract 3 because pose does not include the global rotation of the model
    return torch.exp(pose[:, [55-3, 58-3, 12-3, 15-3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2



def joints_2d_loss(gt_joints_2d=None, joints_2d=None, joint_confidence=None, reproj_loss_scaler=None):
    # Weighted robust reprojection error
    scaleFactor = 112 * 2      # -112 and 112 space originally used in SIMPLify, to balance with other terms. 
    joint_diff = scaleFactor * (gt_joints_2d - joints_2d) / reproj_loss_scaler.reshape(-1, 1, 1)
    joint_diff = gmof(joint_diff, 100) # (B, J, 2)


    if (joint_confidence.shape[-1] == joint_weights.shape[0]):
        _joint_weights = (joint_confidence*joint_weights[:, 0]).unsqueeze(-1) ** 2       # (B, J, 1)
    
        joints_2dloss = (_joint_weights * joint_diff).sum(dim=[1,2]) / (_joint_weights.sum(dim=[1,2]) + 1e-9)
        joints_2dloss = joints_2dloss[_joint_weights.sum([1,2]) > 0] 

        joints_2dloss = joints_2dloss * joint_weights.sum()               # (to set same scale on reprojection loss)
    else:
        raise AssertionError()

    return joints_2dloss


# hand_joints_loss(smpl_verts, batched_cameras, hand_jnts_stack, hand_confs_stack, reproj_loss_scaler)
def hand_joints_loss(smpl_verts, batched_cameras, hand_jnts_stack, hand_confs_stack, reproj_loss_scaler):
    global smplverts2hand_mapping
    hand_joints_3d = torch.index_select(smpl_verts, 1, smplverts2hand_mapping)
    hand_joints_2d = batched_cameras(hand_joints_3d)
    # print(hand_jnts_stack.shape, hand_joints_2d.shape, reproj_loss_scaler.shape)
    # Weighted robust reprojection error
    scaleFactor = 112 * 2      # -112 and 112 space originally used in SIMPLify, to balance with other terms. 
    joint_diff = scaleFactor * (hand_jnts_stack - hand_joints_2d) / reproj_loss_scaler.reshape(-1, 1, 1)
    joint_diff = gmof(joint_diff, 100) # (B, J, 2)
    
    
    hand_confs_stack = hand_confs_stack.unsqueeze(-1)       # (B, J, 1)
    joints_2dloss = (hand_confs_stack * joint_diff).sum(dim=[1,2]) / (hand_confs_stack.sum(dim=[1,2]) + 1e-9)
    joints_2dloss = joints_2dloss * joint_weights.sum()     

    
    return joints_2dloss
    






def smpl_depth_loss(world_smpl_op_j3d, batched_cameras, op_depths, op_confs, opt_scale):
    # 0. get depths
    camera_centers = batched_cameras.global_camera_position     # (B,3)
    op_j3d_depths = ((world_smpl_op_j3d - camera_centers.unsqueeze(1)) ** 2).sum(-1).sqrt()     # (B, J)

    # 1. get filters
    valid_op_joints = (op_confs > 0)
    valid_depths = (op_depths > 0)
    valid_op = valid_op_joints * valid_depths

    # 2. calculate loss
    current_depth = op_j3d_depths[valid_op]
    pseudo_gt_depth = op_depths[valid_op]

    depth_diff = gmof((current_depth - pseudo_gt_depth) / opt_scale.detach(), 5)       # 100 in pixel space (512)
    depth_loss = torch.mean(depth_diff)

    return depth_loss


def joints_temporal_loss(joints_3d, fids, opt_scale):
    """
    get near fids
    """
    if len(fids) <= 1:
        return torch.zeros(1).float().to(joints_3d.device)

    # Joints 3D different
    joints_3d_f = torch.cat([joints_3d[0:1], joints_3d], dim=0)
    joints_3d_b = torch.cat([joints_3d, joints_3d[-1:]], dim=0)
    joints_3d_diff = ((joints_3d_f - joints_3d_b) ** 2).sum(-1).mean(-1) / (opt_scale.detach() **2)

    # find valid fids:
    fids_f = torch.tensor([fids[0], *fids])
    fids_b = torch.tensor([*fids, fids[1]])
    fids_diff = (fids_f - fids_b).abs()
    valid_pair_mask = (fids_diff == 1).to(joints_3d.device)

    # get masked temporal loss
    temporal_loss = (joints_3d_diff * valid_pair_mask).sum()

    return temporal_loss
