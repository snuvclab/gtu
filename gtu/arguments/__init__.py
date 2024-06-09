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

from argparse import ArgumentParser, Namespace
import sys
import os


class HumanOptimizationParams:
    def __init__(self):
        self.position_lr_init = 0.001           # 0.00016
        self.position_lr_final = 0.000002        # 0.00 00 016
        self.position_lr_delay_mult = 0.02      # 0.01
        self.position_lr_max_steps = 6000       # 30_000
        self.feature_lr = 0.0025                  # 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005                 # 0.005 (original)
        self.rotation_lr = 0.001
        self.percent_dense = 0.01                # 0.01   if low : less clone more split. 
        self.lambda_dssim = 0.2
        self.lambda_mask = 0.01
        self.lambda_trans_reg = 0.1             
        self.densification_interval = 500       # originally: 100
        self.opacity_reset_interval = 1500      # 3000 # (do reset on time (after double densification))
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.00005 # x1/4 then original method 


        self.clip_init_smpl_opacity = False
        self.smpl_opacity_clip_min = 0.9 # x1/4 then original method 

        # SMPL related fitting options ()
        self.do_smpl_mod = True
        self.fix_init_smpls_verts = False            #  If True, THE GS corresponding to the initial SMPL vertices does not be pruned and instead, returns initial points
                                                     # Will be deprecated, as its functions is duplicated on "allow_init_smpl_pruning"
        self.track_gs_parent_id = True              # Now, we always need to track it's parent....

        self.allow_init_smpl_splitting = True      # Need more thoughtful thinking here
        self.allow_init_smpl_cloning = True
        self.allow_init_smpl_pruning = False

        # regularizing Long (sharp) gaussian
        self.split_sharp_gaussian = False
        
        # Human Gaussian settings 
        self.sh_degree = 2
        self.view_dir_reg = False
        




DEBUG_FAST_LOADING = False
MV_TRAIN_SAMPLE_INTERVAL = 1
MV_AUX_SAMPLE_INTERVAL = 16
EVAL_LOAD_ALL_MV = True

# if DEBUG_FAST_LOADING:
#     print("\n\n\n\n\n\n[WARN!!!!!!!!!!!!!!!!!!!!!!!!!!!!]\n Debugging Option is ON now\n\n\n")
#     MV_TRAIN_SAMPLE_INTERVAL = 64
#     MV_AUX_SAMPLE_INTERVAL = 256

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    if value:
                        group.add_argument("--no_" + key, default=value, action="store_false")
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()

        skip_lists = []
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                if arg[0] in skip_lists:
                    continue
                setattr(group, arg[0], arg[1])
            elif arg[0][:3] == "no_" and arg[0][3:] in vars(self):
                if not arg[1]:
                    print(f"turning off {arg[0][3:]} {arg[1]}")
                    setattr(group, arg[0][3:], False)
                    skip_lists.append(arg[0][3:])
        return group


class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3                                      # SH degree for scene
        self._source_path = ""
        self._model_path = ""
        self.mask_path = ""
        self.occlusion_mask_path = "none"
        self.frame_sample_ratio = ""
        self.textual_inversion_path = ""
        self.textual_inversion_method = ""
        self.use_canon = False                                  
        self.use_canon_single_camera = False                    # Valid only if it's training canonical (use single forward camera)

        ## Settings on Diffusion
        # opt settings
        self.apply_dgm_every_n = 1                              # Apply Diffusion Guidance every n iter
        self.dgm_start_iter = 1000

        # noise & camera settings
        self.dgm_noise_sched = "time_annealing"
        self.dgm_random_sample = True                           # If False, only upper bound is defined
        self.dgm_camera_sched = "default"                       # camera sampling strategy
        self.dgm_cfg_sched = "default"


        # Textual Inversion Settings
        self.use_ti_in_controlnet = False
        self.use_ti_free_prompt_on_controlnet = False
        self.ti_chkpt_epoch = -1                                # -1 means loading most recent checkpoints

        # diffusion settings
        self.dgm_cfg_scale = 100.                                # cfg scale
        self.dgm_controlnet_weight = 0.7                         # 1.0 default. for inpainting, 0.5 is recommended. for img2img, 0.7 is recommended.
        self.dgm_use_cfg_rescale = True                     
        self.dgm_use_inpaint = False                            # enable inpainting (vsibility based)
        self.dgm_hard_masking = False
        self.dgm_use_optimizer_masking = False                  # Instead of masking with rendered visibility, just mask optimizers
        self.dgm_minimum_mask_thrs = 0.02
        self.dgm_cfg_rescale_weight = 0.8
        self.dgm_use_inpaint_unet = False
        self.dgm_inpaint_guidance_scale = 7.5
        self.dgm_use_joint_diff = False
        self.use_diffusion_guidance = False

        # multi-SDS settings
        self.dgm_use_ddim_denoiser = False                          # use ddim denoiser
        self.dgm_multi_step_ddim_step = 20                          # DDIM step size
        self.dgm_use_fixed_max_step = True                          # if False, it always do DGM_MULTI_STEP_DDIM_STEP for denoising. Else, the maximum denoise step is DGM_MULTI_STEP_DDIM_STEP


        ## Settings on SMPL opt iterations
        self.iter_fix_smpl_init_verts = 1500                        # from this iter, SMPL vertices move freely
        self.iter_smpl_densify = 1500                               # Option, that turning on densify & splitting of human gaussians
        self.iter_densify_smpl_until = 3500                        # Do densify until this iterations (# of densify is different for each people)
        self.person_smpl_reset = -200000                        # do reset right after second densification. (just skip resetting here)
        self.iter_prune_smpl_until = 7000                       # Prune invalid points until 7000 iterations (seems not that effective)
        # (do optimization for 5000 iterations..)

        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.random_background = False
        self._use_mask = False
        self._loss_mask = False
        self._opt_smpl = False
        self.use_trans_grid = False
        self.data_device = "cuda"
        self.main_camera = -1
        self.eval = False

        self.n_person = -1
        self.frames_start = -1
        self.frames_end = -1
        self.flip_bg_mask = False
        
        self.manual_smpl_global_scale = -1.


        self.iter_clip_person_shs = 10                            # clip shs every n iters
        self.smpl_view_dir_reg = True
        self.clip_init_smpl_opacity = False                             # clip SMPL init opacity 
        self.smpl_opacity_clip_min = 0.7
        self.optimize_bg_from_zero = False
        
        self.use_adaptive_rgb_loss = False                      # use adpative loss based on MAX noise range of diffusion
        self.use_lpips_loss = False
        self.use_density_reg_loss = False                       # apply density regularize loss or not. (in DGM rendering)
        self.use_novel_view_density_reg = False                 # If True, calculate novel_view density_reg (Should not use together with use_diffusion_guidance)

        self.eval_with_black_bg = False
        self.render_smpl_emfs = False
        

        # Some debug factors
        self.check_ti_loaded = False
        self.human_sh_degree = 2
        self.human_sh_degree_enlarge_interval = 2000        # prev: 1500
        
        # Some updated features
        self.target_pid = -1
        self.preload_human_masks = False                      # We use it to optimize with individual masks  
        self.train_wo_bg = False
        self.dilate_human_occ_masks_ratio = 0.03                 # In default, we dilate the occlusion mask 3% (0.0 to disable)
        self.use_skip_fids_for_indiv_loading = False             # If true, and "skip_fids.txt" exists, skip written fids for human.scene

        self.apply_occ_mask_on_density_loss = True                # To avoid initial degenerate of density
        
        
        # some features for evaluation on People-Snapshot Datasets
        self.use_data_beta_for_test = False                          # If True, use beta from data (when testing beta is different from training beta)
        self.use_mask_loss = False                                   # If True, get mask loss  



        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        g.main_camera = int(g.main_camera)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.use_wandb = False
        self.view_dir_reg = False
        self.disc_gaussian = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_mask = 0.01      # active if uses
        self.lambda_lpips = 0.1                 # 24.02.06 add LPIPS loss
        self.lambda_density_reg = 0.1           # 24.02.06 add density reg loss (weight from IA) -> Let's set higher value
        self.lambda_trans_reg = 10
        self.lambda_init_smpl_verts_reg = 1e6        # SMPL vertices regularization (initial vertices <-> moved Gaussian center)

        self.lambda_rgb_loss = 1e4
        self.lambda_dg_loss = 1.
        self.lambda_cd_loss = 0

        # it's relative values (when SDS loss is set to 1.0)
        self.lambda_dgm_percep = 1.0
        self.lambda_dgm_rgb = 0.1

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        super().__init__(parser, "Optimization Parameters")
        
        
        
class RenderParams(ParamGroup):
    def __init__(self, parser):
        # self.traj_paths = []
        # self.human_model_paths = []
        self.camera_paths = ""
        
        
        super().__init__(parser, "Combiner Rendering Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
