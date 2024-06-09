#
# Training Background Gaussian roughly before joint optimizations
#
import sys
import os
import cv2
import torch
import numpy as np

from pathlib import Path
from tqdm import tqdm
from random import randint, random
from argparse import ArgumentParser, Namespace

from gtu.renderer.gaussian_renderer import render
from gtu.renderer.renderer_wrapper import render_set
from gtu.dataset.scene import Scene, GaussianModel
from gtu.arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state
from utils.image_utils import tensor2cv, psnr
from utils.loss_utils import l1_loss, ssim


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, view_dir_reg=pipe.view_dir_reg, load_mv_bg=dataset.load_mv_bg, load_aux_mv=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # Load occlusion masks
    if dataset.occlusion_mask_path != "":
        print("try to load front-occlusion masks")
        if scene.cam_name is None:
            raise NotImplementedError(f"[ERROR] mask_path isn't supported for non-mv dataset, yet")

        occ_mask_path = os.path.join(dataset.occlusion_mask_path, scene.cam_name)
        occ_cam_dict = dict()
        occ_mask_path = Path(occ_mask_path)

        for mask_fname in sorted(list(occ_mask_path.glob("*.png"))+list(occ_mask_path.glob("*.jpg"))):
            fid = int(mask_fname.name.split(".")[0])
            occ_cam_dict[fid] = torch.from_numpy(cv2.imread(str(mask_fname), 0)>1).float().squeeze()

        scene.occ_cam_dict = occ_cam_dict
    else:
        scene.occ_cam_dict = None


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    
    white_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    black_bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    if dataset.random_background:
        print("[INFO] using random background!")

    if dataset.use_bg_reg:
        print("[INFO] using Sphere Background Initialization!")
        with torch.no_grad():
            scene_xyzs = gaussians.get_xyz
            dists = (scene_xyzs ** 2).sum(-1).sqrt()
            bg_radius = dists.detach().cpu().numpy().mean()
            LAMBDA_BG_REG = 1e5
            scene.cameras_extent *= (bg_radius)

            print(f"[INFO] INIT BG RADIUS : {bg_radius}")
    else:
        LAMBDA_BG_REG = 0

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
        

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            if len(scene.aux_cam_dict) > 0:
                for k, v in scene.aux_cam_dict.items():
                    viewpoint_stack.extend(v)
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        if dataset.random_background:
            invert_bg_color = random() > 0.5
            background = white_bg if invert_bg_color else black_bg
        
            
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        if not scene.occ_cam_dict is None:
            fid = viewpoint_cam.colmap_id
            if fid in scene.occ_cam_dict:
                # print("applying occlusion mask")
                height = gt_image.shape[-2]
                width = gt_image.shape[-1]

                occ_mask = scene.occ_cam_dict[fid].cuda()[None, None]
                occ_mask = torch.nn.functional.interpolate(occ_mask, (height, width), mode="bilinear", align_corners=False)
                occ_mask = occ_mask[0]

                image = image * occ_mask
                gt_image = gt_image * occ_mask



        # Do filtering if mask exist
        if not (viewpoint_cam.gt_alpha_mask is None):
            mask = viewpoint_cam.gt_alpha_mask.cuda()
            if dataset.reverse_mask:
                mask = 1 - mask
            image = image * mask
            gt_image = gt_image * mask
            
        Ll1 = l1_loss(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "N_gaussian": f"{gaussians.get_n_points}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            background = white_bg if dataset.white_background else black_bg
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                os.makedirs(scene.model_path + "/bg_training", exist_ok=True)

                if len(scene.aux_cam_dict) > 0:
                    cameras = [scene.getTrainCameras()[0]]
                    for k, v in scene.aux_cam_dict.items():
                        cameras.append(v[0])
                    render_set(scene.model_path + "/bg_training", "test", iteration, cameras, scene.gaussians, [], pipe, background, is_canon=False)
                else:
                    render_set(scene.model_path + "/bg_training", "test", iteration, scene.getTrainCameras(), scene.gaussians, [], pipe, background, is_canon=False)
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)


                delete_unseen = True if iteration == 3000 else False

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold * 0.5, 0.005, scene.cameras_extent, size_threshold, delete_unseen)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = False)      # This might be reason of high memory consumptions


            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        # Do regularization
        if LAMBDA_BG_REG > 0:
            if True:
                scene_xyzs = gaussians.get_xyz
                dists = (scene_xyzs ** 2).sum(-1).sqrt()
                bg_rg_loss = ((dists[dists < bg_radius] - bg_radius) ** 2).mean()
                
                loss = LAMBDA_BG_REG * bg_rg_loss
                loss.backward()
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = False)
            else:
                with torch.no_grad():
                    scene_xyzs = gaussians.get_xyz
                    dists = (scene_xyzs ** 2).sum(-1).sqrt()
                    errorneous_mask = dists < bg_radius
                    errorneous_mask = errorneous_mask.unsqueeze(-1).repeat(1,3)

                    gaussians._xyz[errorneous_mask] /= (gaussians._xyz[errorneous_mask] ** 2).sum(-1).sqrt().unsqueeze(-1)
                    gaussians._xyz[errorneous_mask] *= bg_radius

                    bg_rg_loss = ((dists[dists < bg_radius] - bg_radius) ** 2).mean()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--load_mv_bg", action='store_true', default = False)
    parser.add_argument("--bg_radius", type=float, default=10.)
    parser.add_argument("--use_bg_reg", action='store_true')
    parser.add_argument("--reverse_mask", action='store_true')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("[INFO] Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    lp_extracted = lp.extract(args)
    lp_extracted.bg_radius = args.bg_radius
    lp_extracted.use_bg_reg = args.use_bg_reg
    lp_extracted.reverse_mask = args.reverse_mask
    lp_extracted.load_mv_bg = args.load_mv_bg

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp_extracted, op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
