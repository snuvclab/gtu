
import shutil
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from gtu.renderer.gaussian_renderer import render_for_diffusion
from utils.image_utils import gen_videos, img_add_text
from utils.log_utils import print_cli

def test_dgm_loaded_ti_properly(DGM, scene, people_infos, save_dir, pipe, n_test_camera=8):
    people_ids = [pi.human_id for pi in people_infos]
    dg_log_dir = save_dir / "check_loaded_ti"
    dg_log_dir.mkdir(exist_ok=True)

    white_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") # for testing
    black_bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    scene_cameras = scene.getTrainCameras()

    with torch.no_grad():
        for pi in people_infos:
            # load masks 
            pid = pi.human_id
            render_fid = sorted(pi.fids)[0]
            _data_idx = pi.fids.index(render_fid)
            uid = 0
            
            person_save_dir = dg_log_dir / f"{pi.human_id}"
            person_save_dir.mkdir(exist_ok=True)

            # load smpl_param
            beta = pi.beta
            if hasattr(pi, 'smpl_params'):
                smpl_param = pi.smpl_params[_data_idx]
                smpl_param[-10:] = beta

                smpl_param = smpl_param.to('cuda').float()
                smpl_param = smpl_param.unsqueeze(0)
            else:
                smpl_param = torch.cat([
                    pi.smpl_scale.reshape(-1),
                    pi.smpl_global_poses[_data_idx],
                    pi.smpl_local_poses[_data_idx],
                    beta
                ], dim=-1)
                smpl_param = smpl_param.unsqueeze(0)
            smpl_deformer = pi.smpl_deformer

            # cancel out global movement
            smpl_param[0, 0] = 1.       # Fix scale as 1
            smpl_param[0, 1:4] *= 0     # remove global translation
            smpl_param[0, 2] = 0.3     # remove global translation (transl + 0.3 on y direction)
            smpl_param[0, 4:7] *= 0     # remove global rotation


            # load person infos
            cam_dicts, aux_prompt_dicts, aux_neg_prompt_dicts, new_smpl_dicts = DGM.get_test_camera(
                smpl_param = smpl_param,
                smpl_deformer = smpl_deformer, 
                n_cameras=n_test_camera
            )
            # Reset dirs 
            for k in cam_dicts.keys():
                cam_save_dir = person_save_dir / k
                if cam_save_dir.exists():
                    shutil.rmtree(cam_save_dir)


            # Render images
            vid_dirs = []
            for k, v in cam_dicts.items():
                pos_prompts = aux_prompt_dicts[k]
                neg_prompts = aux_neg_prompt_dicts[k]
                smpl_params = new_smpl_dicts[k]

                cam_save_dir = person_save_dir / k
                cam_save_dir.mkdir(exist_ok=True)
                vid_dirs.append(cam_save_dir)
                
                idx = 0
                for cam, pos, neg, _smpl_param in zip(v, pos_prompts, neg_prompts, smpl_params):
                    rendered_output, op_cond, jnt_prompts = render_for_diffusion(
                        mini_cam = cam,
                        pipe = pipe,
                        person_info = pi,
                        smpl_param  = _smpl_param, 
                        uid = uid,
                        bg_color = white_bg,
                    )

                    normal_rendering, op_cond, jnt_prompts = render_for_diffusion(
                        mini_cam = cam,
                        pipe = pipe,
                        person_info = pi,
                        smpl_param  = _smpl_param, 
                        uid = uid,
                        bg_color = white_bg,
                        normal_rendering = True
                    )

                    pos = pos + jnt_prompts
                    # set prompt conditioning
                    print_cli(f"positive: {pos}\nneg:{neg}\n-------", 'debug')

                    DGM.guidance_controlnet.get_text_embeds([pos], [neg])

                    generated_img = DGM.guidance_controlnet.refine(
                        pred_rgb=rendered_output,
                        pid = pid,
                        cond_image = op_cond,
                        guidance_scale=7.5,
                        steps=20,
                        strength=0.
                    )
                    
                    # save images
                    op_cond[0].save(cam_save_dir / f'op_{idx:03}.jpg')

                    generated_img = generated_img.detach().squeeze().cpu().permute(1,2,0).numpy()
                    generated_img = (generated_img * 255).astype(np.uint8)
                    generated_img = generated_img[..., [2,1,0]]    # convert to BGR
                    generated_img = np.concatenate([
                        np.ones((50, generated_img.shape[1], 3), dtype=np.uint8),
                        generated_img
                    ], axis=0)
                    generated_img = img_add_text(generated_img, f"pos: {pos}")
                    cv2.imwrite(str(cam_save_dir / f'generated_{idx:03}.jpg'), generated_img)

                    rendered_output = rendered_output.detach().cpu().squeeze().permute(1,2,0).numpy()
                    rendered_output = (rendered_output * 255).astype(np.uint8)   
                    cv2.imwrite(str(cam_save_dir / f'render_{idx:03}.jpg'), rendered_output)

                    normal_rendering = normal_rendering.detach().cpu().squeeze().permute(1,2,0).numpy()
                    normal_rendering = (normal_rendering * 255).astype(np.uint8)   

                    joint_imgs = np.array(op_cond[0])[..., ::-1]
                    jnts_fg = (joint_imgs.sum(-1, keepdims=True) > 0)
                    normal_rendering = normal_rendering[..., :3] * (1-jnts_fg) + joint_imgs * jnts_fg
                    normal_rendering = normal_rendering.astype(np.uint8)
                    normal_rendering = img_add_text(normal_rendering, f"neg: {neg}")
                    cv2.imwrite(str(cam_save_dir / f'overlay_{idx:03}.jpg'), normal_rendering)

                    idx += 1
            
            # Make Videos
            gen_videos(vid_dirs, is_jpg=True, fps=10, rm_dir=False, regex_fname="op_*.jpg", save_tag="_op")
            gen_videos(vid_dirs, is_jpg=True, fps=10, rm_dir=False, regex_fname="generated_*.jpg", save_tag="_gen")
            gen_videos(vid_dirs, is_jpg=True, fps=10, rm_dir=False, regex_fname="render_*.jpg", save_tag="_raw")
            gen_videos(vid_dirs, is_jpg=True, fps=10, rm_dir=False, regex_fname="overlay_*.jpg", save_tag="_overlay")

