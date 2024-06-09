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

import os
import argparse
import copy
import shutil
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path

from preprocess.humans4d.dataset import load_results as h4d_dataset
from preprocess.humans4d.dataset import load_default_camdicts
from utils.draw_op_jnts import draw_op_img, op25_to_op18, op18_to_op25, dwpose_to_op25
from utils.image_utils import gen_videos, draw_bbox
from utils.io_utils import read_pickle, write_pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='processed dir')   
    parser.add_argument('--viz', action='store_true', help='If true, visualize the tracking results')
    parser.add_argument('--render_smpl', action='store_true', help='If true, render SMPLs on visualizations')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # 0. Load dataset
    # 0.1 Load humans4D data
    pt_results = None
    human4d_resfile = data_dir / "phalp_v2" / "results" / "demo_images_jpg.pkl"
    h4d_results = h4d_dataset(human4d_resfile)

    # 0.1 Load OpenPose data
    openpose_file = data_dir / 'openpose_estimation.pkl'
    if openpose_file.exists():
        openpose_results = read_pickle(openpose_file)

        for pid in sorted(list(h4d_results.keys())):
            if pid not in openpose_results:
                continue
            indiv_vitpose_res = openpose_results[pid]

            for fid in (indiv_vitpose_res.keys()):
                frame_reses = openpose_results[pid][fid]
                if len(frame_reses)==0:
                    # When no person detected
                    continue

                if len(frame_reses) == 1:
                    # when only single person detected
                    frame_res = frame_reses[0]
                    if 'body' not in frame_res:
                        # No valid body pose detection exists
                        continue

                if len(frame_reses) > 1:
                    # Need to find best matching person
                    phalp_j2ds = h4d_results[pid][fid]['phalp_j2ds']        # Which is OP25

                    min_distance = 999999999
                    min_id = -1
                    
                    for op_id, _frame_res in enumerate(frame_reses):
                        distance = 0
                        j_cnt = 0
                        if 'body' in _frame_res:
                            if len(_frame_res['body']) == 19 or len(_frame_res['body']) == 18:      # why 19 is included?
                                op25_est = op18_to_op25(_frame_res['body'])
                            elif len(_frame_res['body']) == 24:
                                op25_est = dwpose_to_op25(_frame_res['body'])
                            else:
                                op25_est = _frame_res['body']

                            for p_j, o_j in zip(phalp_j2ds, op25_est):
                                if o_j is not None:
                                    distance += np.sqrt((p_j[0] - o_j[0]) ** 2 + (p_j[1] - o_j[1]) ** 2)
                                    j_cnt += 1

                        if j_cnt > 0:                                  
                            distance /= j_cnt

                            if distance < min_distance:
                                min_distance = distance
                                min_id = op_id

                    if min_id < 0:
                        # No valid body pose detection exists
                        continue
                    else:
                        frame_res = frame_reses[min_id]

                # Simply select first one
                if len(frame_res['body']) == 19:
                    op25_est = op18_to_op25(frame_res['body'])
                elif len(frame_res['body']) == 24:
                    op25_est = dwpose_to_op25(frame_res['body'])
                else:
                    op25_est = frame_res['body']

                h4d_results[pid][fid]['body_pose'] = op25_est
                h4d_results[pid][fid]['hand_pose'] = dict(
                    left_hand=frame_res['left_hand'], 
                    right_hand=frame_res['right_hand']
                )
                h4d_results[pid][fid]['face'] = frame_res['face']
                
                # Remove phap mask (to save memory)
                del h4d_results[pid][fid]['phalp_mask'] 


    # Load Images
    img_dir = data_dir / 'images'
    img_dict = dict()
    combined_img_dict = dict()
    for img_fname in (list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))):
        img = cv2.imread(str(img_fname))
        if img is None:
            os.remove(img_fname)
            continue
        fid = int(img_fname.name.split(".")[0])
        img_dict[fid] = img
        combined_img_dict[fid] = img.copy()


    # 1. Save results
    final_results = h4d_results
    save_fname = data_dir / "op_phalp_merged.pkl"
    write_pickle(save_fname, final_results)

    # 2.1 make "merge_hids.txt"
    with open(data_dir / "merge_hids.txt", 'w') as f:
        f.write("\n")

    # 3. Visualize 
    if args.viz:
        from utils.draw_op_jnts import draw_bodypose_with_color
        if args.render_smpl:
            from utils.render_smpl import render_w_torch3d
            from gtu.smpl_deformer.smpl_server import SMPLServer
            default_camdicts = load_default_camdicts(human4d_resfile)
            smpl_server = SMPLServer()

            smpl_rendered_dicts, smpl_rendering_alpha_dicts = render_w_torch3d(
                render_camdicts=default_camdicts,
                people_dict=final_results,
                smpl_server=smpl_server,
                render_camera_position=False,
                render_bg_as_pc=False,
                get_smpl_alpha=True,
                for_viz=False,
                skip_text=True
            ) 

            for fid, smpl_rendering in smpl_rendered_dicts.items():
                if fid not in img_dict:
                    continue

                alpha_rendering = smpl_rendering_alpha_dicts[fid] / 255.
                alpha_rendering = alpha_rendering[..., None]
                img = img_dict[fid]
                c_img = combined_img_dict[fid]
                img = img * (1 - alpha_rendering) + smpl_rendering * alpha_rendering
                c_img = c_img * (1 - alpha_rendering) + smpl_rendering * alpha_rendering
                img_dict[fid] = img
                combined_img_dict[fid] = c_img


        # Make Save directories
        overall_save_dir = data_dir / 'merged_trackings'
        if overall_save_dir.exists():
            shutil.rmtree(str(overall_save_dir))
        overall_save_dir.mkdir(exist_ok=True)

        # Choose COLOR of plotting
        AP_COLOR = (0, 255, 0)  
        H4D_COLOR = (255, 0, 0) 
        GREY_COLOR = (127, 127, 127)    # fort merged bbox


        # Make pid folder
        render_folders = []
        for pid, final_result in tqdm(final_results.items(), desc="rendering each person"):
            indiv_save_dir = overall_save_dir / f"{pid:05}"
            indiv_save_dir.mkdir(exist_ok=True)

            for fid, frame_res in final_result.items():
                if fid not in img_dict:
                    continue

                img = img_dict[fid].copy()
                if frame_res['bbox'] is not None:
                    img = draw_bbox(img, frame_res['bbox'], GREY_COLOR, pid=pid)
                if 'bbox_phalp' in frame_res and frame_res['bbox_phalp'] is not None:
                    img = draw_bbox(img, frame_res['bbox_phalp'], H4D_COLOR, pid=pid)
                if 'bbox_ap' in frame_res and frame_res['bbox_ap'] is not None:
                    img = draw_bbox(img, frame_res['bbox_ap'], AP_COLOR, pid=pid)


                if 'phalp_j2ds' in frame_res and frame_res['phalp_j2ds'] is not None:
                    img = draw_bodypose_with_color(img, op25_to_op18(frame_res['phalp_j2ds']), img.shape[:2], H4D_COLOR)
                if 'body_pose' in frame_res and frame_res['body_pose'] is not None:
                    img = draw_bodypose_with_color(img, op25_to_op18(frame_res['body_pose']), img.shape[:2], AP_COLOR)
                cv2.imwrite(str(indiv_save_dir / f"{fid:07}.jpg"), img)


                # Also do same visualize on combined_img_dict
                img = combined_img_dict[fid]
                if frame_res['bbox'] is not None:
                    img = draw_bbox(img, frame_res['bbox'], GREY_COLOR, pid=pid)
                if 'bbox_phalp' in frame_res and frame_res['bbox_phalp'] is not None:
                    img = draw_bbox(img, frame_res['bbox_phalp'], H4D_COLOR, pid=pid)
                if 'bbox_ap' in frame_res and frame_res['bbox_ap'] is not None:
                    img = draw_bbox(img, frame_res['bbox_ap'], AP_COLOR, pid=pid)


                if 'phalp_j2ds' in frame_res and frame_res['phalp_j2ds'] is not None:
                    img = draw_bodypose_with_color(img, op25_to_op18(frame_res['phalp_j2ds']), img.shape[:2], H4D_COLOR)
                if 'body_pose' in frame_res['body_pose'] is not None:
                    img = draw_bodypose_with_color(img, op25_to_op18(frame_res['body_pose']), img.shape[:2], AP_COLOR)
                combined_img_dict[fid] = img
            render_folders.append(indiv_save_dir)
        
        # save combined rendering
        combined_save_dir = overall_save_dir / "_combined"
        combined_save_dir.mkdir(exist_ok=True)
        for fid, img in combined_img_dict.items():
            cv2.imwrite(str(combined_save_dir / f"{fid:07}.jpg"), img)
        render_folders.append(combined_save_dir)


        # Make videos
        gen_videos(render_folders, is_jpg=True, fps=10, rm_dir=True)