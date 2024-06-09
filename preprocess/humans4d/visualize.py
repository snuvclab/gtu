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
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


from preprocess.humans4d.dataset import load_results, load_default_camdicts
from utils.draw_op_jnts import draw_op_img, draw_bodypose_with_color, op25_to_op18
from utils.image_utils import gen_videos, draw_bbox
from utils.render_smpl import render_w_torch3d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='processed dir')
    parser.add_argument('--render_smpl', action='store_true', help='If True, also render SMPLs together')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # 0. Load dataset
    human4d_resfile = data_dir / "phalp_v2" / "results" / "demo_images_jpg.pkl"
    assert human4d_resfile.exists(), f"File Not Found | {str(human4d_resfile)}"
    h4d_results = load_results(human4d_resfile)


    # Load Images
    img_dir = data_dir / 'images'
    img_dict = dict()
    for img_fname in (list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))):
        img = cv2.imread(str(img_fname))
        fid = int(img_fname.name.split(".")[0])
        img_dict[fid] = img
        

    if args.render_smpl:
        from gtu.smpl_deformer.smpl_server import SMPLServer
        default_camdicts = load_default_camdicts(human4d_resfile)
        assert len(default_camdicts) > 0, "No valid detection exists"
            
        smpl_server = SMPLServer()

        smpl_rendered_dicts, smpl_rendering_alpha_dicts = render_w_torch3d(
            render_camdicts=default_camdicts,
            people_dict=h4d_results,
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
            img_dict[fid] = img


    for pid, final_result in tqdm(h4d_results.items()):
        for fid, frame_res in final_result.items():
            img = img_dict[fid]
            if frame_res['bbox'] is not None:
                img = draw_bbox(img, frame_res['bbox'], (255, 0, 0) , pid=pid)
            if frame_res['phalp_j2ds'] is not None:
                jnts = op25_to_op18(frame_res['phalp_j2ds'])
                img = draw_bodypose_with_color(img, jnts, img.shape[:2], (0, 0, 255))

                # raw_jnts_img = draw_op_img(jnts, img.shape[:2])
                # raw_jnts_img = np.array(raw_jnts_img)[..., ::-1]
                # jnts_fg = (raw_jnts_img.sum(-1, keepdims=True) > 0)
                # img = img[..., :3] * (1-jnts_fg) + raw_jnts_img * jnts_fg
                
            img_dict[fid] = img


    combined_save_dir = data_dir / "phalp_v2" / "joint_rendering"
    combined_save_dir.mkdir(exist_ok=True)
    for fid, img in img_dict.items():
        cv2.imwrite(str(combined_save_dir / f"{fid:07}.jpg"), img)
    gen_videos([combined_save_dir], is_jpg=True, fps=10, rm_dir=False)