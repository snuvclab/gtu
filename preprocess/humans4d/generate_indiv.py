# Generate indiv bbox cropping
import argparse
import json
from pathlib import Path
from math import floor, ceil


import cv2
import numpy as np
from tqdm import tqdm


from preprocess.humans4d.dataset import load_results, load_default_camdicts
from utils.draw_op_jnts import draw_op_img, draw_bodypose_with_color, op25_to_op18
from utils.image_utils import gen_videos, draw_bbox, get_crop_img_w_jnts
from utils.render_smpl import render_w_torch3d

VITPOSE_CROP_OFFSET_RATIO = 1.2


def dump_vitpose_json(save_res_dict, json_fname):
    json_dict = dict()
    # Add default dictionaries

    json_dict['categories'] = [
        dict(
            keypoints=[
                "nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle"
            ],
            skeleton=[
                [
                    16,
                    14
                ],
                [
                    14,
                    12
                ],
                [
                    17,
                    15
                ],
                [
                    15,
                    13
                ],
                [
                    12,
                    13
                ],
                [
                    6,
                    12
                ],
                [
                    7,
                    13
                ],
                [
                    6,
                    7
                ],
                [
                    6,
                    8
                ],
                [
                    7,
                    9
                ],
                [
                    8,
                    10
                ],
                [
                    9,
                    11
                ],
                [
                    2,
                    3
                ],
                [
                    1,
                    2
                ],
                [
                    1,
                    3
                ],
                [
                    2,
                    4
                ],
                [
                    3,
                    5
                ],
                [
                    4,
                    6
                ],
                [
                    5,
                    7
                ]
            ],
            id=1,
            supercategory='person',
            name='person'
        )
    ]
    json_dict['images'] = []
    json_dict['annotations'] = []

    # add bbox to json_dict
    rescale = VITPOSE_CROP_OFFSET_RATIO
    for i in range(len(save_res_dict['fid'])):
        bbox = save_res_dict['bbox'][i]
        if isinstance(bbox, np,ndarray):
            bbox = bbox.tolist()
        img_name = save_res_dict['img_name'][i]
        _id = i

        # 1. add image information
        json_dict['images'].append(
            dict(
                file_name = img_name,
                width = save_res_dict['W'][i],
                height = save_res_dict['H'][i],
                id = _id
            )
        )

 
        json_dict['annotations'].append(
            dict(
                category_id = 1,
                iscrowd = 0,
                image_id = _id,
                bbox = bbox,
                j2d = [jnt.tolist() for jnt in save_res_dict['j2d'][i]],
                id = _id
            )
        )

    # dump json
    with open(json_fname, 'w') as f:
        json.dump(json_dict, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='processed dir')
    parser.add_argument('--just_save_crop_json', action='store_true', help='If it is true, instead of making cropped image, just save json for top-down approach')
    parser.add_argument('--for_vitpose', action='store_true', help='If it is true, dump in vitpose json format')
    parser.add_argument('--crop_resize', type=int, default=-1, help='If it is positive, do resize on cropped image into 512')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # 0. Load dataset
    human4d_resfile = data_dir / "phalp_v2" / "results" / "demo_images_jpg.pkl"
    assert human4d_resfile.exists(), f"File Not Found | {str(human4d_resfile)}"
    h4d_results = load_results(human4d_resfile)


    # Load Images
    img_dir = data_dir / 'images'
    img_dict = dict()
    img_fname_dict = dict()
    for img_fname in (list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))):
        img = cv2.imread(str(img_fname))
        fid = int(img_fname.name.split(".")[0])
        img_dict[fid] = img
        img_fname_dict[fid] = img_fname
        

    if not args.just_save_crop_json:
        # Save individuals (really? No We don't need it)
        indiv_dir = data_dir / 'indiv'
        indiv_dir.mkdir(exist_ok=True)

    for pid, final_result in tqdm(h4d_results.items()):
        new_res_dict = dict(
            fid = [],
            bbox = [],
            img_name = [],
            min_bbox = [],
            j2d=[],
            H=[],
            W=[]
        )
        # For vitpose

        for fid, frame_res in final_result.items():
            img = img_dict[fid]
            H,W,_ = img.shape
            bbox = frame_res['bbox']
            op_jnts = frame_res['phalp_j2ds'] # Which is Openpose-25 joints
            crop_offset_ratio = 1.5

            # Get 2D JNTS of SMPLs
            cropped_img, cropped_jnts = get_crop_img_w_jnts(img.copy(), bbox, op_jnts, rescale=crop_offset_ratio, resize=args.crop_resize) # Here we skip resizing part (so that to be able to load data naively

            # Get left-top of the bbox (which is the offset of projected jnts)
            min_x = bbox[0]
            min_y = bbox[1]
            max_x = bbox[0] + bbox[2]
            max_y = bbox[1] + bbox[3]
            
            _w = int((max_x-min_x)*crop_offset_ratio)
            _h = int((max_y-min_y)*crop_offset_ratio)
            c_x = (min_x + max_x) // 2
            c_y = (min_y + max_y) // 2
            
            w = _w if _w>_h else _h
            h = w

            lt_x = floor(c_x - w//2)
            lt_y = floor(c_y - h//2)

            save_dict = dict(
                lefttop=[lt_x, lt_y],
                jnts=cropped_jnts
            )

            new_res_dict['fid'].append(fid)
            new_res_dict['bbox'].append(bbox.tolist())
            new_res_dict['img_name'].append(img_fname_dict[fid].name)
            new_res_dict['j2d'].append(op_jnts) # It's op25
            new_res_dict['H'].append(H)
            new_res_dict['W'].append(W)

            if not args.just_save_crop_json:
                # PID processing
                person_dir = indiv_dir / f"{pid:05}"
                person_dir.mkdir(exist_ok=True)

                np.save(str(person_dir / f"{fid:09}.npy"), save_dict)
                cv2.imwrite(str(person_dir / f"{fid:09}.jpg"), cropped_img)


        # VitPose processing
        indiv_json_fname = data_dir / "phalp_v2" / f"{pid:05}_gt_bbox.json"

        if args.for_vitpose:
            dump_vitpose_json(new_res_dict, indiv_json_fname)
        else:
            del new_res_dict['j2d']
            with open(indiv_json_fname, 'w') as f:
                json.dump(new_res_dict, f)


