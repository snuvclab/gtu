"""
For simplicity, we use OpenPose implemented by controlnet authors here.
You can also use https://github.com/IDEA-Research/DWPose options here, which requires mmpose series.


Currently, only (N x 64) resolution is supported. others are not allowed.
"""
import argparse
import torch
import cv2
import pickle
import json
import numpy as np
from typing import List, Dict

import warnings
# Ignore specific UserWarning
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_5m_224 in registry with controlnet_aux.segment_anything.modeling.tiny_vit_sam.tiny_vit_5m_224.*")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_11m_224 in registry with controlnet_aux.segment_anything.modeling.tiny_vit_sam.tiny_vit_11m_224.*")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_21m_224 in registry with controlnet_aux.segment_anything.modeling.tiny_vit_sam.tiny_vit_21m_224.*")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_21m_384 in registry with controlnet_aux.segment_anything.modeling.tiny_vit_sam.tiny_vit_21m_384.*")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_21m_512 in registry with controlnet_aux.segment_anything.modeling.tiny_vit_sam.tiny_vit_21m_512.*")


from PIL import Image
from pathlib import Path
from tqdm import tqdm
from controlnet_aux import OpenposeDetector
from controlnet_aux.util import HWC3, resize_image
from controlnet_aux.open_pose import util as op_util
from controlnet_aux.dwpose import util as dw_util

from utils.io_utils import write_pickle
from utils.image_utils import get_crop_img


class WrapperDWPose():
    def __init__(self, dwpose):
        self.dwpose = dwpose

    def __call__(self, input_image, bbox=None, detect_resolution=512, image_resolution=512, output_type="pil", **kwargs):
        input_image = cv2.cvtColor(np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR)    # DWPose require BGR

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
        input_image = HWC3(input_image)
        raw_input_image = input_image
        H, W, C = input_image.shape
        raw_H = H
        raw_W = W

        if bbox is not None:
            input_image, new_bbox = get_crop_img(input_image, bbox, rescale=1.1, resize=-1, get_new_bbox=True)
            W = new_bbox[-2]
            H = new_bbox[-1]

        input_image = resize_image(input_image, detect_resolution)
        
        with torch.no_grad():
            candidate, subset = self.dwpose.pose_estimation(input_image)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(detect_resolution)
            candidate[..., 1] /= float(detect_resolution)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = np.copy(subset[:,:18])
            
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            foot = candidate[:,18:24]
            faces = candidate[:,24:92]
            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)
            
            detected_map = self.draw_pose(pose, H, W)
            detected_map = HWC3(detected_map)
            

            if bbox is not None:
                black_bg = np.zeros(raw_input_image.shape, dtype=np.uint8)
                x, y, w, h = new_bbox

                if x+w >= raw_W:
                    w = raw_W-x-1
                    detected_map = detected_map[:, :w]
                if y+h >= raw_H:
                    h = raw_H-y-1
                    detected_map = detected_map[:h]
                if x < 0:
                    detected_map = detected_map[:, -x:]
                    w = w + x
                    x = 0
                if y < 0:
                    detected_map = detected_map[-y:]
                    h = h + y
                    y = 0

                black_bg[y:y+h, x:x+w] = detected_map
                detected_map = black_bg
            else:
                detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
                w = raw_W
                h = raw_H

            # Save detected image
            op_img_mask = (detected_map.sum(-1) > 0)
            detected_map = raw_input_image[...,::-1] * (1-op_img_mask[..., None]) + detected_map*(op_img_mask[..., None])
            detected_map = detected_map.astype(np.uint8)
            if output_type == "pil":
                detected_map = Image.fromarray(detected_map)

            
            # Please refer https://github.com/huggingface/controlnet_aux/blob/master/src/controlnet_aux/dwpose/util.py#L67 to finish this code
            # When we check the pose order, the order is different from DWPose and default OpenPose
            new_poses = []
            for _cand, _score in zip(candidate, subset):
                body = _cand[:24]
                body_score = _score[:24]
                # foot = _cand[18:24]
                # foot_score = _score[18:24]
                faces = _cand[24:92]
                faces_score = _score[24:92]
                left_hands = _cand[92:113]
                left_hands_score = _score[92:113]
                right_hands = _cand[113:]
                right_hands_score = _score[113:]


                _body = []
                for kpt, s in zip(body, body_score):
                    if kpt[0] < 0 and kpt[1] < 0:
                        _body.append(None)
                        continue
                    save_kpt = [kpt[0], kpt[1], s]
                    if bbox is not None:
                        x1, y1, w, h = new_bbox
                        save_kpt[0] = save_kpt[0] * w + x1
                        save_kpt[1] = save_kpt[1] * h + y1
                    _body.append(save_kpt)
                            

                _left_hand = []
                for kpt, s in zip(left_hands, left_hands_score):
                    if kpt[0] < 0 and kpt[1] < 0:
                        _left_hand.append(None)
                        continue
                    save_kpt = [kpt[0], kpt[1], s]
                    if bbox is not None:
                        x1, y1, w, h = new_bbox
                        save_kpt[0] = save_kpt[0] * w + x1
                        save_kpt[1] = save_kpt[1] * h + y1
                    _left_hand.append(save_kpt)
                
                _right_hand = []
                for kpt, s in zip(right_hands, right_hands_score):
                    if kpt[0] < 0 and kpt[1] < 0:
                        _right_hand.append(None)
                        continue
                    save_kpt = [kpt[0], kpt[1], s]
                    if bbox is not None:
                        x1, y1, w, h = new_bbox
                        save_kpt[0] = save_kpt[0] * w + x1
                        save_kpt[1] = save_kpt[1] * h + y1
                    _right_hand.append(save_kpt)


                _face = []
                for kpt, s in zip(right_hands, right_hands_score):
                    if kpt[0] < 0 and kpt[1] < 0:
                        _face.append(None)
                        continue
                    save_kpt = [kpt[0], kpt[1], s]
                    if bbox is not None:
                        x1, y1, w, h = new_bbox
                        save_kpt[0] = save_kpt[0] * w + x1
                        save_kpt[1] = save_kpt[1] * h + y1
                    _face.append(save_kpt)

                new_pose = dict(
                    body = _body,
                    left_hand = _left_hand,
                    right_hand = _right_hand,
                    face = _face,
                )
                new_poses.append(new_pose)

            return detected_map, new_poses

    def draw_pose(self, pose, H, W):
        bodies = pose['bodies']
        faces = pose['faces']
        hands = pose['hands']
        candidate = bodies['candidate']
        subset = bodies['subset']
        
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        canvas = dw_util.draw_bodypose(canvas, candidate, subset)
        canvas = dw_util.draw_handpose(canvas, hands)
        canvas = dw_util.draw_facepose(canvas, faces)

        return canvas

        
class WrapperOPPose():
    def __init__(self, oppose):
        self.oppose = oppose

    def __call__(self, input_image, bbox=None, detect_resolution=512, image_resolution=None, include_body=True, include_hand=False, include_face=False, hand_and_face=None, output_type="pil", **kwargs):
        if hand_and_face is not None:
            warnings.warn("hand_and_face is deprecated. Use include_hand and include_face instead.", DeprecationWarning)
            include_hand = hand_and_face
            include_face = hand_and_face

        if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"
        if type(output_type) is bool:
            warnings.warn("Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
            if output_type:
                output_type = "pil"

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
        input_image = HWC3(input_image)
        raw_input_image = input_image
        H, W, C = input_image.shape
        raw_H = H
        raw_W = W

        if bbox is not None:
            input_image, new_bbox = get_crop_img(input_image, bbox, rescale=1.1, resize=-1, get_new_bbox=True)
            W = new_bbox[-2]
            H = new_bbox[-1]

        input_image = resize_image(input_image, detect_resolution)
        
        poses = self.oppose.detect_poses(input_image, include_hand, include_face)
        canvas = self.draw_poses(poses, H, W) 
        detected_map = canvas
        detected_map = HWC3(detected_map)

        if bbox is not None:
            black_bg = np.zeros(raw_input_image.shape, dtype=np.uint8)
            x, y, w, h = new_bbox

            if x+w >= raw_W:
                w = raw_W-x-1
                detected_map = detected_map[:, :w]
            if y+h >= raw_H:
                h = raw_H-y-1
                detected_map = detected_map[:h]
            if x < 0:
                detected_map = detected_map[:, -x:]
                w = w + x
                x = 0
            if y < 0:
                detected_map = detected_map[-y:]
                h = h + y
                y = 0

            black_bg[y:y+h, x:x+w] = detected_map
            detected_map = black_bg
        else:
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
            w = raw_W
            h = raw_H
        

        op_img_mask = (detected_map.sum(-1) > 0)
        detected_map = raw_input_image * (1-op_img_mask[..., None]) + detected_map*(op_img_mask[..., None])
        detected_map = detected_map.astype(np.uint8)
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)


        new_poses = []
        for _pose in poses:
            _body = []
            if _pose.body is None:
                _body = None
            else:
                for kpt in _pose.body.keypoints:
                    if kpt is None:
                        _body.append(None)
                    else:
                        save_kpt = [kpt.x, kpt.y, kpt.score]
                        if bbox is not None:
                            x1, y1, w, h = new_bbox
                            save_kpt[0] = save_kpt[0] * w + x1
                            save_kpt[1] = save_kpt[1] * h + y1
                        _body.append(save_kpt)
                        
            _left_hand = []
            if _pose.left_hand is None:
                _left_hand = None
            else:
                for kpt in _pose.left_hand:
                    if kpt is None:
                        _left_hand.append(None)
                    else:
                        save_kpt = [kpt.x, kpt.y, kpt.score]
                        if bbox is not None:
                            x1, y1, w, h = new_bbox
                            save_kpt[0] = save_kpt[0] * w + x1
                            save_kpt[1] = save_kpt[1] * h + y1

                        _left_hand.append(save_kpt)

            _right_hand = []
            if _pose.right_hand is None:
                _right_hand = None
            else:
                for kpt in _pose.right_hand:
                    if kpt is None:
                        _right_hand.append(None)
                    else:
                        save_kpt = [kpt.x, kpt.y, kpt.score]
                        if bbox is not None:
                            x1, y1, w, h = new_bbox
                            save_kpt[0] = save_kpt[0] * w + x1
                            save_kpt[1] = save_kpt[1] * h + y1

                        _right_hand.append(save_kpt)

            _face = []
            if _pose.face is None:
                _face = None
            else:
                for kpt in _pose.face:
                    if kpt is None:
                        _face.append(None)
                    else:
                        save_kpt = [kpt.x, kpt.y, kpt.score]
                        if bbox is not None:
                            x1, y1, w, h = new_bbox
                            save_kpt[0] = save_kpt[0] * w + x1
                            save_kpt[1] = save_kpt[1] * h + y1

                        _face.append(save_kpt)


            new_pose = dict(
                body = _body,
                left_hand = _left_hand,
                right_hand = _right_hand,
                face = _face,
            )
            new_poses.append(new_pose)

        return detected_map, new_poses

    
    def draw_poses(self, poses: List, H, W, draw_body=True, draw_hand=True, draw_face=True):
        """
        Draw the detected poses on an empty canvas.

        Args:
            poses (List[PoseResult]): A list of PoseResult objects containing the detected poses.
            H (int): The height of the canvas.
            W (int): The width of the canvas.
            draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
            draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
            draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.

        Returns:
            numpy.ndarray: A 3D numpy array representing the canvas with the drawn poses.
        """
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

        for pose in poses:
            if draw_body:
                canvas = op_util.draw_bodypose(canvas, pose.body.keypoints)

            if draw_hand:
                canvas = op_util.draw_handpose(canvas, pose.left_hand)
                canvas = op_util.draw_handpose(canvas, pose.right_hand)

            if draw_face:
                canvas = op_util.draw_facepose(canvas, pose.face)

        return canvas

        


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Options for OpenPose estimations.")
    parser.add_argument("--use_dw_pose", action='store_true', help="If then, we use DW Pose Detector for higher accuracy")
    parser.add_argument("--data_path", type=str, help="Path of images to estimate.")
    args = parser.parse_args()


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load Pose estimation modules
    if args.use_dw_pose:
        from controlnet_aux import DWposeDetector
        det_config = "submodules/controlnet_aux/src/controlnet_aux/dwpose/yolox_config/yolox_l_8xb8-300e_coco.py"
        det_ckpt = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth"
        pose_config = "submodules/controlnet_aux/src/controlnet_aux/dwpose/dwpose_config/dwpose-l_384x288.py"
        pose_ckpt = "https://huggingface.co/wanghaofan/dw-ll_ucoco_384/resolve/main/dw-ll_ucoco_384.pth"
        pose_estimator = DWposeDetector(det_config=det_config, det_ckpt=det_ckpt, pose_config=pose_config, pose_ckpt=pose_ckpt, device=device)
        pose_estimator = WrapperDWPose(pose_estimator)
    else:
        pose_estimator = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        pose_estimator = pose_estimator.to(device)
        pose_estimator = WrapperOPPose(pose_estimator)
    
    # Load images to estimate
    img_dir = Path(args.data_path) / 'images'
    img_lists = sorted(list(img_dir.glob("*.jpg"))+list(img_dir.glob("*.png")))

    # Make directories to save
    viz_dir = Path(args.data_path) / 'openpose'
    viz_dir.mkdir(exist_ok=True)

    # Load people track infos
    json_flist = (Path(args.data_path) / 'phalp_v2').glob("*.json")
    json_flist = sorted(list(json_flist))
    track_bbox_dicts = dict()
    indiv_viz_dict = dict()
    for track_json_fname in json_flist:
        pid = int(track_json_fname.name.split("_")[0])
        with open(track_json_fname, 'r') as f:
            data = json.load(f)

        person_bbox_dict = dict()
        for _idx, fid in enumerate(data['fid']):
            person_bbox_dict[fid] = data['bbox'][_idx]
        track_bbox_dicts[pid] = person_bbox_dict

        indiv_viz_dict[pid] = viz_dir / f"{pid:05}"
        indiv_viz_dict[pid].mkdir(exist_ok=True)


    # Estimate track infos
    estimation_dicts = dict()
    for pid in track_bbox_dicts.keys():
        estimation_dicts[pid] = dict()

    for img_fname in tqdm(img_lists):
        fid = int(img_fname.name.split(".")[0])
        img = Image.open(img_fname).convert("RGB")
        img_resolution = min(img.size)

        for pid, bbox_dict in track_bbox_dicts.items():
            if fid in bbox_dict:
                bbox = bbox_dict[fid]
                pose_overlay_img, poses = pose_estimator(img, bbox, include_hand=True, include_face=True, detect_resolution=512, image_resolution=img_resolution)

                estimation_dicts[pid][fid] = poses
                pose_overlay_img.save(indiv_viz_dict[pid] / img_fname.name) # save visualized images

    # Save openpose results
    save_fname = Path(args.data_path) / f"openpose_estimation.pkl"
    write_pickle(save_fname, estimation_dicts)

    print(f"[INFO] Finished all pose estimation of {str(img_dir)}")


if __name__ == '__main__':
    main()