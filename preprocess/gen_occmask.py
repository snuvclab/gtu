# You should run this file with detectron installed. 
# Or u can use grounded sam instead
import argparse
import math
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict

import cv2
import json
import numpy as np

from utils.io_utils import read_pickle
from utils.mask_utils import dilate_mask, save_mask, load_mask, erode_mask
from utils.image_utils import stitch_outputs, gen_videos

OVERWRAP_RATIO_THRS = 0.7

# TODO! if it's buggy, update this visualizer! 
# def show_mask(masks, image):
#     fig = plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#     ax = plt.gca()
    
#     for mask in masks:s
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#         h, w = mask.shape[-2:]
#         mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#         ax.imshow(mask_image)
        
    
#     plt.axis('off')
#     fig.canvas.draw()
#     mask_img = np.array(fig.canvas.renderer._renderer)
    
#     plt.clf()
#     plt.close('all')
    
#     return mask_img

def do_class_based_masking(data_dir: Path, save_dir: Path):
    """
        Do masking of dynamic object. Can't handle static object here.
    """
    from preprocess.autocolmap.scene_segment import dir_segment, select_predictor
    img_dir = data_dir / 'images'
    

    m_name = 'COCO_RCNN_50'
    predictor, cfg = select_predictor(m_name)

    dir_segment(
        predictor = predictor, 
        cfg = cfg, 
        _dir = str(img_dir), 
        check_panop = False,
        save_bbox = False, 
        bg_is_zero = False, 
        output_path = save_dir,
        include_human = False,
        save_for_occmask = True
    )


def get_average_depth(depth_map: np.ndarray, mask: np.ndarray, filter_ratio: float=0.3):
    if mask.shape[:2] != depth_map.shape[:2]:
        mask = cv2.resize(mask, depth_map.shape[:2])

    # Get depths of masked area
    depth_map = depth_map.reshape(-1)
    mask = (mask.reshape(-1) > 0)
    valid_depths = depth_map[mask]

    # Do filtering
    valid_depths = np.sort(valid_depths)
    n_valid_points = len(valid_depths)

    n_filter = math.ceil(n_valid_points * filter_ratio)
    if n_valid_points <= n_filter *2:
        valid_depths = valid_depths
    else:
        valid_depths = valid_depths[:n_valid_points][-n_valid_points:]
    
    # Get Average depth
    avg_depth = valid_depths.mean()

    return avg_depth



def estimate_occmask(person_mask: np.ndarray, disparity: np.ndarray, pseg: Dict, filter_corresponding_pmask: bool=True):
    """
        Estimate Occlusion Masks
    """
    skipping_label_values = []
    pseg_mask = pseg['mask']
    pseg_pmask = None


    # 0. Filter corresponding person_mask
    if filter_corresponding_pmask:
        print("[INFO] filter out mask of pseg corresponding to person_mask here")
        pmask = (person_mask > 0)

        pseg_pmask = []
        pseg_labels = []
        for label_value, label_name in pseg['label'].items():
            indiv_mask = (pseg_mask == label_value) 
            bg_indiv_mask = (1 - indiv_mask)   
            bg_indiv_mask = dilate_mask(bg_indiv_mask, kernel_size=5)
            indiv_mask = (bg_indiv_mask == 0)
            

            indiv_points = indiv_mask.sum() + 1
            overwrapped_points = ((indiv_mask * pmask) > 0).sum()
            overwrap_ratio = overwrapped_points / indiv_points
            overwrap_ratio_2 = overwrapped_points / (pmask.sum() + 1)
            overwrap_ratio = overwrap_ratio if overwrap_ratio > overwrap_ratio_2 else overwrap_ratio_2
            
            print(overwrap_ratio)
            if overwrap_ratio > OVERWRAP_RATIO_THRS:
                skipping_label_values.append(label_value)
                pseg_pmask.append(indiv_mask)
                pseg_labels.append(label_name)
        
        if len(pseg_pmask) > 0:
            pseg_pmask = np.stack(pseg_pmask, axis=0).sum(0)
            pseg_pmask = pseg_pmask > 0

            print(f"[INFO] filtered out following things: {pseg_labels}")

            person_mask = (pseg_pmask + person_mask) > 0
        else:
            pseg_pmask = None

    # 1. dilate mask and find corresponding masks
    original_mask = person_mask.copy()
    dilated_mask = dilate_mask(person_mask.copy())
    purely_dilated_mask = (dilated_mask > 0) * (original_mask == 0)

    flatten_pseg_mask = pseg_mask.reshape(-1)
    purely_dilated_mask = purely_dilated_mask.reshape(-1)
    neighboring_pseg_indices = flatten_pseg_mask[purely_dilated_mask]
    neighboring_pseg_indices = np.unique(neighboring_pseg_indices)

    neighbor_indices = []
    neighbor_masks = []
    for pseg_ind in neighboring_pseg_indices:
        if not pseg_ind in skipping_label_values:
            neighbor_indices.append(pseg_ind)
            nmask = (pseg_mask == pseg_ind)
            neighbor_masks.append(nmask)
    if len(neighbor_masks) == 0:
        occmasks = np.zeros_like(original_mask)
        nmasks = np.zeros_like(original_mask)
        
        return occmasks, nmasks, pseg_pmask
    
    # Make neighboring Mask
    nmasks = np.stack(neighbor_masks, axis=0)
    nmasks = nmasks.sum(0)
    nmasks = nmasks > 0


    # 2. Get avg_depth of mask & filter outs
    pmask_avg_depth = get_average_depth(depth_map=disparity, mask=person_mask)
    occmasks = []
    for neighbor_ind in neighbor_indices:
        nmask = (pseg_mask == neighbor_ind)
        bg_nmask = (1 - nmask)   
        bg_nmask = dilate_mask(bg_nmask, kernel_size=5)
        nmask = (bg_nmask == 0)
        p_avg_depth = get_average_depth(depth_map=disparity, mask=nmask)

        if p_avg_depth > pmask_avg_depth:
            occmasks.append(nmask)
    
    if len(occmasks) > 0:
        occmasks = np.stack(occmasks, axis=0)
        occmasks = occmasks.sum(0)
        occmasks = occmasks > 0
    else:
        occmasks = np.zeros_like(person_mask)
    
    # 
    occmasks = dilate_mask(occmasks, kernel_size=5)
    occmasks = dilate_mask(occmasks, kernel_size=5)
    occmasks = occmasks > 0
    
    return occmasks, nmasks, pseg_pmask




def load_panoptic_segmentations(pseg_dir: Path):
    """
        Load panoptic Segmentation mask
    """
    label_fdict = dict()
    mask_fdict = dict()
    for mask_fname in pseg_dir.glob("mask_*.jpg"):
        fid = int(mask_fname.name.split("_")[-1][:-4])
        mask_fdict[fid] = cv2.imread(str(mask_fname), -1)

    for label_fname in pseg_dir.glob("label_*.json"):
        fid = int(label_fname.name.split("_")[-1][:-5])
        with open(label_fname, 'r') as f:
            label_data = json.load(f)['mask']
        
        label_dict = dict()
        for label in label_data:
            label_dict[label['value']] = label['label']
        label_fdict[fid] = label_dict

    
    mask_res_dict = dict()
    for fid in sorted(list(label_fdict.keys())):
        mask_res_dict[fid] = dict(
            label = label_fdict[fid],
            mask = mask_fdict[fid]
        )

    return mask_res_dict




def do_indiv_masking(data_dir: Path, indiv_dir: Path, save_overview: bool=False):
    # Get Camera lists.
    cam_list = []
    for cam_dir in indiv_dir.iterdir():
        if cam_dir.is_dir():
            try:
                cam_id = int(cam_dir.name)
                cam_list.append(cam_dir)
            except:
                print(f"{str(cam_dir)} is not camera directory. skip it")
                continue

    # Also get depth informations.
    depth_cam_dicts = dict()
    panoptic_cam_dicts = dict()
    overview_fname_dicts = dict()
    for cam_dir in cam_list:
        cam_name = cam_dir.name

        # load depth
        data_cam_dir = data_dir / cam_name
        assert data_cam_dir.exists(), f"{str(data_cam_dir)} not exist!"
        depth_data = read_pickle(data_cam_dir / "mono_depth.pkl")
        depth_cam_dicts[cam_name] = depth_data

        # load pseg
        pseg_dir = data_cam_dir / "panoptic_segmentation"
        assert pseg_dir.exists(), f"{str(pseg_dir)} not exist!"
        panoptic_cam_dicts[cam_name] = load_panoptic_segmentations(pseg_dir)

        if save_overview:
            # load input image fdicts
            img_fname_dict = dict()
            for img_fname in (data_cam_dir / 'images').glob("*.png"):
                fid = int(img_fname.name[:-4])
                img_fname_dict[fid] = img_fname
            
            # load pseg image fdicts
            pseg_fname_dict = dict()
            for img_fname in (data_cam_dir / 'panoptic_segmentation').glob("label_output_*.jpg"):
                fid = int(img_fname.name.split("_")[-1][:-4])
                pseg_fname_dict[fid] = img_fname
                
            overview_fname_dicts[cam_name] = dict(
                pseg = pseg_fname_dict,
                image = img_fname_dict
            )


    for cam_dir in tqdm(cam_list, desc="iterating Cameras"):
        people_dict = dict()
        depth_dict = depth_cam_dicts[cam_dir.name]
        pseg_dict = panoptic_cam_dicts[cam_dir.name]
        overview_dict = overview_fname_dicts[cam_dir.name]

        # 0. Find people lists
        for possible_person in cam_dir.iterdir():
            if possible_person.is_dir():
                try:
                    pid = int(possible_person.name)
                    people_dict[pid] = possible_person
                except:
                    # it's not person
                    continue
        
        # 1. Do iteration on each person
        for pid, person_dir in people_dict.items():
            # Make save directories first
            occmask_save_dir = person_dir / 'occ_masks'
            occmask_save_dir.mkdir(exist_ok=True)

            if save_overview:
                overview_save_dir = person_dir / 'overview'
                overview_save_dir.mkdir(exist_ok=True)

            # list-up indiv masks
            pmask_fdict = dict()
            for pmask_fname in (person_dir / 'masks').glob("*.png"):
                fid = int(pmask_fname.name[:-4])
                pmask_fdict[fid] = pmask_fname

            # We only process the "existing frames" only.
            for fid in tqdm(sorted(list(pmask_fdict.keys())), desc=f"pid: {pid}, fid: {fid}"):
                assert (fid in depth_dict), f"fid: {fid} not in depth_dict"
                assert (fid in pseg_dict), f"fid: {fid} not in pseg_dict"

                person_mask = cv2.imread(str(pmask_fdict[fid]), -1)
                disparity = depth_dict[fid]
                pseg = pseg_dict[fid]

                occmask, neighbor_mask, pseg_pmask = estimate_occmask(person_mask, disparity, pseg)
                save_mask(occmask_save_dir / f"{fid:06}.png", occmask)

                if save_overview:
                    #
                    # raw image   person_mask     neighbor mask   occmask
                    #
                    # pseg mask   pseg_pmask      depth_map(disparity)
                    #
                    
                    raw_image = cv2.imread(str(overview_dict['image'][fid]), -1)
                    _H, _W, _ = raw_image.shape
                    H = 1000 if _H > 1000 else _H
                    W = int(H * _W / _H)
                    resize_shape = (H, W)


                    person_mask = (person_mask > 0)
                    person_mask = (np.repeat(person_mask[..., None], 3, axis=-1) * 255).astype(np.uint8)
                    neighbor_mask = (np.repeat(neighbor_mask[..., None], 3, axis=-1) * 255).astype(np.uint8)
                    if pseg_pmask is None:
                        pseg_pmask = np.zeros(resize_shape)
                    pseg_pmask = (np.repeat(pseg_pmask[..., None], 3, axis=-1) * 255).astype(np.uint8)
                    occmask = (np.repeat(occmask[..., None], 3, axis=-1) * 255).astype(np.uint8)
                    pseg_mask = cv2.imread(str(overview_dict['pseg'][fid]), -1)
                    disparity = np.repeat(disparity[..., None], 3, axis=-1)
                    disparity_img = ((disparity - disparity.min()) / (disparity.max() - disparity.min()) * 255.0).astype(np.uint8)


                    # print(person_mask.shape)
                    # print(neighbor_mask.shape)
                    # print(pseg_pmask.shape)
                    # print(occmask.shape)
                    # print(pseg_mask.shape)
                    # print(disparity_img.shape)
                    # print(raw_image.shape)



                    save_img = [
                        dict(
                            raw_image = cv2.resize(raw_image, resize_shape),
                            individual_mask = cv2.resize(person_mask, resize_shape),
                            neighbor_mask = cv2.resize(neighbor_mask, resize_shape),
                            occlusion_mask = cv2.resize(occmask, resize_shape) 
                        ),
                        dict(
                            panoptic_seg = cv2.resize(pseg_mask, resize_shape),
                            corres_pseg = cv2.resize(pseg_pmask, resize_shape),
                            disparity = cv2.resize(disparity_img, resize_shape)
                        )
                    ]
                    overview_img = stitch_outputs(save_img)[0]
                    cv2.imwrite(str(overview_save_dir / f"{fid:06}.jpg"), overview_img)
                    
            
            # 3. Make videos of results (for debugging)
            gen_videos([occmask_save_dir], is_jpg=False, fps=10, rm_dir=False)
            if save_overview:
                gen_videos([overview_save_dir], is_jpg=True, fps=10, rm_dir=False)




    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None, help='overall data dir')
    parser.add_argument('--save_dir', type=str, default=None, help='overall data dir')
    parser.add_argument('--class_based_masking', action='store_true', help='Use class-based filtering of dynamic objects')
    parser.add_argument('--save_overview', action='store_true', help='Save overview of processings')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)

    assert data_dir.exists(), f"Data directory {str(data_dir)} Not exists"

    if args.class_based_masking:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True)
        do_class_based_masking(data_dir, save_dir)
    else:
        print("[INFO] We use depth & panoptic segmentation based masking here!")
        print("[WARNING] Need to query []_hn datafile as save_dir here, for individual segmentation mask")

        save_dir = Path(args.save_dir)
        assert save_dir.exists(), f"Indiv directory {str(save_dir)} Not exists"

        do_indiv_masking(data_dir=data_dir, indiv_dir=save_dir, save_overview=args.save_overview)
 
    print("Done!")
    
    
    
    if False:
        indiv_dir = Path(args.save_dir)
        cam_list = []
        for cam_dir in indiv_dir.iterdir():
            if cam_dir.is_dir():
                try:
                    cam_id = int(cam_dir.name)
                    cam_list.append(cam_dir)
                except:
                    print(f"{str(cam_dir)} is not camera directory. skip it")
                    continue
        
        people_dict = dict()
        for cam_dir in tqdm(cam_list, desc="iterating Cameras"):
            # 0. Find people lists
            for possible_person in cam_dir.iterdir():
                if possible_person.is_dir():
                    try:
                        pid = int(possible_person.name)
                        people_dict[pid] = possible_person
                    except:
                        # it's not person
                        continue
        
        for pid, person_dir in people_dict.items():
            # Make save directories first
            occmask_save_dir = person_dir / 'occ_masks'
            
            for occ_mask_fname in occmask_save_dir.glob("*.png"):
                occmasks = load_mask(occ_mask_fname)
                occmasks = dilate_mask(occmasks, kernel_size=5)
                occmasks = occmasks > 0
                
                save_mask(occmask_save_dir / f"{occ_mask_fname.name}", occmasks)
            gen_videos([occmask_save_dir], is_jpg=False, fps=10, rm_dir=False)


    
    if False:
        # filter out static BG (which generated due to inaccurate cropping)
        data_dir = Path(args.data_dir)
        indiv_dir = Path(args.save_dir)
        cam_list = []
        for cam_dir in indiv_dir.iterdir():
            if cam_dir.is_dir():
                cam_list.append(cam_dir)
        
        
        for cam_dir in tqdm(cam_list, desc="iterating Cameras"):
            people_dict = dict()
            # 0. Find people lists
            for possible_person in cam_dir.iterdir():
                if possible_person.is_dir():
                    try:
                        pid = int(possible_person.name)
                        people_dict[pid] = possible_person
                    except:
                        # it's not person
                        continue
            # Load all image
            cam_name = cam_dir.name
            data_cam_dir = data_dir / cam_name
            img_fname_dict = dict()
            imgs_stack = []
            for img_fname in tqdm((data_cam_dir / 'images').glob("*.png")):
                fid = int(img_fname.name[:-4])
                img_fname_dict[fid] = img_fname

                imgs_stack.append(cv2.imread(str(img_fname)))

            imgs_stack = np.stack(imgs_stack, axis=0)
            max_image = imgs_stack.max(0)
            min_image = imgs_stack.min(0)
            max_min_gap = max_image - min_image
            letterbox = (max_min_gap < 3)
            letterbox_mask = (letterbox.sum(-1) == 3)
            # letterbox_mask = (1-letterbox)


            for pid, person_dir in people_dict.items():
                # Make save directories first
                occmask_save_dir = person_dir / 'occ_masks'
                
                for occ_mask_fname in occmask_save_dir.glob("*.png"):
                    occmasks = load_mask(occ_mask_fname)
                    # occmasks = dilate_mask(occmasks, kernel_size=5)
                    occmasks = occmasks + letterbox_mask
                    occmasks = occmasks > 0
                
                    save_mask(occmask_save_dir / f"{occ_mask_fname.name}", occmasks)
                gen_videos([occmask_save_dir], is_jpg=False, fps=10, rm_dir=False)
            
