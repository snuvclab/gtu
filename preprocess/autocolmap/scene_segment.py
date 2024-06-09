
#from socket import SIO_KEEPALIVE_VALS
import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import os.path as osp
import glob
import tqdm

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

COCO_IND_LISTS = [
    1, # person
    2, # bicycle
    3, #car
    4, #motorcycle
    5, #airplane
    6, #bus
    7, #train
    8, #truck
    9, #boat
    16, #bird
    17, #cat
    18, #dog
    19, #horse
    33, #ball
]
# TODO ; consider handling clothes (segmentation : exclusive)

USE_RCNN_50_FPN = True
DEBUG = False

SAVE_BBOX = True




MODEL_LIST = [
    'COCO_RCNN_50',
    'LVIS_RCNN_101',
    'CITY_RCNN_50',
    'COCO_PANOP_RCNN_101',
]

def select_predictor(model, thrs=0.5):
    cfg = get_cfg()

    if model not in MODEL_LIST:
        print("wrong model")
        assert()

    if model == MODEL_LIST[0]:
        print("select model : {}".format(model))
        
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thrs  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    elif model == MODEL_LIST[1]:
        print("select model : {}".format(model))
        
        cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thrs  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml")
    
    elif model == MODEL_LIST[2]:
        print("select model : {}".format(model))
        
        cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thrs  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")
    
    elif model == MODEL_LIST[3]:
        print("select model : {}".format(model))
        
        cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thrs  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    
    else:
        assert()

    predictor = DefaultPredictor(cfg)

    return predictor, cfg


def get_predictor():
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    
    if USE_RCNN_50_FPN:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    else:
        cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")
    
    predictor = DefaultPredictor(cfg)

    return predictor, cfg







def dir_segment(predictor, cfg, _dir, save_bbox = SAVE_BBOX, bg_is_zero = False, output_path=None, only_dynamic=True, check_panop=False, include_human=True, save_for_occmask=False):
    '''
    We should use 'bg_is_zero'=False to ONLY extract pose from BG
    '''
    img_list = glob.glob(os.path.join(_dir,"*.png"))
    img_list.extend(glob.glob(os.path.join(_dir,"*.PNG")))
    img_list.extend(glob.glob(os.path.join(_dir,"*.jpg")))
    img_list.extend(glob.glob(os.path.join(_dir,"*.JPG")))
    img_list.extend(glob.glob(os.path.join(_dir,"*.jpeg")))
    img_list.extend(glob.glob(os.path.join(_dir,"*.JPEG")))

    if save_for_occmask:
        output_path = output_path
        os.makedirs(_dir, exist_ok=True)
    else:
        if isinstance(output_path, type(None)):
            os.makedirs(os.path.join(_dir, "masks"), exist_ok=True)
            os.makedirs(os.path.join(_dir, "detection_output"), exist_ok=True)
            os.makedirs(os.path.join(_dir, "bbox"), exist_ok=True)
            output_path = _dir
        else:
            os.makedirs(output_path, exist_ok=True)
            os.makedirs(os.path.join(output_path, "masks"), exist_ok=True)
            os.makedirs(os.path.join(output_path, "detection_output"), exist_ok=True)
            os.makedirs(os.path.join(output_path, "bbox"), exist_ok=True)

    
    for ind, img_path in tqdm.tqdm(enumerate(img_list)):
        if ind == 10 and DEBUG:
            print("debug this section")
            assert()
        img = cv2.imread(img_path)

        # get basename
        img_name = os.path.basename(img_path)   
        
        if save_for_occmask:
            mask_path = os.path.join(output_path, img_name)
        else:
            mask_path = os.path.join(output_path, "masks", img_name + ".png")
        comb_path = os.path.join(output_path, "detection_output", img_name + ".png")
        if save_bbox:
            bbox_path = os.path.join(output_path, "bbox", os.path.basename(img_name))
            os.makedirs(bbox_path, exist_ok=True)

        outputs = predictor(img)

        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        if check_panop:
            panop_path = os.path.join(output_path, "panoptic", img_name + ".png")
            os.makedirs(os.path.join(output_path, "panoptic"), exist_ok=True)

            #print(outputs['panoptic_seg'][0])
            out_panoptic = v.draw_panoptic_seg_predictions(outputs['panoptic_seg'][0].to("cpu"), outputs['panoptic_seg'][1])
            cv2.imwrite(panop_path, out_panoptic.get_image()[:, :, ::-1])


        masks = np.asarray(outputs["instances"].pred_masks.to("cpu"))
        classes = np.asarray(outputs["instances"].pred_classes.to("cpu"))

        mask = np.zeros(img.shape[0:2], dtype=np.uint8)
            
        for ind, item_mask in enumerate(masks):
            if only_dynamic:
                if (classes[ind]+1) == 1: # 1 means human
                    if include_human:
                        mask += item_mask
                # ignore non-dynamic targets 
                elif (classes[ind]+1) in COCO_IND_LISTS:  # detectron index start with 0 not 1
                    mask += item_mask
                if not USE_RCNN_50_FPN and not DEBUG:
                    print("need to implement ind lists for other segmentation dataset")
                    assert()
            else:
                # use all targets
                mask += item_mask 
            if save_bbox:
                if classes[ind] == 0:
                    # The case of "person"
                    # I will save both bbox & segmentation mask.
                    np.save(bbox_path+'/'+str(ind).zfill(5)+'.npy', outputs["instances"].pred_boxes[ind].tensor.detach().cpu().numpy())
                    cv2.imwrite(bbox_path+'/'+str(ind).zfill(5)+'.png', (item_mask>0)*255)


                    
        mask = (mask>0) * 255

        if not bg_is_zero:
            mask = 255 - mask


        # save results
        print(mask_path)
        cv2.imwrite(mask_path, mask)
        if not save_for_occmask:
            cv2.imwrite(comb_path, out.get_image()[:, :, ::-1])

