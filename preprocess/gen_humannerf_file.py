"""
Code to generate humannerf-version of people using processed data
"""
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
import os
import shutil
import numpy as np
import cv2
import torch
import trimesh
import json
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.renderer import (
    SfMPerspectiveCameras,
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures
from segment_anything import SamPredictor, sam_model_registry, build_sam
from tqdm import tqdm
from distutils.dir_util import copy_tree
from pathlib import Path
from PIL import Image
from typing import List, Dict


from gtu.smpl_deformer.smpl_server import SMPLServer
from utils.io_utils import read_pickle, write_pickle
from utils.mask_utils import dilate_mask


SMPL2COCO19 = np.array([
    12, # 0  Neck
    24, # 1  Nose
    # -1, # 2  Pelvis (Though it's not center of hips...)
    16, # 3  L Shoulder
    18, # 4  L Elbow
    20, # 5  L Wrist
    1,  # 6  L Hip
    4,  # 7  L Knee
    7,  # 8  L Ankle
    17, # 9  R Shoulder
    19, # 10 R Elbow
    21, # 11 R Wrist
    2,  # 12 R Hip
    5,  # 13 R Knee
    8,  # 14 R Ankle
    26, # 15 L Eey
    28, # 16 L Ear
    25, # 17 R Eye
    27, # 18 R Ear
], dtype=np.int32)

N_VIZ_JNTS_THRS = 3

# GROUNDED_SAM_SEARCH_KEYWORDS = "person"
# GROUNDED_SAM_BOX_THRS = 0.3
# GROUNDED_SAM_TEXT_THRS = 0.25


# def load_model(model_config_path, model_checkpoint_path, device):
#     args = SLConfig.fromfile(model_config_path)
#     args.device = device
#     model = build_model(args)
#     checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
#     load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
#     print(load_res)
#     _ = model.eval()
#     return model

# def load_grounded_sam(device):
#     pj_dir = Path(str(__file__)).parents[1]
#     grounded_sam_dir = pj_dir / "submodules" / "Grounded-Segment-Anything"
#     config_file = grounded_sam_dir / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
#     grounded_checkpoint = grounded_sam_dir / "groundingdino_swint_ogc.pth"
#     model = load_model(str(config_file), str(grounded_checkpoint), device=device)

# def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
#     caption = caption.lower()
#     caption = caption.strip()
#     if not caption.endswith("."):
#         caption = caption + "."
#     model = model.to(device)
#     image = image.to(device)
#     with torch.no_grad():
#         outputs = model(image[None], captions=[caption])
#     logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
#     boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
#     logits.shape[0]

#     # filter output
#     logits_filt = logits.clone()
#     boxes_filt = boxes.clone()
#     filt_mask = logits_filt.max(dim=1)[0] > box_threshold
#     logits_filt = logits_filt[filt_mask]  # num_filt, 256
#     boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
#     logits_filt.shape[0]

#     # get phrase
#     tokenlizer = model.tokenizer
#     tokenized = tokenlizer(caption)
#     # build pred
#     pred_phrases = []
#     for logit, box in zip(logits_filt, boxes_filt):
#         pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
#         if with_logits:
#             pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
#         else:
#             pred_phrases.append(pred_phrase)

#     return boxes_filt, pred_phrases

# def gsam_masking(image, gsam_model, sam_predictor, device):
#     image_pil = Image.fromarray(image).convert("RGB")
#     transform = T.Compose(
#         [
#             T.RandomResize([800], max_size=1333),
#             T.ToTensor(),
#             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ]
#     )
#     image_for_dino, _ = transform(image_pil, None)  # 3, h, w


#     boxes_filt, pred_phrases = get_grounding_output(
#         gsam_model, image_for_dino, GROUNDED_SAM_SEARCH_KEYWORDS, GROUNDED_SAM_BOX_THRS, GROUNDED_SAM_TEXT_THRS, device=device
#     )

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     sam_predictor.set_image(image)

#     size = image_pil.size
#     H, W = size[1], size[0]
#     for i in range(boxes_filt.size(0)):
#         boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
#         boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
#         boxes_filt[i][2:] += boxes_filt[i][:2]

#     boxes_filt = boxes_filt.cpu()
#     transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

#     masks, _, _ = predictor.predict_torch(
#         point_coords = None,
#         point_labels = None,
#         boxes = transformed_boxes.to(device),
#         multimask_output = False,
#     )

#     human_mask = masks.cpu().numpy().sum(0) > 0
#     return (human_mask * 255).astype(np.uint8)




def sam_masking(img, jnts, sam_predictor):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(img)    
    masks, _, _ = sam_predictor.predict(jnts[:, :2], np.ones_like(jnts[:, 0]))
    mask = masks.sum(axis=0) > 0
    return (mask * 255).astype(np.uint8)


def render_trimesh(renderer, mesh, R, T, mode='np', K=None):
    
    verts = torch.tensor(mesh.vertices).cuda().float()[None]
    faces = torch.tensor(mesh.faces).cuda()[None]
    colors = torch.tensor(mesh.visual.vertex_colors).float().cuda()[None,...,:3]/255
    renderer.set_camera(R,T,K)
    image = renderer.render_mesh_recon(verts, faces, colors=colors, mode=mode)[0]
    image = (255*image).data.cpu().numpy().astype(np.uint8)
    
    return image


def combine_meshes(mesh_list):
    """
    Combine multiple meshes into a single mesh.

    Args:
        mesh_list (list): List of PyTorch3D Meshes.

    Returns:
        Meshes: Combined mesh.
    """
    verts_list, faces_list, rgb_list = [], [], []

    # Extract vertices and faces from each mesh
    for mesh in mesh_list:
        verts_list.append(mesh.verts_packed())
        faces_list.append(mesh.faces_packed() + len(verts_list) - 1)

    # Concatenate vertices and faces
    combined_verts = torch.cat(verts_list, dim=0)
    combined_faces = torch.cat(faces_list, dim=0)

    # Create a new combined mesh
    combined_mesh = Meshes(verts=[combined_verts], faces=[combined_faces])

    return combined_mesh

class Renderer():
    def __init__(self, principal_point=None, img_size=None, cam_intrinsic = None):
    
        super().__init__()

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        self.cam_intrinsic = cam_intrinsic
        self.image_size = img_size
        self.render_img_size = np.max(img_size)

        principal_point = [-(self.cam_intrinsic[0,2]-self.image_size[1]/2.)/(self.image_size[1]/2.), -(self.cam_intrinsic[1,2]-self.image_size[0]/2.)/(self.image_size[0]/2.)]  
        self.principal_point = torch.tensor(principal_point, device=self.device).unsqueeze(0)

        self.cam_R = torch.from_numpy(np.array([[-1., 0., 0.],
                                                [0., -1., 0.],
                                                [0., 0., 1.]])).cuda().float().unsqueeze(0)

        self.cam_T = torch.zeros((1,3)).cuda().float()

        half_max_length = max(self.cam_intrinsic[0:2,2])
        self.focal_length = torch.tensor([(self.cam_intrinsic[0,0]/half_max_length).astype(np.float32), \
                                          (self.cam_intrinsic[1,1]/half_max_length).astype(np.float32)]).unsqueeze(0)
        
        self.cameras = SfMPerspectiveCameras(focal_length=self.focal_length, principal_point=self.principal_point, R=self.cam_R, T=self.cam_T, device=self.device)

        self.lights = PointLights(device=self.device,location=[[0.0, 0.0, 0.0]], ambient_color=((1,1,1),),diffuse_color=((0,0,0),),specular_color=((0,0,0),))

        self.raster_settings = RasterizationSettings(image_size=self.render_img_size, faces_per_pixel=10, blur_radius=0, max_faces_per_bin=30000)
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)

        self.shader = SoftPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)

        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)
    
    def set_camera(self, R, T, K=None):
        if K is None:
            focal_length = self.focal_length
            principal_point = self.principal_point
            raster_settings = self.raster_settings
        else:
            focal_length = torch.tensor([K[0,0], K[1,1]]).unsqueeze(0).to(self.device)
            principal_point = torch.tensor([K[0,2], K[1,2]]).unsqueeze(0).to(self.device)
            
            raster_settings = RasterizationSettings(image_size=self.image_size, faces_per_pixel=10, blur_radius=0, max_faces_per_bin=30000)
        
        self.cam_R = R
        self.cam_T = T
        self.cam_R[:, :2, :] *= -1.0
        self.cam_T[:, :1] *= -1.0
        self.cam_T[:, :2] *= -1.0

        # self.cam_R = torch.transpose(self.cam_R,1,2)
        self.cameras = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point, R=self.cam_R, T=self.cam_T, device=self.device, in_ndc=False, image_size=[self.image_size])
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings)
        self.shader = SoftPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)
        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)

    def render_mesh_recon(self, meshes):
        '''
        mode: normal, phong, texture
        '''
        with torch.no_grad():
            verts_ = []
            faces_ = []
            normal_vis_ = []
            object_ids = []
            n_verts = 0
            for k, _mesh in meshes.items():       
                verts = torch.tensor(_mesh.vertices).cuda().float()
                faces = torch.tensor(_mesh.faces).cuda()
                mesh = Meshes(verts[None], faces[None])
                normals = torch.stack(mesh.verts_normals_list())
        
                normals_vis = normals* 0.5 + 0.5
                normals_vis = normals_vis[:,:,[2,1,0]]

                verts_.append(verts)
                faces_.append(faces + n_verts)
                normal_vis_.append(normals_vis.squeeze())
                n_verts += len(verts.squeeze())

                object_ids.append(int(k))

            
            verts_ = torch.cat(verts_, dim=0)
            faces_ = torch.cat(faces_, dim=0)
            normal_vis_ = torch.cat(normal_vis_, dim=0)
            
            render_targets = Meshes([verts_], [faces_], textures=Textures(verts_rgb=[normal_vis_]))
            fragments = self.rasterizer(render_targets)
            pix_to_face = (fragments.pix_to_face).detach().cpu().squeeze()
            dists = fragments.dists.detach().cpu().squeeze()[..., 0]
            # map to 
            pix_to_face = pix_to_face.squeeze()[..., 0]
            pix_to_face[pix_to_face < 0] *= len(_mesh.faces.squeeze())
            mask_ids = pix_to_face // len(_mesh.faces.squeeze())

            res_mask = torch.zeros_like(mask_ids)
            min_dists = dict()
            for i, k in enumerate(object_ids):
                if (mask_ids == i).sum() > 0:
                    res_mask[mask_ids == i] = (torch.ones_like(res_mask[mask_ids == i]) * int(k)).long()
                    min_dists[k] = dists[mask_ids == i].min().data

            sorted_pids = sorted(list(min_dists.keys()), key=lambda x: list(min_dists.values())[list(min_dists.keys()).index(x)])

            # sort by dists

            return res_mask, sorted_pids



@torch.no_grad()
def process_single_cam(cam_dir: Path, save_dir: Path, smpl_models: Dict, smpl_genders: List, sam_predictor, use_person_mask: bool=False):
    print(f"[INFO] processing {str(cam_dir)}")
    # 1. make save dir
    save_dir.mkdir(exist_ok=True)

    reading_dir = cam_dir / 'images'
    assert reading_dir.exists(), f"{str(reading_dir)} does not exist"

    # However, it's waste of storage, copying ALL images.
    # Instead make shared image directory & connect symbolic link
    shared_img_dir = save_dir / '_images'
    shared_img_dir.mkdir(exist_ok=True)
    files = sorted(list(reading_dir.glob('*.png')))

    # Iterate over each file and copy it to the destination directory
    for file in files:
        source_path = str(file)
        destination_path = shared_img_dir / file.name
        # shutil.copy2(source_path, destination_path)
    print(f"Copied: {str(reading_dir)} to :{str(shared_img_dir)}")


    # 2. load images & img_dict
    img_dict = dict()
    img_fname_dict = dict()
    for img_fname in sorted(list((cam_dir / "images").glob("*.png"))+list((cam_dir / "images").glob("*.jpg"))):
        img = cv2.imread(str(img_fname))

        fid = int(img_fname.name[:-4])
        img_dict[fid] = img
        img_fname_dict[fid] = img_fname.name

    # 2.1 load person_mask
    person_fname_dict = dict()
    if use_person_mask:
        person_mask_dir = cam_dir / 'people_segmentation'
        assert reading_dir.exists(), f"{str(person_mask_dir)} does not exist"

        for img_fname in sorted(list((person_mask_dir).glob("*.png"))+list((person_mask_dir).glob("*.jpg"))):
            fid = int(img_fname.name[:-4])
            person_fname_dict[fid] = str(img_fname)

    # Second load SMPL informations
    people_dict = dict()
    people_save_dir_dict = dict()
    people_json_dict = dict()
    for pkl_fname in sorted(list(cam_dir.glob("results_p*.pkl"))):
        data = read_pickle(pkl_fname)

        pid = int(pkl_fname.name[:-4].split("_p")[-1])
        people_dict[pid] = data

        people_save_dir_dict[pid] = save_dir / f"{int(pid):03}"
        people_save_dir_dict[pid].mkdir(exist_ok=True)

        
        (people_save_dir_dict[pid] / 'masks').mkdir(exist_ok=True)
        (people_save_dir_dict[pid] / 'images').mkdir(exist_ok=True)

        people_json_dict[pid] = dict()

    
    # Third do rendering
    fids = sorted(list(img_dict.keys()))

    overview = dict()
    for pid in people_dict.keys():
        overview[f"pid:{pid:03}"] = []


    sorted_pids_dict = dict()
    for fid in tqdm(fids, desc="Rendering single camera"):
        input_image = img_dict[fid]

        # render images
        render_meshes = dict()
        project_jnts = dict()
        gt_jnts = dict()
        gt_bboxs = dict()
        # get rendering
        for pid, person_dict in people_dict.items():
            if fid in person_dict:
                frame_info = person_dict[fid]

                gender = smpl_genders[pid-1]
                smpl_model = smpl_models[gender]

                                
                smpl_params = torch.from_numpy(frame_info['smpl_param']).to(smpl_model.smpl.faces_tensor.device).float()
                smpl_output = smpl_model(smpl_params)
                smpl_vert = smpl_output['smpl_verts'].data.cpu().numpy().squeeze() 
                smpl_jnts = smpl_output['smpl_jnts'].data.cpu().numpy().squeeze()
                smpl_jnts = smpl_jnts[SMPL2COCO19]      # Here converted to COCO joints, to avoid joints such as top of head

                R = frame_info['camera']['rotation'].reshape(3,3)
                T = frame_info['camera']['translation'].reshape(-1)
                K = frame_info['camera']['intrinsic'].reshape(3,3).astype(np.float32)
                img_size = [frame_info['camera']['height'], frame_info['camera']['width']]

                smpl_vert = np.einsum('ij,nj->ni', R, smpl_vert) + np.expand_dims(T, axis=0)
                smpl_mesh = trimesh.Trimesh(smpl_vert, smpl_model.smpl.faces, process=False)

                smpl_jnts = np.einsum('ij,nj->ni', R, smpl_jnts) + np.expand_dims(T, axis=0)
                smpl_jnts = np.einsum('ij,nj->ni', K, smpl_jnts)
                smpl_jnts = (smpl_jnts[:, :2] / smpl_jnts[:, 2:3]).astype(np.int32) # map into int space
                project_jnts[pid] = smpl_jnts
                
                gt_jnt = []
                for _jnt in frame_info['gt_body_pose']:
                    if _jnt is not None:
                        gt_jnt.append(_jnt)
                if len(gt_jnt) > 0:
                    gt_jnts[pid] = np.stack(gt_jnt, axis=0)
                gt_bboxs[pid] = frame_info['gt_bbox']
                render_meshes[pid] = smpl_mesh

        if len(render_meshes) == 0:
            continue
        R = torch.diag(torch.tensor([1, 1, 1]))[None].float()
        T = torch.zeros(3)[None].float() 
        renderer = Renderer(img_size=img_size, cam_intrinsic=K)

        renderer.set_camera(R,T,K)
        rendered_res, sorted_pids = renderer.render_mesh_recon(render_meshes)
        mask_ids = rendered_res.squeeze().detach().cpu().numpy()

        # Load mask of image
        if use_person_mask:
            person_mask = cv2.imread(person_fname_dict[fid], -1)
            if len(person_mask.shape) == 3:
                person_mask = person_mask.sum(-1) > 0
            
            


        # check smpl_jnts exists or not.
        visible_dict = dict()
        for pid, pj_jnts in project_jnts.items():
            # Firstly, Check with gt joints first
            if pid in gt_jnts:
                jnts = gt_jnts[pid]
                valid_jnts_mask = (jnts[:, 0] >= 0) * (jnts[:, 0] < frame_info['camera']['width']) \
                                    * (jnts[:, 1] >= 0) * (jnts[:, 1] < frame_info['camera']['height'])

                if valid_jnts_mask.sum() > 0:
                    valid_jnts = jnts[valid_jnts_mask]

                    # check whether occluded or not (by checking projected in proper rendered SMPL or not)
                    visible_jnts = []
                    for _jnt in valid_jnts:
                        if mask_ids[int(_jnt[1]), int(_jnt[0])] == int(pid):
                            if use_person_mask:
                                if person_mask[int(_jnt[1]), int(_jnt[0])] > 0:
                                    visible_jnts.append(_jnt)
                            else:
                                visible_jnts.append(_jnt)

                    # if (# of vsibile_jnts) > 2, determine as visible (only for gt joitns)
                    if len(visible_jnts) >= 2:
                        visible_dict[pid] = visible_jnts
            
            # Secondly, check with SMPL joints 
            if pid not in visible_dict:
            # Check with gt joints first
                jnts = pj_jnts
                valid_jnts_mask = (jnts[:, 0] >= 0) * (jnts[:, 0] < frame_info['camera']['width']) \
                                    * (jnts[:, 1] >= 0) * (jnts[:, 1] < frame_info['camera']['height'])

                if valid_jnts_mask.sum() > 0:
                    valid_jnts = jnts[valid_jnts_mask]

                    # check whether occluded or not
                    visible_jnts = []
                    for _jnt in valid_jnts:
                        if mask_ids[_jnt[1], _jnt[0]] == int(pid):
                            if use_person_mask:
                                if person_mask[int(_jnt[1]), int(_jnt[0])] > 0:
                                    visible_jnts.append(_jnt)
                            else:
                                visible_jnts.append(_jnt)


                    # if vsibile_jnts > 3, determine as visible
                    if len(visible_jnts) >= N_VIZ_JNTS_THRS:
                        visible_dict[pid] = visible_jnts


        # Now, extract SAM mask from visible jnts
        # Also, save 
        accum_mask = np.zeros_like(input_image[...,0])
        sorted_pids_dict[fid] = sorted_pids
        for pid in sorted_pids:
            if pid in visible_dict:
                img_bbox = gt_bboxs[pid]
                img_bbox = [int(__i) for __i in img_bbox]

                visible_jnt = []
                for v_jnt in visible_dict[pid]:
                    v_jnt[0] -= img_bbox[0]
                    v_jnt[1] -= img_bbox[1]
                    visible_jnt.append(v_jnt)
                visible_jnt = np.array(visible_jnt)

                cropped_img = input_image[img_bbox[1]:img_bbox[1]+img_bbox[3], img_bbox[0]:img_bbox[0]+img_bbox[2]]
                cropped_sam_mask = sam_masking(cropped_img, visible_jnt, sam_predictor)
                sam_mask = np.zeros_like(input_image[...,0])
                sam_mask[img_bbox[1]:img_bbox[1]+img_bbox[3], img_bbox[0]:img_bbox[0]+img_bbox[2]] = cropped_sam_mask
                
                if use_person_mask:
                    # Use Mask only valid regions (we prefer false negative rather than false positive)
                    dilated_person_mask = dilate_mask(person_mask, kernel_size=5)
                    sam_mask[dilated_person_mask==0] *= 0
                    

                # filter out front mask
                sam_mask[accum_mask > 0] *= 0
                accum_mask += sam_mask
                
                cv2.imwrite(str(people_save_dir_dict[pid] / 'masks' / f"{fid:09}.png"), sam_mask)
                if not (people_save_dir_dict[pid] / 'images' / f"{fid:09}.png").exists():
                    pass
                    # os.symlink((shared_img_dir / img_fname_dict[fid]).resolve(), (people_save_dir_dict[pid] / 'images' / f"{fid:09}.png").resolve())

                # add data for saving
                person_dict = people_dict[pid]
                frame_info = person_dict[fid]

                smpl_param = frame_info['smpl_param'].reshape(-1)
                smpl_scale = smpl_param[0]

                smpl_transl = smpl_param[1:4]
                smpl_pose = smpl_param[4:76]
                smpl_beta = smpl_param[-10:]

                R = frame_info['camera']['rotation'].reshape(3,3)
                T = frame_info['camera']['translation'].reshape(-1)

                # We need to cancel out smpl_transl & smpl_scale here.
                new_T = (T + smpl_transl) / smpl_scale
                new_extrinsic = np.concatenate([R, new_T.reshape(1, 3)], axis=0)    # (4,3)
                assert new_extrinsic.shape[0] == 4

                people_json_dict[pid][f"{fid:09}"] = dict(
                    poses = smpl_pose.tolist(),
                    betas = smpl_beta.tolist(),
                    cam_intrinsics = frame_info['camera']['intrinsic'].reshape(3,3).astype(np.float32).tolist(),
                    cam_extrinsics = new_extrinsic.tolist()
                )

    # save sorted_pids just in case
    write_pickle(save_dir / 'sorted_pids.pkl', sorted_pids_dict)

    # After finish getting all mask, save dict
    for pid, save_dict in people_json_dict.items():
        save_fname = people_save_dir_dict[pid] / 'metadata.json'
        with open(save_fname, 'w') as f:
            json.dump(save_dict, f)

        



def convert_to_humannerf(data_dir: Path, save_dir: Path, sam_predictor, use_person_mask: bool=False):
    """
    convert to humannerf

    ### FILE structure
    exp_dir
    ├── p001
        ├── images
        │   └── ${item_id}.png
        ├── masks
        │   └── ${item_id}.png
        └── metadata.json

    ##### metadata.json structure
    {
        // Replace the string item_id with your file name of video frame.
        "item_id": {
                // A (72,) array: SMPL coefficients controlling body pose.
                "poses": [
                    -3.1341, ..., 1.2532
                ],
                // A (10,) array: SMPL coefficients controlling body shape. 
                "betas": [
                    0.33019, ..., 1.0386
                ],
                // A 3x3 camera intrinsic matrix.
                "cam_intrinsics": [
                    [23043.9, 0.0,940.19],
                    [0.0, 23043.9, 539.23],
                    [0.0, 0.0, 1.0]
                ],
                // A 4x4 camera extrinsic matrix.
                "cam_extrinsics": [
                    [1.0, 0.0, 0.0, -0.005],
                    [0.0, 1.0, 0.0, 0.2218],
                    [0.0, 0.0, 1.0, 47.504],
                    [0.0, 0.0, 0.0, 1.0],
                ],
        }
    }


    ### OpenCV camera model (assume PINHOLE)
    - cam_extrinisic: w2c 
    - cam_intrinsic: image space
    """
    device = torch.device("cuda:0")
    smpl_model_neutral = SMPLServer(gender='neutral').to(device)

    smpl_models = dict(
        neutral=smpl_model_neutral,
    )
    smpl_genders = ['neutral' for _ in range(1000)]

    # Convert to handle camera-IDs
    if (data_dir / "0").exists() and len(list(data_dir.glob("*")))==1:
        cam_dir = data_dir / "0" 
    else:
        # move directory under camera name 
        dest_dir = data_dir / "0"
        dest_dir.mkdir(exist_ok=True)

        for comp in data_dir.glob("*"):
            if str(comp) != str(dest_dir):
                # print(comp)
                try:
                    shutil.move(comp, dest_dir / comp.name)
                except:
                    # When permission issue can't be solved
                    copy_tree(str(comp), str(dest_dir / comp.name))
                    # shutil.copy(comp, dest_dir / comp.name)
        cam_dir = data_dir / "0"


    save_dir.mkdir(exist_ok=True)
    save_dir = save_dir / "0"
    save_dir.mkdir(exist_ok=True)

    # Now process each datasets
    # Here, to determine occlusion, we render SMPLs, get masks of each subject, to determine visible keypoints!
    # Let's render all images with mesh renderer
    process_single_cam(cam_dir=cam_dir, save_dir=save_dir, smpl_models=smpl_models, smpl_genders=smpl_genders, sam_predictor=sam_predictor, use_person_mask=use_person_mask)

    with open(save_dir / 'done.txt', 'w') as f:
        f.write(" ")
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessing MV data")
    parser.add_argument('--data_dir', type=str, default=None, help='overall data dir')
    parser.add_argument('--save_dir', type=str, default=None, help="path of processed data")
    parser.add_argument('--sam_checkpoint_path', type=str, default=None, help="get checkpoint dir")
    parser.add_argument('--use_person_mask', action='store_true', help="use person_mask as occlusion masks")
    parser.add_argument('--use_sam_hq', action='store_true', help="use SAM_HQ instead of SAM")
    args = parser.parse_args() 

    SAM_CHECKPOINT = args.sam_checkpoint_path

    if args.use_sam_hq:
        print("Initialize SAM-HQ Predictor")
        from segment_anythig import build_sam_hq
        sam_predictor = SamPredictor(build_sam_hq(checkpoint=args.sam_checkpoint_path).to("cuda"))
    else:
        sam_predictor = SamPredictor(build_sam(checkpoint=args.sam_checkpoint_path).to("cuda"))


    convert_to_humannerf(Path(args.data_dir), Path(args.save_dir), sam_predictor, use_person_mask=args.use_person_mask)