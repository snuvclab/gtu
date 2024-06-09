# We will make sphere based on observations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd 


def create_mesh(center, radius, resolution=100):
    theta = np.linspace(0, 2*np.pi, resolution)
    phi = np.linspace(0, np.pi, resolution)
    theta, phi = np.meshgrid(theta, phi)

    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)

    return x, y, z

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--human_track_method', type=str, default="multiview")
    parser.add_argument('--source_path', type=str)
    parser.add_argument('--cam_name', type=str, default="0")
    args = parser.parse_args()



    human_camera_paths = []
    human_camera_paths_dict = dict()
    # Let's find human camera path
    if args.human_track_method == "alphapose":
        data_path = Path(args.source_path) / "segmentations" / "alphapose" / "indiv"
        human_candidates = data_path.glob("**/romp")
        
        for hc in human_candidates:
            n_human = len(list(hc.glob("*.npz")))
            
            if n_human > 5 and (hc.parent / "points3D.txt").exists():
                human_camera_paths.append(str(hc.parent))
                human_id = os.path.basename(str(hc.parent))
                human_camera_paths_dict[human_id] = hc.parent

    elif args.human_track_method == "phalp":
        data_path = Path(args.source_path) / "segmentations" / "phalp" / "indiv"
        if not data_path.exists():
            data_path = Path(args.source_path) / "segmentations" / "phalp_v2" / "indiv"

        human_candidates = data_path.glob("**/points3D.txt")
        
        print(data_path)
        for hc in human_candidates:
            human_camera_paths.append(str(hc.parent))
            human_id = os.path.basename(str(hc.parent))
            human_camera_paths_dict[human_id] = hc.parent

    elif args.human_track_method == "multiview":
        data_path = Path(args.source_path) / args.cam_name
        data_lists = sorted(list(data_path.glob("results_*.pkl")))
        for data_file in data_lists:
            human_camera_paths.append(str(data_file))
            human_id = data_file.name[:-4].split("results_p")[-1]
            human_camera_paths_dict[human_id] = data_file

    else:
        raise AssertionError("Wrong human_tracker method name")



    # Let's find human camera path
    overall_smpl_locs = []
    for _idx, human_id in enumerate(sorted(list(human_camera_paths_dict.keys()))):
        human_camera_path = human_camera_paths_dict[human_id]
        person_data = pd.read_pickle(human_camera_path)
        for pid, pdata in person_data.items():
            transl = pdata['smpl_param'][0,0] * pdata['smpl_param'][0,1:4]
            overall_smpl_locs.append(transl)
    overall_smpl_locs = np.stack(overall_smpl_locs, axis=0)

    # Get Max / Min value
    max_radius = overall_smpl_locs.max() if overall_smpl_locs.max() > -overall_smpl_locs.min() else -overall_smpl_locs.min()
    radius = (max_radius + 5) * 2


    # Make sphere
    sphere_center = np.zeros(3, dtype=np.float32)
    resolution = 100
    x, y, z = create_mesh(sphere_center, radius, resolution)

    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)

    xyzs = np.stack([x,y,z], axis=-1)
    rgbs = np.ones_like(xyzs, dtype=np.uint8) * np.array([[128, 128, 128]])   # mean gray color

    with open(Path(args.source_path) / f"background_points.txt", 'w') as f:
        for i, p in enumerate(xyzs):
            # first 0: id
            # 1:4 = xyz
            # 4:7 = rgb (uint8)
            # 8 = error
            xyz = p.tolist()
            rgb = rgbs[i].tolist()
            norm_dir = sphere_center - xyz 
            normal = (norm_dir / np.sqrt((norm_dir**2).sum())).tolist()
            error = 0
            f.write(f"{i} {xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]} {error} {normal[0]} {normal[1]} {normal[2]}\n")

    print("Done!")