import cv2
import argparse
import os
import shutil
from pathlib import Path
from tqdm import tqdm


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Extract frames from video at specified FPS and resize factor.")
    parser.add_argument("--data_path", type=str, help="Path to the raw images.")
    parser.add_argument("--bboxes", nargs="+", type=int, help="bbox of x, y, w, h")
    args = parser.parse_args()

    # Move directories
    data_dir = Path(args.data_path)
    img_dir = data_dir / 'images'
    old_img_dir = data_dir / 'initial_raw_image'
    
    if not old_img_dir.exists():
        shutil.move(img_dir, old_img_dir)
    img_dir.mkdir(exist_ok=True)

    bbox = [int(args.bboxes[0]), int(args.bboxes[1]), int(args.bboxes[2]), int(args.bboxes[3])]

    print(f"[PREPROCESS] Cropping bbox{bbox}")

    for img_fname in tqdm(old_img_dir.glob("*.png")):
        img = cv2.imread(str(img_fname))
        img = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        save_fname = img_dir / img_fname.name
        cv2.imwrite(str(save_fname), img)

    print(f"[PREPROCESS] Successfully cropped frames in {str(img_dir)}")

if __name__ == "__main__":
    main()