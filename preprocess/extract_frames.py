import cv2
import argparse
import os
from tqdm import tqdm
from pathlib import Path

def extract_frames(video_path, img_save_dir, fps, resize_factor, jpg_img_save_dir=None):
    # Capture the video from the given path
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {str(video_path)}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    fname_count = 0
    save_frame_every = int(video_fps / fps)
    print(f"[INFO] video fps: {video_fps}, sample every {save_frame_every} frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no frames to read
        
        # Save frames according to the given FPS
        if frame_count % save_frame_every == 0:
            if resize_factor != 1:
                width = int(frame.shape[1] * resize_factor)
                height = int(frame.shape[0] * resize_factor)
                dim = (width, height)
                frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            
            cv2.imwrite(str(img_save_dir / f'{fname_count:09}.png'), frame)
            if jpg_img_save_dir is not None:
                cv2.imwrite(str(jpg_img_save_dir / f'{fname_count:09}.jpg'), frame)
            fname_count += 1
        frame_count += 1
    
    cap.release()
    print(f"Extracted and saved {frame_count // save_frame_every} frames.")

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Extract frames from video at specified FPS and resize factor.")
    parser.add_argument("--video_path", type=str, help="Path to the video file.")
    parser.add_argument("--output_path", type=str, help="Path to the save image files.")
    parser.add_argument("--sample_fps", type=int, required=True, help="Frames per second to extract.")
    parser.add_argument("--resize", type=float, default=1, help="Resize factor for the frames.")
    parser.add_argument("--save_jpg_in_addition", action='store_true', help="Save JPG in for PHALP tracking.")    
    args = parser.parse_args()
    
    # Extract frames
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True)
    img_save_dir = output_path / 'images'
    img_save_dir.mkdir(exist_ok=True)

    jpg_img_save_dir = None
    if args.save_jpg_in_addition:
        jpg_img_save_dir = output_path / 'images_jpg'
        jpg_img_save_dir.mkdir(exist_ok=True)

    if (not Path(args.video_path).exists()) and args.save_jpg_in_addition:
        for img_fname in tqdm(img_save_dir.glob("*.png")):
            img = cv2.imread(str(img_fname))
            img_fname_wo_ext = img_fname.name.split(".")[0]
            cv2.imwrite(str(jpg_img_save_dir / f'{img_fname_wo_ext}.jpg'), img)
        
        print(f"[PREPROCESS] Successfully CONVERTED into JPG frames in {str(img_save_dir)}")
    else:
        extract_frames(args.video_path, img_save_dir, args.sample_fps, args.resize, jpg_img_save_dir=jpg_img_save_dir)

        print(f"[PREPROCESS] Successfully extracted frames in {str(img_save_dir)}")

if __name__ == "__main__":
    main()