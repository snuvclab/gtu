"""Video Downloader"""
import os
from pathlib import Path
from datetime import datetime

import argparse
import ray
import numpy as np
from ray.util.multiprocessing import Pool


def do_system(arg):
    print(f"==== running: {arg}")
    err = True
    while (err):
        err = os.system(arg)
        if err:
            print("download: command failed, retry")

    return 0


DB_PATH = '/mnt/hdd/experiments/gdna/pifu_gdna'
DL_LIST_PATH = 'data_list/youtube_dl/youtube_link.txt'
NUM_PROCESS = 8


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default=DB_PATH,
                        help='directory where to store downloaded videos')
    parser.add_argument('--dl_list_path', type=str, default=DL_LIST_PATH,
                        help='path of .txt file, holding urls to download')
    parser.add_argument('--n_process', type=int, default=8,
                        help='# of process. (default=8)')
    parser.add_argument('--fname_tag', type=str, default="",
                        help='save video file name, appended to the .mp4 file')
    return parser.parse_args()

def main():
    args = get_opts()
    
    res_path = Path(args.save_path)
    res_path.mkdir(exist_ok=True)
    
    with open(args.dl_list_path, 'r') as f:
        links = f.read().splitlines()
        
    # remove empty lines
    links = [ l for l in links if l!=""]
    print("[init] processing url list")
    for l in links:
        print(l)
    print("==========================")
    
    queries = []
    tag = datetime.today().strftime('%Y%m%d%H')
    if args.fname_tag != "":
        tag = tag + "_" + str(args.fname_tag)
    
    with Pool(processes=args.n_process) as pool:
        for i, link in enumerate(links):
            queries.append(dict(
                link = link,
                res_path = res_path / (f"yt_{tag}_{i}")
            ))
        res = pool.map(download_videos, queries)



def download_videos(input_arg):
    link = input_arg['link']
    res_path = input_arg['res_path']
    cmd = f"yt-dlp --check-formats -R infinite -o {str(res_path)} '{link}'"
    print(f"start downloading {res_path.name}")
    do_system(cmd)
    return 0


if __name__ == '__main__':
    ray.init(address="local", num_cpus=NUM_PROCESS)
    main()