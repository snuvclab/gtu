



import os
import random
import torch
import cv2
import numpy as np
from tqdm import tqdm

from utils.image_utils import img_add_text


color_fname = os.path.join(os.path.dirname(__file__), '_colors.txt')
COLOR_DICT = dict()
with open(color_fname, 'r') as f:
    colors = f.readlines()

for color in colors:
    if color[0] == '_':
        color_name = color[1:-1].lower()
        COLOR_DICT[color_name] = []
    else:
        _color = color[1:-1].lower()
        if _color.split("#")[0] in ["50"]:
            continue
        if color_name == 'grey' and _color.split("#")[0] in ["100", "200", "300"]:
            continue

        rgb_color = _color.split("#")[-1]
        assert len(rgb_color) == 6, f"invalid color name {rgb_color}"
        rgb_color = (int(rgb_color[:2], 16), int(rgb_color[2:4], 16), int(rgb_color[4:], 16))
        COLOR_DICT[color_name].append(rgb_color)



def get_color(idx=0, interval=1, get_color_lists=False, theme='green'):
    # use higher value in interval, if you want to sample every n-iters
    # "https://www.materialpalette.com/colors"

    if isinstance(theme, str):
        if theme.lower() not in COLOR_DICT:
            print(f"[WARNING] invalid theme {theme.lower()}")
            rand_idx = random.randint(0, len(COLOR_DICT)-1)
            theme = list(COLOR_DICT.keys())[rand_idx]
            print(f"[WARNING] Randomly chosed following theme {theme}")
        
        # get color
        colors = COLOR_DICT[theme]
        idx = (idx * interval) % len(colors)
        rgb_color = colors[idx]
    else:
        themes = []
        for _theme in theme:
            if _theme.lower() in COLOR_DICT:
                themes.append(_theme)
        print(f"[INFO] selecting from {themes}")
        colors = []
        for _theme in themes:
            colors.extend(COLOR_DICT[_theme])
        idx = (idx * interval) % len(colors)
        rgb_color = colors[idx]

    if get_color_lists:
        return rgb_color, colors
    else:
        return rgb_color


def get_opencv_cam(c2w, scale=0.4):
    p0 = torch.tensor([0,0,0], dtype=torch.float32).to(c2w.device)
    z = 1.5
    p1 = torch.tensor([1,1,z], dtype=torch.float32).to(c2w.device) * scale
    p2 = torch.tensor([1,-1,z], dtype=torch.float32).to(c2w.device) * scale
    p3 = torch.tensor([-1,-1,z], dtype=torch.float32).to(c2w.device) * scale
    p4 = torch.tensor([-1,1,z], dtype=torch.float32).to(c2w.device) * scale

    points = torch.stack([p0, p1, p2, p3, p4], dim=0)   # (b,3)
    points = torch.cat([points, torch.ones(5,1).float().to(points.device)], dim=-1)   # (b,4)
    world_points = torch.einsum('ij,bj->bi', c2w, points)

    return world_points


def draw_cams(image, cam_points, colors, font_colors, thicknesses = 2, skip_text=False):
    """Here we copy image"""
    image = image.copy()

    edges = [
        (0,1),
        (0,2),
        (0,3),
        (0,4),
        (1,2),
        (2,3),
        (3,4),
        (4,1),
    ]

    thicknesses = [thicknesses for _ in colors] if isinstance(thicknesses, int) else thicknesses

    for cam_point, color, thickness in zip(cam_points, colors, thicknesses):
        color = (color[2], color[1], color[0])  # RGB to BGR
        # draw lines
        for edge in edges:
            start_point = (int(cam_point[edge[0]][0]), int(cam_point[edge[0]][1]))
            end_point = (int(cam_point[edge[1]][0]), int(cam_point[edge[1]][1]))
            cv2.line(image, start_point, end_point, color, thickness)

    texts = ["train view", "current test view", "other test view"]
    _font_colors = []
    for color in font_colors:
        _font_colors.append((color[2], color[1], color[0]))

    image = image.copy()
    if not skip_text:
        img_add_text(image, texts, 1.5, _font_colors)

    return image