"""
get dataset general informations

"""

import os
from typing import OrderedDict, List
import random

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torch

from Utils.gutils import isEmpty
from Utils.gutils import Paths, save_as_pickle, load_pickle






def unfiy_frame_size(root_path:str):
    
    cam_names = os.listdir(root_path)
    for cam_name in cam_names:
        cam_path = os.path.join(root_path, cam_name)
        cam_video_names = os.listdir(cam_path)
        for video_name in cam_video_names:
            video_path = os.path.join(cam_path, video_name)
            video_frame_names = [f for f in os.listdir(video_path) if f.endswith(".png")]
            frame0_path = os.path.join(video_path, video_frame_names[0])
            frame0 = Image.open(frame0_path)
            w, h = frame0.size
            framesize = (h, w)
            print(f"{cam_name}: {video_name}: {framesize}")
        
        print("==="*40)




def cvt2Intensity(img:Image):
    if img.mode != "RGB":
        img = img.convert("RGB")

    img_np = np.asarray(img).astype(np.float32)
    img_y = (0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2])/255.0
    return img_y




def get_cam_samples(cam_path, sample_per_patch:int):
    """docs"""
    cam_videos = os.listdir(cam_path)
    



def main():
    """
    docs
    """
    root_path = "/mnt/exthd/Dataset/frames"
    paths = Paths()
    
    unfiy_frame_size(root_path=root_path)
    



    











if __name__ == "__main__":
    main()