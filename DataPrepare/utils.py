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




def get_cam_samples(cam_path, num_samples:int):
    """docs"""
    cam_videos = os.listdir(cam_path)
    All_Triplets = []
    sample_info = dict()
    for video_name in cam_videos:
        video_path = os.path.join(cam_path, video_name)
        video_frames = sorted([f for f in os.listdir(video_path) if f.endswith(".png")])
        num_video_frames = len(video_frames)
        for i in range(0, num_video_frames-3, 3):
            frame0_path = os.path.join(video_path, video_frames[i])
            frame1_path = os.path.join(video_path, video_frames[i+1])
            frame2_path = os.path.join(video_path, video_frames[i+2])
            sample = [frame0_path, frame1_path, frame2_path]
            All_Triplets.append(sample)
    
    num_all_samples = len(All_Triplets)
    print(f"{cam_path}: {num_all_samples}")
    random.shuffle(All_Triplets)
    num_subsample = min(num_all_samples, num_samples)
    sub_samples = random.sample(All_Triplets, num_subsample)
    subsize = len(sub_samples)
    for i in range(0, subsize-3, 4):
        sub0 = sub_samples[i]
        sub1 = sub_samples[i+1]
        sub2 = sub_samples[i+2]
        sub3 = sub_samples[i+3]
        sample_info[i//4] = (sub0, sub1, sub2, sub3)

    return sample_info




def pack3frame(frame_path_list:List):
    frames = []
    for frame_path in frame_path_list:
        frame = Image.open(frame_path)
        frame_y = cvt2Intensity(img=frame)
        frames.append(np.expand_dims(frame_y, axis=0))
    
    pack = np.concatenate(frames, axis=0)
    return pack
    



def patcing(img:np.ndarray, crop_size:List, base_path:str, label:int, sample_cntr:int):
    hc, wc = crop_size
    c, h, w = img.shape
    num_h = h//hc
    num_w = w//wc
    crop_cntr = 0
    for i in range(num_h):
        hi = i*hc
        for j in range(num_w):
            wj = j*wc
            crop = img[:, hi:hi+hc, wj:wj+wc]
            save_path = os.path.join(base_path, f"crop_{crop_cntr}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            data = dict(crop=crop, label=label)
            file_name = f"patch_{sample_cntr}.pkl"
            sample_cntr += 1
            crop_cntr += 1
            save_as_pickle(save_path=save_path, filename=file_name, data=data)

    return sample_cntr




def extract_cam_samples(cam_path:str, cam_id:int, ns:int, crop_size:List, base_save_path:str):
    """docs"""
    sample_counter = 0
    cam_save_path = os.path.join(base_save_path, f"{cam_id}")
    cam_sample_info = get_cam_samples(cam_path=cam_path, num_samples=ns)
    for smpl_idx in cam_sample_info:
        sample = cam_sample_info[smpl_idx]
        for sub_sample in sample:
            stack = pack3frame(frame_path_list=sub_sample)
            sample_counter = patcing(img=stack, crop_size=crop_size, base_path=cam_save_path, label=cam_id, sample_cntr=sample_counter)









def main():
    """
    docs
    """
    root_path = "/mnt/exthd/Dataset/frames"
    paths = Paths()
    crop_size = [64, 64]
    ns = 400
    base_save_path = "/mnt/exthd/Dataset/All_crops/64x64xs"
    camera_names = [int(f) for f in os.listdir(root_path)]
    
    for camera_name in camera_names:
        camera_path = os.path.join(root_path, f"{camera_name}")
        extract_cam_samples(cam_path=camera_path, cam_id=camera_name, ns=ns, crop_size=crop_size, base_save_path=base_save_path)
        # break
    
    



    











if __name__ == "__main__":
    main()