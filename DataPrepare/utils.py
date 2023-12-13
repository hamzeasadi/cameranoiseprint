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
    
    print(f"{cam_path}: {len(All_Triplets)}")
    random.shuffle(All_Triplets)
    sub_samples = random.sample(All_Triplets, num_samples)
    subsize = len(sub_samples)
    for i in range(0, subsize, 4):
        sub0 = sub_samples[i]
        sub1 = sub_samples[i+1]
        sub2 = sub_samples[i+2]
        sub3 = sub_samples[i+3]
        sample_info[i//4] = (sub0, sub1, sub2, sub3)
    
    return sample_info




def extract_cam_samples(cam_path:str, cam_id:int, crop_size:List):
    """docs"""
    






def patchify(img_path, crop_size:List):
    img = Image.open(img_path)
    w, h = img.size
    img_y = cvt2Intensity(img=img)
    plt.imshow(img_y, cmap='gray')
    plt.axis("off")
    # hc, wc = crop_size
    # num_h = h//hc
    # num_w = w//wc
    # fig, axs = plt.subplots(nrows=num_w, ncols=num_h)
    # for i in range(num_h):
    #     hi = i*hc
    #     for j in range(num_w):
    #         wi = j*wc
    #         crop = img_y[hi:hi+hc, wi:wi+wc]
    #         axs[j, i].imshow(crop, cmap='gray')
    #         axs[j, i].axis("off")
    
   
    plt.savefig("patchify.png", bbox_inches='tight', pad_inches=0, transparent = True)
    plt.close()



def main():
    """
    docs
    """
    root_path = "/mnt/exthd/Dataset/frames"
    paths = Paths()
    img_base = "/mnt/exthd/Dataset/frames/100/Eurecom_100_video_001.mp4"
    img_path = os.path.join(img_base, f"img_{1:08d}.png")
    patchify(img_path=img_path, crop_size=[64, 64])
    # get_cam_samples(cam_path=os.path.join(root_path, f"{100}"), num_samples=100)

    



    











if __name__ == "__main__":
    main()