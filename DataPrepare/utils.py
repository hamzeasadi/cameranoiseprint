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






def unfiy_frame_size(cam_path:str):
    video_names = os.listdir(cam_path)
    frame_0_idx = 1
    frame_sizes = dict()
    diff_sizes = []
    count = dict()
    for video_name in video_names:
        video_frames_path = os.path.join(cam_path, video_name)
        num_frames = len([f for f in os.listdir(video_frames_path) if f.endswith(".png")])
        frame_0_path = os.path.join(video_frames_path, f"img_{frame_0_idx:08d}.png")
        frame_0 = Image.open(frame_0_path)
        w, h = frame_0.size
        fsize = (h, w)
        
        if fsize not in diff_sizes:
            diff_sizes.append(fsize)
            count[fsize] = 0
        frame_sizes[video_name] = fsize
        count[fsize] += num_frames
    # if len(diff_sizes)>1:
    # print(frame_sizes)
    # print(diff_sizes)
    print(count)




def cvt2Intensity(img:Image):
    """
    convert RGB to intensity
    args:
        img: the PIL image object
    returns:
        img_y: a y-channel intensity image
    """

    if img.mode != "RGB":
        img = img.convert("RGB")

    img_np = np.asarray(img).astype(np.float32)
    img_y = (0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2])/255.0
    return img_y



def cam_data_gen(cam_path:str, cam_id:int, crop_size:List):
    """
    docs
    """
    hc, wc = crop_size
    paths = Paths()
    base_path = "/mnt/exthd/Dataset/All_crops/64x64xs"
    video_names = os.listdir(cam_path)
    frame_count = []
    for video_name in video_names:
        video_path = os.path.join(cam_path, video_name)
        num_frames = len([f for f in os.listdir(video_path) if f.endswith(".png")])
        frame_count.append(num_frames)

    frame_count = np.array(frame_count)
    idx_sort = frame_count.argsort()
    info = dict()
    for idx in idx_sort[-4:]:
        video_name = video_names[idx]
        video_path = os.path.join(cam_path, video_name)
        frame_names = [f for f in os.listdir(video_path) if f.endswith(".png")]
        num_frames = len(frame_names)
        mid_idx = int((2/3)*num_frames)
        idx_sample = [1, mid_idx, num_frames-3]
        info[video_name] = dict(video_path=video_path, num_frames=num_frames, idx_sample=idx_sample)
    
    sample_counter = 0
    for video_name in info:
        video_path = info[video_name]['video_path']
        indexes = info[video_name]["idx_sample"]
        for index in indexes:
            frame1_path = os.path.join(video_path, f"img_{index:08d}.png")
            frame2_path = os.path.join(video_path, f"img_{index+1:08d}.png")
            frame3_path = os.path.join(video_path, f"img_{index+2:08d}.png")
            frame1 = Image.open(frame1_path)
            w, h = frame1.size
            frame2 = Image.open(frame2_path)
            frame3 = Image.open(frame3_path)
            frame1_y = cvt2Intensity(frame1)
            frame2_y = cvt2Intensity(frame2)
            frame3_y = cvt2Intensity(frame3)
            ff1 = np.expand_dims(frame1_y, axis=0)
            ff2 = np.expand_dims(frame2_y, axis=0)
            ff3 = np.expand_dims(frame3_y, axis=0)
            fff = np.concatenate((ff1, ff2, ff3), axis=0)
            crop_cntr = 0
            num_h = h//hc
            num_w = w//wc
            for h0 in range(num_h):
                hi = h0*hc
                for w0 in range(num_w):
                    wi = w0*wc
                    crop = fff[:, hi:hi+hc, wi:wi+wc]
                    sample = dict(crop=crop, label=cam_id)
                    save_path = os.path.join(base_path, f"{cam_id}", f"crop_{crop_cntr}")
                    paths.crtdir(save_path)
                    filename = f"patch_{sample_counter}.pkl"
                    save_as_pickle(save_path=save_path, data=sample, filename=filename)
                    crop_cntr += 1
                    sample_counter += 1



def get_crop(img_path:str, crop_size:dict):
    img = Image.open(img_path)
    htop = crop_size['topleftcorner_x']
    wtop = crop_size["topleftcorner_y"]
    hbtn = crop_size['h'] + htop
    wbtn = crop_size['w'] + wtop
    crop = img.crop(box=(wtop, htop, wbtn, hbtn))
    if crop.mode != "RGB":
        crop = crop.convert("RGB")

    crop_np = np.asarray(crop).astype(np.float32)
    crop_y = (0.299 * crop_np[:, :, 0] + 0.587 * crop_np[:, :, 1] + 0.114 * crop_np[:, :, 2])/255.0
    
    return crop_y


def get_stack(cam_info:dict, num_stack:int, crop_size:dict):
    num_videos = len(list(cam_info.keys()))
    batch_stack = torch.randn(size=(num_videos, num_stack, crop_size['h'], crop_size['w']))

    for j, video_name in enumerate(cam_info):
        video_path = cam_info[video_name]['video_path']
        video_crops = torch.randn(size=(num_stack, crop_size['h'], crop_size['w']))
        for i in range(num_stack):
            img_path = os.path.join(video_path, f"img_{i+1}.png")
            crop = get_crop(img_path=img_path, crop_size=crop_size)
            video_crops[i] = torch.from_numpy(crop)

        batch_stack[j] = video_crops
    return batch_stack



class NoiseprintSampler:
    """"
    generate samples for camera noiseprint
    """
    def __init__(self, paths:Paths, dataset_name:str, crop_size:dict, num_videos:int, num_stack:int) -> None:
        self.paths = paths
        self.dataset_info = get_videos_prcam(paths=paths, num_videos=num_videos, dataset_name=dataset_name, crop_size=crop_size)
        self.num_stack = num_stack
        self.all_samples = self.get_all_samples()
        random.shuffle(self.all_samples)
        

    def _get_cam_crop(self, cam_name:str, crop_size:dict):
        cam_data = self.dataset_info[cam_name]
        # cam_crops = cam_data['crops']
        # num_crops = len(cam_crops)
        # index = idx%num_crops
        cam_info = cam_data['info']
        # crop_size = cam_crops[index]
        
        sample = get_stack(cam_info=cam_info, num_stack=self.num_stack, crop_size=crop_size)

        return sample
    
    def get_all_samples(self):
        all_samples = []
        cam_map = 0
        for cam_name in self.dataset_info:
            cam_data = self.dataset_info[cam_name]
            crops = cam_data['crops']
            for crop in crops:
                all_samples.append((cam_name, crop, cam_map))
            cam_map += 1

        return all_samples
    


        



def main():
    """
    docs
    """
    # ds_info = "/home/hasadi/project/dataset_camerasource/DatabaseInformation.xlsx"
    paths = Paths()
    # came_name = 115
    cam_names = os.listdir("/mnt/exthd/Dataset/frames")
    for cam_name in cam_names:
        print(cam_name)
        cam_path = f"/mnt/exthd/Dataset/frames/{cam_name}"
        cam_data_gen(cam_path=cam_path, cam_id=int(cam_name), crop_size=[64, 64])
        # break

    # crop_path = "/mnt/exthd/Dataset/All_crops/48x48xs/215/crop_0/patch_4400.pkl"

    # data = load_pickle(file_path=crop_path)

    # crop = data['crop']
    # fig, axs = plt.subplots(nrows=1, ncols=3)
    # axs[0].imshow(crop[0], cmap='gray')
    # axs[0].axis('off')
    # axs[1].imshow(crop[1], cmap='gray')
    # axs[1].axis('off')
    # axs[2].imshow(crop[2], cmap='gray')
    # axs[2].axis('off')
    
    # fig.savefig("crops.png", bbox_inches='tight', pad_inches=0)
    # plt.close()


    











if __name__ == "__main__":
    main()