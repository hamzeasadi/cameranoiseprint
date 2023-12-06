"""
get dataset general informations

"""

import os
from typing import OrderedDict
import random

import numpy as np
from PIL import Image
import torch

from Utils.gutils import isEmpty
from Utils.gutils import Paths






def calc_iframe_patchs(iframe_path:str, crop_size:dict=None):
    iframe = Image.open(fp=iframe_path)
    w, h = iframe.size
    hc, wc = crop_size['h'], crop_size['w']
  
    num_h, num_w = h//hc, w//wc
    Crops = []
    for i in range(num_h):
        hi = i*crop_size["h"]
        for j in range(num_w):
            wj = j*crop_size["w"]
            crop = dict(topleftcorner_x=hi, topleftcorner_y=wj, h=hc, w=wc)
            Crops.append(crop)
    
    return Crops






class Cam_Info:
    """
    gather camera information with regard to number of
    video and iframes for each video
    """
    def __init__(self, paths:Paths, dataset_name:str='socraties') -> None:
        self.root = os.path.join(paths.dataset, dataset_name, "iframes")
        

    def get_videos_info(self, cam_name:str, crop_size:dict=None)->OrderedDict:
        """
        return camera iframes informations
        Args:
            cam_name: name of the required camera to get information for
        
        Returns:
            info: a order dictionary of dictionaries that specify the
            video name, num of video iframe and video abs path
        """
        info = OrderedDict()
        cam_data_path = os.path.join(self.root, cam_name)
        video_names = [f for f in os.listdir(cam_data_path)]
        iframe_path = os.path.join(cam_data_path, video_names[0], "img_1.png")
        crops = calc_iframe_patchs(iframe_path=iframe_path, crop_size=crop_size)
        for video_name in video_names:
            video_path = os.path.join(cam_data_path, video_name)
            if not isEmpty(video_path):
                video_files = [f for f in os.listdir(video_path) if f.endswith(".png")]
                num_iframes = len(video_files)
                info[video_name] = dict(video_path=video_path, num_iframes=num_iframes)
        

        return OrderedDict({cam_name: {'info':info, 'crops': crops}})





def get_videos_prcam(paths:Paths, num_videos:int=None, dataset_name:str='socraties', crop_size:dict=None):
    """
    from each camera in dataset it will get a subsamples of videos
    Args:
        paths: general paths structure of the project
        num_videos: number of video to be return (their information)
        dataset_name: the name of dataset that we want to extract info of
    Returns:
        info: An orderdict that spcify the each camera with Cam_Info stype information
    """
    dataset_info = OrderedDict()

    root = os.path.join(paths.dataset, dataset_name, "iframes")
    cam_info = Cam_Info(paths=paths, dataset_name=dataset_name)
    cam_names = os.listdir(root)
    for cam_name in cam_names:
        this_cam_info = cam_info.get_videos_info(cam_name=cam_name, crop_size=crop_size)
        videos_info = this_cam_info[cam_name]
        cntr = 0
        sub_videos = OrderedDict()
        for video_name, video_info in videos_info['info'].items():
            if video_info['num_iframes']>3 and cntr<num_videos:
                sub_videos[video_name] = video_info
                cntr += 1
        if cntr>=num_videos:
            dataset_info[cam_name] = {'info':sub_videos, 'crops':videos_info['crops']}
    
    empty_cams = []
    for cam_name in dataset_info:

        if not bool(dataset_info[cam_name]['info']) or len(dataset_info[cam_name]['crops'])>1000:
            empty_cams.append(cam_name)
    
    for cam_name in empty_cams:
        del dataset_info[cam_name]
    
    return dataset_info





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
    crop_size = dict(topleftcorner_x=0, topleftcorner_y=0, h=64, w=64)
    
    img_path = "/home/hasadi/project/cameranoiseprint/img_2.png"

    sampler = NoiseprintSampler(paths=paths, dataset_name="socraties", crop_size=crop_size, num_stack=3, num_videos=4)
    sample = sampler._get_cam_crop("100", idx=10)
    all_samples = sampler.get_all_samples()

    print(len(all_samples))
    print(all_samples[0])



    











if __name__ == "__main__":
    main()