"""
docs

"""

import os
from typing import Tuple
from typing import List
from typing import Dict
from itertools import combinations

import numpy as np
from PIL import Image


from Utils.gutils import Paths






def cvt2Intensity(img:Image, crop_size:Tuple):
    """
    convert RGB to intensity
    args:
        img: the PIL image object
        crop_size: the size of crop:(topleft_corner_h, topleft_corner_w, bottomright_corner_h, bottomright_corner_w)
    returns:
        crop_y: an crop with specified size and only intensity channel
    """
    crop = img.crop(box=crop_size)

    if crop.mode != "RGB":
        crop = crop.convert("RGB")

    crop_np = np.asarray(crop).astype(np.float32)
    crop_y = (0.299 * crop_np[:, :, 0] + 0.587 * crop_np[:, :, 1] + 0.114 * crop_np[:, :, 2])/255.0
    return crop_y







def packIntensity(img_list:List, crop_size:Tuple, central_crop:Tuple=None):
    """
    read_images, from paths in img_list, if necesary it will central crop them and 

    and make small patch and return it
    args:
        img_list: list of image paths to be read
        crop_size: size of crop
        central_crop: if its not None the main image before patchify it will be central crop

    returns:
        crop_stack: a stack of patches
    """
    crops = []
    
    for img_path in img_list:
        img = Image.open(img_path)
        if central_crop is not None:
            h, w = img.size
            hc, wc = central_crop[0], central_crop[1]
         
            if h>=hc and w>=wc:
                topcorner_h = (h-hc)//2
                topcorner_w = (w-wc)//2
                bottomcorner_h = topcorner_h + hc
                bottomcorner_w = topcorner_w + wc
                img = img.crop(box=(topcorner_h, topcorner_w, bottomcorner_h, bottomcorner_w)) 
            else:
                raise IndexError(f"img weight and heigh f{h}, {w} are smaller than crop size {hc}, {wc}")
        crop = cvt2Intensity(img=img, crop_size=crop_size)
        crops.append(crop)
    
    return np.array(crops)







def isEmpty(dir_path:str):

    if os.path.exists(dir_path) and not os.path.isfile(dir_path):
        if len(os.listdir(dir_path)) == 0:
            return True
        else:
            return False
    
    raise NotADirectoryError






def cam_info(cam_root_path:str, cam_name:str)->Dict:
    """
    taks the class root and return genral info about the class

    args:
        class_root_path: path tp the class
    
        returns:
            genral_info: general information
            
    """

    class_root_path = os.path.join(cam_root_path, cam_name)

    video_folders = [f for f in os.listdir(class_root_path) if not isEmpty(os.path.join(class_root_path, f))]

    general_info = dict(
        order={f:[] for f in video_folders}, 
        general=[])

    iframes = []

    for video_folder in video_folders:
        
        video_folder_path = os.path.join(class_root_path, video_folder)
        video_folder_files = [f for f in os.listdir(video_folder_path) if f.endswith(".png")]
        sorted_video_files = sorted(video_folder_files)
        for iframe_name in sorted_video_files:
            iframe_path = os.path.join(video_folder_path, iframe_name)
            iframes.append(iframe_path)
            general_info["order"][video_folder].append(iframe_path)

    general_info["general"] = iframes

    return general_info










class DatasetInfo:
    """
    general dataset info
    """

    def __init__(self, dataset_name:str, paths:Paths) -> None:
        self.paths = paths
        self.dataset_name = dataset_name

    def get_dataset_iframes(self):

        dataset_root_path = os.path.join(self.paths.dataset, self.dataset_name, 'iframes')
        cam_names = [f for f in os.listdir(dataset_root_path)]
        dataset_info = dict()

        for cam_name in cam_names:
            dataset_info[cam_name] = cam_info(cam_root_path=dataset_root_path, cam_name=cam_name)
        
        return dataset_info


    def cam_iframe_info(self, cam_name:str, crop_size:Tuple=None):
        cam_path = os.path.join(self.paths.dataset, self.dataset_name,  'iframes', cam_name)
        videos = [f for f in os.listdir(cam_path) if not isEmpty(cam_path)]
        for video in videos:
            video_path = os.path.join(cam_path, video)
            video_files = [f for f in os.listdir(video_path) if f.endswith(".png")]
            iframe_path = os.path.join(video_path, video_files[0])
            img = Image.open(fp=iframe_path)
            break
        w, h = img.size

        hc, wc = crop_size

        num_h, num_w = h//hc, w//wc
        crops = dict()
        cntr = 0
        for i in range(num_h):
            hi = i*hc
            for j in range(num_w):
                wj = j*wc
                crop = [hi, wj, hi+hc, wj+wc]
                crops[cntr] = crop
                cntr+=1
        
        info = dict(h=h, w=w, crops=crops, num_crops=cntr, hc=hc, wc=wc)

        return info



    def get_order_sample(self, seq_len:int, crop_size:Tuple, min_sample_prcam:int, max_sample_prcam:int):
        dataset_root_path = os.path.join(self.paths.dataset, self.dataset_name, 'iframes')
        cam_names = [f for f in os.listdir(dataset_root_path)]
        data_info = self.get_dataset_iframes()

        all_samples = dict()

        for i, cam_name in enumerate(cam_names):
            iframe_info = self.cam_iframe_info(cam_name=cam_name, crop_size=(crop_size))
            cam_info = data_info[cam_name]
            samples = []
            sample_cnt = 0
            for video_name, video_iframes in cam_info['order'].items():
                num_iframes = len(video_iframes)
                if num_iframes>=seq_len:
                    for j in range(num_iframes - seq_len):
                        sample = (video_iframes[j:seq_len+j], i)
                        samples.append(sample)
                        sample_cnt += 1

            if len(samples)>min_sample_prcam and len(samples)<max_sample_prcam:
                all_samples[cam_name] = dict(samples=samples, iframe_info=iframe_info, num_samples=sample_cnt)
        
        return all_samples

    


    def get_random_sample(self, seq_len:int, crop_size:Tuple, min_sample_prcam:int, max_sample_prcam:int):
        dataset_root_path = os.path.join(self.paths.dataset, self.dataset_name, 'iframes')
        cam_names = [f for f in os.listdir(dataset_root_path)]
        data_info = self.get_dataset_iframes()

        all_samples = dict()

        for i, cam_name in enumerate(cam_names):
            iframe_info = self.cam_iframe_info(cam_name=cam_name, crop_size=(crop_size))
            cam_info = data_info[cam_name]

            samples = []
            sample_cnt = 0

            iframes = cam_info['general']

            num_iframes = len(iframes)
            if num_iframes>=seq_len:
                combs = combinations(iframes, r=seq_len)
                for comb in combs:
                    sample = (comb, i)
                    samples.append(sample)
                    sample_cnt += 1
                    if sample_cnt>max_sample_prcam:
                        break

            if len(samples)>min_sample_prcam:
                all_samples[cam_name] = dict(samples=samples, size=(iframe_info['h'], iframe_info['w']), num_samples=sample_cnt)
        
        return all_samples
    

    
    
            

            

    





        






def main():
    """
    docs
    """
    paths = Paths()
    dataset_name = "socraties"

    dataset_info = DatasetInfo(dataset_name=dataset_name, paths=paths)
    samples = dataset_info.get_random_sample(seq_len=3, crop_size=(64, 64), min_sample_prcam=1000, max_sample_prcam=10000)

    for cam_name, cam_sample in samples.items():
        print(f"{cam_name}: {cam_sample['num_samples']} size:{cam_sample['size']}")

    # img_path = "/home/hasadi/project/cameranoiseprint/data/dataset/socraties/iframes/141/Eurecom_141_video_005/img_2.png"
    # img = Image.open(img_path)
    # print(img.size)



if __name__ == "__main__":
    main()





