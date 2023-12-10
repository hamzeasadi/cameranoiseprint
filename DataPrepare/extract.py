import os
from typing import Tuple, List
import numpy as np
from PIL import Image
from distutils.dir_util import copy_tree

from Utils.gutils import Paths
from Utils.gutils import json2dict
from Utils.gutils import save_as_pickle


def get_frames(cam_path:str, paths:Paths):
    cam_name = cam_path.split("/")[-1].strip()
    # save_base = os.path.dirname(os.path.dirname(cam_path))
    save_base = "/mnt/exthd/Dataset"
    save_path = os.path.join(save_base, "frames", cam_name)
    
    for video_name in os.listdir(cam_path):
        video_path = os.path.join(cam_path, video_name)
        video_frames_save_path = os.path.join(save_path, video_name)
        paths.crtdir(video_frames_save_path)
        command = f"ffmpeg -i {video_path} -vsync vfr {video_frames_save_path}/img_%08d.png"
        # command = f'ffmpeg -i {video_path} -vf "select=not(mod(n\,2))" -vsync vfr {video_frames_save_path}/img_%08d.png'
        os.system(command=command)



def get_sim_cams(root:str, paths:Paths):
    cams = []
    cams_dict = dict()
    sub_videos_path = os.path.join(os.path.dirname(root), "sub_videos")

    for cam_name in os.listdir(root):
        cam_path = os.path.join(root, cam_name)

        base_name = "".join(cam_name.split("_")[1:])
        number_name = cam_name.split("_")[0]
        number_name_path = os.path.join(sub_videos_path, number_name)
        # paths.crtdir(number_name_path)

        if base_name in cams:
            cams_dict[base_name].append(cam_name)

        else:
            cams.append(base_name)
            cams_dict[base_name] = [cam_name]

            # copy_tree(cam_path, number_name_path)

    print(cams_dict)




class DataExtract:
    """
    docs
    """
    def __init__(self, paths:Paths) -> None:
        self.paths = paths

    def get_frames(self, dataset_name:str):
        dataset_videos_path = os.path.join(self.paths.dataset, dataset_name, "sub_videos")
        for cam_name in os.listdir(dataset_videos_path):
            cam_path = os.path.join(dataset_videos_path, cam_name)
            get_frames(cam_path=cam_path, paths=self.paths)















class Create_samples:
    """
    docs
    """
    def __init__(self, paths:Paths, crop_size:List) -> None:
        self.paths = paths
        self.crop_size = crop_size



def main():
    """docs"""
    paths = Paths()

    data_extract = DataExtract(paths=paths)
    data_extract.get_frames(dataset_name="socraties")




if __name__ == "__main__":
    main()