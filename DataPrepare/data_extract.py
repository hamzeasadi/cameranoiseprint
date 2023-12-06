import os
from typing import Dict
import numpy as np
import pandas as pd
from glob import glob
import shutil

from Utils.gutils import Paths
from Utils.gutils import jsco2dict
from Utils.gutils import save_as_pickle
from DataPrepare.data_info import RawDataInfo





class ExtractData:
    """
    docs
    """

    def __init__(self, config_name:str, paths:Paths) -> None:
        self.paths = paths


    def extract_sepvideos(self, data_info:Dict):
        base_path = os.path.join(self.paths.dataset, "socraties", "videos")
        Paths.crtdir(base_path)

        for id, name in zip(data_info['camera_id'], data_info['camera_name']):
            sub_dir = os.path.join(base_path, f"{int(id)}_{name}")
            Paths.crtdir(sub_dir)

            videos_path = os.path.join(self.paths.external_dataset, f"Eurecom_{int(id)}_video*")
            video_files_path = glob(videos_path)
            for video_path in video_files_path:
                shutil.copy(src=video_path, dst=sub_dir)



    def extract_iframes(self, save_dir:str, data_info):
        base_path = os.path.join(save_dir, "socraties", "iframes")
        Paths.crtdir(base_path)

        for id, name in zip(data_info['camera_id'], data_info['camera_name']):
            sub_dir = os.path.join(base_path, f"{int(id)}")
            Paths.crtdir(sub_dir)

            videos_path = os.path.join(self.paths.external_dataset, f"Eurecom_{int(id)}_video*")
            video_files_path = glob(videos_path)
            for video_path in video_files_path:
                base_name = os.path.splitext(os.path.basename(video_path))[0].strip()
                base_name = base_name.replace(".", "_")
                base_name = base_name.replace(" ", "")
                
                subsubdir = os.path.join(sub_dir, base_name)
                Paths.crtdir(subsubdir)
                
                command = f"ffmpeg -skip_frame nokey -i {video_path} -vsync vfr {subsubdir}/img_%d.png"
                os.system(command=command)
                # break
            # break






def main():
    """docs"""
    paths = Paths()
    ds_info = ds_info = "/home/hasadi/project/dataset_camerasource/DatabaseInformation.xlsx"
    raw_data = RawDataInfo(config_name=None, paths=Paths(), dataset_info=ds_info)
    data_info = raw_data.get_info()

    save_path = "/media/hasadi/Dataset"
    # Paths.crtdir(save_path)

    # save_as_pickle(save_path=save_path, data=data_info, filename="socraties_meta.pkl")
    extract_frame = ExtractData(config_name=None, paths=Paths())

    extract_frame.extract_iframes(data_info=data_info, save_dir=save_path)






if __name__ == "__main__":
    main()