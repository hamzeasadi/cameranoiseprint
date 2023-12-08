"""
docs

"""

import os
from typing import Tuple
from typing import List

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from DataPrepare.data_info import NoiseprintSampler
from Utils.gutils import Paths





class Noiseprint_Dataset(Dataset):
    """
    noiseprint dataset
    """
    def __init__(self, paths:Paths, general_crop_size:dict, dataset_name:str="socraties", num_stack:int=3, num_videos:int=4) -> None:
        super().__init__()
        dataset_sampler = NoiseprintSampler(paths=paths, dataset_name=dataset_name, 
                                            crop_size=general_crop_size, num_videos=num_videos, num_stack=num_stack)
        self.samples = dataset_sampler.get_all_samples()
        self.sampler = dataset_sampler._get_cam_crop
        self.num_videos = num_videos
    
    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, index):
        sample_info = self.samples[index]
        sample_x = self.sampler(cam_name=sample_info[0], crop_size=sample_info[1])

        return sample_x, torch.ones(size=(self.num_videos, 1))*sample_info[-1]
    



def custome_collate(data):
    X = data[0][0]
    Y = data[0][1]
    for i in range(1, len(data)):
        x, y = data[i][0], data[i][1]
        X = torch.cat((X, x), dim=0)
        Y = torch.cat((Y, y), dim=0)

    return X, Y



def create_loader(dataset:str=None, batch_size:int=10):
    if dataset is None:
        paths = Paths()
        crop_size = dict(topleftcorner_x=None, topleftcorner_y=None, h=64, w=64)
        dataset = Noiseprint_Dataset(paths=paths, general_crop_size=crop_size)
    
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=custome_collate, num_workers=10)
    return loader



def main():
    """
    pass
    """
    paths = Paths()
    crop_size = dict(topleftcorner_x=None, topleftcorner_y=None, h=64, w=64)

    dataset = Noiseprint_Dataset(paths=paths, general_crop_size=crop_size)
    # print(dataset.samples) 
    
    for i in range(len(dataset)):
        pass






if __name__ == "__main__":
    main()




