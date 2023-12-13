"""
docs

"""

import os
from typing import Tuple
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


from Utils.gutils import Paths, load_pickle




class Cam_Dataset(Dataset):

    def __init__(self, dataset_name:str, cam_name:str, paths:Paths) -> None:
        super().__init__()
        self.cam_path:str = os.path.join(paths.dataset, dataset_name, cam_name)
        # self.cam_path = "/mnt/exthd/Dataset/All_crops/64x64xs/104"
        self.camera_sample_info, self.dataset_size = self.get_samples()

    def get_samples(self):
        cam_samples:Dict = dict()
        sample_counter = 0
        crops = [f for f in os.listdir(self.cam_path) if f.startswith("crop_")]
        for crop_name in crops:
            crop_path = os.path.join(self.cam_path, crop_name)
            crop_imgs = [f for f in os.listdir(crop_path) if f.endswith(".pkl")]
            num_crops = len(crop_imgs)
            for i in range(0, num_crops-3, 4):
                crop0_path = os.path.join(crop_path, crop_imgs[i])
                crop1_path = os.path.join(crop_path, crop_imgs[i+1])
                crop2_path = os.path.join(crop_path, crop_imgs[i+2])
                crop3_path = os.path.join(crop_path, crop_imgs[i+3])
                cam_samples[sample_counter] = (crop0_path, crop1_path, crop2_path, crop3_path)
                sample_counter += 1
        return cam_samples, sample_counter


    def __len__(self):
        return self.dataset_size
    

    def __getitem__(self, index):
        sample_info = self.camera_sample_info[index%self.dataset_size]
        X_list, y_list = [], []
        for crop_info in sample_info:
            data = load_pickle(crop_info)
            X_list.append(torch.from_numpy(data['crop']).unsqueeze(dim=0))
            y_list.append(torch.tensor(data['label']))
        X = torch.cat(X_list, dim=0)
        return X, torch.tensor(y_list)





def custome_collate(data):
    X = data[0][0]
    Y = data[0][1]
    for i in range(1, len(data)):
        x, y = data[i][0], data[i][1]
        X = torch.cat((X, x), dim=0)
        Y = torch.cat((Y, y), dim=0)

    return X, Y





def create_loaders(dataset_name:str, paths:Paths):
    root_path = os.path.join(paths.dataset, dataset_name)
    cam_names = os.listdir(root_path)
    cam_loaders = []
    for cam_name in cam_names:
        dataset = Cam_Dataset(dataset_name=dataset_name, cam_name=cam_name, paths=paths)
        loader = DataLoader(dataset=dataset, shuffle=True, batch_size=1)
        cam_loaders.append(loader)
    
    return cam_loaders




def create_batch(batch):
    """docs"""
    X = 0.0
    y = 0.0
    for i, sbatch in enumerate(batch):
        if i==0:
            X = sbatch[0]
            y = sbatch[1]
        else:
            X = torch.cat((X, sbatch[0]), dim=1)
            y = torch.cat((y, sbatch[1]), dim=1)
    
    return X.squeeze(), y.squeeze()




# def create_loader(dataset:str=None, batch_size:int=10):
#     if dataset is None:
#         paths = Paths()
#         dataset = Noiseprint_Dataset(paths=paths, dataset_name="48x48xs")
    
#     loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     return loader



def main():
    """
    pass
    """
    paths = Paths()
    crop_size = dict(topleftcorner_x=None, topleftcorner_y=None, h=64, w=64)

    
    cam_dataset = Cam_Dataset(dataset_name="64x64xs", cam_name="104", paths=paths)
    cam_dataset1 = Cam_Dataset(dataset_name="64x64xs", cam_name="104", paths=paths)
    loader0 = DataLoader(dataset=cam_dataset, shuffle=True, batch_size=1)
    loader1 = DataLoader(dataset=cam_dataset1, shuffle=True, batch_size=1)
    loaders = [loader0, loader1]
    xx = []
    yy = []
    for batch in zip(*loaders):
        X, y = create_batch(batch=batch)
        print(X.shape)
        print(y)
        break



if __name__ == "__main__":
    main()




