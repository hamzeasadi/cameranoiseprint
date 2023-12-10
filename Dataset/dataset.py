"""
docs

"""

import os
from typing import Tuple
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


from Utils.gutils import Paths, load_pickle





class Noiseprint_Dataset(Dataset):
    """
    noiseprint dataset
    """
    def __init__(self, paths:Paths, dataset_name:str="48x48xs") -> None:
        super().__init__()
        self.dataset_path = os.path.join(paths.dataset, dataset_name)
        self.cam_names = os.listdir(self.dataset_path)
        self.cam_info = self.get_caminfo()
        

    def get_caminfo(self):
        cam_info = dict()
        for cam_name in self.cam_names:
            cam_path = os.path.join(self.dataset_path, cam_name)
            crops = [f for f in os.listdir(cam_path) if f.startswith("crop_")]
            num_crops = len(crops)
            cam_info[cam_name] = num_crops
        return cam_info

    def get_sample(self):
        x_list = []
        y_list = []
        for cam_name, num_crops in self.cam_info.items():
            crop_idx = int(np.random.random()*10000)%num_crops
            crop_path = os.path.join(self.dataset_path, cam_name, f"crop_{crop_idx}")
            patches = [f for f in os.listdir(crop_path) if f.startswith("patch_")]
            patchs_idx = np.random.randint(low=0, high=12, size=(4,))
            for i, patch_idx in enumerate(patchs_idx):
                patch_path = os.path.join(crop_path, patches[patch_idx])
                data = load_pickle(file_path=patch_path)
                x_list.append(torch.from_numpy(data['crop']).unsqueeze(dim=0))
                y_list.append(data['label'])
        X = torch.cat(x_list, dim=0)
        y = torch.tensor(y_list)
        return X, y
        

    def __len__(self):
        return 1000
    

    def __getitem__(self, index):
        
        X, y = self.get_sample()

        return X, y
    



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
        dataset = Noiseprint_Dataset(paths=paths, dataset_name="48x48xs")
    
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader



def main():
    """
    pass
    """
    paths = Paths()
    crop_size = dict(topleftcorner_x=None, topleftcorner_y=None, h=64, w=64)

    dataset = Noiseprint_Dataset(paths=paths, dataset_name="asqar")
    
    # x, y = dataset[0]

    # print(x.shape)
    # print(y.shape)
    # print(y)

    for i in range(10):
        print(np.random.random())






if __name__ == "__main__":
    main()




