import os
from os.path import expanduser
import sys
import random
import json
from typing import Optional, Tuple

import cv2
import lmdb
import torch
import numpy as np
from matplotlib import pyplot as plt
# sys.path.append('../..')
# sys.path.append(os.getcwd())

from Module.Data.Base_lmdb_entery import Image_entery
from Module.Data import Base_lmdb_entery
from torch.utils.data import Dataset, DataLoader
from Module.Utils import basic_utils
from Module.Data import denoise 


def crop_and_denoise(img, crop_size, denoise_method, vscale, fup):
    h, w = crop_size
    hc, wc = 720//2, 1280//2
    crop = img[hc-h//2:hc+h//2, wc-w//2:wc+w//2, :]
    crop = crop.astype(np.float32)
    denoise_crop = denoise.skiwt(img=crop, method=denoise_method, vscale=vscale)
    denoise_crop_down = cv2.resize(denoise_crop, None, fx=(1/fup), fy=(1/fup), interpolation=cv2.INTER_LINEAR)
    denoise_crop_up = cv2.resize(denoise_crop_down, None, fx=fup, fy=fup, interpolation=cv2.INTER_LINEAR)

    return crop, denoise_crop_up



class VideoData(Dataset):
    def __init__(self, db_path:str, db_name:str, crop_size:Optional[Tuple]=None, fup:int=2, denoise_method:str='vshrink', vscale:int=1):
        self.cs = crop_size
        self.db_path = db_path
        self.db_name = db_name
        self.dm = denoise_method
        self.vscale = vscale
        self.fup = fup

    def __len__(self):
        with lmdb.open(self.db_path, readonly=True) as conn:
            with conn.begin() as txn:
                stat = txn.stat()
                return stat['entries']
            
    def __getitem__(self, index):
        acc_id = f"{index:08}".encode(encoding='utf-8')
        img, label = Base_lmdb_entery.get_lmdb_entery(database_path=self.db_path, database_name=self.db_name, image_id=acc_id)
        
        patch, dnoise_patch = crop_and_denoise(img=img, crop_size=self.cs, denoise_method=self.dm, vscale=self.vscale, fup=self.fup)
        patch = patch/np.max(patch)
        dnoise_patch = dnoise_patch/np.max(dnoise_patch)
        patch_t = torch.from_numpy(patch).float().permute(2, 0, 1)
        dnoise_patch_t = torch.from_numpy(dnoise_patch).float().permute(2, 0, 1)
        return patch_t, dnoise_patch_t, torch.tensor(label)





if __name__ == '__main__':
    print(42)

    new_db_path = '/home/hasadi/project/cameranoiseprint/data'
    new_db_name = 'vision_720x1280'
    new_db = os.path.join(new_db_path, new_db_name)

    dataset = VideoData(db_path=new_db_path, db_name=new_db_name, crop_size=(510, 510), fup=3)
    img, dimg, lbl = dataset[0]
    
    layer = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1)

    k = layer(img)
    k1 = layer(dimg)

    knp = k.detach().permute(1,2,0).numpy()
    k1np = k1.detach().permute(1,2,0).numpy()

    print(k.shape)

    fig, axs = plt.subplots(nrows=1, ncols=3)

    axs[0].imshow(knp)
    axs[1].imshow(k1np)
    axs[2].imshow(knp-k1np)

    plt.show()
    

    # fig, axs = plt.subplots(nrows=1, ncols=2)
    # axs[0].imshow(img.permute(1, 2, 0).numpy(), cmap='gray')
    # axs[1].imshow(dimg.permute(1, 2, 0).numpy(), cmap='gray')


    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()


    
