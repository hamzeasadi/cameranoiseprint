import os
from os.path import expanduser
import random
import json
from typing import Optional
import torch
import numpy as np
from matplotlib import pyplot as plt

import lmdb
import pickle

from Base_lmdb_entery import Image_entery
import Base_lmdb_entery as ble
from torch.utils.data import Dataset, DataLoader
from Module.Utils import basic_utils


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def all_pathces():
    patches = dict()
    patch_id = 0
    for i in range(11):
        for j in range(20):
            h = i*64
            w = j*64
            patches[patch_id] = (h, w)
            patch_id += 1
    return patches

def get_patch(img, idx, hc:Optional[int]=None, wc:Optional[int]=None, rnd:bool=True):

    if rnd:
        hc = np.random.randint(low=32, high=720-32)
        wc = np.random.randint(low=32, high=1280-32)
        crop = img[hc-32:hc+32, wc-32:wc+32, :]
    else:
        crop = img[hc-32:hc+32, wc-32:wc+32, :]

    return crop/255
        





class Iframe720X1280(Dataset):
    """
    doc
    """
    def __init__(self, database_path, database_name) -> None:
        super().__init__()
        # self.batch_size = batch_size
        # self.epch = epoch
        # self.num_cams = num_cams
        self.dbpath = database_path
        self.dbname = database_name
        # with open(os.path.join(database_path, 'meta_data.json')) as json_file:
        #     self.meta = json.load(json_file)

        # cam_names = list(self.meta.keys())
        # self.sub_cams = random.sample(cam_names, num_cams)
        # self.frprcam = batch_size//num_cams


    def __len__(self):
        with lmdb.open(os.path.join(self.dbpath, self.dbname), readonly=True) as conn:
            with conn.begin() as txn:
                stat = txn.stat()
                return stat['entries']
    
    def __getitem__(self, index):
        key_id = f"{index:08}".encode(encoding='utf-8')
        image, label  = ble.get_lmdb_entery(database_path=self.dbpath, database_name=self.dbname, image_id=key_id)
        imgt = torch.from_numpy(image/255).permute(2, 0, 1).float()
        return imgt, torch.tensor(label)
        # X = torch.randn(size=(self.num_cams*self.frprcam, 3, 64, 64))
        # Y = torch.ones(size=(self.num_cams*self.frprcam, 1))
        # img_ptn = 0
        # for cam_name in self.sub_cams:
        #     frame_ids = np.random.randint(low=0, high=self.meta[cam_name]['num_iframe'], size=self.frprcam)
        #     key_ids = [f"{cam_name}_{fid:08}".encode(encoding='utf-8') for fid in frame_ids]
        #     for key_id in key_ids:
        #         image, label  = ble.get_lmdb_entery(database_path=self.dbpath, database_name=self.dbname, image_id=key_id)
        #         patch = get_patch(img=np.copy(image))
        #         patcht = torch.from_numpy(patch).permute(2, 0, 1).float()
        #         X[img_ptn] = patcht
        #         Y[img_ptn] = torch.tensor(label)
        #         img_ptn+=1

        # return X.float().to(dev), Y.float().to(dev)
    


def create_loader(num_cams, batch_size):
    home = expanduser('~')
    dbpath = os.path.join(home, 'project', 'Datasets', 'dataset')
    dbname = 'lmdb_720x1280'
    db = Iframe720X1280(database_path=dbpath, database_name=dbname, batch_size=batch_size, num_cams=num_cams, epoch=1)
    db_loader = DataLoader(db, batch_size=1)
    return db_loader


def create_loader64(batch_size):
    home = expanduser('~')
    dbpath = os.path.join(home, 'project', 'Datasets', 'dataset64')
    dbname = 'lmdb_64x64'
    db = Iframe720X1280(database_path=dbpath, database_name=dbname)
    db_loader = DataLoader(db, batch_size=batch_size, shuffle=True, pin_memory=True)
    return db_loader








if __name__ == '__main__':
    print(42)
    databasepath = '/home/hasadi/project/cameranoiseprint/dataset'
    databasename = 'lmdb_720x1280'
    # dataset = Iframe720X1280(database_path=databasepath, database_name=databasename, batch_size=40, num_cams=10, epoch=1)
    # len(dataset)
    databasepath64 = '/home/hasadi/project/cameranoiseprint/dataset64'
    databasename64 = 'lmdb_64x64'
    db64 = os.path.join(databasepath64, databasename64)

    dataset = Iframe720X1280(database_path=databasepath64, database_name=databasename64)

    img, label = dataset[0]
    print(img.shape, label)

    # with open(os.path.join(databasepath, 'meta_data.json')) as json_file:
    #     meta = json.load(json_file)
    #     print(len(list(meta.keys())))

    


















































    # with open(os.path.join(databasepath, 'meta_data.json')) as json_file:
    #     meta = json.load(json_file)

    # connw = lmdb.open(db64, map_size=1e+12)
    # with connw.begin(write=True) as txnw:
    #     ky_cnt = 0
    #     for ky, vl in meta.items():
    #         h, w, c = vl['iframe_shape']
    #         num_h = h//64
    #         num_w = w//64

    #         lbl = vl['label']
    #         for i in range(vl['num_iframe']):
    #             acc_ky = f"{ky}_{i:08}".encode(encoding='utf-8')
    #             image, label  = ble.get_lmdb_entery(database_path=databasepath, database_name=databasename, image_id=acc_ky)
    #             for hidx in range(num_h):
    #                 hi = hidx*64
    #                 for widx in range(num_w):
    #                     wi = widx*64
    #                     patch = image[hi:hi+64, wi:wi+64, :]
    #                     key_id = f"{ky_cnt:08}".encode(encoding='utf-8')
    #                     key_value = Image_entery(image=patch, label=label)
    #                     txnw.put(key_id, pickle.dumps(key_value))
    #                     ky_cnt += 1

    # connw.close()

    
