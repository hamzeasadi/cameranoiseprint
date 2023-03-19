import os
import pickle
import csv
import json
import lmdb
from PIL import Image
import numpy as np
import random



class Image_entery():
    """
    doc
    """
    def __init__(self, image, label):
        imshape = image.shape
        self.channel = imshape[2]
        self.size = imshape[:2]
        self.image = image.tobytes()
        self.label = label

    def get_image(self):
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channel)
    

def get_dir_size(dir_path):
    total = 0
    with os.scandir(dir_path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)

    return total





def store_data_lmdb(datasetpath, databasepath, database_name):

    camera_meta = dict()
    list_cams = [f for f in os.listdir(datasetpath) if not f.startswith('.')] 
    byte_size = get_dir_size(datasetpath)
    map_size = byte_size * 5
    
    env = lmdb.open(os.path.join(databasepath, database_name), map_size=map_size)
    
    with env.begin(write=True) as txn:
        for j, cam_name in enumerate(list_cams):
            cam_path = os.path.join(datasetpath, cam_name)
            # stbl_id = cam_name.strip().split("_")
            list_iframes1 = [f for f in os.listdir(cam_path) if not f.startswith('.')]

            num_iframe = min(200, len(list_iframes1))
            list_iframes = random.sample(list_iframes1, num_iframe)

            for i, iframe_name in enumerate(list_iframes):
                iframe_path = os.path.join(cam_path, iframe_name)
                iframe = np.asarray(Image.open(iframe_path), dtype=np.uint8)
                key_id = f"{cam_name}_{i:08}"
                image_label = j
                key_value = Image_entery(image=iframe, label=image_label)
                txn.put(key_id.encode(encoding='utf-8'), pickle.dumps(key_value))
            
            camera_meta[cam_name] = dict(label=j, imshape=iframe.shape, num_iframes=i+1)

    env.close()
    with open(f"{databasepath}/meta_data.json", 'w') as jsonfile:
        json.dump(camera_meta, jsonfile)
    

    


if __name__ == '__main__':
    # dir_path = '/Users/hamzeasadi/python/CameraSource'
    # dir_size = get_dir_size(dir_path)
    # print(dir_size)
    datasetpath = '/Users/hamzeasadi/python/singlevideofingerprint/data/iframes/train'
    databasepath = 'database/lmdb'
    x = dict(x=2, y=3, z='hello', d=dict(w=1, b='ced'))
    with open('myj.json', 'w') as f:
        json.dump(x, f)

