import os
import pickle
import csv
import json
import lmdb
from PIL import Image
import numpy as np
import random
from matplotlib import pyplot as plt


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





def store_data_lmdb(databasepath, database_name):
    data1path = '/media/hasadi/Elements/Socraties/iframe_720x1280'
    data2path = '/media/hasadi/myDrive/Datasets/visionDataset/VISION/iframe_720x1280'
    cam_1_names = [f for f in os.listdir(data1path) if not f.startswith('.')]
    cam_2_names = [f for f in os.listdir(data2path) if not f.startswith('.')]
    cams_dict = dict()
    for cam_name in cam_1_names:
        cams_dict[cam_name] = os.path.join(data1path, cam_name)

    for cam_name in cam_2_names:
        cams_dict[cam_name] = os.path.join(data2path, cam_name)

    camera_meta = dict()

    byte_size = get_dir_size(data1path) + get_dir_size(data2path)
    map_size = byte_size * 1

    database_compelete_path = os.path.join(databasepath, database_name)
    try:
        os.makedirs(database_compelete_path)
    except Exception as e:
        print(e)

    conn = lmdb.open(database_compelete_path, map_size=map_size)
    
    with conn.begin(write=True) as txn:
        for i, (cam_name, cam_path) in enumerate(cams_dict.items()):
            cam_frames = [f for f in os.listdir(cam_path) if not f.startswith('.')]
            num_frames = min(len(cam_frames), 200)
            cam_frames_sample = random.sample(cam_frames, num_frames)

            for j, iframe_name in enumerate(cam_frames_sample):
                iframe_path = os.path.join(cam_path, iframe_name)
                iframe = np.asarray(Image.open(iframe_path), dtype=np.uint8)
                iframe_label = i
                key_id = f"{cam_name}_{j:08}".encode(encoding='utf-8')
                key_value = Image_entery(image=iframe, label=iframe_label)
                txn.put(key_id, pickle.dumps(key_value))


    #     for j, cam_name in enumerate(list_cams):
    #         cam_path = os.path.join(datasetpath, cam_name)
    #         # stbl_id = cam_name.strip().split("_")
    #         list_iframes1 = [f for f in os.listdir(cam_path) if not f.startswith('.')]

    #         num_iframe = min(200, len(list_iframes1))
    #         list_iframes = random.sample(list_iframes1, num_iframe)

    #         for i, iframe_name in enumerate(list_iframes):
    #             iframe_path = os.path.join(cam_path, iframe_name)
    #             iframe = np.asarray(Image.open(iframe_path), dtype=np.uint8)
    #             key_id = f"{cam_name}_{i:08}"
    #             image_label = j
    #             key_value = Image_entery(image=iframe, label=image_label)
    #             txn.put(key_id.encode(encoding='utf-8'), pickle.dumps(key_value))
            
    #         # camera_meta[cam_name] = dict(label=j, imshape=iframe.shape, num_iframes=i+1)

    # conn.close()
    # with open(f"{databasepath}/meta_data.json", 'w') as jsonfile:
    #     json.dump(camera_meta, jsonfile)
    

def get_sample(database_path, database_name, image_id):
    database_full_path = os.path.join(database_path, database_name)
    conn = lmdb.open(database_full_path, readonly=True)
    with conn.begin() as txn:
        data = txn.get(image_id)
        sample = pickle.loads(data)

    return sample.get_image(), sample.label



if __name__ == '__main__':
    # dir_path = '/Users/hamzeasadi/python/CameraSource'
    # dir_size = get_dir_size(dir_path)
    # print(dir_size)
    datasetpath = '/media/hasadi/Elements/Socraties/iframe_720x1280'
    databasepath = '/home/hasadi/project/cameranoiseprint/dataset'
    databasename = 'lmdb_720x1280'



    store_data_lmdb(databasepath=databasepath, database_name=databasename)

    # with open(os.path.join(databasepath, 'meta_data.json')) as json_file:
    #     meta_data = json.load(json_file)

    # cam_names = list(meta_data.keys())
    # print(meta_data[cam_names[0]])
    # img_id = f"{cam_names[10]}_{10:08}".encode(encoding="utf-8")


    # # img_id = f"Eurecom_100_{1:08}".encode(encoding='utf-8')
    # img, lbl = get_sample(database_path=databasepath, database_name=databasename, image_id=img_id)
    # print(lbl)
    # plt.imshow(img)
    # plt.show()