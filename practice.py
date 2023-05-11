import os
import sys
import torch
from torch import nn

# sys.path.append('../..')
# sys.path.append(os.getcwd())


from Module.Data.Base_lmdb_entery import Image_entery
from Module.Data import iframe_720x1280_dataset




if __name__ == '__main__':
    print(42)
    new_db_path = '/home/hasadi/project/cameranoiseprint/data'
    new_db_name = 'vision_720x1280'
    new_db = os.path.join(new_db_path, new_db_name)

    dataset = iframe_720x1280_dataset.VideoData(db_path=new_db_path, db_name=new_db_name, crop_size=(510, 510), fup=3)
    
    print(len(dataset))