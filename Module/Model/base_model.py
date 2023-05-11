import os
import sys
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
sys.path.append('../..')
sys.path.append(os.getcwd())

from Module.Data import iframe_720x1280_dataset
from matplotlib import pyplot as plt
from Module.Data import Base_lmdb_entery


class VideoSource(nn.Module):
    """
    doc
    """
    def __init__(self, inch=3, outch=3):
        super().__init__()
        self.inch = inch
        self.outch = outch
        self.resnet_18 = resnet18(weights=ResNet18_Weights)

    

    def forward(self, x):
        return self.resnet_18(x)



if __name__ == '__main__':
    print(42)
    new_db_path = '/home/hasadi/project/cameranoiseprint/data'
    new_db_name = 'vision_720x1280'
    new_db = os.path.join(new_db_path, new_db_name)
   

    model = VideoSource()
    print(model)