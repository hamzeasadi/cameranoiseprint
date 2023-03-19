import os
import sys

from typing import Optional
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from Module.Data import iframe_720x1280_dataset as dst
from Module.Model import base_model as bsm




def train_step(gen:nn.Module, disc:nn.Module, data_loader:DataLoader, batch_size:int, num_cams:int, ratio:float,
               genopt:Optional[Optimizer]=None, discopt:Optional[Optimizer]=None):
    epoch_loss = 0.0
    gen.train()
    disc.train()
    for X, Y in data_loader:
        X = X.squeeze(dim=0)
        noise = gen(X)
        


if __name__ == '__main__':
    print()




