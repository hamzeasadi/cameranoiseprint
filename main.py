"""
docs
"""


import os
import argparse

import torch
from torch import nn
from torch.optim import Adam

from Utils.gutils import Paths
from Dataset.dataset import create_loader
from Model.noiseprint_model import Noise_Print
from Loss.lossfunction import Loss_Function
from Engine.engine import Engine
from Loss.lossfunction import ExamplePairMiner



def main():
    """
    
    """

    paths = Paths()

    x = torch.tensor([
                    [1, 2, 3],
                    [2, 1, 3],
                    [2, 2, 2],
                    [4, 5, 6],
                    [4, 5, 5],
                    [1, 3, 3]
                ], dtype=torch.float32)
    y = torch.tensor([1,0,1,0,1,0], dtype=torch.long)
    num_label = y.squeeze().size()
    print(x.shape)
    print(y.shape)


    dist = torch.cdist(x1=x, x2=x, p=2)
    
    n = num_label[0]
    dist = dist.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)
    print(dist)
    soft_dist = torch.softmax(-dist, dim=1)
    print(soft_dist)
    for i in range(n):
        lbl = y[i]
        indices = torch.cat((y[:i], y[i+1:]), dim=0)
        indices_ind = indices==lbl
        dist_idx = soft_dist[i]
        print(y)
        print(i, y[i])
        print(indices)
        print(dist_idx)
        print(indices_ind)
        ps = dist_idx[indices_ind]
        print(ps)
        print(torch.sum(ps))
        print("=="*50)

    






if __name__ == "__main__":

    main()