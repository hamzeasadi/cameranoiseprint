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




def main():
    """
    
    """

    paths = Paths()

    model = Noise_Print(input_shape=[1,3,64,64], num_layers=17)
    data_loader = create_loader(batch_size=40)
    opt = Adam(params=model.parameters(), lr=3e-4)
    criterion = Loss_Function()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    engine = Engine(epochs=100, dataset=data_loader, model=model, opt=opt, criterion=criterion, paths=paths, dev=dev)

    engine.run()






if __name__ == "__main__":

    main()