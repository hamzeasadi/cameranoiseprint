"""
docs
"""

import os

import torch
from torch.optim import Adam

from Dataset.dataset import create_loader
from Model.noiseprint_model import Noise_Print
from Loss.lossfunction import Loss_Function







def main():
    """
    docs
    """

    data_loader = create_loader(batch_size=10)
    model = Noise_Print(input_shape=[1, 3, 64, 64], num_layers=17)
    crt = Loss_Function()

    opt = Adam(params=model.parameters(), lr=3e-4)

    for i in range(10):

        for X, y in data_loader:
            print(X.shape)
            print(y.shape)
            out = model(X)
            out = out.squeeze()
            outshape = out.shape
            embeddings = out.view((outshape[0], -1))
            loss = crt(embeddings, y.squeeze())
            print(loss)
            # break

        break






if __name__ == "__main__":

    main()