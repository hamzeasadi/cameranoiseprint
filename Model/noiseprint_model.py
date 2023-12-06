"""
model for camera noiseprint
"""

import os
from typing import List

import torch
from torch import nn


class Const_layer(nn.Module):
    """
    docs
    """



class Noise_Print(nn.Module):
    """
    noiseprint module
    """

    def __init__(self, input_shape:List, num_layers:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers

        self.blk0 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[1], out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.mid_blk = self._mid_blk()

        self.head = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
        )


    def _mid_blk(self):
        layers = []
        for i in range(self.num_layers):
            layer = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=64), nn.ReLU()
            )
            layers.append(layer)
        
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.blk0(x)
        x = self.mid_blk(x)
        out = self.head(x)

        return out







def main():
    """
    
    """

    x = torch.randn(size=(1, 3, 64, 64))

    model = Noise_Print(input_shape=[1,3,64,64], num_layers=17)

    out = model(x)
    print(model)
    print(out)
    print(out.shape)






if __name__ == "__main__":
    
    main()