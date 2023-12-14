"""
model for camera noiseprint
"""

import os
from typing import List

import torch
from torch import nn


class ConstConv(nn.Module):
    """
    const conv
    """
    def __init__(self, inch:int, outch:int, ks:int, stride:int, dev, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dev = dev
        self.ks = ks
        self.inch = inch
        self.conv0 = nn.Conv2d(in_channels=inch, out_channels=outch, kernel_size=ks, stride=stride, padding=0, bias=False)
        self.padding = nn.ZeroPad2d(1)
        self.bn = nn.BatchNorm2d(num_features=outch)
        self.act = nn.ReLU()
        self.data = self._get_data()

    def forward(self, x):
        if self.training:
            out0 = self.conv0(self.data['zero'])
            out1 = self.conv0(self.data['one'])
            out = self.padding(x)
            out = self.conv0(out)
            out = self.act(out)
            return dict(out=out, out1=out1, out0=out0)
        
        x = self.padding(x)
        x = self.conv0(x)
        x = self.act(x)
        return dict(out=x)

    

    def _get_data(self):
        ones = torch.ones(size=(self.inch, self.ks, self.ks), dtype=torch.float32)
        ones[:, self.ks//2, self.ks//2] = 0.0
        zeros = torch.zeros_like(ones)
        zeros[:, self.ks//2, self.ks//2] = 1.0

        return dict(one=ones.unsqueeze(dim=0).to(self.dev), zero=zeros.unsqueeze(dim=0).to(self.dev))

    






class Noise_Print(nn.Module):
    """
    noiseprint module
    """

    def __init__(self, input_shape:List, num_layers:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers

        self.blk0 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[1], out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.mid_blk = self._mid_blk()

        self.head = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU6(inplace=True)
        )


    def _mid_blk(self):
        layers = []
        for i in range(self.num_layers):
            layer = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(num_features=64, momentum=0.9, affine=True, eps=1e-4), nn.ReLU(inplace=True)
            )
            layers.append(layer)
        
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.blk0(x)
        x = self.mid_blk(x)
        out = self.head(x)

        return out






class Noise_PrintConst(nn.Module):
    """
    noiseprint module
    """

    def __init__(self, input_shape:List, num_layers:int, dev,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers

        # self.blk0 = nn.Sequential(
        #     nn.Conv2d(in_channels=input_shape[1], out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.ReLU(inplace=True)
        # )

        self.blk0 = ConstConv(inch=3, outch=64, ks=3, stride=1, dev=dev)

        self.mid_blk = self._mid_blk()

        self.head = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU6(inplace=True)
        )


    def _mid_blk(self):
        layers = []
        for i in range(self.num_layers):
            layer = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(num_features=64, momentum=0.9, affine=True, eps=1e-4), nn.ReLU(inplace=True)
            )
            layers.append(layer)
        
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            out_dict = self.blk0(x)
            x = self.mid_blk(out_dict['out'])
            x = self.mid_blk(x)
            out = self.head(x)
            return dict(out=out, out1=out_dict['out1'], out0=out_dict['out0'])
        else:
            out_dict = self.blk0(x)
            x = self.mid_blk(out_dict['out'])
            x = self.mid_blk(x)
            out = self.head(x)
            return dict(out=out)






def main():
    """
    
    """

    x = torch.randn(size=(1, 3, 64, 64))

    model = Noise_Print(input_shape=[1,3,64,64], num_layers=15)

    cont_layer = ConstConv(inch=3, outch=3, ks=3, stride=1)

    cont_layer.eval()
    out = cont_layer(x)
    print(out)




if __name__ == "__main__":
    
    main()