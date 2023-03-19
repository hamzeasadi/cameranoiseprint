import os
import torch
from torch import nn



class Gen(nn.Module):

    def __init__(self, inch:int=3, depth: int=15) -> None:
        super().__init__()
        self.inch = inch 
        self.depth = depth
        self.noisext = self.blks()
        
    def blks(self):
        firstlayer = nn.Sequential(nn.Conv2d(in_channels=self.inch, out_channels=64, kernel_size=3, stride=1, padding='same'), nn.ReLU())
        lastlayer = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding='same'))
        midelayers = [firstlayer]
        for i in range(self.depth):
            layer=nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'), 
                                nn.BatchNorm2d(num_features=64, momentum=0.9, eps=1e-5), nn.ReLU())
            midelayers.append(layer)
        
        midelayers.append(lastlayer)
        fullmodel = nn.Sequential(*midelayers)
        return fullmodel

    def forward(self, x):
        out = self.noisext(x)   
        return out
    


class Disc(nn.Module):
    def __init__(self, inch) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=inch, out_channels=16, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(0.2), nn.BatchNorm2d(32, momentum=0.8),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(0.2), nn.BatchNorm2d(64, momentum=0.8),
            nn.Conv2d(in_channels=64, out_channels=128 , kernel_size=3, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(0.2), nn.BatchNorm2d(128, momentum=0.8),

            nn.Flatten(),
            nn.Linear(in_features=4*4*128, out_features=1)
        )

    def forward(self, x):
        return self.net(x)
    

if __name__ == '__main__':
    print(42)
    x = torch.randn(10, 3, 64, 64)
    model = Disc(inch=3)
    out = model(x)
    print(out.shape)