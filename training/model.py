import torch
from torch import nn as nn
from torchinfo import summary

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NoisePrint(nn.Module):
    """
    doc
    """
    def __init__(self, inch, depth):
        super().__init__()
        self.inch, self.depth = inch, depth
        self.model = self.xton()


    def _firstblk(self):
        layer = nn.Sequential(
            nn.Conv2d(in_channels=self.inch, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=64, momentum=0.9, eps=1e-5)
        )
        return layer

    def _midblk(self):
        layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=64, momentum=0.9, eps=1e-5), nn.ReLU(inplace=True)
        )
        return layer

    def _finalblk(self):
        layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding='same')
        )
        return layer

    def xton(self):
        mids = [self._midblk() for i in range(self.depth)]
        m = nn.Sequential(
            self._firstblk(), *mids , self._finalblk()
        )
        return m


    def forward(self, x):
        out = self.model(x)
        return x[:,0:1] - out



def main():
    inch = 3
    dp = 15
    x = torch.randn(size=(10, inch, 64, 64))
    mm = NoisePrint(inch=inch, depth=dp)
    # out = mm(x)
    # print(out.shape)
    summary(mm)



if __name__ == '__main__':
    main()
