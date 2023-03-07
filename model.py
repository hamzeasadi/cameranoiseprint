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
            nn.ReLU(inplace=True)
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
        return out



class Disc(nn.Module):
    def __init__(self, inch) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=inch, out_channels=32, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(0.25), nn.BatchNorm2d(64, momentum=0.8),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(0.25), nn.BatchNorm2d(128, momentum=0.8),
            nn.Conv2d(in_channels=128, out_channels=256 , kernel_size=3, stride=2), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(0.25), nn.BatchNorm2d(256, momentum=0.8),
            # nn.Conv2d(in_channels=256, out_channels=512 , kernel_size=3, stride=1)
            nn.AvgPool2d(kernel_size=3),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=1)
            
        )

    def forward(self, x):
        return self.net(x)


def main():
    inch = 3
    dp = 15
    x = torch.randn(size=(10, inch, 64, 64))
    disc = Disc(inch=3)
    out = disc(x)
    print(out.shape)



if __name__ == '__main__':
    main()
