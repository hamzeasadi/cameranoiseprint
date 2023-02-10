import conf as cfg
import torch
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer



def train_setp(net:nn.Module, criterion: nn.Module, datal:DataLoader, optimizer:Optimizer):
    epochloss = 0
    l = len(datal)
    net.train()
    for X in datal:
        out = net(X)
        loss = criterion(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epochloss+=loss.item()
    return epochloss/l


def val_setp(net:nn.Module, criterion: nn.Module, datal:DataLoader, optimizer:Optimizer):
    epochloss = 0
    l = len(datal)
    net.eval()
    with torch.no_grad():
        for X in datal:
            out = net(X)
            loss = criterion(out)
            epochloss+=loss.item()
    return epochloss/l





def main():
    pass



if __name__ == "__main__":
    main()