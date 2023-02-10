import os
import torch
import conf as cfg
from torch import nn as nn
from torch.optim import Optimizer




dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class KeepTrack():
    def __init__(self, path) -> None:
        self.path = path
        self.state = dict(model="", opt="", epoch=1, trainloss=0.1, valloss=0.1)

    def save_ckp(self, model: nn.Module, opt: Optimizer, epoch, fname: str, trainloss=0.1, valloss=0.1):
        self.state['model'] = model.state_dict()
        self.state['opt'] = opt.state_dict()
        self.state['epoch'] = epoch
        self.state['trainloss'] = trainloss
        self.state['valloss'] = valloss
        save_path = os.path.join(self.path, fname)
        torch.save(obj=self.state, f=save_path)

    def load_ckp(self, fname):
        state = torch.load(os.path.join(self.path, fname), map_location=dev)
        return state


def main():
    epochs=100
    M1 = 10000
    M2 = 10000
    for i in range(epochs):
        m1 = max(10, M1-i*200)
        m2 = max(20, M2-i*100)
        print(m1, m2)



if __name__ == '__main__':
    main()