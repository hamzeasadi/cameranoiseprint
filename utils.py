import os
import torch
from itertools import combinations
import random
import conf as cfg
from torch import nn as nn
from torch.optim import Optimizer
import numpy as np



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

def get_pairs(batch_size, frprcam, ratio=2):
    pair_list = list(combinations(list(range(batch_size)), r=2))
    pos_pairs = []
    for i in range(0, batch_size-1, frprcam):
        sub_pos_pair = list(combinations(list(range(i, i+frprcam)), r=2))
        for pair in sub_pos_pair:
            pos_pairs.append(pair)
    indexs = []
    for pair in pair_list:
        if pair in pos_pairs:
            indexs.append([pair[0], pair[1], 0])
        else:
            indexs.append([pair[0], pair[1], 1])

    random.shuffle(indexs)
    indexs_np = np.array(indexs)
    index_1, index_2, labels = indexs_np[:, 0], indexs_np[:, 1], indexs_np[:, 2]

    weights = labels.copy()
    for i, elm in enumerate(weights):
        if elm == 0:
            weights[i] = ratio
        else:
            weights[i] = 1

    labels = torch.from_numpy(labels).view(-1, 1)
    weights = torch.from_numpy(weights).view(-1, 1)*200
    # print(weights)
    return index_1, index_2, labels.float().to(dev), weights.float().to(dev)





def main():
    epochs=100




if __name__ == '__main__':
    print(42)