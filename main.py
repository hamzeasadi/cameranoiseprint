"""
docs
"""


import os
import argparse
import json

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR

from Utils.gutils import Paths
from Dataset.dataset import create_loaders
from Model.noiseprint_model import Noise_Print
from Loss.lossfunction import NP_Loss
from Engine.engine import train_step_np




def main():
    """
    
    """

    paths = Paths()
    # region

    # x = torch.tensor([
    #                 [1, 2, 3],
    #                 [2, 1, 3],
    #                 [2, 2, 2],
    #                 [4, 5, 6],
    #                 [4, 5, 5],
    #                 [1, 3, 3]
    #             ], dtype=torch.float32)
    # y = torch.tensor([1,0,1,0,1,0], dtype=torch.long)
    # num_label = y.squeeze().size()
    # print(x.shape)
    # print(y.shape)


    # dist = torch.cdist(x1=x, x2=x, p=2)
    
    # n = num_label[0]
    # dist = dist.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)
    # print(dist)
    # soft_dist = torch.softmax(-dist, dim=1)
    # print(soft_dist)
    # for i in range(n):
    #     lbl = y[i]
    #     indices = torch.cat((y[:i], y[i+1:]), dim=0)
    #     indices_ind = indices==lbl
    #     dist_idx = soft_dist[i]
    #     print(y)
    #     print(i, y[i])
    #     print(indices)
    #     print(dist_idx)
    #     print(indices_ind)
    #     ps = dist_idx[indices_ind]
    #     print(ps)
    #     print(torch.sum(ps))
    #     print("=="*50)

    # endregion

    parser = argparse.ArgumentParser(prog=os.path.basename(__file__), description="config for training")
    parser.add_argument("--config_name", type=str, required=True)
    args = parser.parse_args()

    with open(os.path.join(paths.config, args.config_name)) as config_file:
        config = json.load(config_file)

    dev = torch.device("cuda")
    model = Noise_Print(input_shape=[1, 3, 64, 64], num_layers=15)
    if config['ckp_num'] is not None:
        ckp_num = config['ckp_num']
        ckp_name = f"ckpoint_{ckp_num}.pt"
        model_path = os.path.join(paths.model, ckp_name)
        state = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state['model'])
        
    model.to(dev)
    criterion = NP_Loss(lamda=config['lamda'], scale=config['scale'])
    loaders = create_loaders(dataset_name="64x64s")
    opt = Adam(params=model.parameters(), lr=config['lr'], weight_decay=0.0005)
    if config['sch'] is not None:
        scheduler = ExponentialLR(opt, gamma=config['gamma'])
    else:
        scheduler = None
    
    for epoch in range(args.epochs):
        out_state = train_step_np(model=model, loader=loaders, opt=opt, crt=criterion, epoch=epoch, dev=dev, scheduler=scheduler)
        print(out_state)





if __name__ == "__main__":

    main()