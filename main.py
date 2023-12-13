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
from Model.noiseprint_model import Noise_Print, RDisc
from Loss.lossfunction import NP_Loss
from Engine.engine import train_step_np, train_step_gan




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
    gen = Noise_Print(input_shape=[1, 3, 64, 64], num_layers=15)
    disc = RDisc(input_shape=[1,1,64,64])
    if config['ckp_num'] is not None:
        ckp_num = config['ckp_num']
        ckp_name = f"ckpoint_{ckp_num}.pt"
        model_path = os.path.join(paths.model, ckp_name)
        state = torch.load(model_path, map_location=torch.device("cpu"))
        gen.load_state_dict(state['model'])
        
    gen.to(dev)
    disc.to(dev)
    gen_crt = NP_Loss(lamda=config['lamda'], scale=config['scale'])
    disc_crt = nn.BCEWithLogitsLoss()

    loaders = create_loaders(dataset_name="64x64s")
    gen_opt = Adam(params=gen.parameters(), lr=config['gen_lr'], weight_decay=0.0005)
    disc_opt = Adam(params=disc.parameters(), lr=config['dis_lr'], weight_decay=0.0005)

    if config['gen_sch'] is not None:
        gen_scheduler = ExponentialLR(gen_opt, gamma=config['gen_gamma'])
    else:
        gen_scheduler = None
    

    if config['dis_sch'] is not None:
        disc_scheduler = ExponentialLR(gen_opt, gamma=config['dis_gamma'])
    else:
        disc_scheduler = None
    
    for epoch in range(args.epochs):
        out_state = train_step_gan(
                        gen=gen, disc=disc, loader=loaders, opt_disc=disc_opt, opt_gen=gen_opt, epoch=epoch,
                        dev=dev, gen_crt=gen_crt, disc_crt=disc_crt, gen_schedure=gen_scheduler,
                        disc_scheduler=disc_scheduler)
        
        print(out_state)





if __name__ == "__main__":

    main()