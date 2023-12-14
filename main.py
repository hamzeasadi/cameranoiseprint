"""
docs
"""


import os
import argparse

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR

from Utils.gutils import Paths
from Dataset.dataset import create_loader, Noiseprint_Dataset
from Model.noiseprint_model import Noise_Print, Noise_PrintConst
from Loss.lossfunction import NP_Loss
from Engine.engine import Engine




def main():
    """
    
    """

    paths = Paths()
  

    parser = argparse.ArgumentParser(prog=os.path.basename(__file__), description="config for training")
    parser.add_argument("--lr", type=float, required=True, default=0.01)
    parser.add_argument("--epochs", type=int, required=True, default=100)
    parser.add_argument("--gamma", type=float, required=True, default=0.9)
    parser.add_argument("--lamda", type=float, required=True, default=0.5)
    parser.add_argument("--psd", action=argparse.BooleanOptionalAction, default=True, required=True)
    parser.add_argument("--ckp_num", default=0, type=int)
    args = parser.parse_args()

    dev = torch.device("cuda")

    dataset = Noiseprint_Dataset(paths=paths)
    dataset_size = len(dataset)

    
    ckp_num = args.ckp_num
    ckp_name = f"ckpoint_{ckp_num}.pt"
    model_path = os.path.join(paths.model, ckp_name)

    # model = Noise_Print(input_shape=[1, 3, 48, 48], num_layers=15)
    model = Noise_PrintConst(input_shape=[1, 3, 48, 48], num_layers=15, dev="cuda")
    if args.ckp_num != 0:
        state = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state['model'])
        
    model.to(dev)
    criterion = NP_Loss(lamda=args.lamda)
    opt = Adam(params=model.parameters(), lr=args.lr, weight_decay=0.0001)
    # opt = SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    scheduler = ExponentialLR(opt, gamma=args.gamma)
    crt_const = nn.BCEWithLogitsLoss()
    crt_sum = nn.MSELoss()
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for i in range(dataset_size):
            X, y = dataset[i]
            out = model(X.to(dev))
            loss1 = criterion(out['out'], y.to(dev), psd_flag=args.psd, epoch=1)

            y_pos = torch.zeros_like(out['out0'], requires_grad=False)
            const_loss = crt_sum(out['out0']+out['out1'], y_pos)
            this_label = torch.ones_like(out['out0'], requires_grad=False)
            another_loss = crt_const(0.1*out['out0'], this_label)
            loss = loss1 + another_loss + const_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            print(loss.item())
        scheduler.step()
        print(f"epoch={epoch} loss={train_loss:1.4f}")
        info = dict(model=model.eval().state_dict(), loss=train_loss/dataset_size)
        
        torch.save(obj=info, f=os.path.join(paths.model, f"ckpointconst_{epoch}.pt"))






if __name__ == "__main__":

    main()