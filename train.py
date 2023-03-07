import conf as cfg
import model as m
import torch
from torch import nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import utils
import datasetup as dst
import argparse
import engine
import os



dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
paths = cfg.Paths()

parser = argparse.ArgumentParser(prog='train.py', description='read setting for training proceduer')

parser.add_argument('--train', action=argparse.BooleanOptionalAction)
parser.add_argument('--test', action=argparse.BooleanOptionalAction)


parser.add_argument('--epochs', type=int, default=100, required=True)
parser.add_argument('--batch_size', type=int, default=200, required=True)
parser.add_argument('--depth', type=int, default=15, required=True)
parser.add_argument('--num_cams', type=int, default=5, required=True)
parser.add_argument('--dl', type=int, default=100, required=True)
parser.add_argument('--ratio', type=float, default=1)

parser.add_argument('--modelname', type=str, required=True)

args = parser.parse_args()



def train():
    kt = utils.KeepTrack(path=paths.model)
    gen = m.NoisePrint(inch=3, depth=args.depth)
    disc = m.Disc(inch=1)
    gen.to(dev)
    disc.to(dev)
    opt = Adam(params=list(disc.parameters())+list(gen.parameters()), lr=3e-4)
    for epoch in range(args.epochs):
        datatrain = dst.create_loader(batch_size=args.batch_size, num_cams=args.num_cams)

        trainl = engine.train_step(gen=gen, disc=disc, disc_opt=opt, data=datatrain, 
                                   batch_size=args.batch_size, num_cams=args.num_cams, ratio=args.ratio)
        print(trainl)
        modelname = f'{args.modelname}_{epoch}.pt'
        if epoch%10 == 0:
            kt.save_ckp(model=gen, opt=opt, epoch=epoch, fname=modelname, trainloss=trainl, valloss=0.1)
        



if __name__ == '__main__':
    print(args)
    if args.train:
        train()