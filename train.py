import conf as cfg
import model as m
import torch
from torch import nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
import lossfunc
import utils
import datasetup as dst
import argparse
import engine




dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(prog='train.py', description='read setting for training proceduer')

parser.add_argument('--train', action=argparse.BooleanOptionalAction)
parser.add_argument('--test', action=argparse.BooleanOptionalAction)
parser.add_argument('--adaptive', action=argparse.BooleanOptionalAction)
parser.add_argument('--coord', action=argparse.BooleanOptionalAction)

parser.add_argument('--epochs', type=int, default=100, required=True)
parser.add_argument('--batch_size', type=int, default=200, required=True)
parser.add_argument('--depth', type=int, default=15, required=True)
parser.add_argument('--reg', type=float, default=1, required=True)
parser.add_argument('--m1', type=float, default=100)
parser.add_argument('--m2', type=float, default=1000)

parser.add_argument('--modelname', type=str, required=True)

args = parser.parse_args()



def getm1m2(M1, M2, epoch):
    m1 = M1//(1+epoch)
    m2 = M2//(1+epoch)



def train(Net:nn.Module, opt:Optimizer, M1, M2):
    kt = utils.KeepTrack(path=cfg.paths['model'])
    for epoch in range(args.epochs):
        m1, m2 = getm1m2(M1=M1, M2=M2, epoch=epoch)
        crttrain = lossfunc.OneClassBCE(batch_size=args.batch_size, num_cam=20, reg=args.reg, m1=m1, m2=m2)
        crtval = lossfunc.OneClassBCE(batch_size=args.batch_size, num_cam=5, reg=args.reg, m1=m1, m2=m2)

        datatrain = dst.VisionDataset(datapath=cfg.paths['train'], numcam=20, batch_size=args.batch_size)
        dataval = dst.VisionDataset(datapath=cfg.paths['val'], numcam=5, batch_size=args.batch_size)
        
        trainl = engine.train_setp(net=Net, criterion=crttrain, datal=datatrain, optimizer=opt)
        vall = engine.val_setp(net=Net, criterion=crtval, datal=dataval, optimizer=opt)
        modelname = f'{args.modelname}.pt'
        kt.save_ckp(model=Net, opt=opt, epoch=epoch, fname=modelname, trainloss=trainl, valloss=vall)
        









def main():
    print(args)
    inch=1
    if args.coord:
        inch=3
    
    network = m.NoisePrint(inch=inch, depth=args.depth)
    optt = torch.optim.Adam(params=network.parameters(), lr=3e-4)
    if args.train:
        train(Net=network, opt=optt, M1=args.m1, M2=args.m2)


if __name__ == '__main__':
    main()