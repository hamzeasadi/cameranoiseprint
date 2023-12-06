"""
docs
"""

import os
from typing import Type

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from Utils.gutils import Paths






class Engine:
    """
    engine 
    """
    def __init__(self, epochs:int, dataset:DataLoader, model:nn.Module, opt:Optimizer, 
                 criterion:nn.Module, paths:Paths, dev:torch.device) -> None:
        self.dev = dev
        self.model = model
        self.model.to(dev)
        self.paths = paths
        self.loader = dataset
        self.criterion = criterion
        self.epochs = epochs
        self.opt = opt
        

    
    def train_step(self, epoch):
        train_loss = 0.0
        self.model.train()
        num_batches = len(self.loader)

        for X, y in self.loader:
            out = self.model(X.to(self.dev))
            outshape = out.shape
            out = out.squeeze().view((outshape[0], -1))
            loss = self.criterion(out, y.to(self.dev).squeeze())
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            train_loss += loss.item()
        
        return dict(loss=train_loss/num_batches)
    

    def train_step(self, epoch):
        valid_loss = 0.0
        self.model.eval()
        num_batches = len(self.loader)
        with torch.no_grad():
            for X, y in self.loader:
                out = self.model(X)
                outshape = out.shape
                out = out.squeeze().view((outshape[0], -1))
                loss = self.criterion(out, y.squeeze())
                
                valid_loss += loss.item()
        
        return dict(loss=valid_loss/num_batches, epoch=epoch)



    def report(self, epoch, min_val_error):
        pass



    def run(self):

        for epoch in range(self.epochs):
            train_states = self.train_step(epoch=epoch)
            print(train_states)
    
