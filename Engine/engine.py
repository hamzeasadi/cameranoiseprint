"""
docs
"""

import os
from typing import Type, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from Utils.gutils import Paths
from Dataset.dataset import create_batch


paths = Paths()

def train_step_np(model:nn.Module, loader:List, opt:Optimizer, crt:nn.Module, epoch:int, dev:torch.device, scheduler=None):
    train_loss = 0.0
    model.train()
    model.to(dev)
    num_batches = 0
    for bbatch in zip(*loader):
        X, y = create_batch(batch=bbatch)
        model_out = model(X.to(dev))
        loss = crt(embeddings=model_out, labels=y.to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss.item()
        num_batches += 1

    if scheduler is not None:
        scheduler.step()

    stat_dict = dict(model=model.eval().state_dict(), loss=train_loss/num_batches, epoch=epoch)
    torch.save(obj=stat_dict, f=os.path.join(paths.model, f"npchpoint_{epoch}.pt"))
    return dict(loss=train_loss, epoch=epoch)





def fake_real_idx(batch_size:int=200):
    evens = torch.arange(start=0, end=200, step=2)
    odds = torch.arange(start=1, end=200, step=2)
    y_real = torch.ones_like(evens)
    y_fake = torch.zeros_like(odds)
    return (evens, y_real), (odds, y_fake)






def train_step_gan(
                gen:nn.Module, disc:nn.Module, loader:List, opt_disc:Optimizer, opt_gen:Optimizer,
                epoch:int, dev:torch.device, gen_crt:nn.Module, disc_crt:nn.Module, gen_schedure=None, disc_scheduler=None):
    train_loss = 0
    dis_loss = 0
    gen.train()
    disc.train()
    gen.to(dev)
    disc.to(dev)
    num_batches = 0

    for bbatch in zip(*loader):
        X, y = create_batch(batch=bbatch)
        gen_out = gen(X.to(dev))

        (real_idx, real_label), (fake_idx, fake_label) = fake_real_idx()
        real_img = gen_out[real_idx].detach()
        fake_img = gen_out[fake_idx]

        real_prd = disc(real_img)
        fake_prd = disc(fake_img)
        disc_loss = disc_crt(real_prd.squeeze() - fake_prd.squeeze(), real_label.to(dev))
        opt_disc.zero_grad()
        disc_loss.backward(retain_graph=True)
        opt_disc.step()

        disc_loss = disc_crt(fake_prd.squeeze()-real_prd.squeeze(), real_label.to(dev))
        gen_loss = gen_crt(embeddings=gen_out, labels=y.to(dev))
        loss = gen_loss + disc_loss
        opt_gen.zero_grad()
        loss.backward()
        opt_gen.step()

        train_loss += loss.item()
        disc_loss += disc_loss.item()
        num_batches += 1
    
    if gen_schedure is not None:
        gen_schedure.step()

    if disc_scheduler is not None:
        disc_scheduler.step()

    return dict(loss=train_loss, disc=disc_loss, epoch=epoch)

    








def main():
    """docs"""

    ev, od = fake_real_idx()
    print(ev)
    print(od)




if __name__ == "__main__":
    main()