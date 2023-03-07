import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import utils



def train_step(gen:nn.Module, disc:nn.Module, disc_opt:Optimizer, data:DataLoader, batch_size, num_cams, ratio):
    epoch_loss = 0
    gen.train()
    disc.train()

    for i, X in enumerate(data):
        X = X.squeeze(dim=0)
        fakeandrealnoise = gen(X)
        idx_1, idx_2, lbls, w = utils.get_pairs(batch_size=batch_size, frprcam=batch_size//num_cams, ratio=ratio)
        crt = nn.BCEWithLogitsLoss(weight=w)
        X1 = fakeandrealnoise[idx_1]
        X2 = fakeandrealnoise[idx_2]

        X1_out = disc(X1)
        X2_out = disc(X2)
        gdisc_loss = crt(X1_out - X2_out, lbls)

        disc_opt.zero_grad()
        gdisc_loss.backward()
        disc_opt.step()
    

        epoch_loss +=  (1/(i+1))*(gdisc_loss.item() - epoch_loss)
    
    return epoch_loss

def main():
    pass



if __name__ == "__main__":
    main()