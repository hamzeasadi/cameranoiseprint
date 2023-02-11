import torch
from torch import nn as nn
from torch.nn import functional as F
import utils
import lossfunc2 as loss2



dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_labels(batch_size, numcams):
    numframes = batch_size//numcams
    lbl_dim = numcams * numframes
    labels = torch.zeros(size=(lbl_dim, lbl_dim), device=dev, dtype=torch.float32)
    for i in range(0, lbl_dim, numframes):
        labels[i:i+numframes, i:i+numframes] = 1
    return labels


def calc_m(batch_size, numcams, m1, m2):
    lbls = calc_labels(batch_size=batch_size, numcams=numcams)
    for i in range(lbls.size()[0]):
        for j in range(lbls.size()[1]):
            if lbls[i, j] == 1:
                lbls[i, j] = m1
            elif lbls[i, j] == 0:
                lbls[i, j] = m2

    return lbls

def calc_psd(x):
    # x = x.squeeze()
    dft = torch.fft.fft2(x)
    avgpsd =  torch.mean(torch.mul(dft, dft.conj()).real, dim=0)
    r = torch.mean(torch.log(avgpsd)) - torch.log(torch.mean(avgpsd))
    return r




class OneClassBCE(nn.Module):
    """
    doc
    """
    def __init__(self, batch_size, num_cam, reg, m1, m2) -> None:
        super().__init__()
        self.reg = reg
        self.m = calc_m(batch_size=batch_size, numcams=num_cam, m1=m1, m2=m2)
        self.label = calc_labels(batch_size=batch_size, numcams=num_cam)
        self.crt = nn.BCEWithLogitsLoss()
        self.newloss = loss2.SoftMLoss(batch_size=batch_size, framepercam=batch_size//num_cam)


    def forward(self, x):
        xs = x.squeeze()
        # distmtx = utils.euclidean_distance_matrix(xs)
        # logits = self.m - torch.square(distmtx)
        # l1 = self.crt(logits, self.label)
        l2 = self.reg*calc_psd(xs)
        l3 = self.newloss(xs)
        return l3 - l2
        # return l1+l3-l2




def main():
    x = torch.randn(size=(200, 1, 64,64))
    # y = torch.randn(size=(3, 3))
    # xy = torch.cat((x.unsqueeze(dim=0), y.unsqueeze(dim=0)), dim=0).unsqueeze(dim=0)
    # onloss = OneClassBCE(num_cam=20, batch_size=200, reg=0, m1=10000, m2=10000)
    # loss = onloss(x)
    # print(loss.item())
    print(calc_labels(batch_size=200, numcams=20))
    print(calc_m(batch_size=200, numcams=20, m1=10, m2=20))
    



if __name__ == '__main__':
    main()