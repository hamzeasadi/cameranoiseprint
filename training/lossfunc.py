import torch
from torch import nn as nn
from torch.nn import functional as F




dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def euclidean_distance_matrix(x):
    eps = 1e-8
    x = torch.flatten(x, start_dim=1)
    dot_product = torch.mm(x, x.t())
    squared_norm = torch.diag(dot_product)
    distance_matrix = squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1)
    distance_matrix = F.relu(distance_matrix)
    mask = (distance_matrix == 0.0).float()
    distance_matrix = distance_matrix.clone() + mask * eps
    distance_matrix = torch.sqrt(distance_matrix)
    distance_matrix = distance_matrix.clone()*(1.0 - mask)
    return distance_matrix

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

    def forward(self, x):
        xs = x.squeeze()
        distmtx = euclidean_distance_matrix(xs)
        print(torch.square(distmtx))
        print(torch.sigmoid(distmtx))
        logits = self.m - torch.square(distmtx)
        print(torch.sigmoid(logits))
        l1 = self.crt(logits, self.label)
        l2 = self.reg*calc_psd(xs)
        return l1-l2





def main():
    x = torch.randn(size=(200, 1, 64,64))
    # y = torch.randn(size=(3, 3))
    # xy = torch.cat((x.unsqueeze(dim=0), y.unsqueeze(dim=0)), dim=0).unsqueeze(dim=0)
    onloss = OneClassBCE(num_cam=20, batch_size=200, reg=0, m1=10000, m2=10000)
    loss = onloss(x)
    print(loss.item())



if __name__ == '__main__':
    main()