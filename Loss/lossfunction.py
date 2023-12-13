"""
loss functions
"""

import torch
from torch import nn
from matplotlib import pyplot as plt
from torch.nn import functional as F
from pytorch_metric_learning.miners import BaseMiner
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu




        


class NP_Loss(nn.Module):
    """
    noiseprint loss implementation
    """
    def __init__(self, lamda:float=10.0, scale:float=100.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lamda = lamda
        self.scale = scale

    def forward(self, embeddings:torch.Tensor, labels:torch.Tensor, psd_flag:bool=True):
        loss = 0.0
        if psd_flag:
            loss += (-self.lamda * self.psd(embeddings=embeddings))
        
        loss += self._np_loss(embeddins=embeddings, y=labels, scale=self.scale)
    

    def _np_loss(self, embeddins:torch.Tensor, y:torch.Tensor, scale:float=100):
        b, c, _, _ = embeddins.shape
        np_acc_loss = 0.0
        embed_flat = embeddins.view(b, -1)
        y_ = y.squeeze()
        num_lbl = y_.size()[0]
        dist_mtx = torch.cdist(x1=embed_flat, x2=embed_flat, p=2)
        dist_mtx_offdiagnal = dist_mtx.flatten()[1:].view(num_lbl-1, num_lbl+1)[:,:-1].reshape(num_lbl, num_lbl-1)
        dist_mtx_offdiagnal_sm = torch.softmax(-dist_mtx_offdiagnal, dim=1)
        for i in range(num_lbl):
            lbl = y_[i]
            distance_sm_lbl = dist_mtx_offdiagnal_sm[i]
            indices = torch.cat((y_[:i], y_[i+1:]), dim=0)
            indices_ind = indices==lbl
            probs = torch.sum(distance_sm_lbl[indices_ind])
            np_acc_loss += -torch.log(probs)
        
        return np_acc_loss/scale
    

    def psd(self, embeddings:torch.Tensor):
        """
        docs
        """
        x = embeddings.squeeze()
        b, h, w = x.shape
        k = h*w
        dft = torch.fft.fft2(x)
        avgpsd =  torch.mean(torch.mul(dft, dft.conj()).real, dim=0)
        loss_psd = torch.clamp((1/k)*torch.sum(torch.log(avgpsd)) - torch.log((1/k)*torch.sum(avgpsd)), min=0.0, max=100.0)
        return loss_psd







def psd(x:torch.Tensor):
    """
    
    """
    b, h, w = x.shape
    k = h*w
    dft = torch.fft.fft2(x)
    avgpsd =  torch.mean(torch.mul(dft, dft.conj()).real, dim=0)

    loss_psd = (1/k)*torch.sum(torch.log(avgpsd)) - torch.log((1/k)*torch.sum(avgpsd))
    print(loss_psd)


    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].imshow(x.squeeze().numpy(), label="img", cmap='gray')
    axs[0].axis("off")
    axs[1].imshow(avgpsd.squeeze().numpy(), label="ft", cmap='gray')
    axs[1].axis("off")
    fig.savefig("ft.png", bbox_inches='tight', pad_inches=0)
    plt.close()





def gen_img(freq:int):
    ones = torch.ones(size=(1, 48))
    zeros = torch.zeros(size=(1, 48))
    ones11 = torch.cat((ones, ones), dim=0)
    zeros01 = torch.cat((zeros, zeros), dim=0)
    
    
    img0 = torch.cat((ones, zeros), dim=0).repeat(repeats=(24, 1))
    img1 = torch.cat((ones11, zeros01), dim=0).repeat(repeats=(12, 1))

    return img0, img1
    





def main():
    """"
    docs
    """
    x = torch.tensor([[1,2,3,4], [2,1,3,4], [2,2,3,3], [7,8,9,10], [6,8,8,10], [7,7,9,9]], dtype=torch.float32)
    y = torch.tensor([0,0,0,1,1,1])
    x_dist = torch.cdist(x1=x, x2=x, p=2)
    print(x_dist)
    x_dist = x_dist.flatten()[1:].view(6-1, 6+1)[:,:-1].reshape(6, 6-1)
    print(x_dist)
    x_dist_sm = torch.softmax(-x_dist, dim=1)
    print(x_dist_sm)
    for i in range(6):
            print("=="*40)
            lbl = y[i]
            distance_sm_lbl = x_dist_sm[i]
            print(distance_sm_lbl)
            indices = torch.cat((y[:i], y[i+1:]), dim=0)
            print(indices)
            indices_ind = indices==lbl
            print(indices_ind)
            probs = torch.sum(distance_sm_lbl[indices_ind])
            print(probs)
            # np_acc_loss += -torch.log(probs)
            # break

    # x = x.view((4, 1, 2, 2))
    # y = torch.tensor([1,1,0,0])
    # crt = NP_Loss(lamda=1, scale=1)

    # out = crt._np_loss(embeddins=x, y=y, scale=1)

    # print(out)

    




   





if __name__ == "__main__":

    main()