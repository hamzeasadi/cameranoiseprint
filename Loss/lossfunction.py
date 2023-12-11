"""
loss functions
"""

import torch
from torch import nn
from matplotlib import pyplot as plt
from torch.nn import functional as F
from pytorch_metric_learning.miners import BaseMiner
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu







class ExamplePairMiner(BaseMiner):
    def __init__(self, margin=0.1, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def mine(self, embeddings, labels, ref_emb=None, ref_labels=None):
        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = lmu.get_all_pairs_indices(labels, ref_labels)
        return a1, p, a2, n





class Loss_Function(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.miner = ExamplePairMiner()
        self.bceloss = nn.BCEWithLogitsLoss()

    def forward(self, embedding, labels):
        a, p, a1, n = self.miner.mine(embeddings=embedding, labels=labels)
        A = embedding[a]
        P = embedding[p]
        A1 = embedding[a1]
        N = embedding[n]
        dist1 = torch.mean(torch.norm(torch.sub(A, P), dim=1))
        dist2 = torch.norm(torch.sub(A1, N), dim=1)
        neg_labels = torch.ones_like(dist2, requires_grad=False)
        loss = 10*self.bceloss(dist2, neg_labels) + dist1
        
        return loss
        


class NP_Loss(nn.Module):
    """
    noiseprint loss implementation
    """
    def __init__(self, lamda:float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lamda = lamda
    
    def forward(self, embeddings:torch.Tensor, labels:torch.Tensor, psd_flag:bool=True):
        b, c, _, _ = embeddings.shape
        loss = 0.0
        if psd_flag:
            loss += (-self.lamda * self.psd(embeddings=embeddings))
        embeddings = embeddings.view(b, -1)
        labels = labels.squeeze()
        num_lbls = labels.size()[0]
        distance_matrix = torch.cdist(x1=embeddings, x2=embeddings, p=2)
        distance_matrix = distance_matrix.flatten()[1:].view(num_lbls-1, num_lbls+1)[:,:-1].reshape(num_lbls, num_lbls-1)
        distance_sm = torch.softmax(input=-distance_matrix, dim=1)
        
        for i in range(num_lbls):
            lbl = labels[i]
            distance_sm_lbl = distance_sm[i]
            indices = torch.cat((labels[:i], labels[i+1:]), dim=0)
            indices_ind = indices==lbl
            probs = torch.sum(distance_sm_lbl[indices_ind])
            loss += -torch.log(probs)/200.0
        
        return loss
    
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
    # x = torch.randn(size=(10, 1, 48, 48))
    # y = torch.randint(low=0, high=3, size=(10, 1))
    # crt = NP_Loss(lamda=0.3)
    # loss = crt(x, y)
    # print(loss)
    x = torch.randn(size=(48, 48))
    x1, x2 = gen_img(freq=1)
    psd(x2.unsqueeze(dim=0))

    




   





if __name__ == "__main__":

    main()