"""
loss functions
"""

import torch
from torch import nn
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
        






def main():
    """"
    docs
    """
    crt = nn.BCEWithLogitsLoss()

    miner = ExamplePairMiner()

    x = torch.tensor([
        [1, 2, 3],
        [1, 2, 3.5],
        [7, 8, 9],
        [1, 2, 3]
    ], dtype=torch.float32)

    y = torch.tensor([1,1, 0,0])

    crt = Loss_Function()

    loss = crt(x, y)

    print(loss)

   





if __name__ == "__main__":

    main()