import torch
import numpy as np



def distmtxidxlbl(batch_size, frprcam):
    indexs = torch.tensor(list(range(batch_size)))
    idxandlbl = dict()
    for blk in range(0, batch_size, frprcam):
        for row in range(blk, blk+frprcam):
            rowidx = []
            rowlbl = []
            for i in range(row+1, blk+frprcam):
                idx = torch.cat(( indexs[:blk], indexs[i:i+1], indexs[blk+frprcam:]), dim=0)
                rowidx.append(idx)
                rowlbl.append(blk)

            idxandlbl[row] = (rowidx, rowlbl)

    return idxandlbl



def main():
    dim = 20
    stp = 5
    # indexs = torch.tensor(list(range(dim)))
    # all_idx = []
    # all_lbl = []
    # idxandlbl = dict()
    # for blk in range(0, dim, stp):
    #     for row in range(blk, blk+stp):
    #         rowidx = []
    #         rowlbl = []
    #         for i in range(row+1, blk+stp):
    #             idx = torch.cat(( indexs[:blk], indexs[i:i+1], indexs[blk+stp:]), dim=0)
    #             rowidx.append(idx)
    #             rowlbl.append(blk//stp)

    #         idxandlbl[f'{row}'] = (rowidx, rowlbl)
    
    idxlbl = distmtxidxlbl(batch_size=12, frprcam=3)

    print(idxlbl)


if __name__ == '__main__':
    main()