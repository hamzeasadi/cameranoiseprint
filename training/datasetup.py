import torch
import os, random, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import conf as cfg
from torch.utils.data import Dataset, DataLoader
import cv2


def gethiwi(camname):
    if (camname == 'D40') or (camname == 'D41') or (camname == 'D42') or (camname == 'D43') or (camname == 'D44'):
        hi = int((1080-364)*random.random() + 100)
        wi = int((1920-64)*random.random())
    else:
        hi = int((1080-64)*random.random())
        wi = int((1920-64)*random.random())
    return hi, wi

def patchtensorify(img, hi, wi):
    imgpatch = img[hi:hi+64, wi:wi+64, 1:2]
    imgpatch = (imgpatch - 127)/255.0
    return torch.from_numpy(imgpatch).permute(2, 0, 1)

def getpatch(campath, camname, patchprcam):
    hi, wi = gethiwi(camname=camname)
    camiframes = cfg.rm_ds(os.listdir(campath))
    subcamiframes = random.sample(camiframes, patchprcam)
    patchsample = torch.zeros(size=(patchprcam, 1, 64, 64))
    for j, iframename in enumerate(subcamiframes):
        iframepath = os.path.join(campath, iframename)
        img = cv2.imread(iframepath)
        patcht = patchtensorify(img=img, hi=hi, wi=wi)
        patchsample[j] = patcht
    
    return patchsample


def getdatasample(datadir, numpatchs):
    cams = cfg.rm_ds(os.listdir(datadir))
    firstpatch = getpatch(campath=os.path.join(datadir, cams[0]), camname=cams[0], patchprcam=numpatchs)
    for i in range(1, len(cams)):
        campath = os.path.join(datadir, cams[0])
        campatch = getpatch(campath=campath, camname=cams[i], patchprcam=numpatchs)
        firstpatch = torch.cat((firstpatch, campatch), dim=0)

    return firstpatch






class VisionDataset(Dataset):
    """
    doc
    """
    def __init__(self, datapath, numcam, batch_size) -> None:
        super().__init__()
        self.dp = datapath
        self.frprcm = batch_size//numcam
        self.cams = cfg.rm_ds(os.listdir(datapath))

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return getdatasample(datadir=self.db, numpatchs=self.frprcm)

    


def main():
    print(42)
    x = VisionDataset(datapath=cfg.paths['train'], numcam=20, batch_size=200)
    print(x.shape)



if __name__ == '__main__':
    main()