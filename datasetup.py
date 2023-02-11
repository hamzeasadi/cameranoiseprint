import torch
import os, random, sys, inspect

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir) 

import conf as cfg
from torch.utils.data import Dataset, DataLoader
import cv2


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# randomly select hi and wi
def gethiwi(camname):
    hi = int((1080-64)*random.random())
    wi = int((1920-64)*random.random())
    if (camname == 'D40') or (camname == 'D41') or (camname == 'D42') or (camname == 'D43') or (camname == 'D44'):
        hi = max(200, min(720-164, int((720-64)*random.random())))
        wi = int((1280-64)*random.random())

    return hi, wi

# extract a patch (green channel) of image in location hi, wi
def patchtensorify(img, hi, wi):
    imgpatch = img[hi:hi+64, wi:wi+64, 1:2]
    imgpatch = (imgpatch - 127)/255.0
    return torch.from_numpy(imgpatch).permute(2, 0, 1)

# extract patchprcam patchs from on random location, hi,wi, from a camera folder
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

# extract a sample which contain numpatchs for each camera and location is each time different.
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
        return 10

    def __getitem__(self, index):
        return getdatasample(datadir=self.dp, numpatchs=self.frprcm).to(dev)

    


def main():
    print(42)
    # print(cfg.paths)
    X = VisionDataset(datapath=cfg.paths['train'], numcam=20, batch_size=400)
    for x in X:
        print(x.shape)
    # for i in range(10):
    #     print(gethiwi(f'D4{i}'))



if __name__ == '__main__':
    main()