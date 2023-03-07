import os 
import random
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import conf as cfg


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
paths = cfg.Paths()

hight = list(np.arange(start=0, stop=1080-64, step=10))
width = list(np.arange(start=0, stop=1920-64, step=10)) 


# randomly select hi and wi
def gethiwi(H:int=1080, W:int=1920):
    hi = random.sample(hight, 1)
    wi = random.sample(width, 1)
    return hi[0], wi[0]



def patchtensorify(img, hi, wi):
    imgpatch = img[hi:hi+64, wi:wi+64, :]
    imgpatch = imgpatch/255.0
    return torch.from_numpy(imgpatch).permute(2, 0, 1)

# extract patchprcam patchs from on random location, hi,wi, from a camera folder
def getpatch(campath, num_patch):
    hi, wi = gethiwi()
    cam_iframes = cfg.rm_ds(os.listdir(campath))
    subcamiframes = random.sample(cam_iframes, num_patch)
    patchsample = torch.zeros(size=(num_patch, 3, 64, 64), dtype=torch.float32)

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
    def __init__(self, datapath, numcam, batch_size, dataset_len) -> None:
        super().__init__()
        self.dp = datapath
        self.num_cams = numcam
        self.dl = dataset_len
        self.frprcm = batch_size//numcam
        self.cams = cfg.rm_ds(os.listdir(datapath))
        
    def __len__(self):
        return self.dl

    def __getitem__(self, index):
        sub_cams = random.sample(self.cams, self.num_cams)
        patchs = torch.randn(size=(1, 3, 64, 64))
   
        for cam_name in sub_cams:
            cam_path = os.path.join(self.dp, cam_name)
            patch = getpatch(campath=cam_path, num_patch=self.frprcm)
            patchs = torch.cat((patchs, patch), dim=0)


        return patchs[1:].float().to(dev)

    
def create_loader(batch_size:int=40, num_cams:int=5, dl:int=10000):
    dataset = VisionDataset(datapath=paths.itrain, numcam=num_cams, batch_size=batch_size, dataset_len=dl)
    data_loader = DataLoader(dataset=dataset, batch_size=1)
    return data_loader

def main():
    print(42)



if __name__ == '__main__':
    main()