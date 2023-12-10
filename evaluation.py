"""
docs
"""

import os
import argparse

from PIL import Image
import torch
from matplotlib import pyplot as plt

from Utils.gutils import Paths
from DataPrepare.utils import cvt2Intensity
from Model.noiseprint_model import Noise_Print




def main():
    """docs"""
    paths = Paths()

    parser = argparse.ArgumentParser(prog=os.path.basename(__file__), description="eval config")
    parser.add_argument("--ckp_num", type=int, required=True)

    args = parser.parse_args()

    valid_data_path = os.path.join(paths.dataset, "valid")
    img_list = [f for f in os.listdir(valid_data_path) if f.endswith(".png")]

    ckp_num = args.ckp_num
    ckp_name = f"ckpoint_{ckp_num}.pt"
    model_path = os.path.join(paths.model, ckp_name)
    state = torch.load(model_path, map_location=torch.device("cpu"))
    model = Noise_Print(input_shape=[1,3,48,48], num_layers=17)
    model.load_state_dict(state['model'])
    print(f"epoch={ckp_num} loss={state['loss']}")
    res = []
    with torch.no_grad():
        
        for img_name in img_list:
            # img_path = "/home/hasadi/project/noiseprintPro/data/images/inpainting.png"
            img_path = os.path.join(valid_data_path, img_name)
            img = Image.open(img_path)
            img_np = cvt2Intensity(img=img)
            img_t = torch.from_numpy(img_np).unsqueeze(dim=0)
            img_t = img_t.repeat(repeats=[3, 1, 1])
            out = model(img_t.unsqueeze(dim=0))
            res.append(out.cpu().detach().squeeze().numpy())
    
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].imshow(res[0], cmap='gray')
    axs[0].axis("off")
    axs[1].imshow(res[1], cmap='gray')
    axs[1].axis("off")
    save_path = os.path.join(paths.report, f"res_{ckp_num}.png")
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()






if __name__ == "__main__":
    main()
