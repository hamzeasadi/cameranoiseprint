"""
general utility functionalities
"""

import os
import json
from dataclasses import dataclass
import pickle
from typing import Dict
from typing import Any



@dataclass
class Paths:
    """
    general directories for the experiments

    """
    root:str = os.path.dirname(os.path.dirname(__file__))
    config:str = os.path.join(root, "Config")
    data:str = os.path.join(root, "data")
    external_dataset:str = os.path.join(os.path.dirname(root), "dataset_camerasource")

    model:str = os.path.join(data, "model")
    dataset:str = os.path.join(data, "dataset")
    report:str = os.path.join(data, "report")
    logs:str = os.path.join(data, "logs")

    @staticmethod
    def crtdir(dir_path:str):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)






def jsco2dict(json_path:str)->Dict:
    """
    read a json file and return a dicrionary
    
    """

    with open(json_path, "r", encoding="utf-8") as jfile:
        data = json.load(jfile)
    return data





def save_as_pickle(save_path:str, data:Any, filename:str):
    """
    save any data as a pickle file
    args:
        save_path: path to save data
        data: object that need to be saved
        filename: the name of file with pkl extension

    """
    with open(os.path.join(save_path, filename), "wb") as pfile:
        pickle.dump(obj=data, file=pfile)





def load_pickle(file_path:str):
    """
    load a pickle file
    args:
        file_path: file path with pkl extention
    """
    with open(file_path, "rb") as pfile:
        data = pickle.load(pfile)
    return data



def main():
    """
    docs
    """
    paths = Paths()

    for _, dir_path in paths.__dict__.items():
        Paths.crtdir(dir_path=dir_path)



if __name__ == "__main__":
    main()