"""
general utility functionalities
"""

import os
import json
from dataclasses import dataclass
import pickle
from typing import Dict
from typing import Any
import csv
import pandas as pd




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






def json2dict(json_path:str)->Dict:
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




def csv2list(file_path:str, HEAD:bool):
    """
    get an csv file path and return it as a dictionary
    args:
        file_path: csv file path with .csv extension
        HEAD: if the current file has header or not
    returns:
        header: the header of the file if HEAD is true None otherwise
        data: a list of rows of csv file
    """

    header = None
    data = []
    with open(file_path, "r", encoding="utf-8") as csv_file:
        csv_data = csv.reader(csv_file)
        if HEAD:
            header = csv_data.next()
        for row in csv_data:
            data.append(row)
    
    return header, data




def dict2csv(dict_data:Dict, save_path:str, filename:str):
    """
    convert a dictionary to csv file
    args:
        dict_data: the dictionary that need to be written as csv file
        save_path: the directory to save the csv
        filename: the file name with csv extension
    """
    column_names = list(dict_data.keys())
    column0_data = dict_data[column_names[0]]
    data = []
    for i in range(len(column0_data)):
        item = []
        for col_name in column_names:
            item.append(dict_data[col_name][i])
        data.append(item)

    with open(os.path.join(save_path, filename), "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        csv_writer.writerow(column_names)
        csv_writer.writerows(data)




def isEmpty(dir_path:str):

    if os.path.exists(dir_path) and not os.path.isfile(dir_path):
        if len(os.listdir(dir_path)) == 0:
            return True
        else:
            return False
    
    raise NotADirectoryError





def main():
    """
    docs
    """
    paths = Paths()

    for _, dir_path in paths.__dict__.items():
        Paths.crtdir(dir_path=dir_path)

    x = dict(name=["ali", "john"], age=[1, 2])
    dict2csv(dict_data=x, save_path=paths.report, filename="sth.csv")


if __name__ == "__main__":
    main()