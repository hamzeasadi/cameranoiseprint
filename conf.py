import os
from os.path import expanduser
from typing import NamedTuple





class Paths(NamedTuple):
    """
    paths for the management of current project
    """
    root:str=os.getcwd()
    home:str=expanduser("~")
    data:str=os.path.join(root, 'data')
    model:str=os.path.join(data, 'model')
    output:str=os.path.join(data, 'output')
    artifacts:str=os.path.join(output, 'artifacts')
    logs:str=os.path.join(output, 'logs')
    figs:str=os.path.join(output, 'figs')
    dataset:str=os.path.join(data, 'dataset')
    videos:str=os.path.join(dataset, 'videos')
    vtrain:str=os.path.join(videos, 'vtrain')
    vtest:str=os.path.join(videos, 'vtest')
    iframes:str=os.path.join(dataset, 'iframes')
    itrain:str=os.path.join(iframes, 'itrain')
    itest:str=os.path.join(iframes, 'itest')


def create_dir(dir_path:str)->None:
    try:
        os.makedirs(dir_path)
    except Exception as e:
        print("it probably already exist")

def rm_ds(arr:list)->list:
    try:
        arr.remove('.DS_Store')
    except Exception as e:
        pass
    return arr


def walk_test(dirpath):
    for path1, dirname1, files in os.walk(dirpath):
        print(path1)
        print(dirname1)
        print(files)
        print("====================================")

if __name__ == "__main__":
    print(42)
    paths = Paths()
    # for path in paths:
    #     create_dir(path)

    walk_test((paths.data))
