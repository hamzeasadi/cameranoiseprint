import os
import random
from os.path import expanduser

root = os.getcwd()
data = os.path.join(root, 'data')
home = expanduser("~")

paths=dict(
    root=root, data=data, home=home,
    model=os.path.join(data, 'model'), modelcoord=os.path.join(data, 'modelcoord'), training=os.path.join(data, 'training'), testing=os.path.join(data, 'testing'), 
    videos=os.path.join(data, 'videos'), iframes=os.path.join(data, 'iframes'),
    train=os.path.join(data, 'training', 'train'), val=os.path.join(data, 'training', 'val'), test=os.path.join(data, 'testing', 'test'), 

)

def create_dir(path):
    try:
        os.makedirs(path)
    except Exception as e:
        print(e)

def rm_ds(mylist:list):
    try:
        mylist.remove('.DS_Store')
    except Exception as e:
        print(e)
    return mylist




def main():
    print(42)
    for k, v in paths.items():
        create_dir(v)
        print(v)
    


if __name__ == '__main__':
    main()