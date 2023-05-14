import os
import sys
import csv
import torch
from torch import nn
import cv2
import argparse
import time


# sys.path.append('../..')
# sys.path.append(os.getcwd())


# from Module.Data.Base_lmdb_entery import Image_entery
# from Module.Data import iframe_720x1280_dataset
parser = argparse.ArgumentParser(prog=os.path.basename(__file__), description="read the saving path and frame rate and resolution")

parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--fps', type=int, required=True, default=30)
parser.add_argument('--width', type=int, default=640)
parser.add_argument('--height', type=int, default=480)

args = parser.parse_args()


if __name__ == '__main__':
    print(42)
    # new_db_path = '/home/hasadi/project/cameranoiseprint/data'
    # new_db_name = 'vision_720x1280'
    # new_db = os.path.join(new_db_path, new_db_name)

    # dataset = iframe_720x1280_dataset.VideoData(db_path=new_db_path, db_name=new_db_name, crop_size=(510, 510), fup=3)
    
    # print(len(dataset))
    """
    camera final resolution 
    1280.0 960.0
    """

    saving_path = os.path.join(os.getcwd(), args.save_path)
    txt_path = os.path.dirname(saving_path)
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    cam = cv2.VideoCapture(0)

    cam.set(cv2.CAP_PROP_FPS, args.fps)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cam.isOpened():
        print("cam is not open")
    else:
        # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_id = 0
        with open(os.path.join(txt_path, 'timestame.txt'), 'w') as f:
            with open(os.path.join(txt_path, 'data.csv'), 'w', newline='\n') as f_data:
                writer = csv.writer(f_data)
                while(True):
                    # Capture frame-by-frame
                    ret, frame = cam.read()
                    time_stamp = time.time_ns()
                    f.write(f'{time_stamp}\n')
                    writer.writerow([time_stamp, f'{time_stamp}.png'])
                    # Display the resulting frame
                    cv2.imshow('preview',frame)
                    cv2.imwrite(os.path.join(saving_path, f'{time_stamp}.png'), frame)

                    # Waits for a user input to quit the application
                    # frame_id += 1
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break