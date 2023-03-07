import os
import conf as cfg
import cv2
import numpy as np

IMG_WIDTH = 1280
IMG_HIGHT = 720

paths = cfg.Paths()

def frame_alignment(img_path:str):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imshape = gray.shape
    if imshape[0]>imshape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    hh = IMG_HIGHT//2
    ww = IMG_WIDTH//2
    h, w, c = img.shape
    hc, wc = h//2, w//2
    cv2.imwrite(filename=img_path, img=img[hc-hh:hc+hh, wc-ww:wc+ww, :])
    

    

def extract_iframe(src_video_path:str, trg_iframe_path:str):
    """
    extract all iframes from a directory
    args:
        src_video_path: a folder path that contain folder with the camera names 
                        that contain video of the specific camera 
                        e.g. videos/{cam1, cam2, ...}/{vid1.mp4, vid2.mp4, ...}
        trg_iframe_path: the target path that the iframes of each camera will be save under the name camera folder
    """
    cam_names = os.listdir(src_video_path)
    cam_names = cfg.rm_ds(cam_names)

    for cam_name in cam_names:
        cam_path = os.path.join(src_video_path, cam_name)
        trg_cam_path = os.path.join(trg_iframe_path, cam_name)
        cam_videos = cfg.rm_ds(os.listdir(cam_path))

        for video_name in cam_videos:
            video_path = os.path.join(cam_path, video_name)
            cmd = f'ffmpeg -skip_frame nokey -i {video_path} -vsync vfr -frame_pts true -x264opts no-deblock {trg_cam_path}/{video_name}_%d.bmp'
            os.system(command=cmd)
        
        cam_iframes = cfg.rm_ds(os.listdir(trg_cam_path))
        for cam_iframe in cam_iframes:
            iframe_path = os.path.join(trg_cam_path, cam_iframe)
            frame_alignment(iframe_path)



if __name__ == "__main__":
    print(42)
    # extract train iframes
    extract_iframe(src_video_path=paths.vtrain, trg_iframe_path=paths.itrain)
    
    # extract validation
    extract_iframe(src_video_path=paths.vtest, trg_iframe_path=paths.itest)
    
        

