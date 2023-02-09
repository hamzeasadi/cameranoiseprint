import os, random
import cv2
import conf as cfg



def videoiframeext(videopath, videoid, videoiframepath):
    framename = f'frame_{videoid}'
    # trgvideoiframepath = 
    command = f"ffmpeg -skip_frame nokey -i {videopath} -vsync vfr -frame_pts true -x264opts no-deblock {videoiframepath}/{framename}_%d.bmp"
    os.system(command=command)

def camvideosiframeext(campath, trgcampath):
    camvideonames = cfg.rm_ds(os.listdir(campath))
    cfg.create_dir(trgcampath)
    for i, camvideoname in enumerate(camvideonames):
        camvideopath = os.path.join(campath, camvideoname)
        videoiframeext(videopath=camvideopath, videoid=i, videoiframepath=trgcampath)


def dataiframeext(camspath, iframespath):
    camsname = cfg.rm_ds(os.listdir(camspath))
    for camname in camsname:
        camerapath=os.path.join(camspath, camname)
        trgcampath = os.path.join(iframespath, camname)
        camvideosiframeext(campath=camerapath, trgcampath=trgcampath)








def main():
    srcpath = cfg.paths['videos']
    trgpath = cfg.paths['iframes']
    dataiframeext(camspath=srcpath, iframespath=trgpath)
    



if __name__ == '__main__':
    main()