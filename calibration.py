import os
import argparse
import yaml
import numpy as np
import cv2 
import glob



parser = argparse.ArgumentParser(prog=os.path.basename(__file__), description='read saving path of the camera calibration data')
parser.add_argument('--calib_path', type=str, required=True)
parser.add_argument('--width', type=int, default=1280)
parser.add_argument('--height', type=int, default=720)
parser.add_argument('--ckboard_h_corners', type=int, default=10)
parser.add_argument('--ckboard_v_corners', type=int, default=7)
parser.add_argument('--capture', action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()



################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (args.ckboard_h_corners,args.ckboard_v_corners)
frameSize = (args.width,args.height)
calib_path = os.path.join(os.getcwd(), args.calib_path, 'images')

if not os.path.exists(calib_path):
    os.makedirs(calib_path)

if args.capture:
    cap = cv2.VideoCapture(0)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameSize[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameSize[1])
    except Exception as e:
        print(e)
    num = 0
    while cap.isOpened():
        succes, img = cap.read()
        k = cv2.waitKey(5)
        if k == 27:
            break
        elif k == ord('s'): # wait for 's' key to save and exit
            cv2.imwrite(os.path.join(calib_path, f'{num}.png'), img)
            print("image saved!")
            num += 1

        cv2.imshow('Img',img)

    cap.release()
    cv2.destroyAllWindows()





# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


images = glob.glob(os.path.join(calib_path, '*.png'))

for image in images:

    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)


cv2.destroyAllWindows()


############## CALIBRATION #######################################################

ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

parametr_path = os.path.dirname(calib_path)


with open(os.path.join(parametr_path, 'intrinsic.npy'), 'wb') as f:
    np.save(f, cameraMatrix)


############## UNDISTORTION #####################################################

print(f'calibration matrix: {cameraMatrix}')
print(dist, dist.shape)

h,  w = img.shape[:2]
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

print(f'optimized calibrated matrix: {newCameraMatrix}')

with open(os.path.join(parametr_path, 'intrinsicNew.npy'), 'wb') as f:
    np.save(f, newCameraMatrix)

calib_dict = [
    dict(
    camera_model='pinhole', frame_width=frameSize[0], frame_height=frameSize[1],
    fx=float(newCameraMatrix[0, 0]), fy=float(newCameraMatrix[1,1]), cx=float(newCameraMatrix[0, 2]), cy=float(newCameraMatrix[1,2]),
    k1=float(dist[0][0]), k2=float(dist[0][1]), p1=float(dist[0][2]), p2=float(dist[0][3]), k3=float(dist[0][4])
    )
]

with open(os.path.join(parametr_path, 'intrinsic.yaml'), 'w') as param:
    yaml.dump(calib_dict, param)


# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )
