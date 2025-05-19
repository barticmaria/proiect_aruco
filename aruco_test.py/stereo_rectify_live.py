import cv2
import numpy as np

cameraMatrix1 = np.load("cameraMatrix1.npy")
cameraMatrix2 = np.load("cameraMatrix2.npy")
distCoeffs1 = np.load("distCoeffs1.npy")
distCoeffs2 = np.load("distCoeffs2.npy")
R = np.load("R.npy")
T = np.load("T.npy")
imageSize = (1280, 720)

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1,
    cameraMatrix2, distCoeffs2,
    imageSize, R, T, alpha=0
)

map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, imageSize, cv2.CV_32FC1)

cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, imageSize[0])
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, imageSize[1])
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, imageSize[0])
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, imageSize[1])

if not cap_left.isOpened() or not cap_right.isOpened():
    print(" Nu s-au deschis camerele.")
    exit()

while True:
    ret1, frame1 = cap_left.read()
    ret2, frame2 = cap_right.read()

    if not ret1 or not ret2:
        print(" Eroare la citirea cadrelor.")
        break

    #
