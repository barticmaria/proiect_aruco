
import cv2
import numpy as np

mtx_l = np.load("stereo_calib/stereo_mtx_l.npy")
dist_l = np.load("stereo_calib/stereo_dist_l.npy")
mtx_r = np.load("stereo_calib/stereo_mtx_r.npy")
dist_r = np.load("stereo_calib/stereo_dist_r.npy")
R = np.load("stereo_calib/R.npy")
T = np.load("stereo_calib/T.npy")

img_l = cv2.imread("left/left_00.jpg")
img_r = cv2.imread("right/right_00.jpg")

gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

image_size = gray_l.shape[::-1]

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, image_size, R, T, alpha=0)

map_l1, map_l2 = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, image_size, cv2.CV_16SC2)
map_r1, map_r2 = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, image_size, cv2.CV_16SC2)

rect_l = cv2.remap(gray_l, map_l1, map_l2, cv2.INTER_LINEAR)
rect_r = cv2.remap(gray_r, map_r1, map_r2, cv2.INTER_LINEAR)

stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)
disparity = stereo.compute(rect_l, rect_r)

disp_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disp_norm = np.uint8(disp_norm)

cv2.imshow("Disparity Map", disp_norm)
cv2.imwrite("disparity_map.png", disp_norm)
cv2.waitKey(0)
cv2.destroyAllWindows()
