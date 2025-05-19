
import cv2
import numpy as np
import glob
import os

pattern_size = (9, 6)

square_size = 0.025

objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints_l = []
imgpoints_r = []

left_images = sorted(glob.glob("left/*.jpg"))
right_images = sorted(glob.glob("right/*.jpg"))

assert len(left_images) == len(right_images), "Numar diferit de imagini stanga vs dreapta"

for left_path, right_path in zip(left_images, right_images):
    img_l = cv2.imread(left_path)
    img_r = cv2.imread(right_path)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size, None)

    if ret_l and ret_r:
        objpoints.append(objp)

        corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1),
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1),
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        imgpoints_l.append(corners2_l)
        imgpoints_r.append(corners2_r)

        print(f" Colțuri detectate: {left_path} & {right_path}")
    else:
        print(f" Eroare colțuri: {left_path} / {right_path}")

ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, gray_l.shape[::-1], None, None)
ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, gray_r.shape[::-1], None, None)

flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    mtx_l, dist_l, mtx_r, dist_r,
    gray_l.shape[::-1], criteria=criteria, flags=flags
)

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r,
    gray_l.shape[::-1], R, T, alpha=0
)

os.makedirs("stereo_calib", exist_ok=True)
np.save("stereo_calib/stereo_mtx_l.npy", mtx_l)
np.save("stereo_calib/stereo_mtx_r.npy", mtx_r)
np.save("stereo_calib/stereo_dist_l.npy", dist_l)
np.save("stereo_calib/stereo_dist_r.npy", dist_r)
np.save("stereo_calib/R.npy", R)
np.save("stereo_calib/T.npy", T)
np.save("stereo_calib/Q.npy", Q)

print(" Calibrare stereo completă. Parametrii salvați în folderul stereo_calib/")
