import cv2
import numpy as np
import os

pattern_size = (9, 6)
square_size = 0.024

left_folder = os.path.join(os.path.dirname(__file__), "left")
right_folder = os.path.join(os.path.dirname(__file__), "right")

objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints_left = []
imgpoints_right = []

left_images = sorted([f for f in os.listdir(left_folder) if f.endswith(".jpg")])
right_images = sorted([f for f in os.listdir(right_folder) if f.endswith(".jpg")])

if len(left_images) != len(right_images):
    print("Imagini lipsă sau nesincronizate.")
    exit()

for l_name, r_name in zip(left_images, right_images):
    img_l = cv2.imread(os.path.join(left_folder, l_name))
    img_r = cv2.imread(os.path.join(right_folder, r_name))

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size, None)

    if ret_l and ret_r:
        objpoints.append(objp)

        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1),
                                     criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1),
                                     criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        imgpoints_left.append(corners_l)
        imgpoints_right.append(corners_r)
    else:
        print(f"Colțuri NEDETECTATE în: {l_name} sau {r_name}")

ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, gray_l.shape[::-1], None, None)
ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, gray_r.shape[::-1], None, None)

flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtx_l, dist_l, mtx_r, dist_r, gray_l.shape[::-1],
    criteria=criteria, flags=flags
)

np.save("stereo_R.npy", R)
np.save("stereo_T.npy", T)
np.save("stereo_mtx_l.npy", mtx_l)
np.save("stereo_mtx_r.npy", mtx_r)
np.save("../stereo_dist_l.npy", dist_l)
np.save("stereo_dist_r.npy", dist_r)

print("\n✅ Calibrare stereo finalizată.")
print("R =", R)
print("T =", T)
