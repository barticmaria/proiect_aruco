import cv2
import numpy as np

cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

mtx_l = np.load("stereo_calib/stereo_mtx_l.npy")
dist_l = np.load("stereo_calib/stereo_dist_l.npy")
mtx_r = np.load("stereo_calib/stereo_mtx_r.npy")
dist_r = np.load("stereo_calib/stereo_dist_r.npy")

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()

while True:
    ret_l, frame_l = cap_left.read()
    ret_r, frame_r = cap_right.read()

    if not ret_l or not ret_r:
        break

    for frame, mtx, dist, name in [(frame_l, mtx_l, dist_l, "Left"), (frame_r, mtx_r, dist_r, "Right")]:
        corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        if ids is not None:
            print(f"[{name}] Marker ID-uri detectate:", ids.flatten())
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.04, mtx, dist)
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.aruco.drawAxis(frame, mtx, dist, rvec, tvec, 0.03)
        else:
            print(f"[{name}] Niciun marker detectat.")
        cv2.imshow(name, frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
mtx_l = np.load("cam_matrix.npy")
dist_l = np.load("dist_coeffs.npy")
mtx_r = mtx_l.copy()
dist_r = dist_l.copy()

