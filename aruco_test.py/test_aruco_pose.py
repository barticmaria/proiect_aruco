import cv2
import numpy as np

camera_matrix = np.load("cam_matrix.npy")
dist_coeffs = np.load("dist_coeffs.npy")

marker_length = 4.0

cap = cv2.VideoCapture(0)

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

        for rvec, tvec in zip(rvecs, tvecs):
            cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length)

    cv2.imshow("Aruco Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
