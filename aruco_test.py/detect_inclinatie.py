import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

camera_matrix = np.load("cam_matrix.npy")
dist_coeffs = np.load("dist_coeffs.npy")

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

        for rvec, tvec in zip(rvecs, tvecs):
            cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

            R, _ = cv2.Rodrigues(rvec)

            pitch_rad = math.atan2(-R[2, 1], R[2, 2])
            pitch_deg = math.degrees(pitch_rad)

            cv2.putText(frame, f"Inclinatie: {pitch_deg:.2f} deg", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            print(f"Inclinatie estimata: {pitch_deg:.2f} grade")

    cv2.imshow("Detectie Inclinatie", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
