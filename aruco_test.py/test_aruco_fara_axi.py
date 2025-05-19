import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cam_matrix = np.load("cam_matrix.npy")
dist_coeffs = np.load("dist_coeffs.npy")

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Nu pot citi de la cameră.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        print(f" Marker detectat cu ID-uri: {ids.ravel()}")

    cv2.imshow("Test ArUco fara axa", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
