import cv2
import numpy as np

cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

camera_matrix = np.load("cam_matrix.npy")
dist_left = np.load("dist_coeffs.npy")
dist_right = np.load("dist_coeffs1.npy")

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()

while True:
    ret_l, frame_l = cap_left.read()
    ret_r, frame_r = cap_right.read()

    if not ret_l or not ret_r:
        print(" Eroare la citirea camerelor.")
        break

    for frame, dist, name in [(frame_l, dist_left, "Left"), (frame_r, dist_right, "Right")]:
        corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        if ids is not None:
            print(f"[{name}] Marker ID-uri detectate:", ids.flatten())
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist)
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.aruco.drawAxis(frame, camera_matrix, dist, rvec, tvec, 0.03)
        else:
            print(f"[{name}] Niciun marker detectat.")

    cv2.imshow("Left", frame_l)
    cv2.imshow("Right", frame_r)

    try:
        stitched = np.hstack((frame_l, frame_r))
        cv2.imshow("Concat View", stitched)
    except:
        print("⚠️ Eroare la concatenarea cadrelor.")

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
