import cv2
import numpy as np
import math

cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

camera_matrix = np.load("cam_matrix.npy")
dist_coeffs = np.load("dist_coeffs.npy")

prev_gray = None
traj = np.zeros((300, 800, 3), dtype=np.uint8)
position = np.array([400, 150], dtype=np.float32)
last_position = position.copy()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()

font = cv2.FONT_HERSHEY_SIMPLEX
inclinatie_text = "Inclinatie: -"
coord_text = ""

while True:
    ret_l, frame_l = cap_left.read()
    ret_r, frame_r = cap_right.read()
    if not ret_l or not ret_r:
        break

    frame_l_small = cv2.resize(frame_l, (400, 300))
    frame_r_small = cv2.resize(frame_r, (400, 300))
    labeled_l = frame_l_small.copy()
    labeled_r = frame_r_small.copy()

    cv2.putText(labeled_l, "CAMERA STANGA", (10, 25), font, 0.6, (255, 255, 255), 2)
    cv2.putText(labeled_r, "CAMERA DREAPTA", (10, 25), font, 0.6, (255, 255, 255), 2)

    corners_l, ids_l, _ = cv2.aruco.detectMarkers(labeled_l, aruco_dict, parameters=parameters)
    if ids_l is not None:
        for i, marker_id in enumerate(ids_l.flatten()):
            if marker_id == 0:
                cv2.aruco.drawDetectedMarkers(labeled_l, [corners_l[i]])
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers([corners_l[i]], 0.05, camera_matrix, dist_coeffs)
                cv2.aruco.drawAxis(labeled_l, camera_matrix, dist_coeffs, rvec[0], tvec[0], 0.03)

                R, _ = cv2.Rodrigues(rvec[0])
                pitch_rad = math.atan2(-R[2, 1], R[2, 2])
                pitch_deg = math.degrees(pitch_rad)
                inclinatie_text = f"Inclinatie: {pitch_deg:.2f} deg"

                x, y, z = tvec[0][0]
                coord_text = f"X:{x:.2f} Y:{y:.2f} Z:{z:.2f}"
                cv2.putText(labeled_l, coord_text, (10, 270), font, 0.6, (0, 255, 255), 2)

    corners_r, ids_r, _ = cv2.aruco.detectMarkers(labeled_r, aruco_dict, parameters=parameters)
    if ids_r is not None:
        for i, marker_id in enumerate(ids_r.flatten()):
            cv2.aruco.drawDetectedMarkers(labeled_r, [corners_r[i]])
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers([corners_r[i]], 0.05, camera_matrix, dist_coeffs)
            cv2.aruco.drawAxis(labeled_r, camera_matrix, dist_coeffs, rvec[0], tvec[0], 0.03)

    gray = cv2.cvtColor(frame_l_small, cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        prev_gray = gray
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
    else:
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
        good_new = next_pts[status == 1]
        good_old = prev_pts[status == 1]

        movement = good_new - good_old
        avg_movement = np.mean(movement, axis=0)
        last_position = position.copy()
        position += avg_movement[::-1] * 0.5

        speed = np.linalg.norm(avg_movement)
        if speed < 1:
            color = (0, 255, 0)
        elif speed < 3:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        p1 = (int(last_position[0]), int(last_position[1]))
        p2 = (int(position[0]), int(position[1]))
        cv2.arrowedLine(traj, p1, p2, color, 2)

        prev_gray = gray.copy()
        prev_pts = good_new.reshape(-1, 1, 2)

    traj_labeled = traj.copy()
    cv2.putText(traj_labeled, "TRASEU CAMERA (Visual Odometry)", (10, 25), font, 0.6, (0, 255, 0), 2)
    cv2.putText(traj_labeled, inclinatie_text, (10, 50), font, 0.6, (0, 255, 255), 2)

    top_row = np.hstack((labeled_l, labeled_r))
    full_view = np.vstack((top_row, traj_labeled))

    cv2.imshow("Colaj Vizual Final cu Sageata", full_view)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.imwrite("traseu_camera.png", traj)
print(" Harta salvată cu săgeți de direcție și înclinare.")

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
