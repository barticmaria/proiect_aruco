import tkinter as tk
import threading
import cv2
import numpy as np
import math

camera_matrix = np.load("cam_matrix.npy")
dist_left = np.load("dist_coeffs.npy")
dist_right = np.load("dist_coeffs1.npy")
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()

# Funcție de detectare generică (pentru 1 cameră)
def detect_aruco_single(camera_index, dist, window_name):
    cap = cv2.VideoCapture(camera_index)
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Eroare cameră {camera_index}")
            break
        corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist)
            for rvec, tvec, corner in zip(rvecs, tvecs, corners):
                cv2.aruco.drawAxis(frame, camera_matrix, dist, rvec, tvec, 0.03)
                pos = f"X:{tvec[0][0]:.2f} Y:{tvec[0][1]:.2f} Z:{tvec[0][2]:.2f}"
                x, y = int(corner[0][0][0]), int(corner[0][0][1])
                cv2.putText(frame, pos, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

# Funcție pentru camerele 0 și 1 + concatenare
def detect_aruco_both_concat():
    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(1)
    while True:
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()
        if not ret_l or not ret_r:
            print("Eroare camere.")
            break
        for frame, dist in [(frame_l, dist_left), (frame_r, dist_right)]:
            corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist)
                for rvec, tvec in zip(rvecs, tvecs):
                    cv2.aruco.drawAxis(frame, camera_matrix, dist, rvec, tvec, 0.03)

        try:
            stitched = np.hstack((frame_l, frame_r))
            cv2.imshow("Concat View", stitched)
        except:
            print("Eroare la concatenare.")

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()


def stitch_by_marker():
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

        # === Detectare ArUco pe camera stanga ===
        corners_l, ids_l, _ = cv2.aruco.detectMarkers(labeled_l, aruco_dict, parameters=parameters)
        if ids_l is not None:
            for i, marker_id in enumerate(ids_l.flatten()):
                if marker_id == 0:
                    cv2.aruco.drawDetectedMarkers(labeled_l, [corners_l[i]])
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers([corners_l[i]], 0.05, camera_matrix,
                                                                        dist_coeffs)
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

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

def monocular_vo():
    camera_matrix = np.load("cam_matrix.npy")
    dist_coeffs = np.load("dist_coeffs.npy")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Nu s-a deschis camera.")
        return

    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    traj = np.zeros((480, 480, 3), dtype=np.uint8)
    pos = np.array([240.0, 240.0])

    ret, prev_frame = cap.read()
    if not ret:
        print(" Eroare la primul cadru.")
        return

    prev_frame = cv2.undistort(prev_frame, camera_matrix, dist_coeffs)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)

        vis = frame.copy()

        if prev_des is not None and des is not None:
            matches = bf.match(prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:100]

            if len(good_matches) >= 8:
                pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                pts_curr = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                E, _ = cv2.findEssentialMat(pts_curr, pts_prev, camera_matrix, method=cv2.RANSAC, prob=0.999,
                                            threshold=1.0)
                if E is not None:
                    _, R, t, _ = cv2.recoverPose(E, pts_curr, pts_prev, camera_matrix)

                    dx, dy = t[0][0] * 100, t[2][0] * 100
                    pos += np.array([dx, dy])
                    cv2.circle(traj, tuple(pos.astype(int)), 2, (0, 255, 0), -1)

            vis = cv2.drawMatches(prev_gray, prev_kp, gray, kp, good_matches, None, flags=2)

        vis_resized = cv2.resize(vis, (640, 480))
        traj_resized = cv2.resize(traj, (640, 480))
        canvas = np.vstack((vis_resized, traj_resized))

        cv2.putText(canvas, "Feature Tracking + Visual Odometry", (10, 25), font, 0.7, (0, 255, 255), 2)

        cv2.imshow("🔍 VO Live - Corespondente + Traiectorie", canvas)

        prev_gray = gray.copy()
        prev_kp, prev_des = kp, des

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def run_threaded(func, *args):
    t = threading.Thread(target=func, args=args)
    t.start()

root = tk.Tk()
root.title("Meniu principal ArUco")
root.geometry("350x300")

tk.Label(root, text="Selecteaza o funcționalitate:", font=("Helvetica", 12)).pack(pady=10)

tk.Button(root, text="Detectare ArUco", width=30,
          command=lambda: run_threaded(detect_aruco_single, 0, dist_left, "Camera Stânga")).pack(pady=5)
tk.Button(root, text="ArUco + Inclinare + Traseu Live", width=30,
          command=lambda: run_threaded(stitch_by_marker)).pack(pady=5)
tk.Button(root, text="Monocular VO (cu calibrare)", width=30,
          command=lambda: run_threaded(monocular_vo)).pack(pady=5)
tk.Button(root, text="Iesire", width=30, command=root.destroy).pack(pady=20)

root.mainloop()
