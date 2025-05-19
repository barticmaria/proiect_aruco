import cv2
import numpy as np

try:
    camera_matrix = np.load("cam_matrix.npy")
    dist_coeffs = np.load("dist_coeffs.npy")
except:
    print("⚠️ Nu ai calibrare, folosesc valori default.")
    camera_matrix = np.array([[800, 0, 320],
                              [0, 800, 240],
                              [0,   0,   1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1))

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

marker_length = 0.05

orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

prev_gray = None
prev_kp = None
prev_des = None
trajectory = np.eye(4)

def draw_cube(frame, rvec, tvec):
    axis = np.float32([
        [0, 0, 0], [0, marker_length, 0], [marker_length, marker_length, 0], [marker_length, 0, 0],
        [0, 0, -marker_length], [0, marker_length, -marker_length], [marker_length, marker_length, -marker_length], [marker_length, 0, -marker_length]
    ])
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.astype(int).reshape(-1, 2)

    frame = cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 2)
    for i in range(4):
        frame = cv2.line(frame, imgpts[i], imgpts[i + 4], (255, 0, 0), 2)
    frame = cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 2)

    return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp, des = orb.detectAndCompute(gray, None)
    if prev_gray is not None and prev_des is not None:
        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) > 8:
            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

            E, _ = cv2.findEssentialMat(pts1, pts2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, pts1, pts2, camera_matrix)

                cv2.putText(frame, "Odometry vector t: [{:.2f}, {:.2f}, {:.2f}]".format(t[0][0], t[1][0], t[2][0]),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    prev_gray = gray
    prev_kp = kp
    prev_des = des

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
        for rvec, tvec in zip(rvecs, tvecs):
            frame = draw_cube(frame, rvec, tvec)

    cv2.imshow("ArUco + Odometry", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()




