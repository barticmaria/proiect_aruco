
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

prev_gray = None
traj = np.zeros((600, 800, 3), dtype=np.uint8)  # hartă de traseu

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
        continue

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)

    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    for (x1, y1), (x2, y2) in zip(good_old, good_new):
        frame = cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(x2), int(y2)), 3, (0, 0, 255), -1)

    prev_gray = gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)

    cv2.imshow("Optical Flow", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()