import cv2
import os

output_dir_left = "stereo_calib/left"
output_dir_right = "stereo_calib/right"
os.makedirs(output_dir_left, exist_ok=True)
os.makedirs(output_dir_right, exist_ok=True)

cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("Eroare: una dintre camere nu funcționează.")
    exit()

count = 0
while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("Eroare la citirea cadrelor.")
        break

    both = cv2.hconcat([frame_left, frame_right])
    cv2.imshow("Stereo Cameras (Press SPACE to capture, ESC to exit)", both)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        left_img_name = os.path.join(output_dir_left, f"left_{count:02d}.jpg")
        right_img_name = os.path.join(output_dir_right, f"right_{count:02d}.jpg")
        cv2.imwrite(left_img_name, frame_left)
        cv2.imwrite(right_img_name, frame_right)
        print(f"Capturate: {left_img_name} și {right_img_name}")
        count += 1

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
