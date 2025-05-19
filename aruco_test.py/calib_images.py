import cv2
import os

folder = "calib_images"
os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Camera nu a putut fi pornită.")
    exit()

count = 0
print(" Apasă tasta SPACE pentru a salva imagine. ESC pentru a ieși.")

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Nu se poate citi de la cameră.")
        break

    cv2.imshow("Captura calibrare", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == 32:
        img_name = os.path.join(folder, f"img_{count:02d}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Imagine salvată: {img_name}")
        count += 1

cap.release()
cv2.destroyAllWindows()
