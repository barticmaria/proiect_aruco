import cv2
import numpy as np
import os

folder = "calib_images"
pattern_size = (9, 6)
square_size = 1.0

objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

objpoints = []
imgpoints = []

images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
if not images:
    print(" Nu s-au găsit imagini în folderul 'calib_images'.")
    exit()

for fname in images:
    img_path = os.path.join(folder, fname)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
        cv2.imshow('Colțuri detectate', img)
        cv2.waitKey(300)
    else:
        print(f" Colțuri NEDETECTATE în: {fname}")

cv2.destroyAllWindows()

if len(objpoints) < 3:
    print("⚠ Prea puține imagini valide pentru calibrare.")
    exit()

# calibrare
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# salvare rezultate
np.save("cam_matrix.npy", camera_matrix)
np.save("dist_coeffs.npy", dist_coeffs)

print(" Calibrare finalizată.")
print("Matrice cameră:\n", camera_matrix)
print("Coeficienți de distorsiune:\n", dist_coeffs)


