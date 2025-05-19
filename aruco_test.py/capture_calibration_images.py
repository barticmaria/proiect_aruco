import cv2
import numpy as np
import glob

nr_cols = 8
nr_rows = 5
dimensiune_tabla = (nr_cols, nr_rows)

objp = np.zeros((nr_rows * nr_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:nr_cols, 0:nr_rows].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob('calib_images/*.jpg')

if len(images) == 0:
    print("Nu s-au găsit imagini în folderul calib_images.")
    exit()

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, dimensiune_tabla, flags)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, dimensiune_tabla, corners2, ret)
        cv2.imshow('✔ Colțuri detectate', img)
        cv2.waitKey(1000)
    else:
        print(f" Colțuri nedetectate în: {fname}")
        cv2.imshow('⚠️ Imagine nerecunoscută', gray)
        cv2.waitKey(500)

cv2.destroyAllWindows()

if len(objpoints) == 0:
    print(" Nu s-au detectat colțuri în nicio imagine.")
    exit()

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

np.save("cam_matrix.npy", camera_matrix)
np.save("dist_coeffs.npy", dist_coeffs)

print("\n Calibrare finalizată!")
print("Matrice cameră:\n", camera_matrix)
print("\n Coeficienți de distorsiune:\n", dist_coeffs)
cv2.imshow("Debug imagine", img)
cv2.waitKey(1000)
