import cv2
import numpy as np

nr_cols = 8
nr_rows = 5
dimensiune_tabla = (nr_cols, nr_rows)

img = cv2.imread("test.jpg")

if img is None:
    print("Imaginea nu a fost găsită. Verifică numele fișierului.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
ret, corners = cv2.findChessboardCorners(gray, dimensiune_tabla, flags)

if ret:
    print(" Colțuri detectate cu succes!")
    cv2.drawChessboardCorners(img, dimensiune_tabla, corners, ret)
else:
    print(" Colțuri nedetectate în imagine.")

cv2.imshow("Rezultat", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
