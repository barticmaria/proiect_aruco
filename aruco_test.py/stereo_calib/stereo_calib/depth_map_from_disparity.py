#aici am detectarea adancimei
import cv2
import numpy as np

# Încarcă matricea Q
Q = np.load("Q.npy")

# Încarcă disparity și verifică
disparity = cv2.imread("C:/Users/barti/PycharmProjects/aruco_test.py/stereo_calib/disparity_map.png", cv2.IMREAD_GRAYSCALE)
if disparity is None:
    raise FileNotFoundError("❌ disparity_map.png nu a fost găsit!")

disparity = np.float32(disparity)
disparity[disparity == 0.0] = 0.1
disparity = cv2.GaussianBlur(disparity, (5, 5), 0)

# Reproiectăm imaginea 3D
points_3D = cv2.reprojectImageTo3D(disparity, Q)
z_values = points_3D[:, :, 2]

print(f"📏 Interval valori Z: {np.min(z_values):.2f} cm → {np.max(z_values):.2f} cm")

# Normalizare pentru afișare
depth_map_norm = cv2.normalize(z_values, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
depth_map_norm = np.uint8(depth_map_norm)
color_map = cv2.applyColorMap(depth_map_norm, cv2.COLORMAP_JET)

# Funcție pentru click
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        Z = points_3D[y, x][2]
        print(f"📍 Adâncime la ({x}, {y}): {Z*100:.2f} cm")
        cv2.circle(color_map, (x, y), 5, (255, 255, 255), -1)
        cv2.imshow("Depth Map Interactiv", color_map)

# Afișare
cv2.imshow("Depth Map Interactiv", color_map)
cv2.setMouseCallback("Depth Map Interactiv", on_mouse)
cv2.imwrite("depth_map_interactiv.png", color_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
