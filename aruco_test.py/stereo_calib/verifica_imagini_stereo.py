import os
import cv2

left_folder = "stereo_calib/left"
right_folder = "stereo_calib/right"

left_images = sorted([f for f in os.listdir(left_folder) if f.endswith(".jpg")])
right_images = sorted([f for f in os.listdir(right_folder) if f.endswith(".jpg")])

print(f"Left images ({len(left_images)}):", left_images)
print(f"Right images ({len(right_images)}):", right_images)

ok = True
for l_name, r_name in zip(left_images, right_images):
    if not l_name.split('_')[-1] == r_name.split('_')[-1]:
        print(f" Mismatch: {l_name} ≠ {r_name}")
        ok = False
        continue

    img_l = cv2.imread(os.path.join(left_folder, l_name))
    img_r = cv2.imread(os.path.join(right_folder, r_name))

    if img_l is None:
        print(f" NU s-a putut citi: {l_name}")
        ok = False
    if img_r is None:
        print(f" NU s-a putut citi: {r_name}")
        ok = False

if ok:
    print(" Toate imaginile sunt perechi valide și pot fi citite.")
else:
    print(" Probleme detectate (vezi mai sus).")
