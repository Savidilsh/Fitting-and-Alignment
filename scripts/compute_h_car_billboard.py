import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
img_src = cv2.imread(str(ROOT / 'images' / 'car_ad.png'))
img_dst = cv2.imread(str(ROOT / 'car_on_billboard.png'))

if img_src is None or img_dst is None:
    print('Error: source or destination image not found')
    raise SystemExit(1)

gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray_src, None)
kp2, des2 = sift.detectAndCompute(gray_dst, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

if len(good) < 4:
    print('Not enough matches found')
    raise SystemExit(2)

src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
if H is None:
    print('Homography estimation failed')
    raise SystemExit(3)

np.set_printoptions(precision=6, suppress=True)
out = ROOT / 'figures' / 'H_car_billboard.txt'
out.write_text('\n'.join(' '.join(f"{v:.6f}" for v in row) for row in H.tolist()))
print('Homography matrix H:')
print(H)
print(f'Saved to {out}')
