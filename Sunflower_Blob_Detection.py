# q1_sunflower_blob.py

import cv2
import numpy as np
from skimage.feature import blob_log
import matplotlib.pyplot as plt

def detect_sunflower_centers(img_path, out_path='figures/q1_sunflower.png'):
    img = cv2.imread(img_path, cv2.IMREAD_REDUCED_COLOR_4)
    if img is None:
        raise FileNotFoundError(f"Image {img_path} not found")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    
    blobs = blob_log(inv, min_sigma=3, max_sigma=12, num_sigma=30, threshold=0.3)
    
    h = gray.shape[0]
    # Filter to lower part of image (field area)
    blobs2 = [b for b in blobs if b[0] > h * 0.4]
    blobs2.sort(key=lambda b: b[2], reverse=True)
    top = blobs2[:10]
    
    draw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for (y, x, s) in top:
        r = int(s * np.sqrt(2))
        cv2.circle(draw, (int(x), int(y)), r, (0, 0, 255), 1)
    
    cv2.imwrite(out_path, draw)
    print("Saved:", out_path)
    
    # also show via matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Detected Sunflowers (top 10)")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    detect_sunflower_centers('the_berry_farms_sunflower_field.jpeg')
