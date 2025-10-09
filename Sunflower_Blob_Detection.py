import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log

# Ensure 'the_berry_farms_sunflower_field.jpeg' is in the same directory
image_file = 'the_berry_farms_sunflower_field.jpeg'
img = cv2.imread(image_file, cv2.IMREAD_REDUCED_COLOR_4)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect blobs using Laplacian of Gaussian
blobs = blob_log(gray_img, min_sigma=10, max_sigma=40, num_sigma=10, threshold=0.1)
blobs[:, 2] = blobs[:, 2] * np.sqrt(2) # Convert sigma to radius

# --- Visualization ---
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
for blob in blobs:
    y, x, r = blob
    circle = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
    ax.add_patch(circle)
ax.set_axis_off()
plt.tight_layout()
plt.savefig('sunflower_detection.png', bbox_inches='tight')
plt.show()

# Report the parameters of the largest 5 circles
blobs_sorted = sorted(blobs, key=lambda b: b[2], reverse=True)
print("Top 5 Largest Sunflower Detections")
for i, blob in enumerate(blobs_sorted[:5]):
    y, x, r = map(int, blob)
    print(f"{i+1}. Center=(y:{y}, x:{x}), Radius={r}")