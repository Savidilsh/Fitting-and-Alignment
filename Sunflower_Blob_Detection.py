# q1_sunflowers.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log

def detect_and_visualize_sunflowers(image_path):
    img_color = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_4)
    gray_img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    # Use blob_log which is an efficient implementation of Laplacian of Gaussian
    blobs = blob_log(gray_img, min_sigma=10, max_sigma=35, num_sigma=10, threshold=0.08)
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2) # Convert sigma to radius

    # For visualization, create a BGR image from grayscale to draw color circles
    img_for_drawing = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    for blob in blobs:
        y, x, r = map(int, blob)
        cv2.circle(img_for_drawing, (x, y), r, (0, 0, 255), 2) # Draw red circle

    cv2.imwrite('sunflower_detection_result.png', img_for_drawing)
    print("Result saved as 'sunflower_detection_result.png'")
    
    # Report the parameters of the largest 5 circles
    blobs_sorted = sorted(blobs, key=lambda b: b[2], reverse=True)
    print("\n--- Top 5 Largest Sunflower Detections ---")
    print("Sigma range used: 10 to 35")
    for i, blob in enumerate(blobs_sorted[:5]):
        y, x, r = map(int, blob)
        print(f"{i+1}. Center=(y:{y}, x:{x}), Radius={r}")
        
    cv2.imshow('Sunflower Detection', img_for_drawing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_and_visualize_sunflowers('the_berry_farms_sunflower_field.jpeg')