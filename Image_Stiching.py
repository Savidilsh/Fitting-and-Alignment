import cv2
import numpy as np

def stitch_images(img1_path, img5_path):
    img1 = cv2.imread(img1_path) # left image
    img5 = cv2.imread(img5_path) # right image (reference)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp5, des5 = sift.detectAndCompute(img5, None)

    # Use FLANN matcher and Lowe's ratio test for good matches
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = matcher.knnMatch(des1, des5, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp5[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography with OpenCV's RANSAC
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print("--- Computed Homography Matrix ---")
    print(H)

    # Stitch images
    h1, w1 = img1.shape[:2]
    h5, w5 = img5.shape[:2]
    result = cv2.warpPerspective(img1, H, (w1 + w5, h5))
    result[0:h5, 0:w5] = img5 # Place the reference image on the left

    cv2.imshow("Stitched Image", result)
    cv2.imwrite('stitching_result.png', result)
    print("Result saved as 'stitching_result.png'. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Make sure 'img1.ppm' and 'img5.ppm' are in the same directory
stitch_images('graf/img1.ppm', 'graf/img5.ppm')