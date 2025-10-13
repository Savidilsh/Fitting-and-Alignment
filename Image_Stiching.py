import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def sift_pipeline(img1_path, img2_path):
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    cv.imwrite('figures/q4_keypoints_img1.png', cv.drawKeypoints(img1, kp1, None))
    cv.imwrite('figures/q4_keypoints_img2.png', cv.drawKeypoints(img2, kp2, None))

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    matches_all = bf.knnMatch(des1, des2, k=2)
    all_matches_img = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches_all[:50], None, flags=2)
    cv.imwrite('figures/q4_initial_matches.png', all_matches_img)

    good = [m for m, n in matches_all if m.distance < 0.75 * n.distance]
    good_img = cv.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
    cv.imwrite('figures/q4_good_matches.png', good_img)

    if len(good) > 4:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv.findHomography(src, dst, cv.RANSAC, 5.0)
        result = cv.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
        result[0:img2.shape[0], 0:img2.shape[1]] = img2
        cv.imwrite('figures/q4_stitched_panorama.png', result)

    print("✅ All Q4 figures saved in 'figures/' folder")

if __name__ == "__main__":
    sift_pipeline('graf/img1.ppm', 'graf/img5.ppm')
