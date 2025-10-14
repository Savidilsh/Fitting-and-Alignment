import cv2 as cv
import numpy as np
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Resolve project paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_DIR = ROOT / 'graf'
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)


def stitch_images(img1, img5, H):
    """Stitch img1 onto img5 using homography H (img1 -> img5)."""
    h1, w1 = img1.shape[:2]
    h5, w5 = img5.shape[:2]

    # corners of img1
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners1_transformed = cv.perspectiveTransform(corners1, H)

    # corners of img5
    corners5 = np.float32([[0, 0], [w5, 0], [w5, h5], [0, h5]]).reshape(-1, 1, 2)
    all_corners = np.concatenate([corners5, corners1_transformed], axis=0)

    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # translation
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    output_size = (x_max - x_min, y_max - y_min)
    warped_img1 = cv.warpPerspective(img1, translation @ H, output_size)

    # create result and paste img5
    result = warped_img1.copy()
    tx, ty = -x_min, -y_min
    result[ty:ty+h5, tx:tx+w5] = img5

    return result


def run_stitch(imgA_path, imgB_path, save_prefix='q4'):
    imgA = cv.imread(str(imgA_path))
    imgB = cv.imread(str(imgB_path))
    if imgA is None or imgB is None:
        raise FileNotFoundError(f"Could not read images: {imgA_path}, {imgB_path}")

    grayA = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)
    grayB = cv.cvtColor(imgB, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kpA, desA = sift.detectAndCompute(grayA, None)
    kpB, desB = sift.detectAndCompute(grayB, None)

    # save keypoint visualisations
    cv.imwrite(str(FIG_DIR / f"{save_prefix}_keypoints_img1.png"), cv.drawKeypoints(imgA, kpA, None))
    cv.imwrite(str(FIG_DIR / f"{save_prefix}_keypoints_img2.png"), cv.drawKeypoints(imgB, kpB, None))

    # matching
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(desA, desB, k=2)

    # draw initial few knn matches for inspection
    knn_for_draw = [m for m in knn[:50]]
    all_matches_img = cv.drawMatchesKnn(imgA, kpA, imgB, kpB, knn_for_draw, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(str(FIG_DIR / f"{save_prefix}_initial_matches.png"), all_matches_img)

    # ratio test
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    good_img = cv.drawMatches(imgA, kpA, imgB, kpB, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(str(FIG_DIR / f"{save_prefix}_good_matches.png"), good_img)

    if len(good) < 4:
        raise RuntimeError("Not enough good matches to compute homography")

    src_pts = np.float32([kpA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("cv.findHomography failed to find a homography")

    # stitch
    result = stitch_images(imgA, imgB, H)
    cv.imwrite(str(FIG_DIR / f"{save_prefix}_stitched_panorama.png"), result)
    cv.imwrite(str(ROOT / f"stitched_image.jpg"), result)

    print(f"Saved figures to {FIG_DIR} and stitched image to {ROOT / 'stitched_image.jpg'}")

    # optionally show with matplotlib
    try:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Stitched panorama')
        plt.tight_layout()
        plt.show()
    except Exception:
        pass


if __name__ == '__main__':
    img1_path = DATA_DIR / 'img1.ppm'
    img5_path = DATA_DIR / 'img5.ppm'
    run_stitch(img1_path, img5_path)
