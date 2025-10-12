# q3_homography.py
import cv2
import numpy as np

points = []
def select_points_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(param['image'], (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4 Points", param['image'])

def warp_image(dest_img_path, src_img_path, output_filename):
    global points
    points = []
    
    dest_img = cv2.imread(dest_img_path)
    src_img = cv2.imread(src_img_path)
    dest_clone = dest_img.copy()
    
    cv2.namedWindow("Select 4 Points")
    cv2.setMouseCallback("Select 4 Points", select_points_callback, {'image': dest_clone})
    print(f"Working on {output_filename}: Select 4 points clockwise, starting from top-left.")
    
    while len(points) < 4:
        cv2.imshow("Select 4 Points", dest_clone)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    if len(points) == 4:
        h, w = src_img.shape[:2]
        src_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
        dest_pts = np.array(points, dtype=np.float32)
        
        H, _ = cv2.findHomography(src_pts, dest_pts)
        warped = cv2.warpPerspective(src_img, H, (dest_img.shape[1], dest_img.shape[0]))
        
        # Create a mask to only blend where the warped image is
        mask = np.zeros_like(dest_img, dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(dest_pts)], (255, 255, 255))
        mask_inv = cv2.bitwise_not(mask)
        
        # Black out the area in the destination image
        dest_bg = cv2.bitwise_and(dest_img, mask_inv)
        # Get only the warped region
        warped_fg = cv2.bitwise_and(warped, mask)
        
        # Blend using addWeighted for a semi-transparent effect
        # result = cv2.add(dest_bg, warped_fg) # For solid overlay
        result = cv2.addWeighted(dest_bg, 1, warped_fg, 0.8, 0)

        cv2.imwrite(output_filename, result)
        print(f"Result saved as '{output_filename}'.")
        cv2.imshow("Final Result", result)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # You need to find your own images for this
    # Example 1
    # warp_image('path/to/building.jpg', 'path/to/flag.png', 'homography_result_1.png')
    
    # Example 2
    # warp_image('path/to/billboard.jpg', 'path/to/poster.png', 'homography_result_2.png')
    print("Please replace placeholder paths and run the script for each example.")