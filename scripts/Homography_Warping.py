# Homography_Warping.py
import cv2
import numpy as np

points = []

def select_points_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        # Draw a circle on the clicked point for visual feedback
        cv2.circle(param['image'], (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4 Points", param['image'])

def warp_image(dest_img_path, src_img_path, output_filename):
    global points
    points = [] # Reset points for each new run
    
    dest_img = cv2.imread(dest_img_path)
    src_img = cv2.imread(src_img_path)
    
    if dest_img is None or src_img is None:
        print(f"Error: Could not load images. Check paths: {dest_img_path}, {src_img_path}")
        return

    dest_clone = dest_img.copy()
    
    cv2.namedWindow("Select 4 Points")
    cv2.setMouseCallback("Select 4 Points", select_points_callback, {'image': dest_clone})
    print(f"Click 4 points on the destination image ('{dest_img_path}') clockwise.")
    
    while len(points) < 4:
        cv2.imshow("Select 4 Points", dest_clone)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            cv2.destroyAllWindows()
            return

    if len(points) == 4:
        h, w = src_img.shape[:2]
        src_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
        
        dest_pts = np.array(points, dtype=np.float32)
        
        H, _ = cv2.findHomography(src_pts, dest_pts)
        warped = cv2.warpPerspective(src_img, H, (dest_img.shape[1], dest_img.shape[0]))
        
        mask = np.zeros_like(dest_img, dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(dest_pts)], (255, 255, 255))
        mask_inv = cv2.bitwise_not(mask)
        
        dest_bg = cv2.bitwise_and(dest_img, mask_inv)
        warped_fg = cv2.bitwise_and(warped, mask)
        
        result = cv2.addWeighted(dest_bg, 1, warped_fg, 0.9, 0)

        cv2.imwrite(output_filename, result)
        print(f"Success! Result saved as '{output_filename}'.")
        cv2.imshow("Final Result", result)    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    # Example 1: Warp the car ad onto the billboard
    warp_image(
        dest_img_path='images/bill_board.png',
        src_img_path='images/car_ad.png',
        output_filename='car_on_billboard.png'
    )
    
    # Example 2: Warp the dog image onto the TV screen
    warp_image(
        dest_img_path='images/tv.png',
        src_img_path='images/dog.png',
        output_filename='dog_on_tv.png'
    )