import cv2
import numpy as np

points = []
def select_points_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(param['image'], (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Destination Image", param['image'])

def warp_image(destination_img_path, source_img_path):
    dest_img = cv2.imread(destination_img_path)
    source_img = cv2.imread(source_img_path)
    dest_clone = dest_img.copy()
    
    cv2.namedWindow("Destination Image")
    cv2.setMouseCallback("Destination Image", select_points_callback, {'image': dest_clone})
    print("Select 4 points clockwise, starting from top-left. Press 'q' to quit.")
    
    while len(points) < 4:
        cv2.imshow("Destination Image", dest_clone)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    if len(points) == 4:
        h, w = source_img.shape[:2]
        source_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
        dest_pts = np.array(points, dtype=np.float32)
        
        H, _ = cv2.findHomography(source_pts, dest_pts)
        warped = cv2.warpPerspective(source_img, H, (dest_img.shape[1], dest_img.shape[0]))
        
        mask = np.zeros_like(dest_img, dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(dest_pts)], (255, 255, 255))
        
        # Blend the warped image onto the destination
        result = cv2.bitwise_and(dest_img, cv2.bitwise_not(mask))
        result = cv2.add(result, cv2.bitwise_and(warped, mask))
        
        cv2.imshow("Final Result", result)
        cv2.imwrite('homography_result.png', result)
        print("Result saved as 'homography_result.png'. Press any key to exit.")
        cv2.waitKey(0)
    cv2.destroyAllWindows()

# Create dummy images or replace with your chosen files
cv2.imwrite('my_building.png', np.full((600, 800, 3), 200, np.uint8))
cv2.imwrite('my_logo.png', np.full((200, 400, 3), (255, 0, 0), np.uint8))
warp_image('my_building.png', 'my_logo.png')