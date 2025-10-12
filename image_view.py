# visualize_graf_images.py
import cv2
import matplotlib.pyplot as plt

# List of the image filenames in the graf folder
image_files = [f'graf/img{i}.ppm' for i in range(1, 7)]

# Create a figure and a set of subplots (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

print("Loading and displaying images...")

# Loop through the image files and the axes to display each image
for i, ax in enumerate(axes.flat):
    # Read the image
    img = cv2.imread(image_files[i])
    
    if img is not None:
        # Convert from BGR (OpenCV's default) to RGB for correct color in Matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display the image
        ax.imshow(img_rgb)
        ax.set_title(f'img{i+1}.ppm')
    else:
        ax.set_title(f'Failed to load img{i+1}.ppm')
        
    # Hide the x and y axes for a cleaner look
    ax.axis('off')

# Adjust layout to prevent titles from overlapping and show the plot
plt.tight_layout()
plt.show()