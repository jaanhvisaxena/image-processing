import cv2
import os
import numpy as np

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])  # Sharpening kernel
    sharpened = cv2.filter2D(image, -1, kernel)  # Apply filter
    return sharpened

# Directory containing the images
image_directory = 'train_images'
output_directory = 'train_images_sharpened'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Sharpen all images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_directory, filename)
        img = cv2.imread(image_path)  # Load image

        if img is not None:
            sharpened_img = sharpen_image(img)  # Sharpen image

            # Save the sharpened image in the output directory
            output_path = os.path.join(output_directory, filename)
            cv2.imwrite(output_path, sharpened_img)
            print(f"Processed {filename}")
        else:
            print(f"Could not open {filename}")

