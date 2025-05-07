import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_canny_filter(image_path, low_threshold=30, high_threshold=100):
    """
    Applies Canny edge detection to an image and displays the result.
    
    Parameters:
    - image_path: Path to the input image
    - low_threshold: Lower threshold for hysteresis procedure
    - high_threshold: Upper threshold for hysteresis procedure
    """
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Apply Gaussian blur to reduce noise
    img_blur = cv2.GaussianBlur(img, (5, 5),0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(img_blur, low_threshold, high_threshold)
    
    # Display the original and edge-detected images
    plt.figure(figsize=(10, 5))
    
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge Detection'), plt.xticks([]), plt.yticks([])
    
    plt.show()

# Hardcoded image path (replace with your image path)
IMAGE_PATH = '/home/panzer/Desktop/Desk/Work/Chairs.jpg'  # Change this to your image file

# Apply Canny filter
apply_canny_filter(IMAGE_PATH)


