
import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_custom_filters(image_path, output_path, log_threshold=0):
    """
    Apply custom filters to an image and display results.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the processed image
        log_threshold: Threshold value for LOG display only (0 for automatic)
    """
    # Define the kernels
    log_kernel = np.array([
        [-1,  -1, -1],
        [-1, 8, -1],
        [-1,  -1, -1]
    ])
    
    sharpen_kernel = np.array([
        [-1,  -1, -1],
        [-1, 9, -1],
        [-1,  -1, -1]
    ]) 
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur first (sigma=1.0)
    blurred = cv2.GaussianBlur(gray, (3, 3), 1.0)
    
    # Apply LOG kernel for edge detection
    log_filtered = cv2.filter2D(gray, -1, log_kernel)
    log_abs = cv2.convertScaleAbs(log_filtered)
    
    # Create a version of the LOG result for display (with thresholding)
    if log_threshold == 0:
        _, edges_display = cv2.threshold(log_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, edges_display = cv2.threshold(log_abs, log_threshold, 255, cv2.THRESH_BINARY)
    
    # Convert LOG result to 3-channel and normalize for adding to original image
    log_colored = cv2.cvtColor(log_abs, cv2.COLOR_GRAY2BGR)
    
    # Combine original image with LOG result (without thresholding)
    combined_edges = cv2.addWeighted(image, 1.0, log_colored, 0.5, 0)
    
    # Apply sharpen kernel to original image
    sharpened = cv2.filter2D(image, -1, sharpen_kernel)
    #sharpened = cv2.convertScaleAbs(sharpened)
    
    # Convert images for display (BGR to RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    combined_rgb = cv2.cvtColor(combined_edges, cv2.COLOR_BGR2RGB)
    sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
    
    # Create matplotlib figure
    plt.figure(figsize=(20, 15))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # LOG filtered image
    plt.subplot(2, 3, 2)
    plt.imshow(log_abs, cmap='gray')
    plt.title('3x3 LOG Filtered')
    plt.axis('off')
    
    # Thresholded edges (for display only)
    plt.subplot(2, 3, 3)
    plt.imshow(edges_display, cmap='gray')
    plt.title('Thresholded Edges (Display Only)')
    plt.axis('off')
    
    # Combined with LOG result (no thresholding)
    plt.subplot(2, 3, 4)
    plt.imshow(combined_rgb)
    plt.title('Original with LOG Result')
    plt.axis('off')
    
    # Sharpened image
    plt.subplot(2, 3, 5)
    plt.imshow(sharpened_rgb)
    plt.title('Sharpened Image')
    plt.axis('off')
    
    # Combined sharpened with LOG result (no thresholding)
    combined_sharp_edges = cv2.addWeighted(sharpened, 1.0, log_colored, 0.5, 0)
    combined_sharp_edges_rgb = cv2.cvtColor(combined_sharp_edges, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 3, 6)
    plt.imshow(combined_sharp_edges_rgb)
    plt.title('Sharpened with LOG Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save the combined image with LOG result (no thresholding)
    cv2.imwrite(output_path, combined_edges)
    print(f"Processed image with LOG result saved to {output_path}")

# Hard-coded image path
input_image = "/home/panzer/Desktop/Desk/Uni/sampleg.jpg"
output_image = "/home/panzer/Desktop/Desk/Uni/season_8_result.jpg"

# Apply custom filters
apply_custom_filters(input_image, output_image, log_threshold=120)
