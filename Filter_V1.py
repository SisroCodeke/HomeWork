
    
import cv2
import numpy as np

def apply_custom_filter(image, kernel):
    """
    Apply a custom filter to an image without using cv2.filter2D or similar built-in functions.
    
    Args:
        image (numpy.ndarray): Input image as a NumPy array
        kernel (list of lists): Filter kernel matrix
        
    Returns:
        numpy.ndarray: Filtered image
    """
    # Convert kernel to numpy array and normalize
    kernel = np.array(kernel, dtype=np.float32)
    if kernel.sum() != 0:
        kernel = kernel / kernel.sum()
    
    # Get image dimensions and kernel dimensions
    img_height, img_width = image.shape[:2]
    kernel_height, kernel_width = kernel.shape
    
    # Calculate padding needed
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2
    
    # Create output image
    filtered_image = np.zeros_like(image)
    
    # Pad the image to handle borders
    padded_image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
    
    # Apply filter to each channel if image is color
    if len(image.shape) == 3:
        for channel in range(image.shape[2]):
            for y in range(img_height):
                for x in range(img_width):
                    # Extract the region of interest
                    region = padded_image[y:y+kernel_height, x:x+kernel_width, channel]
                    # Apply the kernel
                    filtered_value = np.sum(region * kernel)
                    filtered_image[y, x, channel] = filtered_value
    else:  # Grayscale image
        for y in range(img_height):
            for x in range(img_width):
                region = padded_image[y:y+kernel_height, x:x+kernel_width]
                filtered_value = np.sum(region * kernel)
                filtered_image[y, x] = filtered_value
    
    return filtered_image

# Hardcoded smoothing filters
box_blur_3x3 = [
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]

gaussian_blur_3x3 = [
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]

def main():
    # Load image
    input_path = '/home/panzer/Desktop/DESK/CV/samplec.jpg'
    output_path_box = '/home/panzer/Desktop/DESK/CV/output_box_blur.jpg'
    output_path_gaussian = '/home/panzer/Desktop/DESK/CV/output_gaussian_blur.jpg'
    
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not read image from {input_path}")
        return
    
    # Apply box blur
    print("Applying custom 3x3 box blur...")
    box_blurred = apply_custom_filter(image, box_blur_3x3)
    cv2.imwrite(output_path_box, box_blurred)
    print(f"Box blur result saved to {output_path_box}")
    
    # Apply Gaussian blur
    print("\nApplying custom 3x3 Gaussian blur...")
    gaussian_blurred = apply_custom_filter(image, gaussian_blur_3x3)
    cv2.imwrite(output_path_gaussian, gaussian_blurred)
    print(f"Gaussian blur result saved to {output_path_gaussian}")

if __name__ == "__main__":
    main()
