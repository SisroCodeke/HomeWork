import cv2
import numpy as np

def apply_linear_filter(image_path, filter_matrix, output_path='filtered_image.jpg'):
    """
    Apply a smoothing linear filter to an image.
    
    Args:
        image_path (str): Path to the input image
        filter_matrix (list of lists): The filter kernel matrix
        output_path (str): Path to save the filtered image
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Convert filter matrix to numpy array
    kernel = np.array(filter_matrix, dtype=np.float32)
    
    # Normalize the kernel if it doesn't sum to 1 (for smoothing filters)
    if kernel.sum() != 0:
        kernel = kernel / kernel.sum()
    
    # Apply the filter to each color channel separately
    filtered_image = cv2.filter2D(image, -1, kernel)
    
    # Save the filtered image
    cv2.imwrite(output_path, filtered_image)
    print(f"Filtered image saved to {output_path}")

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

# Example usage
if __name__ == "__main__":
    input_image = 'input.jpg'  # Change this to your image path
    output_image = 'output.jpg'
    
    # Apply box blur filter
    print("Applying 3x3 box blur filter...")
    apply_linear_filter(input_image, box_blur_3x3, output_image)
    
    # Apply Gaussian blur filter
    print("\nApplying 3x3 Gaussian blur filter...")
    apply_linear_filter(input_image, gaussian_blur_3x3, 'gaussian_output.jpg')
    
    
    
    
