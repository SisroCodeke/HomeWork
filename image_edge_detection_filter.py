import cv2
import numpy as np

def apply_linear_filter(image_path, filter_matrix, output_path='filtered_image.jpg'):
    """
    Apply a linear filter to an image and display original + filtered images side by side.
    
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
    # Skip normalization for edge detection filters
    if abs(kernel.sum()) > 0 and not (kernel.sum() == 0 or np.any(kernel < 0)):
        kernel = kernel / kernel.sum()
    
    # Apply the filter
    filtered_image = cv2.filter2D(image, -1, kernel)
    
    # Create side-by-side comparison
    comparison = np.hstack((image, filtered_image))
    
    # Save the filtered image
    cv2.imwrite(output_path, comparison)
    
    # Display the comparison
    #cv2.imshow('Original (left) vs Filtered (right)', comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Filtered image saved to {output_path}")

# Hardcoded filters
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

sobel_x = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]

sobel_y = [
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
]

laplacian = [
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
]

# Example usage
if __name__ == "__main__":
    input_image = '/home/panzer/Desktop/Desk/Work/sampleg.jpg'  # Change this to your image path
    
    # Apply filters
    print("Applying 3x3 box blur filter...")
    apply_linear_filter(input_image, box_blur_3x3, 'box_blur_output.jpg')
    
    print("\nApplying 3x3 Gaussian blur filter...")
    apply_linear_filter(input_image, gaussian_blur_3x3, 'gaussian_output.jpg')
    
    print("\nApplying Sobel X filter (horizontal edges)...")
    apply_linear_filter(input_image, sobel_x, 'sobel_x_output.jpg')
    
    print("\nApplying Sobel Y filter (vertical edges)...")
    apply_linear_filter(input_image, sobel_y, 'sobel_y_output.jpg')
    
    print("\nApplying Laplacian edge detection filter...")
    apply_linear_filter(input_image, laplacian, 'laplacian_output.jpg')
