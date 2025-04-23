# Import OpenCV library for image processing
import cv2
# Import NumPy for numerical operations
import numpy as np
# Import matplotlib for displaying images and histograms
import matplotlib.pyplot as plt

# Define a class for Look-Up Table (LUT) based image processing
class LUT:
    # Initialize the class with an image path
    def __init__(self, image_path):
        # Store the image path
        self.image_path = image_path
        # Load the image using the get_image method
        self.image = self.get_image()
    
    # Method to load an image from the specified path
    def get_image(self):
        """Get image based on hard path and return as numpy array"""
        # Read the image using OpenCV
        image = cv2.imread(self.image_path)
        # Check if image was loaded successfully
        if image is None:
            raise ValueError("Image not found at the specified path")
        return image
    
    # Method to create a negative of the image
    def negative_image(self):
        """Create negative of the image"""
        # Create a copy of the original image
        negative = self.image.copy()
        # Get image dimensions (rows, columns, channels)
        rows, cols, channels = negative.shape
        # Loop through each pixel and channel
        for i in range(rows):
            for j in range(cols):
                for k in range(channels):
                    # Subtract each pixel value from 255 to get negative
                    negative[i, j, k] = 255 - negative[i, j, k]
        return negative
    
    # Method to apply thresholding to the image
    def thresholding(self, threshold=127):
        """Apply thresholding to the image"""
        # Convert to grayscale if image is color
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()
        
        # Create a copy of the grayscale image
        thresh = gray.copy()
        # Get image dimensions (rows, columns)
        rows, cols = thresh.shape
        # Loop through each pixel
        for i in range(rows):
            for j in range(cols):
                # Apply threshold - set to 255 if above threshold, else 0
                thresh[i, j] = 255 if thresh[i, j] > threshold else 0
        return thresh
    
    # Method to apply gamma correction
    def gamma_correction(self, gamma=1.0, c=1):
        """Apply gamma correction: s = c * r^gamma"""
        # Create a copy of the image with float32 type for calculations
        corrected = self.image.copy().astype(np.float32)
        # Get image dimensions
        rows, cols, channels = corrected.shape
        # Loop through each pixel and channel
        for i in range(rows):
            for j in range(cols):
                for k in range(channels):
                    # Apply gamma correction formula
                    corrected[i, j, k] = c * (corrected[i, j, k] / 255.0) ** gamma
        # Scale back to 0-255 range and convert to uint8
        return (corrected * 255).astype(np.uint8)
    
    # Method to apply contrast stretching
    def contrast_stretching(self, r1=0, r2=255, s1=0, s2=255):
        """Apply contrast stretching"""
        # Create a copy of the image
        stretched = self.image.copy()
        # Get image dimensions
        rows, cols, channels = stretched.shape
        # Loop through each pixel and channel
        for i in range(rows):
            for j in range(cols):
                for k in range(channels):
                    # Get current pixel value
                    pixel = stretched[i, j, k]
                    # Apply contrast stretching formula
                    if pixel <= r1:
                        stretched[i, j, k] = s1
                    elif pixel >= r2:
                        stretched[i, j, k] = s2
                    else:
                        stretched[i, j, k] = s1 + ((pixel - r1) / (r2 - r1)) * (s2 - s1)
        return stretched
    
    # Method to adjust brightness
    def adjust_brightness(self, value=0):
        """Adjust brightness by adding/subtracting value from all pixels"""
        # Create a copy of the image with int16 type to handle negative values
        bright = self.image.copy().astype(np.int16)
        # Get image dimensions
        rows, cols, channels = bright.shape
        # Loop through each pixel and channel
        for i in range(rows):
            for j in range(cols):
                for k in range(channels):
                    # Add brightness value and clip to 0-255 range
                    bright[i, j, k] = np.clip(bright[i, j, k] + value, 0, 255)
        # Convert back to uint8 after clipping
        return bright.astype(np.uint8)
    
    # Helper method to display histograms
    def show_histogram(self, image, title, pos, color='gray'):
        """Helper function to show histogram for a single image"""
        # Create a subplot at the specified position
        plt.subplot(4, 4, pos)
        # For color images
        if len(image.shape) == 3:
            # Define colors for each channel (BGR)
            colors = ('b', 'g', 'r')
            # Calculate and plot histogram for each channel
            for i, col in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
        else:
            # For grayscale images, calculate and plot single histogram
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            plt.plot(hist, color=color)
        # Set plot title and labels
        plt.title(title)
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
    
    # Method to display all processed images and their histograms
    def display_all(self):
        """Display all processed images and their histograms in one figure"""
        # Process all images using the various methods
        negative = self.negative_image()
        threshold = self.thresholding(127)
        gamma_corrected = self.gamma_correction(gamma=0.5)
        contrast_stretched = self.contrast_stretching(r1=50, r2=200, s1=30, s2=220)
        equalized = self.histogram_equalization()
        brighter = self.adjust_brightness(50)
        darker = self.adjust_brightness(-50)
        
        # Convert BGR images to RGB for matplotlib display
        original_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        negative_rgb = cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)
        gamma_rgb = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB)
        contrast_rgb = cv2.cvtColor(contrast_stretched, cv2.COLOR_BGR2RGB)
        equalized_rgb = cv2.cvtColor(equalized, cv2.COLOR_BGR2RGB)
        brighter_rgb = cv2.cvtColor(brighter, cv2.COLOR_BGR2RGB)
        darker_rgb = cv2.cvtColor(darker, cv2.COLOR_BGR2RGB)
        
        # Create a large figure for all subplots
        plt.figure(figsize=(20, 20))
        
        # Original image subplot
        plt.subplot(4, 4, 1)
        plt.imshow(original_rgb)
        plt.title('Original Image')
        plt.axis('off')
        
        # Original histogram
        self.show_histogram(self.image, 'Original Histogram', 2)
        
        # Negative image subplot
        plt.subplot(4, 4, 3)
        plt.imshow(negative_rgb)
        plt.title('Negative Image')
        plt.axis('off')
        
        # Negative histogram
        self.show_histogram(negative, 'Negative Histogram', 4)
        
        # Threshold image subplot
        plt.subplot(4, 4, 5)
        plt.imshow(threshold, cmap='gray')
        plt.title('Threshold Image')
        plt.axis('off')
        
        # Threshold histogram
        self.show_histogram(threshold, 'Threshold Histogram', 6)
        
        # Gamma corrected image subplot
        plt.subplot(4, 4, 7)
        plt.imshow(gamma_rgb)
        plt.title('Gamma Corrected (γ=0.5)')
        plt.axis('off')
        
        # Gamma histogram
        self.show_histogram(gamma_corrected, 'Gamma Histogram', 8)
        
        # Contrast stretched image subplot
        plt.subplot(4, 4, 9)
        plt.imshow(contrast_rgb)
        plt.title('Contrast Stretched')
        plt.axis('off')
        
        # Contrast histogram
        self.show_histogram(contrast_stretched, 'Contrast Histogram', 10)
        
        # Histogram equalized image subplot
        plt.subplot(4, 4, 11)
        plt.imshow(equalized_rgb)
        plt.title('Histogram Equalized')
        plt.axis('off')
        
        # Equalized histogram
        self.show_histogram(equalized, 'Equalized Histogram', 12)
        
        # Brighter image subplot
        plt.subplot(4, 4, 13)
        plt.imshow(brighter_rgb)
        plt.title('Brighter (+50)')
        plt.axis('off')
        
        # Brighter histogram
        self.show_histogram(brighter, 'Brighter Histogram', 14)
        
        # Darker image subplot
        plt.subplot(4, 4, 15)
        plt.imshow(darker_rgb)
        plt.title('Darker (-50)')
        plt.axis('off')
        
        # Darker histogram
        self.show_histogram(darker, 'Darker Histogram', 16)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Display the figure
        plt.show()
    
    # Method to perform histogram equalization
    def histogram_equalization(self):
        """Apply histogram equalization"""
        # For color images
        if len(self.image.shape) == 3:
            # Convert to YCrCb color space (Y is luminance channel)
            ycrcb = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCrCb)
            # Extract the Y channel
            y = ycrcb[:, :, 0]
            
            # Calculate histogram for Y channel
            hist = [0] * 256
            rows, cols = y.shape
            for i in range(rows):
                for j in range(cols):
                    hist[y[i, j]] += 1
            
            # Calculate cumulative distribution function (CDF)
            cdf = [0] * 256
            cdf[0] = hist[0]
            for i in range(1, 256):
                cdf[i] = cdf[i-1] + hist[i]
            
            # Normalize CDF
            cdf_min = min(cdf)
            total_pixels = rows * cols
            for i in range(256):
                cdf[i] = round((cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255)
            
            # Apply equalization to Y channel
            equalized = y.copy()
            for i in range(rows):
                for j in range(cols):
                    equalized[i, j] = cdf[y[i, j]]
            
            # Replace Y channel with equalized values and convert back to BGR
            ycrcb[:, :, 0] = equalized
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            # For grayscale images
            # Calculate histogram
            hist = [0] * 256
            rows, cols = self.image.shape
            for i in range(rows):
                for j in range(cols):
                    hist[self.image[i, j]] += 1
            
            # Calculate CDF
            cdf = [0] * 256
            cdf[0] = hist[0]
            for i in range(1, 256):
                cdf[i] = cdf[i-1] + hist[i]
            
            # Normalize CDF
            cdf_min = min(cdf)
            total_pixels = rows * cols
            for i in range(256):
                cdf[i] = round((cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255)
            
            # Apply equalization
            equalized = self.image.copy()
            for i in range(rows):
                for j in range(cols):
                    equalized[i, j] = cdf[self.image[i, j]]
            
            return equalized


# Example usage
if __name__ == "__main__":
    # Create LUT instance with image path
    lut = LUT("/home/panzer/Desktop/DESK/CV/samplec.jpg")
    
    # Display all processed images and histograms in one figure
    lut.display_all()
