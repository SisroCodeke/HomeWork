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
    
    # Method to convert BGR to RGB manually
    def bgr_to_rgb(self, image):
        """Convert BGR image to RGB using loops"""
        rgb = image.copy()
        rows, cols, channels = rgb.shape
        for i in range(rows):
            for j in range(cols):
                # Swap B and R channels
                rgb[i, j, 0], rgb[i, j, 2] = rgb[i, j, 2], rgb[i, j, 0]
        return rgb
    
    # Method to convert BGR to grayscale manually
    def bgr_to_gray(self, image):
        """Convert BGR image to grayscale using loops"""
        gray = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        rows, cols = gray.shape
        for i in range(rows):
            for j in range(cols):
                # Weighted average for grayscale conversion
                gray[i, j] = 0.114 * image[i, j, 0] + 0.587 * image[i, j, 1] + 0.299 * image[i, j, 2]
        return gray
    
    # Method to convert BGR to YCrCb manually
    def bgr_to_ycrcb(self, image):
        """Convert BGR image to YCrCb using loops"""
        ycrcb = np.zeros_like(image, dtype=np.float32)
        rows, cols, channels = ycrcb.shape
        for i in range(rows):
            for j in range(cols):
                b, g, r = image[i, j]
                # Y component
                ycrcb[i, j, 0] = 0.299 * r + 0.587 * g + 0.114 * b
                # Cr component
                ycrcb[i, j, 1] = (r - ycrcb[i, j, 0]) * 0.713 + 128
                # Cb component
                ycrcb[i, j, 2] = (b - ycrcb[i, j, 0]) * 0.564 + 128
        return ycrcb.astype(np.uint8)
    
    # Method to convert YCrCb to BGR manually
    def ycrcb_to_bgr(self, image):
        """Convert YCrCb image to BGR using loops"""
        bgr = np.zeros_like(image, dtype=np.float32)
        rows, cols, channels = bgr.shape
        for i in range(rows):
            for j in range(cols):
                y, cr, cb = image[i, j]
                # R component
                r = y + 1.403 * (cr - 128)
                # G component
                g = y - 0.714 * (cr - 128) - 0.344 * (cb - 128)
                # B component
                b = y + 1.773 * (cb - 128)
                # Clip values to 0-255 range
                bgr[i, j, 0] = max(0, min(255, b))
                bgr[i, j, 1] = max(0, min(255, g))
                bgr[i, j, 2] = max(0, min(255, r))
        return bgr.astype(np.uint8)
    
    # Method to create a negative of the image
    def negative_image(self):
        """Create negative of the image"""
        negative = self.image.copy()
        rows, cols, channels = negative.shape
        for i in range(rows):
            for j in range(cols):
                for k in range(channels):
                    negative[i, j, k] = 255 - negative[i, j, k]
        return negative
    
    # Method to apply thresholding to the image
    def thresholding(self, threshold=127):
        """Apply thresholding to the image"""
        if len(self.image.shape) == 3:
            gray = self.bgr_to_gray(self.image)
        else:
            gray = self.image.copy()
        
        thresh = gray.copy()
        rows, cols = thresh.shape
        for i in range(rows):
            for j in range(cols):
                thresh[i, j] = 255 if thresh[i, j] > threshold else 0
        return thresh
    
    # Method to apply gamma correction
    def gamma_correction(self, gamma=1.0, c=1):
        """Apply gamma correction: s = c * r^gamma"""
        corrected = self.image.copy().astype(np.float32)
        rows, cols, channels = corrected.shape
        for i in range(rows):
            for j in range(cols):
                for k in range(channels):
                    corrected[i, j, k] = c * (corrected[i, j, k] / 255.0) ** gamma
        return (corrected * 255).astype(np.uint8)
    
    # Method to apply contrast stretching
    def contrast_stretching(self, r1=0, r2=255, s1=0, s2=255):
        """Apply contrast stretching"""
        stretched = self.image.copy()
        rows, cols, channels = stretched.shape
        for i in range(rows):
            for j in range(cols):
                for k in range(channels):
                    pixel = stretched[i, j, k]
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
        bright = self.image.copy().astype(np.int16)
        rows, cols, channels = bright.shape
        for i in range(rows):
            for j in range(cols):
                for k in range(channels):
                    bright[i, j, k] = np.clip(bright[i, j, k] + value, 0, 255)
        return bright.astype(np.uint8)
    
    # Helper method to display histograms
    def show_histogram(self, image, title, pos, color='gray'):
        """Helper function to show histogram for a single image"""
        plt.subplot(4, 4, pos)
        if len(image.shape) == 3:
            colors = ('b', 'g', 'r')
            for i, col in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
        else:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            plt.plot(hist, color=color)
        plt.title(title)
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
    
    # Method to display all processed images and their histograms
    def display_all(self):
        """Display all processed images and their histograms in one figure"""
        negative = self.negative_image()
        threshold = self.thresholding(127)
        gamma_corrected = self.gamma_correction(gamma=0.5)
        contrast_stretched = self.contrast_stretching(r1=50, r2=200, s1=30, s2=220)
        equalized = self.histogram_equalization()
        brighter = self.adjust_brightness(50)
        darker = self.adjust_brightness(-50)
        
        # Convert images using our manual methods
        original_rgb = self.bgr_to_rgb(self.image)
        negative_rgb = self.bgr_to_rgb(negative)
        gamma_rgb = self.bgr_to_rgb(gamma_corrected)
        contrast_rgb = self.bgr_to_rgb(contrast_stretched)
        equalized_rgb = self.bgr_to_rgb(equalized)
        brighter_rgb = self.bgr_to_rgb(brighter)
        darker_rgb = self.bgr_to_rgb(darker)
        
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
        
        plt.tight_layout()
        plt.show()
    
    # Method to perform histogram equalization
    def histogram_equalization(self):
        """Apply histogram equalization"""
        if len(self.image.shape) == 3:
            # Convert to YCrCb using our manual method
            ycrcb = self.bgr_to_ycrcb(self.image)
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
            
            # Replace Y channel and convert back to BGR
            ycrcb[:, :, 0] = equalized
            return self.ycrcb_to_bgr(ycrcb)
        else:
            # For grayscale images
            hist = [0] * 256
            rows, cols = self.image.shape
            for i in range(rows):
                for j in range(cols):
                    hist[self.image[i, j]] += 1
            
            cdf = [0] * 256
            cdf[0] = hist[0]
            for i in range(1, 256):
                cdf[i] = cdf[i-1] + hist[i]
            
            cdf_min = min(cdf)
            total_pixels = rows * cols
            for i in range(256):
                cdf[i] = round((cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255)
            
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
