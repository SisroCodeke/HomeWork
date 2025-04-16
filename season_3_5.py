import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import pow

class LUT:
    def __init__(self, image_path):
        """
        Initialize the LUT class with an image path.
        
        Parameters:
        - image_path: Path to the input image file
        
        Attributes:
        - image_path: Stores the path of the image
        - image: Stores the loaded image as numpy array
        - height: Height of the image in pixels
        - width: Width of the image in pixels
        """
        self.image_path = image_path
        self.image = self.get_image()
        if self.image is not None:
            self.height, self.width = self.image.shape[:2]
    
    def get_image(self):
        """
        Load image from the specified path and return as numpy array.
        
        Process:
        1. Uses OpenCV's imread to load the image
        2. Checks if image was loaded successfully
        3. Returns the image array or None if loading failed
        
        Note: OpenCV loads images in BGR format by default
        """
        try:
            image = cv2.imread(self.image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found at {self.image_path}")
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def negative_image(self):
        """
        Create negative of the image by inverting pixel values.
        
        Process:
        1. Creates a blank image with same dimensions
        2. For each pixel in each channel, subtracts from 255
        3. Returns the negative image
        
        Formula: negative_pixel = 255 - original_pixel
        """
        if self.image is None:
            return None
            
        negative = np.zeros_like(self.image)
        for i in range(self.height):
            for j in range(self.width):
                for k in range(3 if len(self.image.shape) == 3 else 1):
                    negative[i, j, k] = 255 - self.image[i, j, k]
        return negative
    
    def thresholding(self, threshold=127):
        """
        Apply binary thresholding to the image.
        
        Parameters:
        - threshold: Pixel value threshold (default 127)
        
        Process:
        1. Converts image to grayscale if it's color
        2. For each pixel, sets to 255 if above threshold, else 0
        3. Returns the thresholded image
        """
        if self.image is None:
            return None
            
        # Convert to grayscale if needed
        if len(self.image.shape) == 3:
            gray = self._convert_to_grayscale()
        else:
            gray = self.image.copy()
            
        thresholded = np.zeros_like(gray)
        for i in range(self.height):
            for j in range(self.width):
                thresholded[i, j] = 255 if gray[i, j] > threshold else 0
        return thresholded
    
    def gamma_correction(self, gamma=1.0, c=1.0):
        """
        Apply gamma correction to the image.
        
        Parameters:
        - gamma: Gamma value (default 1.0)
        - c: Constant multiplier (default 1.0)
        
        Formula: s = c * r^gamma
        Process:
        1. Normalizes pixel values to [0,1]
        2. Applies gamma correction formula
        3. Scales back to [0,255] range
        """
        if self.image is None:
            return None
            
        corrected = np.zeros_like(self.image)
        for i in range(self.height):
            for j in range(self.width):
                for k in range(3 if len(self.image.shape) == 3 else 1):
                    normalized = self.image[i, j, k] / 255.0
                    corrected_value = c * pow(normalized, gamma)
                    corrected[i, j, k] = np.clip(int(corrected_value * 255), 0, 255)
        return corrected
    
    def contrast_stretching(self, r1=0, r2=255, s1=0, s2=255):
        """
        Apply contrast stretching to enhance image contrast.
        
        Parameters:
        - r1, r2: Input range to be stretched
        - s1, s2: Output range for stretching
        
        Process:
        1. Pixels <= r1 are mapped to s1
        2. Pixels >= r2 are mapped to s2
        3. Others are linearly interpolated between s1 and s2
        """
        if self.image is None:
            return None
            
        stretched = np.zeros_like(self.image)
        for i in range(self.height):
            for j in range(self.width):
                for k in range(3 if len(self.image.shape) == 3 else 1):
                    pixel = self.image[i, j, k]
                    if pixel <= r1:
                        stretched[i, j, k] = s1
                    elif pixel >= r2:
                        stretched[i, j, k] = s2
                    else:
                        stretched[i, j, k] = s1 + ((pixel - r1) * (s2 - s1) / (r2 - r1))
        return stretched
    
    def adjust_brightness(self, value=0):
        """
        Adjust image brightness by adding/subtracting a constant value.
        
        Parameters:
        - value: Brightness adjustment value (can be positive or negative)
        
        Process:
        1. Adds the value to each pixel component
        2. Clips values to stay within [0,255] range
        """
        if self.image is None:
            return None
            
        brightened = np.zeros_like(self.image)
        for i in range(self.height):
            for j in range(self.width):
                for k in range(3 if len(self.image.shape) == 3 else 1):
                    brightened[i, j, k] = np.clip(self.image[i, j, k] + value, 0, 255)
        return brightened
    
    def histogram(self, image=None):
        """
        Calculate and return histogram of the image.
        
        Parameters:
        - image: Optional input image (uses class image if None)
        
        Returns:
        - hist: 256-bin histogram array
        """
        if image is None:
            image = self.image
            if image is None:
                return None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = self._convert_to_grayscale(image)
        else:
            gray = image.copy()
            
        hist = np.zeros(256)
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                intensity = gray[i, j]
                hist[intensity] += 1
        
        return hist
    
    def histogram_equalization(self):
        """
        Perform histogram equalization to improve image contrast.
        
        Process:
        1. Calculates image histogram
        2. Computes cumulative distribution function (CDF)
        3. Normalizes CDF to [0,255] range
        4. Maps pixel values using normalized CDF
        """
        if self.image is None:
            return None
            
        # Convert to grayscale if needed
        if len(self.image.shape) == 3:
            gray = self._convert_to_grayscale()
        else:
            gray = self.image.copy()
            
        # Calculate histogram
        hist = self.histogram(gray)
        
        # Calculate cumulative distribution function (CDF)
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        
        # Create equalized image
        equalized = np.zeros_like(gray)
        for i in range(self.height):
            for j in range(self.width):
                equalized[i, j] = cdf_normalized[gray[i, j]]
        
        return equalized
    
    def _convert_to_grayscale(self, image=None):
        """
        Convert color image to grayscale using luminance formula.
        
        Formula: gray = 0.299*R + 0.587*G + 0.114*B
        """
        if image is None:
            image = self.image
            if image is None:
                return None
        
        if len(image.shape) == 2:  # Already grayscale
            return image.copy()
            
        gray = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                b, g, r = image[i, j]
                gray[i, j] = int(0.299 * r + 0.587 * g + 0.114 * b)
        return gray
    
    def show_all_results(self, brightness_value=50):
        """
        Display all processed images and their histograms in a single figure.
        
        Parameters:
        - brightness_value: Value for brightness adjustment
        """
        if self.image is None:
            print("No image loaded!")
            return
        
        # Process images using all functions
        processed_images = {
            "Original": self.image,
            "Negative": self.negative_image(),
            "Thresholded": self.thresholding(128),
            "Gamma Corrected": self.gamma_correction(gamma=0.5),
            "Contrast Stretched": self.contrast_stretching(50, 200, 0, 255),
            "Brightness Adjusted": self.adjust_brightness(brightness_value),
            "Histogram Equalized": self.histogram_equalization()
        }
        
        # Create figure with subplots
        plt.figure(figsize=(20, 20))
        
        # Display images and their histograms
        for idx, (title, image) in enumerate(processed_images.items(), 1):
            # Show image
            plt.subplot(4, 4, idx*2-1)
            if len(image.shape) == 2:
                plt.imshow(image, cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis('off')
            
            # Show histogram
            plt.subplot(4, 4, idx*2)
            hist = self.histogram(image)
            plt.bar(range(256), hist, color='gray')
            plt.title(f"{title} Histogram")
            plt.xlim([0, 255])
        
        plt.tight_layout()
        plt.show()


# Example usage:
if __name__ == "__main__":
    # Initialize with image path
    lut = LUT("CV/samplec.jpg")
    
    # Display all results in a single figure
    lut.show_all_results(brightness_value=50)