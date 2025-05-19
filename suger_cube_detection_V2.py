import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io

# === Configuration ===
INPUT_FOLDER = "/home/panzer/Desktop/Desk/Uni/images"
OUTPUT_FOLDER = "/home/panzer/Desktop/Desk/Uni/results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Parameters for processing
THRESHOLD_VALUE = 240  # Pixel intensity threshold for binarization
MIN_AREA = 100         # Minimum contour area to be considered a sugar cube
PLOT_DPI = 300         # DPI for output images
BOX_THICKNESS = 10     # Thickness for bounding boxes
FONT_SCALE = 3         # Font size for annotations


def load_and_threshold_image(image_path, threshold_value):
    """
    Load an image and convert it to a binary mask using thresholding.
    
    Parameters:
    image_path (str): Path to input image
    threshold_value (int): Pixel intensity threshold (0-255)
    
    Returns:
    tuple: (original BGR image, binary thresholded image)
    """
    print(f"\n[1/4] Loading and thresholding: {os.path.basename(image_path)}")
    
    # Read image and convert to grayscale
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    print(f"Applied threshold: {threshold_value}. Image shape: {gray.shape}")
    
    return img, binary


def find_sugar_cubes(binary_image, min_area):
    """
    Identify potential sugar cubes using contour detection and area filtering.
    
    Parameters:
    binary_image (numpy array): Thresholded image
    min_area (int): Minimum contour area in pixels
    
    Returns:
    list: Valid contours representing sugar cubes
    """
    print("[2/4] Finding contours...")
    
    # Find external contours (simple approximation)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Initial contours found: {len(contours)}")
    
    # Filter contours by area
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    print(f"Contours after area filtering ({min_area}px): {len(filtered)}")
    
    return filtered


def draw_bounding_boxes(image, contours):
    """
    Annotate image with bounding boxes and count display.
    
    Parameters:
    image (numpy array): Original image to annotate
    contours (list): Valid contours to draw
    
    Returns:
    tuple: (annotated image, cube count)
    """
    print("[3/4] Drawing bounding boxes...")
    
    # Draw rectangles around each contour
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x+w, y+h), 
                     (0, 255, 0), BOX_THICKNESS)
    
    # Add count text overlay
    count = len(contours)
    text_org = (50, 100)
    cv2.putText(image, f"Cubes: {count}", text_org, 
               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
               (0, 255, 255), BOX_THICKNESS//2)
    
    print(f"Annotated {count} cubes on image")
    return image, count


def create_result_plot(original_img, binary_img, filename, count, 
                       threshold, min_area):
    """
    Create a combined visualization plot with processing results.
    
    Parameters:
    original_img (numpy array): Annotated BGR image
    binary_img (numpy array): Thresholded image
    filename (str): Source filename
    count (int): Detected cube count
    threshold (int): Used threshold value
    min_area (int): Used area filter
    
    Returns:
    numpy array: RGB image of the combined plot
    """
    # Convert images for matplotlib display
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    binary_rgb = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Configure main title
    info_text = (f"File: {filename}\n"
                 f"Detected Cubes: {count} | "
                 f"Threshold: {threshold} | "
                 f"Min Area: {min_area}px")
    fig.suptitle(info_text, fontsize=14, y=0.95)

    # Original image subplot
    ax1.imshow(original_rgb)
    ax1.set_title("Processed Image with Detections", fontsize=12)
    ax1.axis('off')

    # Binary image subplot
    ax2.imshow(binary_img, cmap='gray')
    ax2.set_title("Thresholding Mask Used for Detection", fontsize=12)
    ax2.axis('off')

    # Render plot to numpy array
    canvas = FigureCanvas(fig)
    canvas.draw()
    plot_array = np.array(canvas.renderer.buffer_rgba())
    plt.close()

    # Convert RGBA to RGB
    return cv2.cvtColor(plot_array, cv2.COLOR_RGBA2RGB)


def process_and_save_image(image_path):
    """
    Complete processing pipeline for a single image.
    
    Parameters:
    image_path (str): Path to input image
    
    Returns:
    numpy array: Combined result plot in RGB format
    """
    print(f"\n{'='*50}\nProcessing: {os.path.basename(image_path)}")
    
    # Execute processing pipeline
    img, binary = load_and_threshold_image(image_path, THRESHOLD_VALUE)
    cubes = find_sugar_cubes(binary, MIN_AREA)
    result_img, count = draw_bounding_boxes(img.copy(), cubes)
    
    # Create combined visualization plot
    plot_rgb = create_result_plot(
        result_img, binary,
        os.path.basename(image_path), count,
        THRESHOLD_VALUE, MIN_AREA
    )
    
    # Save results
    filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_result.jpg"
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(output_path, cv2.cvtColor(plot_rgb, cv2.COLOR_RGB2BGR))
    print(f"[4/4] Saved result to: {output_path}")
    
    return plot_rgb


def summarize_results(images, cols=4):
    """
    Create and save a summary grid of all processed images.
    
    Parameters:
    images (list): List of RGB images to display
    cols (int): Number of columns in grid
    """
    print("\nCreating summary visualization...")
    
    rows = int(np.ceil(len(images) / cols))
    figsize = (24, 6 * rows)  # Adjusted for better visibility
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle("Sugar Cube Detection Results Summary", fontsize=18, y=0.98)
    
    # Flatten axes array for easy iteration
    axes = axes.ravel() if rows > 1 else [axes]
    
    for idx, img in enumerate(images):
        axes[idx].imshow(img)
        axes[idx].set_title(f"Case {idx+1}", fontsize=14)
        axes[idx].axis('off')
    
    # Hide empty subplots
    for idx in range(len(images), len(axes)):
        axes[idx].axis('off')
    
    # Save high-quality summary
    summary_path = os.path.join(OUTPUT_FOLDER, "detection_summary.jpg")
    plt.savefig(summary_path, bbox_inches='tight', dpi=PLOT_DPI)
    plt.close()
    print(f"Saved summary visualization to: {summary_path}")


def main():
    """Main execution function"""
    print("Starting sugar cube detection pipeline...")
    print(f"Input directory: {INPUT_FOLDER}")
    print(f"Output directory: {OUTPUT_FOLDER}")
    
    processed_images = []
    
    # Process all JPG images in input directory
    for filename in sorted(os.listdir(INPUT_FOLDER)):
        if filename.lower().endswith(".jpg"):
            image_path = os.path.join(INPUT_FOLDER, filename)
            result_plot = process_and_save_image(image_path)
            processed_images.append(result_plot)
    
    # Create and save summary
    if processed_images:
        summarize_results(processed_images)
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
