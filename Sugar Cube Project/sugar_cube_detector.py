
# Import necessary libraries
import os  # For interacting with the operating system (e.g., paths, directories)
import cv2  # OpenCV library for image processing
import numpy as np  # For numerical operations and array handling
import matplotlib.pyplot as plt  # For creating visual plots
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # For rendering matplotlib figures as images
import io  # For handling in-memory I/O (unused in final code)

# === Configuration ===

# Path to the input images folder
INPUT_FOLDER = "/home/panzer/Desktop/Desk/Uni/mehrdad_images"

# Path to the output folder where results will be saved
OUTPUT_FOLDER = "/home/panzer/Desktop/Desk/Uni/results"

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Parameters for image processing
THRESHOLD_VALUE = 150  # Threshold value for binarization (grayscale pixel intensity)
MIN_AREA = 500         # Minimum area of contour to be considered a sugar cube
PLOT_DPI = 300         # DPI for saved plots
BOX_THICKNESS = 10     # Thickness of bounding boxes drawn on detections
FONT_SCALE = 3         # Font size for count annotation

# === Function Definitions ===

def load_and_threshold_image(image_path, threshold_value):
    """
    Loads an image and applies binary thresholding.
    Returns both the original image and the thresholded binary mask.
    """
    print(f"\n[1/4] Loading and thresholding: {os.path.basename(image_path)}")
    
    img = cv2.imread(image_path)  # Load the image as BGR
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert BGR to grayscale

    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)  # Apply binary threshold
    print(f"Applied threshold: {threshold_value}. Image shape: {gray.shape}")
    
    return img, binary  # Return original image and binary mask

def find_sugar_cubes(binary_image, min_area):
    """
    Finds contours (sugar cubes) in the binary image after separation.
    Filters out small contours based on area.
    """
    print("[2/4] Finding contours...")
    
    # Separate touching objects first using watershed method
    separated = separate_touching_objects(binary_image)

    # Find contours in the separated image
    contours, _ = cv2.findContours(separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area
    sugar_cubes = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    return sugar_cubes  # Return list of valid sugar cube contours

def separate_touching_objects(binary_img):
    """
    Applies morphological operations and watershed to separate touching sugar cubes.
    """
    binary_img = binary_img.astype(np.uint8)  # Ensure binary image is in correct format

    kernel = np.ones((3, 3), np.uint8)  # Define 3x3 kernel for morphology
    opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=2)  # Noise removal

    sure_bg = cv2.dilate(opening, kernel, iterations=3)  # Dilate to get sure background

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # Compute distance transform
    _, sure_fg = cv2.threshold(dist_transform, 0.25 * dist_transform.max(), 255, 0)  # Threshold for sure foreground
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)  # Subtract to get unknown regions

    _, markers = cv2.connectedComponents(sure_fg)  # Label connected components
    markers = markers + 1  # Increment all labels (0 becomes 1, 1 becomes 2, etc.)
    markers[unknown == 255] = 0  # Mark unknown regions with 0

    img_color = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)  # Convert to BGR for watershed
    cv2.watershed(img_color, markers)  # Apply watershed algorithm

    separated = np.uint8((markers > 1) * 255)  # Convert non-boundary markers to binary mask

    return separated  # Return separated binary image

def draw_bounding_boxes(image, contours):
    """
    Draws bounding boxes around contours and annotates image with total count.
    """
    print("[3/4] Drawing bounding boxes...")
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # Get bounding box from contour
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), BOX_THICKNESS)  # Draw green rectangle

    count = len(contours)  # Total number of sugar cubes detected
    text_org = (50, 100)  # Position to place text
    cv2.putText(image, f"Cubes: {count}", text_org,
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                (0, 255, 255), BOX_THICKNESS//2)  # Draw text
    
    print(f"Annotated {count} cubes on image")
    return image, count  # Return annotated image and count

def create_result_plot(original_img, binary_img, filename, count, threshold, min_area):
    """
    Creates a side-by-side plot of the annotated image and threshold mask.
    Returns the plot as a numpy array.
    """
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    binary_rgb = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB for display

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))  # Create 2-panel figure

    # Title with processing details
    info_text = (f"File: {filename}\n"
                 f"Detected Cubes: {count} | "
                 f"Threshold: {threshold} | "
                 f"Min Area: {min_area}px")
    fig.suptitle(info_text, fontsize=14, y=0.95)

    ax1.imshow(original_rgb)  # Show original image
    ax1.set_title("Processed Image with Detections", fontsize=12)
    ax1.axis('off')

    ax2.imshow(binary_img, cmap='gray')  # Show binary mask
    ax2.set_title("Thresholding Mask Used for Detection", fontsize=12)
    ax2.axis('off')

    canvas = FigureCanvas(fig)  # Create canvas for rendering
    canvas.draw()
    plot_array = np.array(canvas.renderer.buffer_rgba())  # Convert to numpy array
    plt.close()

    return cv2.cvtColor(plot_array, cv2.COLOR_RGBA2RGB)  # Convert to RGB and return

def process_and_save_image(image_path):
    """
    Complete pipeline for processing a single image and saving the result.
    """
    print(f"\n{'='*50}\nProcessing: {os.path.basename(image_path)}")

    img, binary = load_and_threshold_image(image_path, THRESHOLD_VALUE)  # Step 1: Load and threshold image

    refined_binary = separate_touching_objects(binary)  # Step 2: Separate touching cubes
    cubes = find_sugar_cubes(refined_binary, MIN_AREA)  # Step 3: Find valid contours

    result_img, count = draw_bounding_boxes(img.copy(), cubes)  # Step 4: Annotate image

    plot_rgb = create_result_plot(  # Step 5: Create result plot
        result_img, refined_binary,
        os.path.basename(image_path), count,
        THRESHOLD_VALUE, MIN_AREA
    )

    # Save result image
    filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_result.jpg"
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(output_path, cv2.cvtColor(plot_rgb, cv2.COLOR_RGB2BGR))  # Save RGB as BGR
    print(f"[4/4] Saved result to: {output_path}")

    return plot_rgb  # Return result image

def summarize_results(images, cols=4):
    """
    Create and save a summary plot containing all processed images in a grid.
    """
    print("\nCreating summary visualization...")

    rows = int(np.ceil(len(images) / cols))  # Calculate number of rows
    figsize = (24, 6 * rows)  # Define figure size

    fig, axes = plt.subplots(rows, cols, figsize=figsize)  # Create grid of subplots
    fig.suptitle("Sugar Cube Detection Results Summary", fontsize=18, y=0.98)

    axes = axes.ravel() if rows > 1 else [axes]  # Flatten axes for iteration

    for idx, img in enumerate(images):  # Plot each image
        axes[idx].imshow(img)
        axes[idx].set_title(f"Case {idx+1}", fontsize=14)
        axes[idx].axis('off')

    for idx in range(len(images), len(axes)):  # Hide empty plots
        axes[idx].axis('off')

    summary_path = os.path.join(OUTPUT_FOLDER, "detection_summary.jpg")  # Output file path
    plt.savefig(summary_path, bbox_inches='tight', dpi=PLOT_DPI)  # Save plot
    plt.close()
    print(f"Saved summary visualization to: {summary_path}")

def main():
    """Main entry point for script execution."""
    print("Starting sugar cube detection pipeline...")
    print(f"Input directory: {INPUT_FOLDER}")
    print(f"Output directory: {OUTPUT_FOLDER}")

    processed_images = []  # List to hold all result images

    # Process each image in input folder
    for filename in sorted(os.listdir(INPUT_FOLDER)):
        if filename.lower().endswith(".jpg"):  # Only process JPG images
            image_path = os.path.join(INPUT_FOLDER, filename)
            result_plot = process_and_save_image(image_path)
            processed_images.append(result_plot)

    if processed_images:
        summarize_results(processed_images)  # Create summary if any image was processed

    print("\nProcessing complete!")

# Run main function if script is executed directly
if __name__ == "__main__":
    main()
