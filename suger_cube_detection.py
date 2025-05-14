import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Configuration ===
input_folder = "/home/panzer/Desktop/Desk/Uni/images"
output_folder = "/home/panzer/Desktop/Desk/Uni/results"
os.makedirs(output_folder, exist_ok=True)


def load_and_threshold_image(image_path):
    """
    Load an image, convert it to grayscale, and apply binary thresholding
    to isolate white sugar cubes from the background.
    """
    img = cv2.imread(image_path)  # Load color image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Apply threshold: anything brighter than 240 becomes white (255), else black (0)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    return img, binary


def find_sugar_cubes(binary_image, min_area=100):
    """
    Find contours in the thresholded image that correspond to sugar cubes.
    Only keep contours above a minimum area threshold.
    """
    # Extract only the outer contours of white blobs
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out noise: keep only contours that are large enough to be sugar cubes
    cubes = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return cubes





def draw_bounding_boxes(image, contours):
    """
    Draw green bounding boxes around detected sugar cubes on the image.
    Also return a count of how many cubes were detected.
    """
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # Get bounding box coordinates
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 10)  # Draw rectangle

    # Display cube count in top-left corner of image
    count = len(contours)
    cv2.putText(image, f"Cubes: {count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)
    return image, count


def process_and_save_image(image_path):
    """
    Full pipeline for one image:
    - Load & threshold
    - Detect cubes
    - Draw bounding boxes & count
    - Save result image
    - Return RGB image for plotting
    """
    img, binary = load_and_threshold_image(image_path)  # Step 1: Preprocessing
    cubes = find_sugar_cubes(binary)                    # Step 2: Contour detection
    result_img, count = draw_bounding_boxes(img.copy(), cubes)  # Step 3: Draw boxes

    # Build result filename with predicted count
    base = os.path.basename(image_path)
    name, ext = os.path.splitext(base)
    result_name = f"{name}_pred_{count}.jpg"
    result_path = os.path.join(output_folder, result_name)

    cv2.imwrite(result_path, result_img)  # Save annotated image

    # Return image in RGB format for matplotlib plotting
    return cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)


def summarize_results(images, cols=4):
    """
    Plot all result images in a grid and save the summary image.
    """
    rows = int(np.ceil(len(images) / cols))  # Compute number of grid rows
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)  # Convert flat index to grid (row, col)
        ax = axes[r, c] if rows > 1 else axes[c]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Image {idx + 1}")

    # Turn off any unused subplots
    for i in range(len(images), rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c] if rows > 1 else axes[c]
        ax.axis('off')

    # Save final grid plot
    plt.tight_layout()
    summary_path = os.path.join(output_folder, "summary_all_detections.jpg")
    plt.savefig(summary_path)
    print(f"Summary saved at: {summary_path}")


def main():
    """
    Main function to process all images and create summary plot.
    """
    all_result_images = []  # List to hold results for summary plot

    # Walk through input folder and process each .jpg file
    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith(".jpg"):
            path = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")
            result_rgb = process_and_save_image(path)
            all_result_images.append(result_rgb)

    # Plot and save all processed images in a grid
    summarize_results(all_result_images)
    print("All done!")


if __name__ == "__main__":
    main()

