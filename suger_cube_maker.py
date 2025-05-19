import os
import random
import cv2
import numpy as np

# Image size and square size
img_size = 4096
square_size = img_size // 24

# Output folder
output_folder = "/home/panzer/Desktop/Desk/Uni/images"
os.makedirs(output_folder, exist_ok=True)

def squares_overlap(square1, square2):
    """Check if two axis-aligned squares overlap."""
    x1, y1, size1 = square1
    x2, y2, size2 = square2
    return not (x1 + size1 <= x2 or x2 + size2 <= x1 or y1 + size1 <= y2 or y2 + size2 <= y1)

def generate_non_overlapping_squares(img_size, square_size, count):
    """Generate non-overlapping square positions."""
    squares = []
    max_attempts = 10000
    attempts = 0

    while len(squares) < count and attempts < max_attempts:
        x = random.randint(0, img_size - square_size)
        y = random.randint(0, img_size - square_size)
        new_square = (x, y, square_size)

        if not any(squares_overlap(new_square, existing) for existing in squares):
            squares.append(new_square)
        attempts += 1

    return squares

def draw_rotated_square(image, x, y, size, angle):
    """Draw a rotated square (white) on the image."""
    center = (x + size // 2, y + size // 2)

    # Define the square corners relative to its center
    half = size // 2
    corners = np.array([
        [-half, -half],
        [ half, -half],
        [ half,  half],
        [-half,  half]
    ], dtype=np.float32)

    # Build rotation matrix and rotate corners
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
    rotated_corners = np.dot(corners, rotation_matrix[:, :2].T).astype(np.int32)
    rotated_corners += np.array(center, dtype=np.int32)

    # Draw filled rotated square
    cv2.fillConvexPoly(image, rotated_corners, (255, 255, 255))

# Generate images with non-overlapping rotated white squares
for i in range(1, 16):
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    squares = generate_non_overlapping_squares(img_size, square_size, i)

    for x, y, size in squares:
        angle = random.randint(0, 360)
        draw_rotated_square(img, x, y, size, angle)

    img_name = f"image_{i}.jpg"
    cv2.imwrite(os.path.join(output_folder, img_name), img)

print("Images with clean rotated white squares generated successfully!")
su
