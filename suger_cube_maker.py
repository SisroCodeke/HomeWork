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

# import os
# import random
# import cv2
# import numpy as np

# # Image size and square size
# img_size = 4096
# square_size = img_size // 24

# # Output folder
# output_folder = "/home/panzer/Desktop/Desk/Uni/images"
# os.makedirs(output_folder, exist_ok=True)

# def squares_overlap(square1, square2):
#     """Check if two axis-aligned squares overlap."""
#     x1, y1, size1 = square1
#     x2, y2, size2 = square2
#     return not (x1 + size1 <= x2 or x2 + size2 <= x1 or y1 + size1 <= y2 or y2 + size2 <= y1)

# def generate_non_overlapping_squares(img_size, square_size, count):
#     """Generate non-overlapping square positions."""
#     squares = []
#     max_attempts = 10000
#     attempts = 0

#     while len(squares) < count and attempts < max_attempts:
#         x = random.randint(0, img_size - square_size)
#         y = random.randint(0, img_size - square_size)
#         new_square = (x, y, square_size)

#         if not any(squares_overlap(new_square, existing) for existing in squares):
#             squares.append(new_square)
#         attempts += 1

#     return squares

# def draw_rotated_square(image, x, y, size, angle):
#     """Draw a rotated square (white) on the image."""
#     center = (x + size // 2, y + size // 2)

#     # Define the square corners relative to its center
#     half = size // 2
#     corners = np.array([
#         [-half, -half],
#         [ half, -half],
#         [ half,  half],
#         [-half,  half]
#     ], dtype=np.float32)

#     # Build rotation matrix and rotate corners
#     rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
#     rotated_corners = np.dot(corners, rotation_matrix[:, :2].T).astype(np.int32)
#     rotated_corners += np.array(center, dtype=np.int32)

#     # Draw filled rotated square
#     cv2.fillConvexPoly(image, rotated_corners, (255, 255, 255))

# def create_textured_background(size):
#     """Create a textured background with noise and patterns."""
#     # Base color (slightly off-black)
#     background = np.full((size, size, 3), (10, 10, 10), dtype=np.uint8)
    
#     # Add Perlin-like noise (simplified)
#     for _ in range(3):  # Multiple layers of noise
#         noise = np.random.randint(0, 30, (size, size), dtype=np.uint8)
#         noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=random.uniform(1, 3))
#         noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
#         background = cv2.add(background, noise)
    
#     # Add subtle gradient
#     gradient = np.linspace(0, 15, size, dtype=np.uint8)
#     gradient = np.tile(gradient, (size, 1))
#     gradient = cv2.merge([gradient]*3)
#     background = cv2.add(background, gradient)
    
#     # Add some random speckles
#     speckles = np.random.choice([0, 1, 2], size=(size, size, 3), p=[0.98, 0.01, 0.01])
#     background = cv2.add(background, speckles)
    
#     return background

# # Generate images with non-overlapping rotated white squares
# for i in range(1, 16):
#     # Create textured background
#     img = create_textured_background(img_size)
    
#     # Add Gaussian noise to the entire image
#     noise = np.random.normal(0, 3, img.shape).astype(np.int16)
#     img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
#     # Generate and draw squares
#     squares = generate_non_overlapping_squares(img_size, square_size, i)
#     for x, y, size in squares:
#         angle = random.randint(0, 360)
#         draw_rotated_square(img, x, y, size, angle)
    
#     # Add final subtle noise layer
#     final_noise = np.random.randint(-5, 5, img.shape, dtype=np.int16)
#     img = np.clip(img + final_noise, 0, 255).astype(np.uint8)
    
#     img_name = f"image_{i}.jpg"
#     cv2.imwrite(os.path.join(output_folder, img_name), img)

# print("Images with textured backgrounds and rotated white squares generated successfully!")
