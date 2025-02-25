import os
import random
import math
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageEnhance

# --- Utility functions ---

def generate_background(width, height):
    """
    Generate a random background. Randomly chooses among:
    - noise: each pixel is a random color,
    - gradient: a vertical linear gradient between two random colors,
    - shapes: a solid base with several random ellipses/rectangles.
    """
    style = random.choice(["noise", "gradient", "shapes"])
    image = Image.new("RGB", (width, height))
    
    if style == "noise":
        for x in range(width):
            for y in range(height):
                image.putpixel((x, y), (random.randint(0, 255),
                                          random.randint(0, 255),
                                          random.randint(0, 255)))
    elif style == "gradient":
        start_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        end_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for y in range(height):
            ratio = y / height
            r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
            g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
            b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
            for x in range(width):
                image.putpixel((x, y), (r, g, b))
    else:  # "shapes"
        base_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image = Image.new("RGB", (width, height), base_color)
        draw = ImageDraw.Draw(image)
        for _ in range(10):
            shape_type = random.choice(["ellipse", "rectangle"])
            x0 = random.randint(0, width - 1)
            y0 = random.randint(0, height - 1)
            x1 = random.randint(x0, width)
            y1 = random.randint(y0, height)
            shape_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if shape_type == "ellipse":
                draw.ellipse([x0, y0, x1, y1], fill=shape_color, outline=shape_color)
            else:
                draw.rectangle([x0, y0, x1, y1], fill=shape_color, outline=shape_color)
    return image

def draw_irregular_line(draw, start, end, color, width):
    """
    Draw a line between start and end that is perturbed by random offsets,
    simulating a hand-drawn, irregular line.
    """
    points = [start]
    num_segments = random.randint(3, 6)
    for i in range(1, num_segments):
        t = i / num_segments
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        offset = 2
        x += random.uniform(-offset, offset)
        y += random.uniform(-offset, offset)
        points.append((x, y))
    points.append(end)
    draw.line(points, fill=color, width=width)

def draw_board(draw, bbox, board_color, line_width):
    """
    Draw the Tic-Tac-Toe board inside the bounding box 'bbox'.
    The board consists of two vertical and two horizontal irregular lines.
    """
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    for i in range(1, 3):
        x = x0 + i * width / 3
        draw_irregular_line(draw, (x, y0), (x, y1), board_color, line_width)
    for i in range(1, 3):
        y = y0 + i * height / 3
        draw_irregular_line(draw, (x0, y), (x1, y), board_color, line_width)

def draw_mark_X(draw, cell_bbox, mark_color, line_width):
    """
    Draw a hand-drawn X inside the cell defined by cell_bbox.
    """
    x0, y0, x1, y1 = cell_bbox
    draw_irregular_line(draw, (x0, y0), (x1, y1), mark_color, line_width)
    draw_irregular_line(draw, (x1, y0), (x0, y1), mark_color, line_width)

def draw_mark_O(draw, cell_bbox, mark_color, line_width):
    """
    Draw a hand-drawn O inside the cell defined by cell_bbox.
    Approximates a circle with a series of connected points with random perturbations.
    """
    x0, y0, x1, y1 = cell_bbox
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    r = min(x1 - x0, y1 - y0) / 2 * 0.8
    num_points = 20
    points = []
    for i in range(num_points + 1):
        angle = 2 * math.pi * i / num_points
        r_jitter = r + random.uniform(-r * 0.1, r * 0.1)
        x = cx + r_jitter * math.cos(angle)
        y = cy + r_jitter * math.sin(angle)
        points.append((x, y))
    draw.line(points, fill=mark_color, width=line_width)

def scale_bbox(bbox, scale):
    """
    Scale the bounding box about its center by a factor 'scale'.
    """
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    new_width = (x1 - x0) * scale
    new_height = (y1 - y0) * scale
    return (cx - new_width/2, cy - new_height/2, cx + new_width/2, cy + new_height/2)

def _get_random_affine_coeffs(size):
    """
    Generate random coefficients for an affine transformation.
    """
    a = 1 + random.uniform(-0.1, 0.1)
    d = 1 + random.uniform(-0.1, 0.1)
    b = random.uniform(-0.1, 0.1)
    e = random.uniform(-0.1, 0.1)
    c = random.uniform(-5, 5)
    f = random.uniform(-5, 5)
    return (a, b, c, e, d, f)

def apply_random_transform(image):
    """
    Apply a random rotation and affine transformation to simulate different camera angles.
    """
    angle = random.uniform(-15, 15)
    image = image.rotate(angle, resample=Image.BICUBIC, expand=0)
    coeffs = _get_random_affine_coeffs(image.size)
    image = image.transform(image.size, Image.AFFINE, coeffs, resample=Image.BICUBIC)
    return image

def adjust_lighting(image):
    """
    Randomly adjust brightness and contrast to simulate various lighting conditions.
    """
    enhancer = ImageEnhance.Brightness(image)
    brightness_factor = random.uniform(0.7, 1.3)
    image = enhancer.enhance(brightness_factor)
    
    enhancer = ImageEnhance.Contrast(image)
    contrast_factor = random.uniform(0.7, 1.3)
    image = enhancer.enhance(contrast_factor)
    return image

def generate_tic_tac_toe_image():
    """
    Generate one Tic-Tac-Toe image with:
    - A random background.
    - A hand-drawn board.
    - Randomly placed X's and O's in the 3x3 grid with variable mark sizes.
    - An annotation array with 0 for empty, 1 for X, 2 for O.
      (The annotation is flattened to a 1D array of length 9.)
    - Random transformations simulating different camera angles and lighting.
    """
    img_size = 224
    background = generate_background(img_size, img_size)
    draw = ImageDraw.Draw(background)
    
    margin = img_size * 0.15
    board_bbox = (margin, margin, img_size - margin, img_size - margin)
    
    board_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    board_line_width = random.randint(2, 4)
    draw_board(draw, board_bbox, board_color, board_line_width)
    
    annotations = np.zeros((3, 3), dtype=np.int32)
    
    x0, y0, x1, y1 = board_bbox
    cell_width = (x1 - x0) / 3
    cell_height = (y1 - y0) / 3
    
    for i in range(3):
        for j in range(3):
            cell_bbox = (x0 + j * cell_width, y0 + i * cell_height,
                         x0 + (j + 1) * cell_width, y0 + (i + 1) * cell_height)
            mark_choice = random.choices([None, "X", "O"], weights=[0.4, 0.3, 0.3])[0]
            if mark_choice is not None:
                scale_factor = random.uniform(0.6, 1.0)
                scaled_bbox = scale_bbox(cell_bbox, scale_factor)
                mark_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                mark_line_width = random.randint(2, 4)
                if mark_choice == "X":
                    draw_mark_X(draw, scaled_bbox, mark_color, mark_line_width)
                    annotations[i, j] = 1
                else:
                    draw_mark_O(draw, scaled_bbox, mark_color, mark_line_width)
                    annotations[i, j] = 2

    transformed = apply_random_transform(background)
    final_img = adjust_lighting(transformed)
    
    flat_annotation = annotations.flatten()
    return final_img, flat_annotation

# --- Main generation loop ---

def main():
    output_dir = "tic_tac_toe_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_images = 5000
    for _ in range(num_images):
        img, annotation = generate_tic_tac_toe_image()
        # Generate a unique filename based on the current timestamp
        unique_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_filename = os.path.join(output_dir, f"tic_tac_toe_{unique_id}")
        img.save(base_filename + ".png")
        np.save(base_filename + ".npy", annotation)
        print(f"Generated image and annotation: {base_filename}")
    
    print("Image generation complete.")

if __name__ == "__main__":
    main()
