import numpy as np
import cv2
import os
import random
from PIL import Image, ImageDraw, ImageFilter
from scipy.interpolate import interp1d
from tqdm import tqdm

# Configuration
IMG_SIZE = 512
GRID_SIZE = 3
SAVE_PATH = "readable_hand_drawn_ttt_dataset/"
NUM_SAMPLES = 5000
SYMBOL_SIZE_RATIO = 0.3  # Taille des symboles par rapport à la cellule

def generate_valid_board():
    board = np.zeros(9, dtype=np.uint8)
    players = [1, 2]
    turn = 0
    moves = list(range(9))
    random.shuffle(moves)
    
    for move in moves:
        player = players[turn % 2]
        board[move] = player
        if is_win(board, player):
            break
        turn += 1
    
    return board

def is_win(board, player):
    board = board.reshape(3, 3)
    return (np.any(np.all(board == player, axis=1)) or
            np.any(np.all(board == player, axis=0)) or
            np.all(np.diag(board) == player) or
            np.all(np.diag(np.fliplr(board)) == player))

def generate_simple_background():
    """Fond simple et uniforme."""
    bg = np.full((IMG_SIZE, IMG_SIZE, 3), random.randint(150, 200), dtype=np.uint8)
    return bg

def apply_glass_effect(img):
    """Applique un effet verre léger."""
    overlay = np.full_like(img, 255)
    alpha = 0.1  # Réduit l'intensité de l'effet
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img

def natural_curve(start, end, control_points=3, smoothness=50):
    """Courbes plus douces et contrôlées."""
    points = [start]
    for _ in range(control_points):
        points.append((
            random.randint(min(start[0], end[0]), max(start[0], end[0])),
            random.randint(min(start[1], end[1]), max(start[1], end[0]))
        ))
    points.append(end)
    
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    
    t = np.linspace(0, 1, smoothness)
    x_smooth = interp1d(np.linspace(0, 1, len(points)), x, kind='quadratic')(t)  # 'quadratic'
    y_smooth = interp1d(np.linspace(0, 1, len(points)), y, kind='quadratic')(t)
    
    return list(zip(x_smooth.astype(int), y_smooth.astype(int)))

def draw_natural_line(draw, start, end, width, color):
    points = natural_curve(start, end)
    draw.line(points, fill=color, width=width, joint="curve")

def draw_readable_symbol(draw, pos, symbol, cell_size):
    """Symboles plus clairs et lisibles."""
    x, y = pos
    symbol_size = int(cell_size * SYMBOL_SIZE_RATIO)
    color = (random.randint(180, 255), 0, 0) if symbol == 1 else (0, 0, random.randint(180, 255))
    
    if symbol == 1:  # X
        start = (x - symbol_size, y - symbol_size)
        end = (x + symbol_size, y + symbol_size)
        draw_natural_line(draw, start, end, random.randint(7, 10), color)
        
        start = (x + symbol_size, y - symbol_size)
        end = (x - symbol_size, y + symbol_size)
        draw_natural_line(draw, start, end, random.randint(7, 10), color)
        
    else:  # O
        points = natural_curve((x, y - symbol_size), (x, y - symbol_size), control_points=5, smoothness=100)
        draw.line(points + [points[0]], fill=color, width=random.randint(7, 10), joint="curve")

def generate_image(board):
    bg = generate_simple_background()
    img = Image.fromarray(bg)
    draw = ImageDraw.Draw(img)
    
    # Draw grid
    for i in range(1, GRID_SIZE):
        line_pos = i * IMG_SIZE // GRID_SIZE
        start_x = random.randint(-10, 10)
        end_x = random.randint(-10, 10)
        start_y = random.randint(-10, 10)
        end_y = random.randint(-10, 10)
        draw_natural_line(draw, (line_pos + start_x, 0), (line_pos + end_x, IMG_SIZE), random.randint(3, 5), (180, 180, 180))
        draw_natural_line(draw, (0, line_pos + start_y), (IMG_SIZE, line_pos + end_y), random.randint(3, 5), (180, 180, 180))
    
    cell_size = IMG_SIZE // GRID_SIZE
    for idx, val in enumerate(board):
        if val != 0:
            row, col = divmod(idx, GRID_SIZE)
            pos = (col * cell_size + cell_size // 2 + random.randint(-10, 10),
                   row * cell_size + cell_size // 2 + random.randint(-10, 10))
            draw_readable_symbol(draw, pos, val, cell_size)
    
    img = img.filter(ImageFilter.SMOOTH_MORE)
    img = np.array(img)
    img = apply_glass_effect(img)
    
    return img

def generate_dataset(num_samples):
    os.makedirs(f"{SAVE_PATH}images", exist_ok=True)
    os.makedirs(f"{SAVE_PATH}annotations", exist_ok=True)
    
    for i in tqdm(range(num_samples)):
        board = generate_valid_board()
        img = generate_image(board)
        
        base_name = f"readable_hand_drawn_ttt_{i:05d}"
        cv2.imwrite(f"{SAVE_PATH}images/{base_name}.png", img)
        np.save(f"{SAVE_PATH}annotations/{base_name}.npy", board)

if __name__ == "__main__":
    generate_dataset(NUM_SAMPLES)
    print(f"Dataset généré avec succès : {NUM_SAMPLES} images dans {SAVE_PATH}")
