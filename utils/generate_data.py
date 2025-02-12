import numpy as np
import cv2
import os
import random
from PIL import Image, ImageDraw
from tqdm import tqdm
from sklearn.utils import shuffle

# Configuration
IMG_SIZE = 224
GRID_SIZE = 3
SAVE_PATH = "tic_tac_toe_dataset/"
NUM_SAMPLES = 5000

def generate_valid_board():
    """Génère un plateau valide avec logique de jeu réaliste"""
    board = np.zeros(9, dtype=np.uint8)
    players = [1, 2]
    turn = 0
    moves = list(range(9))
    random.shuffle(moves)
    
    for move in moves:
        player = players[turn % 2]
        board[move] = player
        if is_win(board.reshape(3, 3), player):
            break
        turn += 1
    
    return board

def is_win(board, player):
    """Vérification optimisée des conditions de victoire"""
    return (np.any(np.all(board == player, axis=0)) | 
            np.any(np.all(board == player, axis=1)) |
            (board[0,0] == board[1,1] == board[2,2] == player) |
            (board[0,2] == board[1,1] == board[2,0] == player))

def draw_symbol(draw, pos, symbol, size):
    """Dessin avec variations réalistes"""
    x, y = pos
    color = random.choice(['#FF0000', '#0000FF'])  # Rouge pour X, Bleu pour O
    thickness = random.randint(3,5)
    
    if symbol == 1:  # X
        offset = int(size * 0.3)
        draw.line([(x-offset, y-offset), (x+offset, y+offset)], 
                 fill=color, width=thickness, joint='curve')
        draw.line([(x+offset, y-offset), (x-offset, y+offset)], 
                 fill=color, width=thickness, joint='curve')
    else:  # O
        offset = int(size * 0.25)
        for i in range(thickness):
            draw.ellipse([(x-offset+i, y-offset+i), 
                        (x+offset-i, y+offset-i)], 
                        outline=color, width=1)

def generate_image(board):
    """Génère une image réaliste avec textures"""
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), '#F0F0F0')
    draw = ImageDraw.Draw(img)
    
    # Grille avec imperfections
    for i in range(1, GRID_SIZE):
        jitter = random.randint(-3, 3)
        draw.line([(i*IMG_SIZE//3 + jitter, 0), 
                 (i*IMG_SIZE//3 + jitter, IMG_SIZE)], 
                fill='#404040', width=random.randint(2,4))
        draw.line([(0, i*IMG_SIZE//3 + jitter), 
                 (IMG_SIZE, i*IMG_SIZE//3 + jitter)], 
                fill='#404040', width=random.randint(2,4))
    
    # Placement des symboles
    for idx, val in enumerate(board):
        if val == 0: continue
        row, col = divmod(idx, 3)
        x = col*IMG_SIZE//3 + IMG_SIZE//6 + random.randint(-10,10)
        y = row*IMG_SIZE//3 + IMG_SIZE//6 + random.randint(-10,10)
        draw_symbol(draw, (x,y), val, IMG_SIZE//3)
    
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def generate_dataset(num_samples):
    """Génère les données avec sauvegarde pairée image/annotation"""
    # Création des répertoires
    os.makedirs(f"{SAVE_PATH}images", exist_ok=True)
    os.makedirs(f"{SAVE_PATH}annotations", exist_ok=True)
    
    X = []
    y = []
    
    for i in tqdm(range(num_samples), desc="Génération"):
        board = generate_valid_board()
        img = generate_image(board)
        
        # Nom de base commun
        base_name = f"tic_tac_toe_{i:04d}"
        
        # Sauvegarde de l'image
        img_path = f"{SAVE_PATH}images/{base_name}.png"
        cv2.imwrite(img_path, img)
        
        # Sauvegarde de l'annotation
        annotation_path = f"{SAVE_PATH}annotations/{base_name}.npy"
        np.save(annotation_path, board)
        
        # Stockage pour le fichier global
        X.append(img)
        y.append(board)
    
    # Sauvegarde des fichiers numpy globaux
    X, y = shuffle(np.array(X), np.array(y))
    np.save(f"{SAVE_PATH}X_global.npy", X)
    np.save(f"{SAVE_PATH}y_global.npy", y)

if __name__ == "__main__":
    generate_dataset(NUM_SAMPLES)
    print(f"Dataset généré dans {SAVE_PATH}")
    print(f"Images: {len(os.listdir(SAVE_PATH+'images'))} fichiers PNG")
    print(f"Annotations: {len(os.listdir(SAVE_PATH+'annotations'))} fichiers .npy")
