from flask import Flask, jsonify, request
import cv2
import torch
import numpy as np
import time
import sqlite3
from flask_cors import CORS

# Import des fonctions existantes
from src.models.yolo_model import BoardDetector
from src.models.cnn_model import create_model
from src.utils.image_processing import capture_board_image, recenter_board_image, process_cropped_image
from src.detection.board_detector import detect_board_state
from src.utils.game_logic import check_winner, ai_move, display_board_state

app = Flask(__name__)
CORS(app)

# Variables globales pour stocker l'état du jeu
current_board = np.zeros(9, dtype=int)
player_turn = True
game_over = False
winner = 0
player_name = ""
player_email = ""
game_started = False
current_game_id = None

# Configuration de la base de données
DATABASE = 'tic_tac_toe.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
    CREATE TABLE IF NOT EXISTS games (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_name TEXT,
        player_email TEXT,
        created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

# Initialiser la base de données au démarrage
init_db()

# Initialisation des modèles
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
board_detector = BoardDetector(model_path='models/yolov5_board.pt', confidence_threshold=0.6)
cnn_model = create_model()
cnn_model.load_state_dict(torch.load('models/tic_tac_toe_model_best.pth', map_location=device))
cnn_model.to(device)
cnn_model.eval()

@app.route('/api/game/state', methods=['GET'])
def get_game_state():
    global current_board, player_turn, game_over, winner
    
    try:
        # Capture et analyse de l'image (comme dans game_loop.py)
        image = capture_board_image()
        recentered, detected = recenter_board_image(image, board_detector)
        
        if detected:
            processed = process_cropped_image(recentered)
            detected_board = detect_board_state(processed, cnn_model)
            
            if detected_board is not None:
                # Mise à jour de l'état du jeu
                current_board = detected_board
                
                # Vérification des conditions de fin de jeu
                winner = check_winner(current_board)
                if winner != 0:
                    game_over = True
        
        # Conversion pour JSON
        board_list = current_board.tolist()
        
        return jsonify({
            'board': board_list,
            'playerTurn': player_turn,
            'gameOver': game_over,
            'winner': winner,
            'gameStarted': game_started
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/game/ai-move', methods=['POST'])
def make_ai_move():
    global current_board, player_turn, game_over, winner
    
    if not player_turn and not game_over and game_started:
        # L'IA joue son coup (comme dans game_loop.py)
        current_board = ai_move(current_board)
        player_turn = True
        
        # Vérification des conditions de fin de jeu
        winner = check_winner(current_board)
        if winner != 0:
            game_over = True
    
    return jsonify({
        'board': current_board.tolist(),
        'playerTurn': player_turn,
        'gameOver': game_over,
        'winner': winner
    })

@app.route('/api/game/reset', methods=['POST'])
def reset_game():
    global current_board, player_turn, game_over, winner, game_started, current_game_id
    
    # Réinitialisation de l'état du jeu
    current_board = np.zeros(9, dtype=int)
    game_over = False
    winner = 0
    
    # Déterminer qui commence
    data = request.json
    player_turn = data.get('playerStarts', True)
    game_started = True
    
    # Créer une nouvelle partie dans la base de données
    if player_name:
        conn = get_db_connection()
        cursor = conn.execute('INSERT INTO games (player_name, player_email) VALUES (?, ?)', 
                            (player_name, player_email))
        current_game_id = cursor.lastrowid
        conn.commit()
        conn.close()
    
    return jsonify({
        'board': current_board.tolist(),
        'playerTurn': player_turn,
        'gameOver': game_over,
        'winner': winner,
        'gameStarted': game_started
    })

@app.route('/api/player/info', methods=['POST'])
def set_player_info():
    global player_name, player_email
    
    data = request.json
    player_name = data.get('name', '')
    player_email = data.get('email', '')
    
    return jsonify({
        'success': True,
        'name': player_name,
        'email': player_email
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
