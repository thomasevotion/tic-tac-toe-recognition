from flask import Flask, jsonify, request
from flask.json import JSONEncoder
from flask_cors import CORS
import sqlite3
import threading
import time
import numpy as np
import cv2
import torch

# Importer les modules existants
from src.models.yolo_model import BoardDetector
from src.models.cnn_model import create_model
from src.utils.image_processing import capture_board_image, recenter_board_image, process_cropped_image
from src.detection.board_detector import detect_board_state
from src.utils.game_logic import check_winner, ai_move, display_board_state
from src.game.game_loop import wait_for_empty_board

# Créer un encodeur JSON personnalisé pour gérer les types NumPy
class NumpyJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app = Flask(__name__)
app.json_encoder = NumpyJSONEncoder
CORS(app)

# Variables globales pour stocker l'état du jeu
game_state = {
    "board": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    "player_turn": True,
    "game_over": False,
    "winner": 0,
    "game_started": False,
    "waiting_for_board": False,
    "board_visible": False,
    "ai_playing": False
}

player_info = {
    "name": "",
    "email": ""
}

# Initialisation des modèles (comme dans main.py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
board_detector = BoardDetector(model_path='models/yolov5_board.pt', confidence_threshold=0.6)
cnn_model = create_model()
cnn_model.load_state_dict(torch.load('models/tic_tac_toe_model_best.pth', map_location=device))
cnn_model.to(device)
cnn_model.eval()

# Base de données pour stocker les informations des joueurs
DATABASE = 'tic_tac_toe.db'

def init_db():
    conn = sqlite3.connect(DATABASE)
    conn.execute('''
    CREATE TABLE IF NOT EXISTS games (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_name TEXT,
        player_email TEXT,
        created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

init_db()

# Fonction pour attendre un plateau vide dans un thread séparé
def wait_for_empty_board_thread():
    global game_state
    game_state["waiting_for_board"] = True
    print("Attente d'un plateau vide...")
    
    try:
        wait_for_empty_board(board_detector, cnn_model, consecutive_empty_required=3)
        game_state["board"] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        game_state["game_started"] = True
        game_state["waiting_for_board"] = False
        game_state["board_visible"] = True
        print("Plateau vide détecté, jeu prêt!")
    except Exception as e:
        print(f"Erreur lors de l'attente du plateau: {e}")
        game_state["waiting_for_board"] = False

# Fonction pour mettre à jour l'état du jeu en continu
def update_game_state_thread():
    global game_state
    
    # Variables pour suivre les changements de plateau
    previous_board = np.array(game_state["board"])
    consecutive_stable_detections = 0
    last_detected_board = None
    
    # Variables pour suivre la confirmation de victoire
    consecutive_win_detections = 0
    last_winner = 0
    
    while True:
        if game_state["game_started"] and not game_state["waiting_for_board"]:
            try:
                # Si c'est le tour de l'IA, faire jouer l'IA automatiquement
                if not game_state["player_turn"] and not game_state["game_over"] and not game_state["ai_playing"]:
                    print("Tour de l'IA...")
                    board = np.array(game_state["board"])
                    board = ai_move(board)
                    game_state["board"] = board.tolist()
                    game_state["ai_playing"] = True
                    game_state["expected_board"] = board.tolist()
                    print("L'IA a joué:")
                    display_board_state(board)
                
                image = capture_board_image()
                recentered, detected = recenter_board_image(image, board_detector)
                
                if detected:
                    game_state["board_visible"] = True
                    processed = process_cropped_image(recentered)
                    detected_board = detect_board_state(processed, cnn_model)
                    
                    if detected_board is not None:
                        print("État du plateau détecté:")
                        display_board_state(detected_board)
                        
                        # Si l'IA a joué et attend confirmation
                        if game_state["ai_playing"]:
                            # Vérifier si le O de l'IA est détecté sur le plateau
                            expected_board = np.array(game_state["expected_board"])
                            if np.array_equal(detected_board, expected_board):
                                game_state["board"] = detected_board.tolist()
                                game_state["player_turn"] = True
                                game_state["ai_playing"] = False
                                print("Coup de l'IA confirmé, c'est au tour du joueur.")
                                previous_board = detected_board.copy()
                        
                        # Vérifier si un nouveau coup a été joué par le joueur
                        elif game_state["player_turn"] and not game_state["game_over"]:
                            # Vérifier que le joueur a bien ajouté exactement un X
                            x_count_before = np.sum(previous_board == 1)
                            x_count_after = np.sum(detected_board == 1)
                            o_count_before = np.sum(previous_board == 2)
                            o_count_after = np.sum(detected_board == 2)
                            
                            # Le joueur doit ajouter exactement un X et ne pas modifier les O
                            if x_count_after == x_count_before + 1 and o_count_after == o_count_before:
                                # Confirmer la stabilité de la détection
                                if last_detected_board is not None and np.array_equal(detected_board, last_detected_board):
                                    consecutive_stable_detections += 1
                                else:
                                    consecutive_stable_detections = 1
                                    last_detected_board = detected_board.copy()
                                
                                # Considérer le mouvement comme valide après 3 détections stables
                                if consecutive_stable_detections >= 3:
                                    # Mettre à jour l'état du jeu
                                    game_state["board"] = detected_board.tolist()
                                    previous_board = detected_board.copy()
                                    
                                    # Changer de tour
                                    game_state["player_turn"] = False
                                    consecutive_stable_detections = 0
                                    print("Mouvement du joueur détecté, c'est au tour de l'IA.")
                            else:
                                consecutive_stable_detections = 0
                        
                        # Vérifier s'il y a un gagnant
                        winner = check_winner(detected_board)
                        
                        # Si un gagnant potentiel est détecté
                        if winner != 0:
                            # Vérifier si c'est le même gagnant que la dernière fois
                            if winner == last_winner:
                                consecutive_win_detections += 1
                            else:
                                consecutive_win_detections = 1
                                last_winner = winner
                            
                            # Ne déclarer la victoire qu'après plusieurs détections consécutives
                            if consecutive_win_detections >= 2:
                                game_state["game_over"] = True
                                game_state["winner"] = winner
                                print(f"Fin de partie confirmée: {'Joueur' if winner == 1 else 'IA' if winner == 2 else 'Match nul'}")
                        else:
                            # Réinitialiser le compteur si aucun gagnant n'est détecté
                            consecutive_win_detections = 0
                            last_winner = 0
                else:
                    game_state["board_visible"] = False
                    
            except Exception as e:
                print(f"Erreur lors de la mise à jour de l'état: {e}")
        
        time.sleep(0.5)  # Vérifier l'état toutes les 500ms

# Routes API
@app.route('/api/game/state', methods=['GET'])
def get_game_state():
    return jsonify(game_state)

@app.route('/api/game/reset', methods=['POST'])
def reset_game():
    global game_state, player_info
    
    # Réinitialiser l'état du jeu
    game_state["game_over"] = False
    game_state["winner"] = 0
    game_state["player_turn"] = True
    game_state["game_started"] = False
    game_state["board_visible"] = False
    game_state["ai_playing"] = False
    
    # Enregistrer la partie dans la base de données
    if player_info["name"]:
        conn = sqlite3.connect(DATABASE)
        conn.execute('INSERT INTO games (player_name, player_email) VALUES (?, ?)', 
                    (player_info["name"], player_info["email"]))
        conn.commit()
        conn.close()
        print(f"Partie enregistrée pour {player_info['name']}")
    
    # Démarrer l'attente du plateau vide dans un thread séparé
    threading.Thread(target=wait_for_empty_board_thread, daemon=True).start()
    
    return jsonify(game_state)

@app.route('/api/player/info', methods=['POST'])
def set_player_info():
    global player_info
    
    data = request.json
    player_info["name"] = data.get('name', '')
    player_info["email"] = data.get('email', '')
    
    print(f"Joueur: {player_info['name']}, Email: {player_info['email']}")
    
    return jsonify({
        'success': True,
        'name': player_info["name"],
        'email': player_info["email"]
    })

if __name__ == '__main__':
    # Démarrer le thread de mise à jour de l'état du jeu
    update_thread = threading.Thread(target=update_game_state_thread, daemon=True)
    update_thread.start()
    
    print("Démarrage du serveur de jeu de morpion...")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
