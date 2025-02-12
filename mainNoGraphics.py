#!/usr/bin/env python3
import time
import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
from torchvision import models
from xarm.wrapper import XArmAPI
from robot.o_move import RobotMain  # Importation de la classe RobotMain depuis votre fichier original

#############################################
# Définition du modèle (même architecture qu'à l'entraînement)
#############################################
class TicTacToeResNet(nn.Module):
    def __init__(self, base_model):
        super(TicTacToeResNet, self).__init__()
        self.base = base_model

    def forward(self, x):
        x = self.base(x)  # sortie attendue : (batch, 27)
        x = x.view(-1, 9, 3)
        return x

def create_model():
    # Chargement du ResNet18 pré-entraîné
    base_model = models.resnet18(pretrained=True)
    in_features = base_model.fc.in_features
    # Remplacer la dernière couche pour obtenir 27 sorties (9 cases x 3 classes)
    base_model.fc = nn.Linear(in_features, 9 * 3)
    model = TicTacToeResNet(base_model)
    return model

#############################################
# Classe BoardDetector utilisant YOLOv5
#############################################
class BoardDetector:
    def __init__(self, model_path='models/yolov5_board.pt'):
        """
        Charge un modèle YOLO personnalisé entraîné pour détecter le plateau.
        Le modèle doit renvoyer une bounding box pour le plateau.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Chargement du modèle via torch.hub (assurez-vous d'avoir installé yolov5)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        self.model.to(self.device)
        self.model.eval()

    def detect_board(self, image):
        """
        Prend une image (BGR) et renvoie les coins de la bounding box du plateau sous forme
        de liste de tuples [(x1, y1), (x2, y1), (x2, y2), (x1, y2)].
        Si aucune détection n'est faite, retourne None.
        """
        results = self.model(image)
        detections = results.xyxy[0].cpu().numpy()
        if len(detections) == 0:
            return None
        best_detection = detections[0]
        x1, y1, x2, y2 = best_detection[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return corners

#############################################
# Chargement du modèle CNN pour l'état du plateau
#############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Créer l'instance du modèle
cnn_model = create_model()
# Charger le state_dict dans le modèle
state_dict = torch.load('models/tic_tac_toe_model_best.pth', map_location=device)
cnn_model.load_state_dict(state_dict)
cnn_model.to(device)
cnn_model.eval()

# Transformation appliquée aux images destinées au CNN
transform_cnn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Initialisation du robot avec son adresse IP
arm = XArmAPI('192.168.1.241', baud_checkset=False)

# Création d'une instance de RobotMain
robot = RobotMain(arm)

#############################################
# Fonctions utilitaires pour la capture et le traitement des images
#############################################
def capture_board_image():
    """
    Capture une image depuis la caméra.
    Retourne une image au format BGR (format natif OpenCV).
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Impossible de capturer une image depuis la caméra.")
    return frame  # image en BGR

def recenter_board_image(image, board_detector):
    """
    Utilise BoardDetector pour détecter le plateau dans l'image.
    Si une bounding box est détectée, recadre l'image sur cette zone.
    Sinon, renvoie l'image complète.
    """
    corners = board_detector.detect_board(image)
    if corners is not None:
        (x1, y1), (x2, _), (_, y2), _ = corners
        cropped = image[y1:y2, x1:x2]
    else:
        cropped = image
    cropped = cv2.resize(cropped, (224, 224))
    return cropped

def process_cropped_image(cropped):
    """
    Convertit l'image recadrée (BGR) en RGB et applique la transformation pour le CNN.
    """
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    processed = transform_cnn(cropped_rgb).unsqueeze(0).to(device)
    return processed

def detect_board_state(processed_image):
    """
    Utilise le modèle CNN pour obtenir l'état du plateau.
    La sortie est un vecteur de 9 entiers (0 = case vide, 1 = X, 2 = O).
    """
    with torch.no_grad():
        output = cnn_model(processed_image)  # forme attendue : (1, 9, 3)
        board_state = torch.argmax(output, dim=2).cpu().numpy().flatten()
    return board_state

#############################################
# Fonctions de logique du jeu
#############################################
def check_winner(board):
    """
    Vérifie l'état du plateau.
    Retourne :
      0  : partie en cours,
      1  : victoire du joueur (X),
      2  : victoire de l'IA (O),
     -1  : match nul.
    """
    win_conditions = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),   # Lignes
        (0, 3, 6), (1, 4, 7), (2, 5, 8),   # Colonnes
        (0, 4, 8), (2, 4, 6)               # Diagonales
    ]
    for (a, b, c) in win_conditions:
        if board[a] == board[b] == board[c] and board[a] != 0:
            return board[a]
    if 0 not in board:
        return -1  # Match nul
    return 0  # Partie en cours

def ai_move(board):
    """
    L'IA joue un coup simple : elle occupe la première case vide.
    """
    board_list = list(board)
    for i in range(9):
        if board_list[i] == 0:
            board_list[i] = 2  # L'IA joue O
            return np.array(board_list)
    return board

def wait_for_empty_board(board_detector):
    """
    Attend activement qu'un plateau vide soit détecté avant de démarrer une nouvelle partie.
    """
    print("Attente d'un plateau vide...")
    while True:
        image = capture_board_image()
        recentered = recenter_board_image(image, board_detector)
        processed = process_cropped_image(recentered)
        board = detect_board_state(processed)
        if np.all(board == 0):
            print("Plateau vide détecté.")
            return
        time.sleep(1)

#############################################
# Boucle principale du jeu
#############################################
def game_loop():
    board_detector = BoardDetector(model_path='models/yolov5_board.pt')
    wait_for_empty_board(board_detector)
    
    while True:
        print("\n=== Nouveau jeu démarré ===")
        current_board = np.zeros(9, dtype=int)
        game_over = False
        player_turn = True  # Le joueur commence

        while not game_over:
            if player_turn:
                print("Tour du joueur. Veuillez jouer votre coup...")
                move_detected = False
                while not move_detected:
                    image = capture_board_image()
                    recentered = recenter_board_image(image, board_detector)
                    processed = process_cropped_image(recentered)
                    new_board = detect_board_state(processed)
                    if not np.array_equal(new_board, current_board):
                        current_board = new_board
                        move_detected = True
                    time.sleep(0.5)

                print(f"Plateau après le coup du joueur: {current_board}")
                result = check_winner(current_board)
                if result == 1:
                    print("Le joueur a gagné !")
                    game_over = True
                    continue
                elif result == -1:
                    print("Match nul !")
                    game_over = True
                    continue
                player_turn = False

            else:
                print("Tour de l'IA...")
                robot.run()
                current_board = ai_move(current_board)
                print(f"Plateau après le coup de l'IA: {current_board}")
                result = check_winner(current_board)
                if result == 2:
                    print("L'IA a gagné !")
                    game_over = True
                    continue
                elif result == -1:
                    print("Match nul !")
                    game_over = True
                    continue
                player_turn = True

        print("Fin de partie. Attente d'un nouveau plateau vide pour redémarrer...")
        time.sleep(3)
        wait_for_empty_board(board_detector)

def main():
    try:
        game_loop()
    except KeyboardInterrupt:
        print("Interruption par l'utilisateur. Fin du programme.")

if __name__ == '__main__':
    main()
