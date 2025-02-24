#!/usr/bin/env python3
import time
import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
from torchvision import models
# from xarm.wrapper import XArmAPI
# from robot.o_move import RobotMain  # Importation de la classe RobotMain depuis votre fichier original

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
    def __init__(self, model_path='models/yolov5_board.pt', confidence_threshold=0.7):
        """
        Charge un modèle YOLO personnalisé entraîné pour détecter le plateau.
        Le modèle doit renvoyer une bounding box pour le plateau.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Chargement du modèle via torch.hub (assurez-vous d'avoir installé yolov5)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        self.model.conf = confidence_threshold  # Seuil de confiance élevé pour réduire les faux positifs
        self.model.to(self.device)
        self.model.eval()
        
        # Compteur pour la stabilisation des détections
        self.board_detection_counter = 0
        self.last_detection = None
        self.required_consecutive_detections = 3  # Nombre de détections consécutives requises

    def detect_board(self, image):
        """
        Prend une image (BGR) et renvoie les coins de la bounding box du plateau sous forme
        de liste de tuples [(x1, y1), (x2, y1), (x2, y2), (x1, y2)].
        Si aucune détection n'est faite, retourne None.
        """
        results = self.model(image)
        detections = results.xyxy[0].cpu().numpy()
        
        if len(detections) == 0:
            self.board_detection_counter = 0
            self.last_detection = None
            return None
        
        best_detection = detections[0]
        x1, y1, x2, y2 = best_detection[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        current_detection = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        
        # Vérifier si c'est une détection similaire à la précédente
        if self.last_detection is not None:
            # Calcul de la différence moyenne des coordonnées
            diffs = []
            for i in range(4):
                for j in range(2):
                    diffs.append(abs(current_detection[i][j] - self.last_detection[i][j]))
            avg_diff = sum(diffs) / len(diffs)
            
            if avg_diff < 20:  # Si les détections sont proches
                self.board_detection_counter += 1
            else:
                self.board_detection_counter = 1  # Nouvelle détection stable
        else:
            self.board_detection_counter = 1
        
        self.last_detection = current_detection
        
        # Ne renvoyer la détection que si elle est stable sur plusieurs frames
        if self.board_detection_counter >= self.required_consecutive_detections:
            return current_detection
        return None

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
# arm = XArmAPI('192.168.1.241', baud_checkset=False)

# Création d'une instance de RobotMain
# robot = RobotMain(arm)

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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Attendre que la caméra s'initialise
    time.sleep(0.2)
    
    # Capturer plusieurs images et garder la dernière (pour stabiliser la caméra)
    for _ in range(3):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Impossible de capturer une image depuis la caméra.")
        time.sleep(0.1)
    
    cap.release()
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
        # Vérifier que la bounding box est assez grande pour être un plateau
        width = x2 - x1
        height = y2 - y1
        
        # Si la bounding box est trop petite, c'est probablement une fausse détection
        min_size = min(image.shape[0], image.shape[1]) * 0.2  # Au moins 20% de l'image
        if width < min_size or height < min_size:
            print(f"Détection ignorée car trop petite: {width}x{height} pixels")
            return None, False
            
        # Ajouter une marge autour du plateau
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(image.shape[1], x2 + margin)
        y2 = min(image.shape[0], y2 + margin)
        
        cropped = image[y1:y2, x1:x2]
        cropped_resized = cv2.resize(cropped, (224, 224))
        return cropped_resized, True
    else:
        return None, False

def process_cropped_image(cropped):
    """
    Convertit l'image recadrée (BGR) en RGB et applique la transformation pour le CNN.
    """
    if cropped is None:
        return None
        
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    processed = transform_cnn(cropped_rgb).unsqueeze(0).to(device)
    return processed

def detect_board_state(processed_image):
    """
    Utilise le modèle CNN pour obtenir l'état du plateau.
    La sortie est un vecteur de 9 entiers (0 = case vide, 1 = X, 2 = O).
    """
    if processed_image is None:
        return None
        
    with torch.no_grad():
        output = cnn_model(processed_image)  # forme attendue : (1, 9, 3)
        # Extraire les probabilités pour chaque classe
        probabilities = torch.nn.functional.softmax(output, dim=2)
        # Définir un seuil de confiance pour chaque classe
        confidence = torch.max(probabilities, dim=2)[0]
        # Obtenir les prédictions
        predictions = torch.argmax(output, dim=2).cpu().numpy().flatten()
        confidence = confidence.cpu().numpy().flatten()
        
        # Appliquer un seuil de confiance (0.7) pour les prédictions
        for i in range(9):
            if confidence[i] < 0.7:
                predictions[i] = 0  # En cas de doute, considérer la case comme vide
                
    return predictions

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
    L'IA joue un coup simple mais stratégique.
    Priorité: gagner > bloquer l'adversaire > centre > coins > côtés
    """
    board_copy = board.copy()
    
    # Vérifier si l'IA peut gagner
    for i in range(9):
        if board_copy[i] == 0:
            board_copy[i] = 2
            if check_winner(board_copy) == 2:
                return board_copy
            board_copy[i] = 0
    
    # Bloquer une victoire potentielle du joueur
    for i in range(9):
        if board_copy[i] == 0:
            board_copy[i] = 1
            if check_winner(board_copy) == 1:
                board_copy[i] = 2
                return board_copy
            board_copy[i] = 0
    
    # Jouer au centre si disponible
    if board_copy[4] == 0:
        board_copy[4] = 2
        return board_copy
    
    # Jouer dans un coin disponible
    corners = [0, 2, 6, 8]
    for corner in corners:
        if board_copy[corner] == 0:
            board_copy[corner] = 2
            return board_copy
    
    # Jouer sur un côté disponible
    sides = [1, 3, 5, 7]
    for side in sides:
        if board_copy[side] == 0:
            board_copy[side] = 2
            return board_copy
    
    return board_copy  # Au cas où

def wait_for_empty_board(board_detector, consecutive_empty_required=5):
    """
    Attend activement qu'un plateau vide soit détecté avant de démarrer une nouvelle partie.
    Exige plusieurs détections consécutives pour confirmer que le plateau est bien vide.
    """
    print("Attente d'un plateau vide...")
    consecutive_empty = 0
    
    while consecutive_empty < consecutive_empty_required:
        try:
            image = capture_board_image()
            # Afficher l'image capturée pour déboguer
            cv2.imshow("Caméra", cv2.resize(image, (640, 480)))
            cv2.waitKey(1)
            
            recentered, detected = recenter_board_image(image, board_detector)
            
            if not detected:
                print("Pas de plateau détecté.")
                consecutive_empty = 0
                time.sleep(1)
                continue
                
            # Afficher le plateau recadré pour déboguer
            cv2.imshow("Plateau détecté", recentered)
            cv2.waitKey(1)
            
            processed = process_cropped_image(recentered)
            board = detect_board_state(processed)
            
            if board is None:
                consecutive_empty = 0
                continue
                
            print(f"État du plateau détecté: {board}")
            
            if np.all(board == 0):
                consecutive_empty += 1
                print(f"Plateau vide détecté ({consecutive_empty}/{consecutive_empty_required})")
            else:
                consecutive_empty = 0
                print("Plateau non vide détecté. Attente...")
                
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Erreur lors de la détection: {e}")
            consecutive_empty = 0
            time.sleep(1)
    
    print("Plateau vide confirmé! Prêt à jouer.")
    return

def display_board_state(board):
    """
    Affiche l'état du plateau sous forme de grille 3x3 dans la console.
    """
    symbols = [" ", "X", "O"]
    print("-" * 13)
    for i in range(3):
        row = "| "
        for j in range(3):
            row += symbols[board[i*3 + j]] + " | "
        print(row)
        print("-" * 13)

#############################################
# Boucle principale du jeu
#############################################
def game_loop():
    # Créer le détecteur de plateau avec un seuil de confiance élevé
    board_detector = BoardDetector(model_path='models/yolov5_board.pt', confidence_threshold=0.6)
    
    # Attendre qu'un plateau vide soit détecté de manière fiable
    wait_for_empty_board(board_detector, consecutive_empty_required=3)
    
    while True:
        print("\n=== Nouveau jeu démarré ===")
        current_board = np.zeros(9, dtype=int)
        display_board_state(current_board)
        
        game_over = False
        player_turn = True  # Le joueur commence
        move_counter = 0    # Compteur de coups pour détecter les problèmes

        while not game_over:
            if player_turn:
                print("Tour du joueur. Veuillez jouer votre coup...")
                move_detected = False
                detection_attempts = 0
                consecutive_stable_detections = 0
                last_detected_board = None
                
                # Boucle de détection du mouvement du joueur
                while not move_detected and detection_attempts < 30:  # Limite de 30 tentatives
                    detection_attempts += 1
                    
                    try:
                        image = capture_board_image()
                        recentered, detected = recenter_board_image(image, board_detector)
                        
                        if not detected:
                            print("Plateau non détecté, réessayez...")
                            consecutive_stable_detections = 0
                            time.sleep(1)
                            continue
                            
                        # Afficher le plateau détecté pour déboguer
                        cv2.imshow("Plateau détecté", recentered)
                        cv2.waitKey(1)
                            
                        processed = process_cropped_image(recentered)
                        new_board = detect_board_state(processed)
                        
                        if new_board is None:
                            consecutive_stable_detections = 0
                            continue
                            
                        # Vérifier si un nouveau coup a été joué
                        if not np.array_equal(new_board, current_board):
                            # Vérifier que le joueur a bien ajouté exactement un X
                            x_count_before = np.sum(current_board == 1)
                            x_count_after = np.sum(new_board == 1)
                            o_count_before = np.sum(current_board == 2)
                            o_count_after = np.sum(new_board == 2)
                            
                            # Le joueur doit ajouter exactement un X et ne pas modifier les O
                            if x_count_after == x_count_before + 1 and o_count_after == o_count_before:
                                # Confirmer la stabilité de la détection
                                if last_detected_board is not None and np.array_equal(new_board, last_detected_board):
                                    consecutive_stable_detections += 1
                                else:
                                    consecutive_stable_detections = 1
                                    last_detected_board = new_board.copy()
                                
                                # Considérer le mouvement comme valide après 3 détections stables
                                if consecutive_stable_detections >= 3:
                                    current_board = new_board.copy()
                                    move_detected = True
                                    print(f"Mouvement du joueur détecté après {detection_attempts} tentatives.")
                            else:
                                consecutive_stable_detections = 0
                                
                        time.sleep(0.5)
                        
                    except Exception as e:
                        print(f"Erreur pendant la détection: {e}")
                        time.sleep(1)
                
                if not move_detected:
                    print("Impossible de détecter un mouvement valide après plusieurs tentatives.")
                    print("Redémarrage du jeu...")
                    break
                
                print("\nPlateau après le coup du joueur:")
                display_board_state(current_board)
                
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
                move_counter += 1

            else:
                print("\nTour de l'IA...")
                time.sleep(1)  # Pause pour simuler la réflexion
                
                # Commenter la ligne ci-dessous si vous voulez utiliser le robot physique
                # robot.run()
                
                # L'IA joue
                current_board = ai_move(current_board)
                
                print("L'IA a joué:")
                display_board_state(current_board)
                
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
                move_counter += 1
                
                # Vérification de sécurité: si trop de coups, quelque chose ne va pas
                if move_counter > 15:  # Un jeu de morpion ne peut pas avoir plus de 9 coups
                    print("Erreur: Trop de coups joués. Réinitialisation du jeu.")
                    break

        print("Fin de partie. Attendez quelques secondes...")
        time.sleep(3)
        
        # Fermer les fenêtres OpenCV
        cv2.destroyAllWindows()
        
        # Attendre qu'un plateau vide soit détecté pour redémarrer
        wait_for_empty_board(board_detector, consecutive_empty_required=3)

def main():
    try:
        # Création des fenêtres persistantes dès le début
        cv2.namedWindow("Caméra", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Plateau détecté", cv2.WINDOW_NORMAL)
        game_loop()
    except KeyboardInterrupt:
        print("\nInterruption par l'utilisateur. Fin du programme.")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()