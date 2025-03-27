import cv2
import torch
import os
import time
from datetime import datetime
from src.models.yolo_model import BoardDetector
from src.utils.image_processing import capture_board_image, recenter_board_image

# Créer les dossiers de sauvegarde s'ils n'existent pas
os.makedirs("cam", exist_ok=True)
os.makedirs("board", exist_ok=True)

# Initialiser le détecteur de plateau
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
board_detector = BoardDetector(model_path='models/yolov5_board.pt', confidence_threshold=0.6)

print("Programme de capture d'images démarré. Appuyez sur Ctrl+C pour arrêter.")

try:
    while True:
        # Générer un timestamp pour le nom de fichier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Capturer l'image de la caméra
        image = capture_board_image()
        
        # Sauvegarder l'image de la caméra
        cam_path = os.path.join("cam", f"camera_{timestamp}.jpg")
        cv2.imwrite(cam_path, image)
        print(f"Image de caméra sauvegardée: {cam_path}")
        
        # Recadrer sur le plateau si détecté
        recentered, detected = recenter_board_image(image, board_detector)
        
        if detected:
            # Sauvegarder l'image du plateau recentré
            board_path = os.path.join("board", f"board_{timestamp}.jpg")
            cv2.imwrite(board_path, recentered)
            print(f"Plateau recentré sauvegardé: {board_path}")
        else:
            print("Aucun plateau détecté dans cette image")
        
        # Attendre 1 seconde
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\nProgramme arrêté par l'utilisateur.")
finally:
    cv2.destroyAllWindows()
    print(f"Images sauvegardées dans les dossiers 'cam' et 'board'.")
