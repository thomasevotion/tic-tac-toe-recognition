import cv2
import time
import torch
from torchvision import transforms

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cropped is None:
        return None
        
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    processed = transform_cnn(cropped_rgb).unsqueeze(0).to(device)
    return processed

# Définir ici la transformation pour le CNN
transform_cnn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
