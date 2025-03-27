import torch
from src.models.yolo_model import BoardDetector
import torch.nn.functional as F

def detect_board_state(processed_image, cnn_model):
    """
    Utilise le modèle CNN pour obtenir l'état du plateau.
    La sortie est un vecteur de 9 entiers (0 = case vide, 1 = X, 2 = O).
    """
    if processed_image is None:
        return None

    with torch.no_grad():
        output = cnn_model(processed_image)  # On s'attend à (1, 9, 3)
        # Si la sortie est en 2D (ex: (1, 27)), on la reformate en (1, 9, 3)
        if output.dim() == 2:
            output = output.view(1, 9, 3)

        # Calcul des probabilités pour chaque classe
        probabilities = torch.nn.functional.softmax(output, dim=2)
        # Récupération de la confiance et des prédictions
        confidence = torch.max(probabilities, dim=2)[0]
        predictions = torch.argmax(output, dim=2).cpu().numpy().flatten()
        confidence = confidence.cpu().numpy().flatten()

        # Seuil de confiance : si la confiance est faible, considérer la case comme vide
        for i in range(9):
            if confidence[i] < 0.7:
                predictions[i] = 0

    return predictions
