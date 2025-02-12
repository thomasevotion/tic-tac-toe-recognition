#!/usr/bin/env python3
import os
import torch
from torchvision import transforms, models
from PIL import Image
from torch import nn
import argparse

# Définition du modèle qui utilise un modèle de base pré-entraîné
class TicTacToeCNN(nn.Module):
    def __init__(self, base_model):
        super(TicTacToeCNN, self).__init__()
        self.base = base_model
        # La couche fully connected a déjà été remplacée dans base_model
    def forward(self, x):
        x = self.base(x)  # La sortie sera de forme (batch, 27)
        x = x.view(-1, 9, 3)
        return x

# Prétraitement identique à celui de l'entraînement
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image)
    return tensor.unsqueeze(0)

# Mapping des classes (0 : vide, 1 : X, 2 : O)
class_map = {0: ' ', 1: 'X', 2: 'O'}

def predict_board(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.to(device))
    # Appliquer softmax sur la dimension des classes pour chaque case
    probs = torch.nn.functional.softmax(output[0], dim=1)
    predictions = torch.argmax(probs, dim=1).cpu().numpy()
    # Reformater en plateau 3x3
    board = [[class_map[predictions[i*3+j]] for j in range(3)] for i in range(3)]
    return board

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Modèle non trouvé: {args.model_path}")
    
    # Charger le modèle de base (ici ResNet18) et adapter sa dernière couche
    base_model = models.resnet18(pretrained=True)
    in_features = base_model.fc.in_features
    base_model.fc = torch.nn.Linear(in_features, 9 * 3)
    
    # Instancier le modèle complet avec le modèle de base
    model = TicTacToeCNN(base_model).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Modèle chargé depuis {args.model_path}")
    
    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"Image non trouvée: {args.image_path}")
    
    image_tensor = preprocess_image(args.image_path)
    board = predict_board(model, image_tensor, device)
    
    print("Plateau prédit :")
    for row in board:
        print(" | ".join(row))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script d'inférence pour prédire un plateau de Tic Tac Toe à partir d'une image."
    )
    parser.add_argument("--model_path", type=str, default="models/tic_tac_toe_model_best.pth",
                        help="Chemin vers le fichier modèle sauvegardé.")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Chemin vers l'image test.")
    args = parser.parse_args()
    main(args)
