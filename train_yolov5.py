#!/usr/bin/env python3
import os

# Commande pour entraîner YOLOv5 (ici on utilise le modèle "yolov5s" par défaut)
cmd = (
    "python yolov5/train.py "
    "--img 640 "                # taille d'image d'entraînement
    "--batch-size 16 "          # taille de batch (à adapter selon votre GPU)
    "--epochs 50 "              # nombre d'époques
    "--data board_data.yaml "   # fichier de config des données
    "--cfg yolov5s.yaml "       # configuration du modèle (vous pouvez aussi personnaliser)
    "--name board_detector"     # nom de l'expérience
)

os.system(cmd)
