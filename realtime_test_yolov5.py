import cv2
import torch
import time
import sys
import os

# Installer YOLOv5 si nécessaire
if not os.path.exists('yolov5'):
    print("Installation de YOLOv5...")
    os.system('git clone https://github.com/ultralytics/yolov5')
    os.system('pip install -r yolov5/requirements.txt')

# Ajouter YOLOv5 au path
sys.path.append('yolov5')

# Charger le modèle YOLOv5 (à partir du repo original)
try:
    model = torch.hub.load('yolov5', 'custom', path='models/yolov5_board.pt', source='local')
    model.conf = 0.7  # Seuil de confiance
    print("Modèle chargé avec succès")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    exit()

# Initialiser la capture vidéo
cap = cv2.VideoCapture(0)

# Vérifier si la caméra s'est ouverte correctement
if not cap.isOpened():
    print("Erreur: Impossible d'accéder à la caméra.")
    exit()

# Configurer les propriétés de la caméra
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Paramètres pour améliorer la détection
last_detection_time = 0
detection_cooldown = 0.5  # Limiter les détections à une tous les 0.5 secondes

frame_counter = 0
window_name = "Détection de plateau de morpion"

try:
    # Créer la fenêtre après avoir vérifié que la caméra est fonctionnelle
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur lors de la capture vidéo.")
            break

        frame_counter += 1
        if frame_counter % 10 == 0:  # Afficher tous les 10 frames pour réduire la verbosité
            print(f"Traitement de la frame {frame_counter}")

        current_time = time.time()
        
        # Exécuter l'inférence YOLOv5
        results = model(frame)
        
        # Vérifier si des détections significatives ont été faites
        detections = results.xyxy[0]  # Résultats au format [x1, y1, x2, y2, confiance, classe]
        valid_detections = detections[detections[:, 4] >= model.conf]  # Filtrer par confiance
        
        # Annoter l'image
        annotated_frame = frame.copy()
        for det in valid_detections:
            x1, y1, x2, y2, conf, cls = det.tolist()
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Plateau: {conf:.2f}", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Afficher des informations sur les détections
        num_detections = len(valid_detections)
        if num_detections > 0 and current_time - last_detection_time > detection_cooldown:
            last_detection_time = current_time
            print(f"Détection! Nombre de plateaux: {num_detections}")
            
            # Afficher les détails des détections
            for i, det in enumerate(valid_detections):
                x1, y1, x2, y2, conf, cls = det.tolist()
                print(f"Plateau {i+1}: Position: ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)}), Confiance: {conf:.2f}")
        
        # Afficher l'image
        cv2.imshow(window_name, annotated_frame)
        
        # Sortir si 'q' est pressé
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Sortie demandée par l'utilisateur.")
            break

except Exception as e:
    print(f"Une erreur s'est produite : {e}")
    import traceback
    traceback.print_exc()

finally:
    # Libérer les ressources
    if cap.isOpened():
        cap.release()
    
    # S'assurer que toutes les fenêtres sont fermées proprement
    try:
        cv2.destroyWindow(window_name)
    except:
        pass
    
    try:
        cv2.destroyAllWindows()
    except:
        pass

print("Script terminé.")