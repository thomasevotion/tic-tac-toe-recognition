import torch

class BoardDetector:
    def __init__(self, model_path='models/yolov5_board.pt', confidence_threshold=0.7):
        """
        Charge un modèle YOLO personnalisé entraîné pour détecter le plateau.
        Le modèle doit renvoyer une bounding box pour le plateau.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        self.model.conf = confidence_threshold
        self.model.to(self.device)
        self.model.eval()
        self.board_detection_counter = 0
        self.last_detection = None
        self.required_consecutive_detections = 3

    def detect_board(self, image):
        """
        Prend une image (BGR) et renvoie les coins de la bounding box du plateau sous forme
        de liste de tuples [(x1, y1), (x2, y1), (x2, y2), (x1, y2)].
        Si aucune détection n'est faite, retourne None.
        """
        results = self.model(image)
        detections = results.xyxy[0].cpu().numpy()
        
        # Si aucune détection, utiliser la dernière détection stable si disponible
        if len(detections) == 0:
            if self.last_detection is not None:
                self.board_detection_counter += 1
                return self.last_detection
            self.board_detection_counter = 0
            return None
        
        # Prendre la meilleure détection
        best_detection = detections[0]
        x1, y1, x2, y2 = best_detection[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        current_detection = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        
        # Comparer avec la dernière détection pour vérifier la stabilité
        if self.last_detection is not None:
            diffs = []
            for i in range(4):
                for j in range(2):
                    diffs.append(abs(current_detection[i][j] - self.last_detection[i][j]))
            avg_diff = sum(diffs) / len(diffs)
            # Tolérance augmentée à 40 pixels pour accepter de légères variations
            if avg_diff < 40:
                self.board_detection_counter += 1
            else:
                self.board_detection_counter = 1  # Nouvelle détection considérée comme stable
        else:
            self.board_detection_counter = 1
        
        self.last_detection = current_detection
        
        # Ne renvoyer la détection que si stable sur plusieurs frames
        if self.board_detection_counter >= self.required_consecutive_detections:
            return current_detection
        return None
