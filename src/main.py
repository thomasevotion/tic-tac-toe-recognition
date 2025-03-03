import cv2
import torch
from src.models.yolo_model import BoardDetector
from src.models.cnn_model import create_model
from src.game.game_loop import game_loop

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    board_detector = BoardDetector(model_path='models/yolov5_board.pt', confidence_threshold=0.6)
    cnn_model = create_model()
    cnn_model.load_state_dict(torch.load('models/tic_tac_toe_model_best.pth', map_location=device))
    cnn_model.to(device)
    cnn_model.eval()

    try:
        game_loop(board_detector, cnn_model)
    except KeyboardInterrupt:
        print("\nInterruption par l'utilisateur. Fin du programme.")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
