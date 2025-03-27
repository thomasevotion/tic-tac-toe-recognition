import numpy as np
import cv2
import time
from src.utils.image_processing import capture_board_image, recenter_board_image, process_cropped_image
from src.utils.game_logic import check_winner, ai_move, display_board_state
from src.detection.board_detector import detect_board_state
from src.models.yolo_model import BoardDetector

def wait_for_empty_board(board_detector, cnn_model, consecutive_empty_required=5):
    """
    Attend activement qu'un plateau vide soit détecté avant de démarrer une nouvelle partie.
    Exige plusieurs détections consécutives pour confirmer que le plateau est bien vide.
    """
    print("Attente d'un plateau vide...")
    handle_camera_print()
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
            board = detect_board_state(processed, cnn_model)
            
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


def game_loop(board_detector, cnn_model):
    # Créer le détecteur de plateau avec un seuil de confiance élevé
    board_detector = BoardDetector(model_path='models/yolov5_board.pt', confidence_threshold=0.6)
    
    # Attendre qu'un plateau vide soit détecté de manière fiable
    wait_for_empty_board(board_detector, cnn_model, consecutive_empty_required=3)
    
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
                        new_board = detect_board_state(processed, cnn_model)
                        
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
        time.sleep(5)
        
        # Fermer les fenêtres OpenCV
        cv2.destroyAllWindows()
        
        # Attendre qu'un plateau vide soit détecté pour redémarrer
        wait_for_empty_board(board_detector,cnn_model, consecutive_empty_required=3)

def handle_camera_print():
    cv2.destroyAllWindows()
    cv2.namedWindow("Caméra", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Plateau détecté", cv2.WINDOW_NORMAL)
