import numpy as np

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
    
    return board_copy

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
