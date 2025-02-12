import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
from pathlib import Path

class EditeurAnnotations:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Éditeur d'Annotations")
        
        # Créer le cadre principal
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Cadre pour l'image
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.grid(row=0, column=0, padx=5)
        self.label_image = ttk.Label(self.image_frame)
        self.label_image.pack()
        
        # Cadre pour les boutons d'annotation
        annotation_frame = ttk.Frame(main_frame)
        annotation_frame.grid(row=0, column=1, padx=5)
        
        # Créer 9 boutons pour l'annotation
        self.boutons_annotation = []
        for i in range(3):
            for j in range(3):
                index = i * 3 + j
                btn = tk.Button(annotation_frame, text="", width=10, height=5,
                              command=lambda idx=index: self.toggle_annotation(idx))
                btn.grid(row=i, column=j)
                self.boutons_annotation.append(btn)
        
        # Cadre pour les contrôles de navigation
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        ttk.Button(control_frame, text="Précédent", 
                  command=self.precedent).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Suivant", 
                  command=self.suivant).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Sauvegarder", 
                  command=self.sauvegarder).grid(row=0, column=2, padx=5)
        
        self.label_fichier = ttk.Label(control_frame, text="")
        self.label_fichier.grid(row=1, column=0, columnspan=3, pady=5)
        
        # Variables de suivi
        self.fichiers_images = []
        self.index_actuel = 0
        self.annotation_actuelle = np.zeros(9, dtype=np.int8)
        
    def charger_fichiers(self, chemin_images, chemin_labels=None):
        self.chemin_images = Path(chemin_images)
        self.chemin_labels = Path(chemin_labels) if chemin_labels else self.chemin_images.parent / 'labels'
        
        # Récupérer la liste des fichiers image
        extensions_image = ['.jpg', '.jpeg', '.png']
        self.fichiers_images = sorted([
            f for f in os.listdir(self.chemin_images) 
            if any(f.lower().endswith(ext) for ext in extensions_image)
        ])
        
        if self.fichiers_images:
            self.charger_fichier_actuel()
    
    def charger_fichier_actuel(self):
        if not self.fichiers_images:
            return
        
        # Charger l'image
        fichier_image = self.fichiers_images[self.index_actuel]
        chemin_image = self.chemin_images / fichier_image
        self.afficher_image(chemin_image)
        
        # Essayer de charger l'annotation correspondante
        base_name = os.path.splitext(fichier_image)[0]
        chemin_annotation = self.chemin_labels / f"{base_name}.npy"
        
        # Initialiser l'annotation
        if os.path.exists(chemin_annotation):
            self.annotation_actuelle = np.load(chemin_annotation)
        else:
            self.annotation_actuelle = np.zeros(9, dtype=np.int8)
        
        # Mettre à jour l'affichage des boutons
        self.update_boutons_annotation()
        
        # Mettre à jour le libellé
        self.label_fichier.config(text=f"Fichier {self.index_actuel + 1}/{len(self.fichiers_images)}: {fichier_image}")
    
    def afficher_image(self, chemin_image):
        # Charger et redimensionner l'image
        image = Image.open(chemin_image)
        # Redimensionner en gardant le ratio
        ratio = min(600/image.width, 600/image.height)
        nouvelle_taille = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(nouvelle_taille, Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(image)
        self.label_image.config(image=photo)
        self.label_image.image = photo  # Garder une référence
    
    def toggle_annotation(self, index):
        # Cycle entre 0 (vide), 1 (X), 2 (O)
        self.annotation_actuelle[index] = (self.annotation_actuelle[index] + 1) % 3
        self.update_boutons_annotation()
    
    def update_boutons_annotation(self):
        symboles = {0: "", 1: "X", 2: "O"}
        for i, bouton in enumerate(self.boutons_annotation):
            bouton["text"] = symboles[self.annotation_actuelle[i]]
    
    def precedent(self):
        if self.index_actuel > 0:
            self.index_actuel -= 1
            self.charger_fichier_actuel()
    
    def suivant(self):
        if self.index_actuel < len(self.fichiers_images) - 1:
            self.index_actuel += 1
            self.charger_fichier_actuel()
    
    def sauvegarder(self):
        if self.fichiers_images:
            # Nom du fichier image actuel
            fichier_image = self.fichiers_images[self.index_actuel]
            base_name = os.path.splitext(fichier_image)[0]
            
            # Créer le dossier des labels si nécessaire
            os.makedirs(self.chemin_labels, exist_ok=True)
            
            # Chemin de sauvegarde de l'annotation
            chemin_annotation = self.chemin_labels / f"{base_name}.npy"
            
            # Sauvegarder l'annotation
            np.save(chemin_annotation, self.annotation_actuelle)
            
            print(f"Annotation sauvegardée pour {fichier_image}")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    editeur = EditeurAnnotations()
    # Remplacez ces chemins par vos chemins réels
    editeur.charger_fichiers(
        "data/dataset/train/images",  # chemin vers vos images
        "data/dataset/train/labels"   # chemin vers vos annotations (optionnel)
    )
    editeur.run()