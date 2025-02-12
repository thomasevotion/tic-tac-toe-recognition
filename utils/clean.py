#!/usr/bin/env python3
import os
import glob
import numpy as np
import argparse
import shutil

def flatten_annotation(annotation_file, output_file):
    """
    Charge un fichier .npy contenant une annotation sous forme de matrice 3x3,
    le transforme en vecteur 1D de 9 valeurs et sauvegarde le résultat.
    """
    try:
        annotation = np.load(annotation_file, allow_pickle=True)
    except Exception as e:
        print(f"Erreur lors du chargement de {annotation_file} : {e}")
        return False

    if annotation.shape != (3, 3):
        print(f"Le fichier {annotation_file} n'a pas la forme (3,3) (forme trouvée : {annotation.shape}).")
        return False

    flattened = annotation.flatten()
    try:
        np.save(output_file, flattened)
        print(f"Annotation aplatie sauvegardée dans {output_file}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de {output_file} : {e}")
        return False

    return True

def process_directory(directory, remove_unlabeled):
    # Extensions d'images à rechercher
    image_extensions = ('*.jpg', '*.jpeg', '*.png')
    # Créer un sous-dossier pour sauvegarder les annotations aplaties
    flattened_dir = os.path.join(directory, "flattened_annotations")
    os.makedirs(flattened_dir, exist_ok=True)

    # Parcourir toutes les images
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(directory, ext)))
    
    print(f"Nombre d'images trouvées : {len(images)}")
    removed_images = 0
    processed_annotations = 0

    for img_path in images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        annotation_file = os.path.join(directory, base_name + ".npy")
        if os.path.exists(annotation_file):
            output_file = os.path.join(flattened_dir, base_name + ".npy")
            if flatten_annotation(annotation_file, output_file):
                processed_annotations += 1
        else:
            print(f"Aucune annotation trouvée pour {img_path}.")
            if remove_unlabeled:
                try:
                    os.remove(img_path)
                    print(f"Image {img_path} supprimée.")
                    removed_images += 1
                except Exception as e:
                    print(f"Erreur lors de la suppression de {img_path} : {e}")
            else:
                print("Image ignorée.")

    print(f"Traitement terminé : {processed_annotations} annotations traitées, {removed_images} images supprimées.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script pour aplatir les annotations de morpion (3x3) en vecteur de 9 valeurs et supprimer les images sans annotation."
    )
    parser.add_argument("--dir", required=True, help="Répertoire contenant les images et les annotations (.npy)")
    parser.add_argument("--remove-unlabeled", action="store_true", help="Supprimer les images sans annotation correspondante")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Le répertoire {args.dir} n'existe pas.")
        exit(1)

    process_directory(args.dir, args.remove_unlabeled)
