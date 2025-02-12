#!/usr/bin/env python3
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

#############################################
# 1. Dataset personnalisé pour Tic Tac Toe  #
#############################################
class TicTacToeDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        """
        images_dir : dossier contenant les images (.jpg, .jpeg, .png)
        labels_dir : dossier contenant les annotations (.npy) associées
        transform  : transformations à appliquer sur les images
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            self.image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        # Ne conserver que les images ayant une annotation associée
        self.image_files = [img for img in self.image_files 
                            if os.path.exists(os.path.join(labels_dir, os.path.splitext(os.path.basename(img))[0] + ".npy"))]
        if not self.image_files:
            raise RuntimeError(f"Aucune image avec annotation trouvée dans {images_dir} et {labels_dir}.")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(self.labels_dir, base_name + ".npy")
        # Annotation : vecteur de 9 valeurs (0 = vide, 1 = X, 2 = O)
        annotation = np.load(label_path, allow_pickle=True)
        annotation = torch.tensor(annotation, dtype=torch.long)
        return image, annotation

#############################################
# 2. Modèle pré-entraîné adapté (ResNet18)     #
#############################################
class TicTacToeResNet(nn.Module):
    def __init__(self, base_model):
        """
        base_model : modèle ResNet18 modifié pour avoir 27 sorties (9x3)
        """
        super(TicTacToeResNet, self).__init__()
        self.base = base_model
        
    def forward(self, x):
        # x est passé dans le modèle pré-entraîné
        x = self.base(x)  # x a une forme (batch, 27)
        x = x.view(-1, 9, 3)
        return x

def create_model():
    # Charge un ResNet18 pré-entraîné
    base_model = models.resnet18(pretrained=True)
    in_features = base_model.fc.in_features
    # Remplacez la couche fully connected pour obtenir 27 sorties
    base_model.fc = nn.Linear(in_features, 9 * 3)
    model = TicTacToeResNet(base_model)
    return model

#############################################
# 3. Focal Loss avec gamma augmenté          #
#############################################
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=3, reduction='mean'):
        """
        alpha   : facteur de pondération pour chaque classe (tensor de taille (C,))
        gamma   : paramètre de focalisation (ici on utilise gamma=3 pour accentuer)
        reduction: 'mean' ou 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            ce_loss = at * ce_loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

#############################################
# 4. Calcul automatique des poids des classes#
#############################################
def compute_class_weights(dataset):
    freq = torch.zeros(3)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for _, labels in loader:
        for label in labels.view(-1):
            freq[label] += 1
    total = freq.sum().item()
    weights = total / (3 * freq)
    return weights

#############################################
# 5. Boucle d'entraînement modifiée         #
#############################################
def train_model(train_images, train_labels, val_images=None, val_labels=None,
                num_epochs=150, batch_size=16, lr=1e-3, checkpoint_path=None):
    # Transformations identiques en entraînement
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = TicTacToeDataset(train_images, train_labels, transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    if val_images and val_labels and os.path.isdir(val_images) and os.path.isdir(val_labels):
        val_dataset = TicTacToeDataset(val_images, val_labels, transform_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    else:
        val_loader = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)
    # Si plusieurs GPUs, utilisez DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Si un checkpoint est fourni, rechargez-le
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Checkpoint chargé depuis {checkpoint_path}")
    
    # Calcul des poids de classes (vous pouvez également les ajuster manuellement si besoin)
    class_weights = compute_class_weights(train_dataset).to(device)
    print("Poids des classes calculés:", class_weights)
    
    criterion = FocalLoss(alpha=class_weights, gamma=3, reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    save_path = os.path.join(os.path.dirname(train_images), "tic_tac_toe_model_best.pth")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_samples = 0
        for images, annotations in train_loader:
            images = images.to(device, non_blocking=True)
            annotations = annotations.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)  # sortie de forme (batch, 9, 3)
                loss = criterion(outputs.view(-1, 3), annotations.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
            num_samples += images.size(0)
        
        epoch_loss = running_loss / num_samples
        scheduler.step()
        print(f"[Epoch {epoch+1}/{num_epochs}] Training Loss: {epoch_loss:.4f}")
        
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for images, annotations in val_loader:
                    images = images.to(device, non_blocking=True)
                    annotations = annotations.to(device, non_blocking=True)
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs.view(-1, 3), annotations.view(-1))
                    val_loss += loss.item() * images.size(0)
                    val_samples += images.size(0)
            avg_val_loss = val_loss / val_samples
            print(f"           Validation Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
                print(f"           -> Nouveau meilleur modèle sauvegardé à l'époque {epoch+1}")
    
    final_model_path = os.path.join(os.path.dirname(train_images), "tic_tac_toe_model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Modèle final sauvegardé dans {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entraîne un modèle ResNet18 pré-entraîné pour la reconnaissance du plateau de morpion avec Focal Loss."
    )
    parser.add_argument("--data_dir", required=True, help="Chemin vers le dossier data/dataset/")
    parser.add_argument("--epochs", type=int, default=150, help="Nombre d'époques")
    parser.add_argument("--batch_size", type=int, default=16, help="Taille du batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Taux d'apprentissage")
    parser.add_argument("--checkpoint", type=str, default=None, help="Chemin vers le checkpoint du modèle à reprendre")
    args = parser.parse_args()

    # Structure attendue : data/dataset/ avec les sous-dossiers train/images, train/labels, (optionnel) val/images, val/labels
    train_images = os.path.join(args.data_dir, "train", "images")
    train_labels = os.path.join(args.data_dir, "train", "labels")
    val_images = os.path.join(args.data_dir, "val", "images")
    val_labels = os.path.join(args.data_dir, "val", "labels")
    
    if not os.path.isdir(train_images) or not os.path.isdir(train_labels):
        raise RuntimeError("Les dossiers d'images et/ou de labels d'entraînement n'existent pas.")
    if not os.path.isdir(val_images) or not os.path.isdir(val_labels):
        print("Dossiers de validation introuvables, entraînement sans validation.")
        val_images = None
        val_labels = None

    train_model(train_images, train_labels, val_images, val_labels,
                num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                checkpoint_path=args.checkpoint)
