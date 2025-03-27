import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet

class TicTacToeDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.split('.')[0] + '.npy')
        
        image = Image.open(img_path).convert('RGB')
        label = np.load(label_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.from_numpy(label).long()

def create_model(num_classes=27):
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, 3), labels.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, 3), labels.view(-1))
                val_loss += loss.item()
                _, predicted = outputs.view(-1, 3).max(1)
                total += labels.view(-1).size(0)
                correct += predicted.eq(labels.view(-1)).sum().item()
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {running_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'Val Accuracy: {100.*correct/total:.2f}%')

# Paramètres
image_dir = 'data/dataset/train/images'
label_dir = 'data/dataset/train/labels'
batch_size = 16
num_epochs = 50
learning_rate = 1e-4

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Chargement des données
dataset = TicTacToeDataset(image_dir, label_dir, transform=transform)
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

# Création du modèle
model = create_model()

# Chargement des poids pré-entraînés
model.load_state_dict(torch.load('models/tic_tac_toe_model_best.pth'))

# Critère et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Entraînement
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# Sauvegarde du modèle final
torch.save(model.state_dict(), 'tic_tac_toe_model_finetuned.pth')
