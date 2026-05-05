import os
import argparse
import logging
import subprocess
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt  # WICHTIG: Import hinzugefügt
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# Pfade absolut definieren
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'mlops_biomass_data', 'images_med_res')
CSV_PATH = os.path.join(BASE_DIR, 'digital_biomass_labels.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOG_FILE = os.path.join(BASE_DIR, 'training.log')

# Hilfsfunktion für den Git-Hash
def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return "No Git found"

# Logging Setup
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s', force=True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.000001) #learning rate kleiner gemacht
    parser.add_argument('--batch_size', type=int, default=16)
    return parser.parse_args()

class PlantDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        img = Image.open(img_path).convert('RGB')
        label = torch.tensor(float(row['fresh_weight_total']), dtype=torch.float32)
        if self.transform: img = self.transform(img)
        return img, label

def main():
    args = get_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Daten laden und vorverarbeiten
    df = pd.read_csv(CSV_PATH, decimal=',')
    df = df.dropna(subset=['fresh_weight_total']) 
    
    # Label-Scaling
    max_val = df['fresh_weight_total'].max()
    df['fresh_weight_total'] = df['fresh_weight_total'] / max_val
    
    # Train-Val Split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    #Transform samples into fitting format for ResNet-18
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(PlantDataset(train_df, IMG_DIR, transform), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(PlantDataset(val_df, IMG_DIR, transform), batch_size=args.batch_size)

    # 2. Modell-Setup
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Listen initialisieren für den Plot
    train_losses = []
    val_losses = []
    
    git_hash = get_git_revision_hash()
    logging.info(f"Start: Git={git_hash}, Epochs={args.epochs}, LR={args.lr}, MaxWeight={max_val}")
    
    print(f"Training startet auf: {device}")
    for epoch in range(args.epochs):
        # Schaltet das Modell in den Trainings-Modus (wichtig für Layer wie Dropout oder BatchNorm)
        model.train()
        train_loss = 0
        
        # Schleife über alle Daten-Pakete (Batches) im Trainings-Set
        for imgs, labels in train_loader:
            # Daten auf die GPU (falls vorhanden) schieben und Label-Form anpassen (Spalte statt Zeile)
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            
            # 1. Alte Korrekturhinweise (Gradienten) löschen, damit wir frisch starten
            optimizer.zero_grad()
            
            # 2. Forward Pass: Das Modell rät das Gewicht; Criterion (MSE) berechnet den Fehler zum echten Label
            loss = criterion(model(imgs), labels)
            
            # 3. Backpropagation: Der Fehler wird rückwärts durch das Netz geleitet ("Wer war schuld?")
            loss.backward()
            
            # 4. Optimizer-Schritt: Die Gewichte werden minimal angepasst, um den Fehler zu verringern
            optimizer.step()
            
            # Fehlerwert des aktuellen Batches für die Statistik aufsummieren
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
                loss = criterion(model(imgs), labels)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        # Werte für Plot speichern
        train_losses.append(avg_train) 
        val_losses.append(avg_val)
        
        print(f"Epoch {epoch+1}: Train Loss {avg_train:.4f} | Val Loss {avg_val:.4f}")
        logging.info(f"Epoch {epoch+1}: Train_Loss={avg_train:.4f}, Val_Loss={avg_val:.4f}")

    # 3. Ergebnisse speichern
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Final_Train_Loss: {avg_train:.4f}\nFinal_Val_Loss: {avg_val:.4f}\nMax_Weight_Scale: {max_val}")
    
    # Plot erstellen
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(RESULTS_DIR, 'training_curves.png')
    plt.savefig(plot_path)
    
    print(f"Trainingskurven gespeichert unter: {plot_path}")
    print(f"Training abgeschlossen. Ergebnisse in {RESULTS_DIR} gespeichert.")

if __name__ == '__main__':
    main()