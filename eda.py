import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# --- 1. Dataset Klasse ---
class PlantBiomassDataset(Dataset):
    """
    Eigene Dataset-Klasse, um Bilder und Metadaten (Biomasse) zu verknüpfen.
    """
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Dateiname aus der CSV/Excel-Tabelle lesen
        img_name = self.df.iloc[idx]['filename'] 
        img_path = os.path.join(self.img_dir, img_name)
        
        # Bild laden und in RGB konvertieren
        image = Image.open(img_path).convert("RGB")
        
        # Zielwert (Biomasse) aus der Spalte 'fresh_weight_total' extrahieren 
        label = torch.tensor(float(self.df.iloc[idx]['fresh_weight_total']), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

# --- 2. Training-Funktion ---
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

# --- 3. Hauptprogramm ---
def main():
    # Einstellungen & Pfade
    CSV_FILE = 'digital_biomass_labels.csv'  # CSV-Datei
    IMG_DIR = 'images_med_res' # Ordner mit den Bildern
    BATCH_SIZE = 16
    EPOCHS = 3 # Geringe Anzahl reicht laut Aufgabe für den "Working Flow" 
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformationen (ResNet erwartet 224x224)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Daten laden und Splitten 
    if not os.path.exists(CSV_FILE):
        print(f"Fehler: {CSV_FILE} nicht gefunden!")
        return

    df = pd.read_csv(CSV_FILE)
    # Train/Validation Split (80/20) 
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Datasets & Loader
    train_dataset = PlantBiomassDataset(train_df, IMG_DIR, transform=data_transforms)
    val_dataset = PlantBiomassDataset(val_df, IMG_DIR, transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Modell definieren (ResNet-basiert) 
    # Wir nutzen ein vortrainiertes ResNet18 und passen den Output für Regression an
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    # Output ist 1, da wir nur einen Wert (Gewicht) vorhersagen
    model.fc = nn.Linear(num_ftrs, 1) 
    model = model.to(DEVICE)

    # Verlustfunktion (MSE für Regression) und Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Trainings-Loop
    print(f"Starte Training auf: {DEVICE}")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}")

    # Modell speichern (Wichtig: Nicht ins Git hochladen laut .gitignore!)
    torch.save(model.state_dict(), "model_checkpoint.pth")
    print("Training abgeschlossen und Modell lokal gespeichert.")

if __name__ == "__main__":
    main()
