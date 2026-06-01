import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from dagster import asset, AssetExecutionContext, Definitions
from dagster_mlflow import mlflow_tracking
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from PIL import Image

# ==========================================
# CONFIGURATION & HELPER FUNCTIONS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'mlops_biomass_data', 'images_med_res')
CSV_CANDIDATES = [
    os.path.join(BASE_DIR, 'digital_biomass_labels.csv'),
    os.path.join(BASE_DIR, 'mlops_biomass_data', 'digital_biomass_labels.xlsx'),
]

# Define hyperparameters centrally
EPOCHS = 3
LEARNING_RATE = 0.0001
BATCH_SIZE = 16

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

def select_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

# ==========================================
# DAGSTER ASSETS
# ==========================================

@asset
def raw_dataset(context: AssetExecutionContext) -> dict:
    """Load plant images and metadata from disk"""
    context.log.info("Searching for label file...")
    
    data_path = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)
    if data_path is None:
        raise FileNotFoundError("No label file found.")

    context.log.info(f"Loading data from: {data_path}")
    
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path, decimal=',')
        
    initial_len = len(df)
    df = df.dropna(subset=['fresh_weight_total'])
    context.log.info(f"Removed {initial_len - len(df)} rows with missing labels (NaN).")
    
    image_paths = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    return {"image_dir": IMG_DIR, "image_paths": image_paths, "metadata": df}


@asset
def preprocessed_data(context: AssetExecutionContext, raw_dataset: dict) -> dict:
    """Preprocess images (resize, normalize) and create train/val split"""
    df = raw_dataset["metadata"]
    img_dir = raw_dataset["image_dir"]
    
    max_val = df['fresh_weight_total'].max()
    df['fresh_weight_total'] = df['fresh_weight_total'] / max_val
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(PlantDataset(train_df, img_dir, transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PlantDataset(val_df, img_dir, transform), batch_size=BATCH_SIZE)
    
    context.log.info(f"DataLoaders created. Batch Size: {BATCH_SIZE}")
    return {"train_loader": train_loader, "val_loader": val_loader, "max_weight_val": max_val}


@asset
def trained_model(context: AssetExecutionContext, preprocessed_data: dict) -> dict:
    """Train the ResNet model with MLflow tracking"""
    train_loader = preprocessed_data["train_loader"]
    val_loader = preprocessed_data["val_loader"]
    device = select_device()
    context.log.info(f"Starting training on device: {device}")
 
    # Initialize model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
 
    train_losses = []
    val_losses = []
 
    # Start MLflow run
    with mlflow.start_run(run_name="resnet18_biomass_training"):
        # Log at least 3 parameters (requirement met)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("model_architecture", "resnet18")
        mlflow.log_param("optimizer", "Adam")
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
                optimizer.zero_grad()
                loss = criterion(model(imgs), labels)
                loss.backward()
                optimizer.step()
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
            train_losses.append(avg_train)
            val_losses.append(avg_val)
            
            context.log.info(f"Epoch {epoch+1}: Train Loss {avg_train:.4f} | Val Loss {avg_val:.4f}")
            
            # Log metrics per epoch for the MLflow UI
            mlflow.log_metric("train_loss", avg_train, step=epoch)
            mlflow.log_metric("val_loss", avg_val, step=epoch)
 
        # Save model as artifact (requirement met)
        mlflow.pytorch.log_model(model, "model")
        context.log.info("Model successfully saved to MLflow.")
 
    # Pass the trained model and loss history to the next asset
    return {"model": model, "train_losses": train_losses, "val_losses": val_losses}



@asset
def model_evaluation(
    context: AssetExecutionContext,
    trained_model: dict,
    preprocessed_data: dict,
) -> dict:
    """Evaluate model on validation set, log final metrics and create plots"""
    model = trained_model["model"]
    train_losses = trained_model["train_losses"]
    val_losses = trained_model["val_losses"]
    val_loader = preprocessed_data["val_loader"]
    max_weight_val = preprocessed_data["max_weight_val"]
 
    device = select_device()
    model = model.to(device)
    model.eval()
 
    # Final evaluation on validation set (in original units = grams)
    criterion = nn.MSELoss()
    total_loss_scaled = 0.0
    total_loss_grams = 0.0
    n_batches = 0
 
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            preds = model(imgs)
            # Loss in scaled units [0, 1]
            total_loss_scaled += criterion(preds, labels).item()
            # Loss in original units (grams) for interpretability
            preds_g = preds * max_weight_val
            labels_g = labels * max_weight_val
            total_loss_grams += criterion(preds_g, labels_g).item()
            n_batches += 1
 
    final_val_mse_scaled = total_loss_scaled / n_batches
    final_val_mse_grams = total_loss_grams / n_batches
    final_val_rmse_grams = final_val_mse_grams ** 0.5
 
    context.log.info(
        f"Final Val MSE (scaled): {final_val_mse_scaled:.4f} | "
        f"Final Val MSE (grams): {final_val_mse_grams:.4f} | "
        f"Final Val RMSE (grams): {final_val_rmse_grams:.4f}"
    )
 
    # Log final metrics into the same MLflow experiment (new run for evaluation)
    with mlflow.start_run(run_name="resnet18_biomass_evaluation", nested=False):
        mlflow.log_metric("final_val_mse_scaled", final_val_mse_scaled)
        mlflow.log_metric("final_val_mse_grams", final_val_mse_grams)
        mlflow.log_metric("final_val_rmse_grams", final_val_rmse_grams)
 
    # Generate the training-curves plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, EPOCHS + 1), val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
 
    plot_path = os.path.join(BASE_DIR, 'dagster_training_curves.png')
    plt.savefig(plot_path)
    plt.close()
    context.log.info(f"Training curves saved to: {plot_path}")
 
    return {
        "final_val_mse_scaled": final_val_mse_scaled,
        "final_val_mse_grams": final_val_mse_grams,
        "final_val_rmse_grams": final_val_rmse_grams,
        "plot_path": plot_path,
    }



# ==========================================
# DAGSTER DEFINITIONS & RESOURCES
# ==========================================

defs = Definitions(
    assets=[raw_dataset, preprocessed_data, trained_model, model_evaluation],
    resources={
        "mlflow": mlflow_tracking.configured({
            "experiment_name": "plant_biomass_pipeline",
            "mlflow_tracking_uri": "sqlite:///mlflow.db"
        })
    }
)