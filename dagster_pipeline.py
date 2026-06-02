import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import base64
from io import BytesIO
from dagster import MetadataValue
from dagster import asset, AssetExecutionContext, Definitions
from dagster_mlflow import mlflow_tracking
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from dagster import Config

class PreprocessConfig(Config):
    test_size: float = 0.2
    random_state: int = 42
    image_size: int = 224
    batch_size: int = 16

class TrainingConfig(Config):
    epochs: int = 1
    learning_rate: float = 0.0001
    batch_size: int = 16

# ==========================================
# CONFIGURATION & HELPER FUNCTIONS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'mlops_biomass_data', 'images_med_res')
CSV_CANDIDATES = [
    os.path.join(BASE_DIR, 'digital_biomass_labels.csv'),
    os.path.join(BASE_DIR, 'mlops_biomass_data', 'digital_biomass_labels.xlsx'),
]

_MLFLOW_DB_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"

# Restore deleted experiment so dagster_mlflow resource can activate it
def _restore_deleted_experiment(name: str) -> None:
    client = mlflow.tracking.MlflowClient(_MLFLOW_DB_URI)
    exp = client.get_experiment_by_name(name)
    if exp is not None and exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)

_restore_deleted_experiment("plant_biomass_pipeline")

def plot_to_markdown(fig) -> str:
    """Konvertiert eine matplotlib Figure in einen Markdown-Bild-String."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_b64 = base64.b64encode(buf.read()).decode()
    return f"![plot](data:image/png;base64,{image_b64})"

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
def raw_dataset(context: AssetExecutionContext, config: TrainingConfig) -> dict:
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
def preprocessed_data(context: AssetExecutionContext, raw_dataset: dict, config: PreprocessConfig) -> dict:
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
    
    train_loader = DataLoader(PlantDataset(train_df, img_dir, transform), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(PlantDataset(val_df, img_dir, transform), batch_size=config.batch_size)

    context.log.info(f"DataLoaders created. Batch Size: {config.batch_size}")
    return {"train_loader": train_loader, "val_loader": val_loader, "max_weight_val": max_val}


@asset(required_resource_keys={"mlflow"})
def trained_model(context: AssetExecutionContext, preprocessed_data: dict, config: TrainingConfig) -> dict:
    """Train the ResNet model with MLflow tracking"""
    train_loader = preprocessed_data["train_loader"]
    val_loader = preprocessed_data["val_loader"]
    device = select_device()
    context.log.info(f"Starting training on device: {device}")
 
    # Initialize model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    # dagster_mlflow resource already started a run — log directly into it
    mlflow.log_param("epochs", config.epochs)
    mlflow.log_param("learning_rate", config.learning_rate)
    mlflow.log_param("batch_size", config.batch_size)
    mlflow.log_param("model_architecture", "resnet18")
    mlflow.log_param("optimizer", "Adam")

    for epoch in range(config.epochs):
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

        mlflow.log_metric("train_loss", avg_train, step=epoch)
        mlflow.log_metric("val_loss", avg_val, step=epoch)

    mlflow.pytorch.log_model(model, "model")
    context.log.info("Model successfully saved to MLflow.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
    ax.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training and Validation Loss Over Time')
    ax.legend()
    ax.grid(True)

    mlflow.log_figure(fig, "training_curves.png")
    plot_path = os.path.join(BASE_DIR, 'dagster_training_curves.png')
    fig.savefig(plot_path)
    plot_md = plot_to_markdown(fig)
    plt.close(fig)

    context.add_output_metadata({
        "final_train_loss": MetadataValue.float(train_losses[-1]),
        "final_val_loss": MetadataValue.float(val_losses[-1]),
        "epochs_trained": MetadataValue.int(len(train_losses)),
        "device": MetadataValue.text(str(device)),
        "model_architecture": MetadataValue.text("ResNet-18"),
        "training_curves": MetadataValue.md(plot_md),
        "plot_file": MetadataValue.path(plot_path),
        "mlflow_ui": MetadataValue.url("http://localhost:5000"),
    })

    return {"model": model, "train_losses": train_losses, "val_losses": val_losses}



@asset(required_resource_keys={"mlflow"})
def model_evaluation(
    context: AssetExecutionContext,
    trained_model: dict,
    config: TrainingConfig,
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
 
    # dagster_mlflow resource already started a run — log directly into it
    mlflow.log_metric("final_val_mse_scaled", final_val_mse_scaled)
    mlflow.log_metric("final_val_mse_grams", final_val_mse_grams)
    mlflow.log_metric("final_val_rmse_grams", final_val_rmse_grams)
 
    # Generate the training-curves plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', marker='o')
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
            "mlflow_tracking_uri": f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"
        })
    }
)