import csv
import os
import numpy as np
from datetime import datetime
from pathlib import Path
import gradio as gr
import mlflow.pyfunc
from PIL import Image

mlflow.set_tracking_uri("sqlite:///mlflow.db")

try:
    model = mlflow.pyfunc.load_model("models:/biomass_resnet@champion")
except Exception as e:
    model = None
    print(f"[ERROR] Failed to load model: {e}")

PROD_DIR = Path("production_data")
IMAGES_DIR = PROD_DIR / "images"
LOGS_CSV = PROD_DIR / "logs.csv"

MEASURED_DIR = Path("production_data_measured")
MEASURED_IMAGES_DIR = MEASURED_DIR / "images"
MEASURED_LOGS_CSV = MEASURED_DIR / "logs.csv"

def log_inference(image: Image.Image, prediction: float, actual_value=None) -> str:
    has_actual_value = actual_value is not None
    images_dir = MEASURED_IMAGES_DIR if has_actual_value else IMAGES_DIR
    logs_csv = MEASURED_LOGS_CSV if has_actual_value else LOGS_CSV

    images_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = images_dir / f"{timestamp}.png"
    image.save(image_path)

    row = [timestamp, str(image_path), prediction]
    header = ["timestamp", "image_path", "prediction"]
    if has_actual_value:
        row.append(actual_value)
        header.append("actual_value")

    write_header = not logs_csv.exists()
    with open(logs_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
    return timestamp

def predict(image: Image.Image, actual_value: float = None) -> str:
    if model is None:
        return "Error: model could not be loaded. Check that 'biomass_resnet@champion' exists in MLflow."
    if image is None:
        return "Please upload an image."

    # MLflow Batch-Inferenz ausführen (Liste übergeben)
    result = model.predict([image])

    # Sicherstellen, dass wir eine native Python-Zahl extrahieren
    if hasattr(result, "item"):
        prediction = result.item()
    elif isinstance(result, (list, np.ndarray)):
        prediction = float(result[0])
    else:
        prediction = float(result)

    log_inference(image, prediction, actual_value)
    return f"Predicted biomass: {prediction:.2f}g"

actual_value_input = gr.Number(label="Actual Biomass Value (g, optional)", value=None)

iface = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil", label="Plant Image"), actual_value_input],
    outputs=gr.Textbox(label="Prediction"),
    title="Plant Biomass Predictor",
    description="Upload a plant image to predict its fresh weight in grams.",
)

if __name__ == "__main__":
    iface.launch(server_port=7860)