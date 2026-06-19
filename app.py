import csv
import os
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


def log_inference(image: Image.Image, prediction: float) -> str:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = IMAGES_DIR / f"{timestamp}.png"
    image.save(image_path)

    write_header = not LOGS_CSV.exists()
    with open(LOGS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "image_path", "prediction"])
        writer.writerow([timestamp, str(image_path), prediction])

    return timestamp


def predict(image: Image.Image) -> str:
    if model is None:
        return "Error: model could not be loaded. Check that 'biomass_resnet@Champion' exists in MLflow."
    if image is None:
        return "Please upload an image."

    result = model.predict([image])
    prediction = result[0]

    log_inference(image, prediction)

    return f"Predicted biomass: {prediction:.2f}g"


iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Plant Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Plant Biomass Predictor",
    description="Upload a plant image to predict its fresh weight in grams.",
)

if __name__ == "__main__":
    iface.launch(server_port=7860)
