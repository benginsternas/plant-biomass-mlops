"""Minimal Gradio app: dropdown of registered MLflow model versions -> predict on uploaded images.

Usage:
    Set MLFLOW_TRACKING_URI if needed (defaults to http://localhost:5000)
    python app.py

Dropdown entries: model_name:v<version>
Loads model via models:/model_name/version
"""
from __future__ import annotations

import os
import torch
import time
from functools import lru_cache
from typing import List, Optional

import mlflow
from mlflow.tracking import MlflowClient
import gradio as gr
import pandas as pd
import csv
from datetime import datetime
from pathlib import Path

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"

# Basis-Konfiguration für Standard-Produktionsdaten
LOG_DIR = Path("production_data")
IMAGES_DIR = LOG_DIR / "images"
LOG_FILE = LOG_DIR / "logs.csv"

# Verzeichnisse für Standard-Produktionsdaten sicherstellen
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Standard-CSV Header initialisieren, falls Datei nicht existiert
if not LOG_FILE.exists():
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "filepath", "prediction", "model_version"])


def log_inference(filename: str, image_bytes: bytes, prediction: float, model_uri: str, actual_value: Optional[float] = None):
    """Logs an inference request to disk. Diverts to measured folder if actual value is given."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # 1. Zielverzeichnisse basierend auf Vorhandensein des echten Labels bestimmen
    if actual_value is not None:
        target_dir = Path("production_data_measured")
        target_images_dir = target_dir / "images"
        target_log_file = target_dir / "logs.csv"
        headers = ["timestamp", "filepath", "prediction", "model_version", "actual_value"]
        row_data = [timestamp, "", prediction, model_uri, actual_value]
    else:
        target_dir = LOG_DIR
        target_images_dir = IMAGES_DIR
        target_log_file = LOG_FILE
        headers = ["timestamp", "filepath", "prediction", "model_version"]
        row_data = [timestamp, "", prediction, model_uri]

    # Ordnerstruktur erstellen
    target_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Eindeutigen Dateinamen für das Bild generieren und speichern
    unique_filename = f"{timestamp}_{filename}"
    image_path = target_images_dir / unique_filename
    with open(image_path, "wb") as img_f:
        img_f.write(image_bytes)
        
    # Relativen Pfad für die CSV setzen
    relative_filepath = os.path.join(target_dir.name, "images", unique_filename)
    row_data[1] = relative_filepath

    # In die entsprechende CSV-Datei loggen
    write_header = not target_log_file.exists()
    with open(target_log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        writer.writerow(row_data)


def _init_tracking_uri() -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    return uri


def _list_registered_model_versions(client: MlflowClient) -> List[str]:
    names = [rm.name for rm in client.search_registered_models()]

    uris: List[str] = []
    for name in names:
        try:
            versions = client.search_model_versions(f"name='{name}'")
        except Exception:
            continue
        for v in versions:
            full_version = client.get_model_version(name, v.version)
            aliases = full_version.aliases

            alias_str = f" 🏷️ {', '.join(aliases)}" if aliases else ""
            label = f"{name} v{v.version}{alias_str}"
            uris.append(label)
    print(uris)
    return sorted(uris)


@lru_cache(maxsize=128)
def _load_model(model_uri: str):
    """Load and cache the pyfunc model for a model registry URI (models:/name/version)."""
    return mlflow.pyfunc.load_model(model_uri)


def get_model_choices() -> List[str]:
    _init_tracking_uri()
    client = MlflowClient()
    return _list_registered_model_versions(client)


def predict(model_uri: str, files: Optional[List[gr.File]], actual_value_str: Optional[str]) -> pd.DataFrame:
    if not model_uri:
        return pd.DataFrame([{"error": "No model selected"}])
    if not files:
        return pd.DataFrame([{"error": "No images uploaded"}])
        
    # Echten Wert validieren und parsen, falls vorhanden
    actual_value = None
    if actual_value_str and actual_value_str.strip():
        try:
            actual_value = float(actual_value_str.replace(",", "."))
        except ValueError:
            print(f"Warning: Invalid actual biomass value entered: '{actual_value_str}'")

    model_info = model_uri.split(" 🏷️")[0] if " 🏷️" in model_uri else model_uri
    parts = model_info.rsplit(" v", 1)
    if len(parts) == 2:
        name, version = parts
        actual_uri = f"models:/{name}/{version}"
    else:
        actual_uri = model_uri
        
    model = _load_model(actual_uri)
    
    payload: List[bytes] = []
    names: List[str] = []
    for f in files:
        with open(f.name, "rb") as fh:
            content = fh.read()
            payload.append(content)
        names.append(os.path.basename(f.name))

    preds: List[float] = model.predict(payload)  # type: ignore

    # Iterativ loggen
    for name, content, pred in zip(names, payload, preds):
        try:
            log_inference(name, content, float(pred), model_uri, actual_value)
        except Exception as e:
            print(f"Failed to log inference for {name}: {e}")

    df = pd.DataFrame({"filename": names, "prediction": preds})
    return df


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Biomass Model Server") as demo:
        gr.Markdown("# Biomass Model Inference\nSelect a registered model version and upload images.")
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=["(loading...)"] ,
                value="(loading...)",
                label="Model Version",
                interactive=True,
                allow_custom_value=True,
            )
            refresh_btn = gr.Button("Refresh", variant="secondary")
            
        files = gr.Files(label="Images", file_types=["image"], file_count="multiple")
        
        # Optionales Eingabefeld für Ground-Truth/Ist-Werte hinzugefügt
        actual_val_input = gr.Textbox(
            label="Actual Biomass Value (Optional ground truth)", 
            placeholder="e.g. 85.50 (Leave blank for standard production log)"
        )
        
        predict_btn = gr.Button("Predict", variant="primary")
        output_df = gr.Dataframe(label="Predictions", interactive=False)

        def _refresh_choices():
            uris = get_model_choices()
            if not uris:
                return gr.Dropdown(
                    choices=["(none)"],
                    value="(none)",
                    label="Model Version",
                    interactive=True,
                    allow_custom_value=True,
                )
            return gr.Dropdown(
                choices=uris,
                value=uris[0],
                label="Model Version",
                interactive=True,
                allow_custom_value=True,
            )

        def _do_predict(selection, file_list, actual_val):
            df = predict(selection, file_list, actual_val)
            return df
            
        # UI Event-Bindings
        refresh_btn.click(fn=_refresh_choices, outputs=[model_dropdown])
        predict_btn.click(fn=_do_predict, inputs=[model_dropdown, files, actual_val_input], outputs=[output_df])
        demo.load(fn=_refresh_choices, outputs=[model_dropdown])
    return demo


if __name__ == "__main__":
    _init_tracking_uri()

    iface = build_interface()

    root_path = os.getenv("GRADIO_ROOT_PATH", "")
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    iface.launch(
        server_name=server_name,
        server_port=server_port,
        root_path=root_path
    )