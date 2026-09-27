import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_IMG_DIR = os.path.join(BASE_DIR, 'mlops_biomass_data', 'images_med_res')
CSV_CANDIDATES = [
    os.path.join(BASE_DIR, 'digital_biomass_labels.csv'),
    os.path.join(BASE_DIR, 'mlops_biomass_data', 'digital_biomass_labels.xlsx'),
]
TARGET_DIR = os.path.join(BASE_DIR, 'production_data_measured')
TARGET_IMG_DIR = os.path.join(TARGET_DIR, 'images')

def main():
    os.makedirs(TARGET_IMG_DIR, exist_ok=True)
    
    # Finde originale Label-Datei
    data_path = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path, decimal=',')
    
    df = df.dropna(subset=['fresh_weight_total']).reset_index(drop=True)
    
    # Nimm die ersten 10 Bilder für die Simulation
    sample_df = df.head(10).copy()
    new_rows = []
    
    print(f"Simuliere Drift: Erhöhe Helligkeit für 10 Bilder...")
    for idx, row in sample_df.iterrows():
        src_path = os.path.join(SRC_IMG_DIR, row['filename'])
        if not os.path.exists(src_path):
            continue
            
        # Bild laden und Helligkeit extrem erhöhen (Simulierter Drift)
        with Image.open(src_path) as img:
            enhancer = ImageEnhance.Brightness(img)
            drifted_img = enhancer.enhance(3.5) # Extrem hell
            
            # Speicher unter neuem Namen im neuen Ordner
            new_filename = f"drifted_{row['filename']}"
            dest_path = os.path.join(TARGET_IMG_DIR, new_filename)
            drifted_img.save(dest_path)
            
        # Speichere relativen Pfad passend zur erwarteten Pipeline-Logik
        new_rows.append({
            'filename': os.path.join('production_data_measured', 'images', new_filename),
            'fresh_weight_total': row['fresh_weight_total']
        })
        
    # Erstelle die neue CSV-Struktur im Zielordner
    prod_metadata = pd.DataFrame(new_rows)
    prod_metadata.to_csv(os.path.join(TARGET_DIR, 'labels.csv'), index=False)
    print(f"Erfolgreich {len(prod_metadata)} Bilder und 'labels.csv' in {TARGET_DIR} erstellt.")

if __name__ == "__main__":
    main()