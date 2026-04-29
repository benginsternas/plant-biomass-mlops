import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np

# Verzeichnisse erstellen
os.makedirs('figures', exist_ok=True)

# Daten laden (Pfade anpassen falls nötig)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'digital_biomass_labels.csv')
IMG_DIR = os.path.join(BASE_DIR, 'mlops_biomass_data', 'images_med_res')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

# 2. Ordner erstellen
os.makedirs(FIGURES_DIR, exist_ok=True)

# 3. DANN die Daten laden (mit dem richtigen Pfad-Objekt)
df = pd.read_csv(CSV_PATH, decimal=',')

# 1. Target distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['fresh_weight_total'], kde=True)
plt.title('Distribution of Fresh Weight Total')
plt.savefig('figures/target_distribution.png')

# 2. Sample images grid
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
samples = df.sample(6)
for i, (idx, row) in enumerate(samples.iterrows()):
    ax = axes[i//3, i%3]
    img = Image.open(os.path.join(IMG_DIR, row['filename']))
    ax.imshow(img)
    ax.set_title(f"Weight: {row['fresh_weight_total']}")
    ax.axis('off')
plt.tight_layout()
plt.savefig('figures/sample_images.png')

# 3. Metadata correlations
#plt.figure(figsize=(8, 6))
#sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
#plt.title('Correlation Heatmap')
#plt.savefig('figures/correlation_heatmap.png')
# 3. Optimierte Correlation Heatmap
plt.figure(figsize=(12, 10)) # Deutlich größeres Bild

# Nur relevante, numerische Spalten auswählen (IDs und Zeitstempel weglassen)
relevant_cols = [
    'fresh_weight_total', 'fresh_weight_shoot', 'fresh_weight_root', 
    'height_shoot', 'age_days', 'temperature', 'humidity', 
    'luminancelux', 'total_leaves'
]

# Prüfen, welche dieser Spalten tatsächlich in deiner CSV existieren
existing_cols = [c for c in relevant_cols if c in df.columns]
corr_matrix = df[existing_cols].corr()

# Heatmap zeichnen
sns.heatmap(
    corr_matrix, 
    annot=True,          # Zahlen anzeigen
    fmt=".2f",           # Nur 2 Nachkommastellen
    cmap='coolwarm',     # Schöner Blau-Rot-Kontrast
    linewidths=0.5,      # Kleine Lücken zwischen den Kästchen
    annot_kws={"size": 8} # Kleinere Schrift für die Zahlen
)

plt.title('Relevante Merkmals-Korrelationen', fontsize=15)
plt.xticks(rotation=45, ha='right') # Labels drehen, damit sie nicht überlappen
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'correlation_heatmap.png'))
# 4. Temporal analysis
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='age_days', y='fresh_weight_total')
plt.title('Biomass vs Age')
plt.savefig('figures/age_vs_biomass.png')

# 5. Image pixel analysis (Mittlere Helligkeit)
brightness = []
for img_name in df.sample(100)['filename']:
    img = Image.open(os.path.join(IMG_DIR, img_name)).convert('L')
    brightness.append(np.mean(img))
plt.figure(figsize=(8, 5))
sns.histplot(brightness, color='gray')
plt.title('Pixel Intensity Distribution (Sample)')
plt.savefig('figures/image_pixel_analysis.png')

print("EDA abgeschlossen. Plots in 'figures/' gespeichert.")