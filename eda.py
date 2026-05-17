import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np

# 1. Verzeichnisse und Pfade
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'digital_biomass_labels.csv')
IMG_DIR = os.path.join(BASE_DIR, 'mlops_biomass_data', 'images_med_res')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# 2. Daten laden und reinigen
df = pd.read_csv(CSV_PATH, decimal=',')
df = df.dropna(subset=['fresh_weight_total'])

# --- 1. SMART SAMPLES (diverse_samples.png -> jetzt wieder sample_images.png) ---
# Wir zeigen die gesamte Bandbreite: 2 kleinste, 2 mittlere, 2 größte Pflanzen
df_sorted = df.sort_values('fresh_weight_total')
indices = [0, 1, len(df)//2, len(df)//2 + 1, len(df)-2, len(df)-1]
smart_samples = df_sorted.iloc[indices]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, (idx, row) in enumerate(smart_samples.iterrows()):
    ax = axes[i//3, i%3]
    img_path = os.path.join(IMG_DIR, row['filename'])
    if os.path.exists(img_path):
        img = Image.open(img_path)
        ax.imshow(img)
        # Beschriftung zeigt die Kategorie für den besseren Überblick
        cat = 'Small' if i < 2 else 'Medium' if i < 4 else 'Large'
        ax.set_title(f"Weight: {row['fresh_weight_total']:.4f}\n({cat})")
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'sample_images.png'))

# --- 2. TARGET-FOCUSED HEATMAP (correlation_heatmap.png) ---
# Zeigt nur die Korrelationen mit dem Zielwert (fresh_weight_total)
plt.figure(figsize=(6, 10))
relevant_cols = [
    'fresh_weight_total', 'fresh_weight_shoot', 'fresh_weight_root', 
    'height_shoot', 'age_days', 'temperature', 'humidity', 
    'luminancelux', 'total_leaves'
]
existing_cols = [c for c in relevant_cols if c in df.columns]
# Korrelation berechnen und nach Stärke sortieren
target_corr = df[existing_cols].corr()[['fresh_weight_total']].sort_values(by='fresh_weight_total', ascending=False)

sns.heatmap(target_corr, annot=True, fmt=".2f", cmap='RdYlGn', linewidths=0.5)
plt.title('Correlation with Total Weight', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'correlation_heatmap.png'))

# --- 3. RGB PIXEL ANALYSIS (image_pixel_analysis.png) ---
# Farbanalyse statt Graustufen
r_means, g_means, b_means = [], [], []
for img_name in df.sample(min(100, len(df)))['filename']:
    img_path = os.path.join(IMG_DIR, img_name)
    if os.path.exists(img_path):
        img_np = np.array(Image.open(img_path).convert('RGB'))
        r_means.append(np.mean(img_np[:,:,0]))
        g_means.append(np.mean(img_np[:,:,1]))
        b_means.append(np.mean(img_np[:,:,2]))

plt.figure(figsize=(10, 6))
sns.kdeplot(r_means, color='red', label='Red Channel', fill=True, alpha=0.1)
sns.kdeplot(g_means, color='green', label='Green Channel', fill=True, alpha=0.1)
sns.kdeplot(b_means, color='blue', label='Blue Channel', fill=True, alpha=0.1)
plt.title('RGB Intensity Distribution')
plt.xlabel('Mean Pixel Intensity (0-255)')
plt.ylabel('Density')
plt.legend()
plt.savefig(os.path.join(FIGURES_DIR, 'image_pixel_analysis.png'))

# 4. Zusätzliche Standard-Plots (bleiben gleich)
plt.figure(figsize=(8, 5))
sns.histplot(df['fresh_weight_total'], kde=True)
plt.title('Distribution of Fresh Weight Total')
plt.savefig(os.path.join(FIGURES_DIR, 'target_distribution.png'))

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='age_days', y='fresh_weight_total')
plt.title('Biomass vs Age')
plt.savefig(os.path.join(FIGURES_DIR, 'age_vs_biomass.png'))

print(f"EDA completed. All plots saved in '{FIGURES_DIR}' with original filenames.")
