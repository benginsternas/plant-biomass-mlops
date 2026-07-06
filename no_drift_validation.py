import os
import numpy as np
import pandas as pd
from PIL import Image
from evidently import Report
from evidently.presets import DataDriftPreset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "mlops_biomass_data", "images_med_res")
OUT_HTML = os.path.join(BASE_DIR, "results", "drift_report_no_drift_validation.html")
HOLDOUT_SIZE = 250
SEED = 99


def mean_pixel_intensity(path: str) -> float:
    with Image.open(path) as img:
        return float(np.array(img.convert("L")).mean())


def main():
    """Validate the drift monitor's true-negative behaviour: hold out a random
    sample of training images (never touched by the model or a production
    upload), compare it against the remaining training images as reference,
    and confirm Evidently correctly reports no drift when none exists."""
    all_files = [f for f in os.listdir(IMG_DIR) if f.endswith((".png", ".jpg", ".jpeg"))]
    rng = np.random.default_rng(SEED)
    holdout_files = rng.choice(all_files, size=HOLDOUT_SIZE, replace=False)
    holdout_set = set(holdout_files)
    reference_files = [f for f in all_files if f not in holdout_set]

    reference_vals = [mean_pixel_intensity(os.path.join(IMG_DIR, f)) for f in reference_files]
    current_vals = [mean_pixel_intensity(os.path.join(IMG_DIR, f)) for f in holdout_files]

    reference = pd.DataFrame({"mean_pixel_intensity": reference_vals})
    current = pd.DataFrame({"mean_pixel_intensity": current_vals})

    print(f"Reference: n={len(reference_vals)} mean={np.mean(reference_vals):.2f} std={np.std(reference_vals):.2f}")
    print(f"Current (holdout): n={len(current_vals)} mean={np.mean(current_vals):.2f} std={np.std(current_vals):.2f}")

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(current_data=current, reference_data=reference)
    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
    snapshot.save_html(OUT_HTML)

    for metric in snapshot.dict()["metrics"]:
        name = metric["metric_name"]
        if name.startswith("DriftedColumnsCount") or name.startswith("ValueDrift"):
            print(name, metric.get("value"), metric.get("config"))

    print(f"Saved to {OUT_HTML}")


if __name__ == "__main__":
    main()
