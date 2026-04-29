Plant Biomass Prediciton - MLOps Praktikum 1 - Gruppe 2

This project implements a machine learning pipeling for plant biomass using top-down images of plants. 
The machine learning pipeline includes Exploratory Data Analysis, a model training workflow using PyTorch,
and automated logging.

1. Dataset Description & Statistics

    Samples: 4294 images
    Target Variable: fresh_weight_total
    Features: Temporal data, sensor data
    Input Data: colored images located in "mlops_biomass_data"/"images_med_res"

2. Exploratory Data Analysis

    2.1 Target Distribution

    2.2 Correlation Heatmap

    2.3 Biomass vs Age

    2.4 Image Pixel Analysis

    2.5 Sample images

3. Data Quality Issues

    Missing Labels:
    df.dropna() found empty rows, so we had to implement, that invalid labeled samples are removed automatically.

    Target Scale:
    Initial loss was "NaN" due to high mean squared error values. Labels ware scaled from 0 to 1 by dividing each value by the highest value.

4. Model Architecture

5. Training Procedure

    The training was conducted on a CPU/GPU-agnostic pipeline with the following settings:

    Optimizer: Adam

    Loss Function: Mean Squared Error (MSE)

    Learning Rate: 0.0001

    Batch Size: 16

    Epochs: 3

    Data Split: 80% Training / 20% Validation (Seed: 42).

6. Results

    6.1 Training Curves
    The loss decreases over each epoch. The Validation Loss follows the Training Loss. The Model is not overfitting.

7. Challenges & Improvements

    7.1 Challenges
    Exploding Gradients: Initial training failed with "NaN" loss because the biomass weights were too large for an unscaled regression head.

    7.2 Improvements

9. Reproduction Instructions

    1. Clone the Repository

        git clone https://gitlab.nt.fh-koeln.de/gitlab/mlops/praktikum/mlops_2/MLOps_P1_2.git
        cd MLOps_P1_2

    2. Setup Environment

        python -m venv .venv
        source .venv/bin/activate  
        pip install -r requirements.txt

    3. Run the EDA

        python eda.py

    4. Run the Model Training

        python train_model.py --epochs 3 --lr 0.0001