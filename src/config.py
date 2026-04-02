import os

# Centralizing configuration ensures reproducibility across training and serving.
# In MLOps, we avoid hardcoding so that pipelines can be easily moved/scaled.

# File Paths
DATA_PATH = "data.csv"
MLFLOW_DB = "sqlite:///mlflow.db"

# Experiment Tracking
EXPERIMENT_NAME = "House_Price_Prediction_DagsHub"
REFERENCE_YEAR = 2014  # Fixed year for house_age to ensure deterministic features.

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# DagsHub / Remote MLflow Configuration
# In a real pipeline, these would be set as environment variables.
REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
REPO_NAME = os.getenv("DAGSHUB_REPO_NAME")
