import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")
MODEL_ARTIFACT_PATH = os.path.join(BASE_DIR, "data", "titanic_model_artifact.joblib")

# Model features
NUMERICAL_FEATURES = ['Age', 'Fare', 'FamilySize', 'SibSp', 'Parch']
CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone']
MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# API Configuration
API_TITLE = "Sophisticated Titanic Prediction Service"
API_DESCRIPTION = "Predicts survival with FE, HPO, Metadata, Validation, JSON Logs"
API_VERSION = "1.1"

# Optuna Configuration
OPTUNA_TRIALS = 20