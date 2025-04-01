import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")
MODEL_ARTIFACT_PATH = os.path.join(BASE_DIR, "data", "titanic_model_artifact.joblib")

# Model features after one-hot encoding
# Base numerical features + Pclass + engineered features
BASE_FEATURES = ['Age', 'Fare', 'FamilySize', 'Pclass', 'IsAlone'] 
# One-hot encoded features derived from the training run output
ONE_HOT_FEATURES = [
    'Sex_male', 
    'Embarked_Q', 
    'Embarked_S', 
    'Title_Col', 
    'Title_Countess', 
    'Title_Don', 
    'Title_Dr', 
    'Title_Jonkheer', 
    'Title_Lady', 
    'Title_Major', 
    'Title_Master', 
    'Title_Miss', 
    'Title_Mlle', 
    'Title_Mme', 
    'Title_Mr', 
    'Title_Mrs', 
    'Title_Ms', 
    'Title_Rev', 
    'Title_Sir'
]
MODEL_FEATURES = BASE_FEATURES + ONE_HOT_FEATURES

# API Configuration
API_TITLE = "Sophisticated Titanic Prediction Service"
API_DESCRIPTION = "Predicts survival with FE, HPO, Metadata, Validation, JSON Logs"
API_VERSION = "1.1"

# Optuna Configuration
OPTUNA_TRIALS = 20
