import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score
import joblib

from app.config import MODEL_FEATURES, MODEL_ARTIFACT_PATH, OPTUNA_TRIALS
from app.services.feature_engineering import FeatureEngineer # Import FeatureEngineer

class TitanicModel:
    def __init__(self):
        self.model = None
        self.model_version = "1.0"
        self.feature_engineer = FeatureEngineer() # Instantiate FeatureEngineer

    def train(self, df: pd.DataFrame):
        # Apply feature engineering first
        df_transformed = self.feature_engineer.transform(df)
        
        # Select features and target from the *transformed* DataFrame
        X = df_transformed[MODEL_FEATURES] 
        y = df_transformed['Survived']
        
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self._objective(trial, X, y), n_trials=OPTUNA_TRIALS)
        
        best_params = study.best_params
        self.model = lgb.LGBMClassifier(**best_params)
        self.model.fit(X, y)
        
        joblib.dump(self.model, MODEL_ARTIFACT_PATH)
    
    def _objective(self, trial, X, y):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 50)
        }
        model = lgb.LGBMClassifier(**params)
        scores = cross_val_score(model, X, y, cv=5)
        return scores.mean()
    
    def predict(self, features: pd.DataFrame) -> float:
        if self.model is None:
            self.load_model()
        return self.model.predict_proba(features)[0][1]
    
    def load_model(self):
        self.model = joblib.load(MODEL_ARTIFACT_PATH)
