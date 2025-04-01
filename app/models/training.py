import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score
import joblib
import sys

from app.config import MODEL_FEATURES, MODEL_ARTIFACT_PATH, OPTUNA_TRIALS
from app.services.feature_engineering import FeatureEngineer # Import FeatureEngineer

class TitanicModel:
    def __init__(self):
        self.model = None
        self.model_version = "1.0"
        self.feature_engineer = FeatureEngineer() # Instantiate FeatureEngineer

    def train(self, df: pd.DataFrame):
        # Debug print df shape and columns
        print(f"Training data shape: {df.shape}")
        print(f"Training data columns: {df.columns.tolist()}")
    
        # Ensure target variable is extracted before feature engineering
        if 'Survived' not in df.columns:
            print(f"ERROR: 'Survived' column not found in {df.columns.tolist()}")
            sys.exit(1)
            
        y = df['Survived'].copy()
        
        # Normalize column names for consistency
        df.columns = [col.lower() for col in df.columns]
        print(f"After lowercase in training: {df.columns.tolist()}")
        
        # Apply feature engineering with lowercase data
        try:
            df_transformed = self.feature_engineer.transform(df)
            print(f"Transformed data columns: {df_transformed.columns.tolist()}")
        except Exception as e:
            print(f"Error during feature transformation: {str(e)}")
            raise e
        
        # Validate MODEL_FEATURES against available columns
        missing_features = [f for f in MODEL_FEATURES if f not in df_transformed.columns]
        if missing_features:
            print(f"ERROR: Missing features: {missing_features}")
            print(f"Available features: {df_transformed.columns.tolist()}")
            # Add placeholder columns with default values for missing features
            for feature in missing_features:
                df_transformed[feature] = 0.0
        
        # Select features from the transformed DataFrame
        X = df_transformed[MODEL_FEATURES]
        print(f"Final X shape: {X.shape}")
        
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self._objective(trial, X, y), n_trials=OPTUNA_TRIALS)
        
        best_params = study.best_params
        print(f"Best parameters: {best_params}")
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
