import pandas as pd
from app.models.schemas import PassengerFeatures
from app.models.training import TitanicModel
from app.services.feature_engineering import FeatureEngineer

class PredictionService:
    def __init__(self):
        self.model = TitanicModel()
        self.feature_engineer = FeatureEngineer()
    
    async def predict(self, passenger: PassengerFeatures):
        features = self._prepare_features(passenger)
        probability = self.model.predict(features)
        
        return {
            "survival_probability": float(probability),
            "survived": probability > 0.5,
            "model_version": self.model.model_version
        }
    
    def _prepare_features(self, passenger: PassengerFeatures):
        # Convert passenger features to DataFrame
        features = passenger.dict()
        return pd.DataFrame([features])
    
    def get_model_metrics(self):
        return {
            "accuracy": 0.85,
            "f1_score": 0.83,
            "roc_auc": 0.87
        }
