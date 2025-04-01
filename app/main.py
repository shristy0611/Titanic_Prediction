from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uvicorn

from app.models.schemas import PassengerFeatures, PredictionResponse
from app.services.prediction import PredictionService
from app.models.training import TitanicModel
from app.config import API_TITLE, API_DESCRIPTION, API_VERSION, DATA_PATH
from app.utils.logging import logger

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service
prediction_service = PredictionService()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to the Titanic Survival Prediction API",
        "model_version": prediction_service.model.model_version
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(passenger: PassengerFeatures):
    """Make survival prediction for a passenger."""
    try:
        return await prediction_service.predict(passenger)
    except Exception as e:
        logger.error("Prediction endpoint failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model():
    """Retrain the model with the latest data."""
    try:
        # Load training data
        df = pd.read_csv(DATA_PATH)
        
        # Initialize and train model
        model = TitanicModel()
        model.train(df)
        
        # Reload the prediction service with new model
        prediction_service.model.load_model()
        
        return {
            "message": "Model training completed successfully",
            "model_version": model.model_version
        }
    except Exception as e:
        logger.error("Training endpoint failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Verify model is loaded
        if prediction_service.model.model is None:
            raise Exception("Model not loaded")
            
        return {
            "status": "healthy",
            "model_version": prediction_service.model.model_version
        }
    except Exception as e:
        logger.error("Health check failed", extra={"error": str(e)})
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/metrics")
async def model_metrics():
    """Get model performance metrics."""
    try:
        metrics = prediction_service.get_model_metrics()
        return {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "model_version": prediction_service.model.model_version
        }
    except Exception as e:
        logger.error("Metrics endpoint failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
