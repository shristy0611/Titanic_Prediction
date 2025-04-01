import pandas as pd
from app.models.training import TitanicModel
from app.config import DATA_PATH
from app.utils.logging import logger

def run_training():
    """Loads data and trains the Titanic model."""
    logger.info("Starting model training script...")
    try:
        # Load training data
        logger.info(f"Loading data from: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        
        # Initialize and train model
        logger.info("Initializing TitanicModel...")
        model = TitanicModel()
        
        logger.info("Starting model training...")
        model.train(df) # Pass the original df, train method handles transformation now
        
        logger.info(f"Model training completed successfully. Model version: {model.model_version}")
        logger.info(f"Model artifact saved to: {model.MODEL_ARTIFACT_PATH}") # Access class attribute directly

    except FileNotFoundError:
        logger.error(f"Training data file not found at: {DATA_PATH}")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)

if __name__ == "__main__":
    run_training()
