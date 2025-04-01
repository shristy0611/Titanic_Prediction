# Titanic Survival Prediction Service

A sophisticated ML service that predicts Titanic passenger survival using FastAPI.

## Features
- FastAPI-based REST API
- LightGBM model with Optuna hyperparameter optimization
- Feature engineering pipeline
- Model versioning and metrics
- JSON logging
- Data validation

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the server: `uvicorn app.main:app --reload`
3. Access API docs: `http://localhost:8000/docs`

## API Endpoints
- `GET /`: Health check and version
- `POST /predict`: Make survival prediction
- `POST /train`: Retrain model
- `GET /metrics`: Model performance metrics

## Project Structure
```
titanic_service/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   └── training.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py
│   │   └── prediction.py
│   └── utils/
│       ├── __init__.py
│       └── logging.py
├── data/
│   └── train.csv
├── docs/
│   ├── project_overview.md
│   └── project_email.md
├── requirements.txt
└── README.md
```

## Documentation
See `docs/project_overview.md` for detailed technical documentation.
