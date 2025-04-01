from pydantic import BaseModel, Field, validator, root_validator

class PassengerFeatures(BaseModel):
    age: float = Field(..., ge=0, le=120)
    fare: float = Field(..., ge=0)
    sex: str = Field(..., pattern="^(male|female)$")
    pclass: int = Field(..., ge=1, le=3)
    embarked: str = Field(..., pattern="^[CQS]$")
    sibsp: int = Field(..., ge=0)
    parch: int = Field(..., ge=0)
    
    @validator('age')
    def validate_age(cls, v):
        if v < 0 or v > 120:
            raise ValueError('Age must be between 0 and 120')
        return v

class PredictionResponse(BaseModel):
    survival_probability: float
    survived: bool
    model_version: str
