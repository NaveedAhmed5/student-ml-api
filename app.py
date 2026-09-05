"""
Student ML API - FastAPI Application
Assignment 1: MLOps
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import math

# --- App Initialization ---
app = FastAPI(
    title="Student ML API",
    description="A simple ML prediction API built for the MLOps Assignment.",
    version="1.0.0",
)

# --- Request / Response Models ---
class PredictRequest(BaseModel):
    value: float

    @field_validator("value")
    @classmethod
    def value_must_be_finite(cls, v):
        if not math.isfinite(v):
            raise ValueError("value must be a finite number (not NaN or Inf)")
        return v

class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    input: float
    prediction: float
    model_version: str

class HealthResponse(BaseModel):
    status: str
    api_version: str

# --- Endpoints ---
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Returns the current health status of the API.
    """
    return HealthResponse(status="healthy", api_version="1.0.0")


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Accepts a numeric value and returns a simple polynomial prediction.
    Model: y = 2x^2 + 3x + 1
    """
    x = request.value
    prediction = 2 * (x ** 2) + 3 * x + 1
    return PredictResponse(
        input=x,
        prediction=prediction,
        model_version="1.0.0",
    )
