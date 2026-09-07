"""
Tests for Student ML API
Assignment 1: MLOps

Covers:
  1. Health check returns 200 with correct status
  2. Health check returns correct application name and version (v1.1.0)
  3. Successful prediction with valid input
  4. Prediction math is correct
  5. Missing request body returns 422
  6. Invalid input type (string) returns 422
  7. Edge case: prediction at value=0
  8. Negative number input
"""

import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


# ─────────────────────────────────────────────
# Part 1: Health Endpoint Tests
# ─────────────────────────────────────────────

def test_health_check_returns_200():
    """GET /health should return HTTP 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_check_status_is_healthy():
    """GET /health should return status='healthy'."""
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "healthy"


def test_health_check_returns_application_name():
    """GET /health should return correct application name."""
    response = client.get("/health")
    data = response.json()
    assert data["application"] == "student-ml-api"


def test_health_check_returns_application_version():
    """GET /health should return application_version='1.1.0'."""
    response = client.get("/health")
    data = response.json()
    assert "application_version" in data
    assert data["application_version"] == "1.1.0"


def test_health_check_returns_model_version():
    """GET /health should return model_version='model-1'."""
    response = client.get("/health")
    data = response.json()
    assert data["model_version"] == "model-1"


# ─────────────────────────────────────────────
# Part 2: Predict Endpoint Tests
# ─────────────────────────────────────────────

def test_predict_returns_200_with_valid_input():
    """POST /predict with valid input should return HTTP 200 OK."""
    response = client.post("/predict", json={"value": 5.0})
    assert response.status_code == 200


def test_predict_math_is_correct():
    """POST /predict should compute y = 2x^2 + 3x + 1 correctly."""
    # For x=5: 2*(25) + 3*(5) + 1 = 50 + 15 + 1 = 66
    response = client.post("/predict", json={"value": 5.0})
    data = response.json()
    assert data["prediction"] == pytest.approx(66.0)


def test_predict_at_zero():
    """POST /predict with x=0 should return prediction=1."""
    # For x=0: 2*(0) + 3*(0) + 1 = 1
    response = client.post("/predict", json={"value": 0.0})
    data = response.json()
    assert data["prediction"] == pytest.approx(1.0)


def test_predict_with_negative_number():
    """POST /predict should handle negative inputs correctly."""
    # For x=-2: 2*(4) + 3*(-2) + 1 = 8 - 6 + 1 = 3
    response = client.post("/predict", json={"value": -2.0})
    data = response.json()
    assert data["prediction"] == pytest.approx(3.0)


def test_predict_response_contains_all_fields():
    """POST /predict response must contain input, prediction, and model_version."""
    response = client.post("/predict", json={"value": 3.0})
    data = response.json()
    assert "input" in data
    assert "prediction" in data
    assert "model_version" in data


def test_predict_missing_body_returns_422():
    """POST /predict with no body should return HTTP 422 Unprocessable Entity."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_invalid_string_input_returns_422():
    """POST /predict with a string value should return HTTP 422 Unprocessable Entity."""
    response = client.post("/predict", json={"value": "not-a-number"})
    assert response.status_code == 422
