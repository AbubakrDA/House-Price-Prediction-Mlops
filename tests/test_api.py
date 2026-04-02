import pytest
from fastapi.testclient import TestClient
from app.main import app

# In MLOps, API health checks are standard for deployment readiness.
client = TestClient(app)

def test_read_health():
    """ Verify health endpoint is reachable. """
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_prediction_endpoint_with_model():
    """ 
    Verify multi-model response structure.
    """
    payload = {
        "date": "2014-05-02 00:00:00",
        "bedrooms": 3.0,
        "bathrooms": 1.5,
        "sqft_living": 1340,
        "sqft_lot": 7912,
        "floors": 1.5,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "sqft_above": 1340,
        "sqft_basement": 0,
        "yr_built": 1955,
        "yr_renovated": 2005,
        "city": "Shoreline",
        "statezip": "WA 98133"
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "champion" in data
            assert "champion_prediction" in data
            assert isinstance(data["predictions"], dict)
        else:
            # If training wasn't run yet or registry missing
            assert response.status_code == 503
