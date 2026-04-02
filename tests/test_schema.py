import pytest
from app.schemas import HouseDataInput

def test_valid_input_schema():
    """ Verify schema accepts full payload. """
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
    house = HouseDataInput(**payload)
    assert house.city == "Shoreline"
    assert house.sqft_living == 1340

def test_invalid_categorical_bounds():
    """ Verify Pydantic catches out-of-range values. """
    payload = {
        "date": "2014-05-02 00:00:00",
        "bedrooms": 3.0,
        "bathrooms": 1.5,
        "sqft_living": 1340,
        "sqft_lot": 7912,
        "floors": 1.5,
        "waterfront": 2, # Out of range (max 1)
        "view": 0,
        "condition": 3,
        "sqft_above": 1340,
        "yr_built": 1955,
        "city": "Shoreline",
        "statezip": "WA 98133"
    }
    with pytest.raises(ValueError):
        HouseDataInput(**payload)
