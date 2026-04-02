from pydantic import BaseModel, Field
from typing import Optional

class HouseDataInput(BaseModel):
    """
    Validation schema for house prediction requests.
    Using Pydantic ensures the API layer protects the ML pipeline from Malformed data.
    """
    date: str = Field(..., description="Date of sale (e.g., '2014-05-02 00:00:00')")
    bedrooms: float
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int = Field(0, ge=0, le=1)
    view: int = Field(0, ge=0, le=4)
    condition: int = Field(3, ge=1, le=5)
    sqft_above: int
    sqft_basement: int = 0
    yr_built: int
    yr_renovated: int = 0
    city: str
    statezip: str

class PredictionOutput(BaseModel):
    """
    Comparative output for multi-model serving.
    - predictions: Dictionary of {model_name: price}
    - champion: The name of the best-performing model
    - champion_prediction: The price from the champion
    """
    predictions: dict[str, float]
    champion: str
    champion_prediction: float
