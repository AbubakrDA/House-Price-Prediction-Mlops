import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from app.schemas import HouseDataInput, PredictionOutput
from app.model_loader import get_all_active_models

# Standard logging facilitates production observability.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Seattle House Price Prediction API",
    description="Production-grade ML API connected to MLflow/DagsHub registry.",
    version="1.0.0"
)

# Global variables to cache models in memory for performance.
models_cache = {}
champion_name = "unknown"

@app.on_event("startup")
def load_models():
    """ Load all models from MLflow registry on startup. """
    global models_cache, champion_name
    try:
        models_cache, champion_name = get_all_active_models()
        logger.info(f"Loaded {len(models_cache)} models. Champion: {champion_name}")
    except Exception as e:
        logger.error(f"Startup failing: Model load error -> {e}")

@app.get("/health")
def health_check():
    """ Readiness check for monitoring systems (e.g. K8s). """
    return {"status": "healthy", "models_loaded_count": len(models_cache)}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: HouseDataInput):
    """
    Multi-model inference endpoint.
    Returns predictions from all available models for comparison.
    """
    if not models_cache:
        raise HTTPException(status_code=503, detail="Model registry unavailable.")
    
    try:
        # Pipeline expects a DataFrame to correctly trigger custom transformers.
        input_data = pd.DataFrame([data.dict()])
        
        predictions = {}
        for m_name, m_obj in models_cache.items():
            predictions[m_name] = float(m_obj.predict(input_data)[0])
        
        return PredictionOutput(
            predictions=predictions,
            champion=champion_name,
            champion_prediction=predictions[champion_name]
        )
    except Exception as e:
        logger.error(f"Multi-inference error: {e}")
        raise HTTPException(status_code=500, detail="Error during multi-model calculation.")
