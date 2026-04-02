import mlflow
import mlflow.sklearn
import logging
from src.config import EXPERIMENT_NAME, MLFLOW_DB
from src.utils import setup_mlflow

logger = logging.getLogger(__name__)

def get_all_active_models():
    """
    Fetches all models from the current experiment in MLflow.
    Identifies the champion (lowest RMSE) to prioritize results.
    """
    setup_mlflow()
    
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        raise ValueError(f"Experiment {EXPERIMENT_NAME} not found.")
        
    # Get all successful runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'"
    )
    
    if runs.empty:
        raise ValueError("No successful models found in MLflow.")

    # Identify champion based on RMSE
    champion_row = runs.sort_values("metrics.rmse", ascending=True).iloc[0]
    champion_name = champion_row["tags.mlflow.runName"]
    
    models = {}
    for _, run in runs.iterrows():
        run_name = run["tags.mlflow.runName"]
        run_id = run.run_id
        model_uri = f"runs:/{run_id}/model"
        
        logger.info(f"Loading model: {run_name} from {model_uri}")
        models[run_name] = mlflow.sklearn.load_model(model_uri)
        
    return models, champion_name
