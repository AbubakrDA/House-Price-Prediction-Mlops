import mlflow
import mlflow.sklearn
import logging

# Set up logging for educational purposes
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_best_model_run_id(experiment_name: str) -> str:
    """
    Fetches the best model run ID from MLflow based on RMSE.
    """
    # Only set local tracking URI if it's not already set (e.g. by DagsHub)
    if not mlflow.get_tracking_uri() or mlflow.get_tracking_uri().startswith("file:"):
        logger.info("Setting local MLflow tracking URI.")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # 2. Get the experiment by name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    
    # 3. Search for runs and sort by RMSE
    # We want the lowest RMSE.
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1
    )
    
    if not runs.empty:
        best_run_id = runs.iloc[0]['run_id']
        logger.info(f"Best run ID found: {best_run_id} with RMSE: {runs.iloc[0]['metrics.rmse']}")
        return best_run_id
    else:
        raise ValueError("No runs found in experiment.")

def load_model_from_mlflow(run_id: str):
    """
    Loads the scikit-learn model logged in the specified run.
    """
    model_uri = f"runs:/{run_id}/model"
    logger.info(f"Loading model from {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    return model
