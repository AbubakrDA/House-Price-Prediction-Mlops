import mlflow
import dagshub
import logging
from src.config import REPO_OWNER, REPO_NAME, MLFLOW_DB

# Logging remains essential for MLOps monitoring and debugging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_mlflow():
    """
    Configures MLflow tracking URI.
    Priority: DagsHub (if credentials present) -> local SQLite.
    """
    if REPO_OWNER and REPO_NAME:
        logger.info(f"Connecting to DagsHub: {REPO_OWNER}/{REPO_NAME}")
        dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    else:
        logger.info("DagsHub credentials not found. Falling back to local sqlite database.")
        mlflow.set_tracking_uri(MLFLOW_DB)
