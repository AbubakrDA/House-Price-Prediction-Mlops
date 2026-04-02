import mlflow
import mlflow.sklearn
import logging
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.config import DATA_PATH, EXPERIMENT_NAME, RANDOM_STATE, TEST_SIZE, REFERENCE_YEAR
from src.utils import setup_mlflow
from src.data import load_raw_data, clean_data
from src.features import HouseFeatureExtractor, get_preprocessing_pipeline
from src.evaluate import get_metrics
from src.model_selection import select_best_model

logger = logging.getLogger(__name__)

def run_training():
    # 1. Initialize tracking
    setup_mlflow()
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 2. Extract and split data
    df_raw = load_raw_data(DATA_PATH)
    df = clean_data(df_raw)
    
    X = df.drop(columns=['price'])
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 3. Model configurations
    numeric_features = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
        'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement'
    ]
    categorical_features = ['city', 'statezip']

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
    }

    run_results = []

    # 4. Training Loop
    for model_name, model_obj in models.items():
        with mlflow.start_run(run_name=model_name):
            logger.info(f"🚀 Training {model_name}...")
            
            # Encapsulating entire logic in a Pipeline prevents Training/Serving Skew.
            pipeline = Pipeline([
                ('feature_eng', HouseFeatureExtractor(ref_year=REFERENCE_YEAR)),
                ('preprocessor', get_preprocessing_pipeline(numeric_features, categorical_features)),
                ('regressor', model_obj)
            ])

            # Train
            pipeline.fit(X_train, y_train)

            # Evaluate
            predictions = pipeline.predict(X_test)
            metrics = get_metrics(y_test, predictions)

            # MLflow Logging: Use model name and parameters for tracking performance over time.
            mlflow.log_params(model_obj.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            
            run_results.append({
                "run_name": model_name,
                "metrics": metrics,
                "params": model_obj.get_params()
            })
            
            logger.info(f"✅ {model_name} logged. RMSE: {metrics['rmse']:.2f}")

    # 5. Programmatic selection of the Champion
    select_best_model(run_results)

if __name__ == "__main__":
    run_training()
