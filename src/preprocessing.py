import pandas as pd
import numpy as np
import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# Set up logging for educational purposes
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to perform feature engineering.
    - house_age = reference_year - yr_built
    - renovation_flag = 1 if yr_renovated > 0 else 0
    - Drop original yr_built, yr_renovated, and date since we've extracted the value.
    """
    def __init__(self, reference_year: int = 2014):
        self.reference_year = reference_year
        logger.info(f"Initialized FeatureEngineer with reference year {self.reference_year}")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Calculate house age
        X['house_age'] = self.reference_year - X['yr_built']
        
        # Create renovation binary flag
        X['renovation_flag'] = X['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
        
        # We drop the date and original years as they're no longer needed for prediction.
        cols_to_drop = ['date', 'yr_built', 'yr_renovated']
        X = X.drop(columns=cols_to_drop)
        
        return X

def get_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Creates a preprocessor using ColumnTransformer.
    - numeric_features get StandardScaler
    - categorical_features get OneHotEncoder
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        # handle_unknown='ignore' is critical for production API serving.
        # It ensures that if the API receives a city it hasn't seen during training, it doesn't crash.
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor
