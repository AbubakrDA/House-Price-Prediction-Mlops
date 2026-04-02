import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import REFERENCE_YEAR

class HouseFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for House Price feature engineering.
    - house_age: Calculated using a fixed reference year (2014) for reproducibility.
    - is_renovated: Binary flag indicating if renovations occurred.
    - Drops original time columns to avoid redundancy.
    """
    def __init__(self, ref_year: int = REFERENCE_YEAR):
        self.ref_year = ref_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Calculated features
        X['house_age'] = self.ref_year - X['yr_built']
        X['is_renovated'] = X['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
        
        # We drop 'date' and the original year columns as the relevant variance 
        # is now captured in 'house_age'.
        return X.drop(columns=['date', 'yr_built', 'yr_renovated'])

def get_preprocessing_pipeline(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Creates a column transformer for standardized scaling and encoding.
    """
    # handle_unknown='ignore' is crucial for deployment to handle new/rare cities or zips gracefully.
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
