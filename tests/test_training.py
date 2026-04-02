import pandas as pd
import numpy as np
import os
import sys
import pytest

# Adding root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import clean_data
from src.preprocessing import FeatureEngineer

def test_data_cleaning():
    """
    Tests if the cleaning logic correctly:
    - Drops 'street' and 'country'
    - Removes zero prices
    """
    data = {
        'date': ['2014-05-02', '2014-05-03'],
        'price': [313000.0, 0.0], # One zero price
        'bedrooms': [3.0, 4.0],
        'street': ['A', 'B'],
        'country': ['USA', 'USA'],
        'city': ['Shoreline', 'Seattle'],
        'statezip': ['WA 98133', 'WA 98119']
    }
    df = pd.DataFrame(data)
    
    cleaned_df = clean_data(df)
    
    # Assertions
    assert 'street' not in cleaned_df.columns
    assert 'country' not in cleaned_df.columns
    assert 0.0 not in cleaned_df['price'].values
    assert len(cleaned_df) == 1

def test_feature_engineering():
    """
    Tests if the custom transformer correctly:
    - Calculates age (2014 - built)
    - Sets renovation flag
    - Drops original columns
    """
    data = {
        'date': ['2014-05-02'],
        'yr_built': [1955],
        'yr_renovated': [2005]
    }
    df = pd.DataFrame(data)
    
    fe = FeatureEngineer(reference_year=2014)
    transformed_df = fe.transform(df)
    
    # Assertions
    assert transformed_df.iloc[0]['house_age'] == (2014 - 1955)
    assert transformed_df.iloc[0]['renovation_flag'] == 1
    assert 'date' not in transformed_df.columns
    assert 'yr_built' not in transformed_df.columns
    assert 'yr_renovated' not in transformed_df.columns
