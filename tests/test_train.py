import pytest
import pandas as pd
from src.data import clean_data
from src.model_selection import select_best_model

def test_cleaning_removes_high_cardinality():
    """ Verify 'street' column is dropped during cleaning. """
    df_raw = pd.DataFrame({
        'price': [100.0],
        'street': ['123 Main St'],
        'country': ['USA'],
        'city': ['Seattle']
    })
    df_cleaned = clean_data(df_raw)
    assert 'street' not in df_cleaned.columns
    assert 'country' not in df_cleaned.columns

def test_model_selection_logic():
    """ Verify lowest RMSE logic. """
    results = [
        {'run_name': 'A', 'metrics': {'rmse': 500}},
        {'run_name': 'B', 'metrics': {'rmse': 300}}
    ]
    best = select_best_model(results)
    assert best['run_name'] == 'B'
