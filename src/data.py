import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw data from a CSV file.
    """
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw dataset.
    - Removes zero prices (invalid target)
    - Drops high-cardinality/useless columns (street, country)
    - Handles duplicates
    """
    # 1. Target Data Quality: House prices must be positive to be useful for regression.
    initial_count = len(df)
    df = df[df['price'] > 0]
    dropped_zero = initial_count - len(df)
    
    # 2. Duplicate Removal: Ensures the validation set is independent.
    df = df.drop_duplicates()
    
    # 3. High-Cardinality: 'street' has 4000+ unique values (almost one per row).
    # Including it would lead to extreme overfitting or memory issues with OneHot.
    # 'country' is 'USA' for all rows (no variance).
    cols_to_drop = ['street', 'country']
    df = df.drop(columns=cols_to_drop)
    
    logger.info(f"Cleaned data: dropped {dropped_zero} zero-price rows. Remaining: {len(df)}")
    return df
