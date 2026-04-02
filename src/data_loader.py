import pandas as pd
import logging

# Set up logging for educational purposes
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the house price dataset from a CSV file.
    """
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by:
    - Dropping useless columns
    - Handling zero prices (target data cleaning)
    """
    logger.info("Cleaning data...")
    
    # 1. Drop high-cardinality/useless columns
    # 'street' has 4600+ unique values, which is too many for OneHotEncoding.
    # 'country' is always 'USA', so it has no predictive power (zero variance).
    cols_to_drop = ['street', 'country']
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped columns: {cols_to_drop}")

    # 2. Filter out houses with price 0 (if any)
    # Price is our target; zero prices represent missing or invalid data.
    df = df[df['price'] > 0]
    
    return df
