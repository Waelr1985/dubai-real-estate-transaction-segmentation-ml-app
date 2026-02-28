import pandas as pd
import logging
from src.config import RAW_DATA_PATH, FEATURES_TO_KEEP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path: str = RAW_DATA_PATH, sample_frac: float = 1.0, nrows: int = None) -> pd.DataFrame:
    '''
    Loads the dataset. Because the dataset is large (~1GB), allows sampling or reading limited rows.
    '''
    try:
        logging.info(f"Loading data from {file_path}...")
        
        # Read only required columns to save memory if needed
        # But some columns might be missing so we read all and filter
        df = pd.read_csv(file_path, nrows=nrows)
        
        # Filter columns
        existing_cols = [c for c in FEATURES_TO_KEEP if c in df.columns]
        df = df[existing_cols]
        
        if sample_frac < 1.0:
            logging.info(f"Sampling {sample_frac*100}% of the data...")
            df = df.sample(frac=sample_frac, random_state=42)
            
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise
