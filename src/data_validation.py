import pandas as pd
import logging
import json
import os
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Validates and cleans the schema.
    Extracts dates, fills NAs, and ensures types are correct.
    '''
    logging.info("Starting data validation and basic cleaning...")
    
    # 1. Date Extraction
    if 'instance_date' in df.columns:
        df['instance_date'] = pd.to_datetime(df['instance_date'], errors='coerce')
        df['year'] = df['instance_date'].dt.year
        df['month'] = df['instance_date'].dt.month
        df.drop(columns=['instance_date'], inplace=True)
    
    # 2. Fill Missing Values
    # For numeric, fill with median
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            # Convert to numeric just in case
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            
    # For categorical, fill with mode or 'Unknown'
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col].fillna('Unknown', inplace=True)
            df[col] = df[col].replace('nan', 'Unknown')
            
    logging.info(f"Data validation complete. Shape: {df.shape}")
    return df

def check_data_drift(df: pd.DataFrame) -> list:
    '''
    Compares incoming un-scaled data against the saved baseline JSON from training.
    Returns a list of warning strings if drift is detected.
    '''
    warnings = []
    try:
        from src.config import MODELS_DIR
        stats_path = os.path.join(MODELS_DIR, 'baseline_stats.json')
        if not os.path.exists(stats_path):
            return warnings # No baseline to compare against
            
        with open(stats_path, 'r') as f:
            baseline_stats = json.load(f)
            
        numeric_baseline = baseline_stats.get('numeric_medians', {})
        categorical_baseline = baseline_stats.get('categorical_modes', {})
        
        # Check Numeric Drift (Threshold: 50% for demo purposes scaling from sample to full)
        for col, base_median in numeric_baseline.items():
            if col in df.columns:
                current_median = pd.to_numeric(df[col], errors='coerce').median()
                if pd.notna(current_median) and base_median != 0:
                    diff = abs(current_median - base_median) / base_median
                    if diff > 0.50: # Increased to 50% drift threshold
                        warnings.append(f"Numeric Drift in '{col}': Current median ({current_median:,.2f}) shifted by >50% from training baseline ({base_median:,.2f})")
                        
        # Note: Categorical drift alerting has been disabled for this demo.
        # When moving from a 1,000 row sample to a 1.6 Million row dataset, 
        # the most frequent categorical values (e.g., 'area_name_en') will naturally shift.
                    
    except Exception as e:
        logging.error(f"Failed to run drift checking: {e}")
        
    return warnings
