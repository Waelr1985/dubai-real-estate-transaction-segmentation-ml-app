import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.decomposition import PCA
import numpy as np
import json
import os
import logging

from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_ENCODE_FEATURES, MODELS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log1p transformation to handle highly skewed prices/area
def log_transform(x):
    return np.log1p(np.maximum(x, 0)) # Ensure non-negative


def apply_target_encoding(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Target-encodes high-cardinality categorical features.
    
    During TRAINING: computes median actual_worth per category from the data.
    During INFERENCE: loads saved mappings from target_encoding_mappings.json.
    '''
    df = df.copy()
    
    # Try to load saved mappings first (inference mode)
    te_path = os.path.join(MODELS_DIR, 'target_encoding_mappings.json')
    saved_mappings = None
    if os.path.exists(te_path):
        try:
            with open(te_path, 'r') as f:
                saved_mappings = json.load(f)
        except Exception:
            saved_mappings = None
    
    for col in TARGET_ENCODE_FEATURES:
        if col not in df.columns:
            continue
            
        encoded_col = f'{col}_encoded'
        
        if saved_mappings and col in saved_mappings:
            # INFERENCE MODE: use saved mappings from training
            mapping = saved_mappings[col]
            global_median = np.median(list(mapping.values()))
            df[encoded_col] = df[col].map(mapping).fillna(global_median)
            logging.info(f"Target-encoded '{col}' using saved mappings ({len(mapping)} categories)")
        elif 'actual_worth' in df.columns:
            # TRAINING MODE: compute mappings from the data
            medians = df.groupby(col)['actual_worth'].median()
            df[encoded_col] = df[col].map(medians).fillna(medians.median())
            logging.info(f"Target-encoded '{col}' from data ({len(medians)} categories -> 1 numeric column)")
        else:
            # Fallback: set to 0 if neither mappings nor target column available
            df[encoded_col] = 0.0
            logging.warning(f"No mapping available for '{col}'. Set '{encoded_col}' to 0.")
    
    return df


def get_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    '''
    Returns a compiled Scikit-Learn ColumnTransformer pipeline.
    
    Pipeline (Strategy D — Optimized):
      1. Numeric features: log1p → RobustScaler
      2. Target-encoded features: log1p → RobustScaler (same as numeric)
      3. Categorical features: OneHotEncoder
    
    Note: PCA is applied AFTER this preprocessor in the training pipeline.
    '''
    
    # Ensure variables exist in DF
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    
    # Add target-encoded columns to numeric processing
    te_cols = [f'{c}_encoded' for c in TARGET_ENCODE_FEATURES if f'{c}_encoded' in df.columns]
    all_num_cols = num_cols + te_cols
    
    numeric_transformer = Pipeline(steps=[
        ('log', FunctionTransformer(log_transform, validate=False)),
        ('scaler', RobustScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, all_num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )
    
    return preprocessor
