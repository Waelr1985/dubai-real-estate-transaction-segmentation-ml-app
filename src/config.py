import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# The raw data is at E:\DLD_gemini_v2\Transactions.csv
# But we can assume it can be copied to data/ or referenced absolutely
RAW_DATA_PATH = os.path.join("E:", os.sep, "DLD_gemini_v2", "Transactions.csv")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# List of columns to keep (Dropping Arabic and redundant ones)
FEATURES_TO_KEEP = [
    'transaction_id', 'instance_date',
    'trans_group_en', 'procedure_name_en', 'property_type_en',
    'property_sub_type_en', 'property_usage_en', 'reg_type_en',
    'area_name_en', 'nearest_landmark_en', 'nearest_metro_en',
    'nearest_mall_en', 'rooms_en', 'has_parking', 'procedure_area',
    'actual_worth', 'meter_sale_price', 'rent_value', 'meter_rent_price',
    'no_of_parties_role_1', 'no_of_parties_role_2', 'no_of_parties_role_3'
]

# Numeric and categorical variable names
# Note: rent_value and meter_rent_price were REMOVED because 97.8% of rows
# contain the same placeholder value (1,020,141 / 7,249.30). This is a data
# quality issue in the raw dataset — these fields are not real rent data.
NUMERIC_FEATURES = [
    'procedure_area', 'actual_worth', 'meter_sale_price', 
    'no_of_parties_role_1', 'no_of_parties_role_2', 'no_of_parties_role_3'
]

CATEGORICAL_FEATURES = [
    'trans_group_en', 'procedure_name_en', 'property_type_en',
    'property_sub_type_en', 'property_usage_en', 'reg_type_en',
    'rooms_en', 'has_parking'
]

# Features to target-encode (replaced with median actual_worth per category)
# This converts high-cardinality categoricals into a single numeric column
# instead of hundreds of sparse OHE columns that hurt K-Means performance.
TARGET_ENCODE_FEATURES = ['area_name_en']

