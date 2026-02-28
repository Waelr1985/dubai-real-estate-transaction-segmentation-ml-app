import json
import logging
import os
import pickle
import pandas as pd
from src.data_validation import validate_data

def init():
    """
    This function is called when the container is initialized/started.
    Here we load the trained scikit-learn pipeline (preprocessor + kmeans).
    """
    global model
    logging.info("Initializing the segmentation model")
    
    # AZUREML_MODEL_DIR is an environment variable created during deployment
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "models"), "segmentation_pipeline.pkl")
    
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        raise

def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/predicting.
    """
    logging.info("Received request for scoring.")
    try:
        # 1. Parse JSON data into pandas DataFrame
        data_dict = json.loads(raw_data)
        df_input = pd.DataFrame.from_dict(data_dict)
        
        # 2. Validate and clean schema as done during training
        df_clean = validate_data(df_input)
        
        # 3. Predict clusters
        predictions = model.predict(df_clean)
        
        # 4. Return as JSON
        result = {"clusters": predictions.tolist()}
        return json.dumps(result)
    
    except Exception as e:
        logging.error(f"Error during scoring: {e}")
        return json.dumps({"error": str(e)})
