import pandas as pd
import logging
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_clusters(df: pd.DataFrame, preprocessor, kmeans_model):
    '''
    Evaluates the trained KMeans model using Silhouette and Davies-Bouldin metrics.
    Prints out the profile of each cluster.
    '''
    logging.info("--- Evaluating Clusters ---")
    
    # 1. Transform data
    X_processed = preprocessor.transform(df)
    labels = kmeans_model.predict(X_processed)
    
    # Stratified sampling for silhouette score to maintain cluster proportions and avoid MemoryError
    if len(df) > 50000:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=50000, random_state=42)
        
        for train_index, _ in sss.split(X_processed, labels):
            X_sampled = X_processed[train_index]
            labels_sampled = labels[train_index]
            
        silhouette = silhouette_score(X_sampled, labels_sampled)
    else:
        silhouette = silhouette_score(X_processed, labels)
        
    db_index = davies_bouldin_score(X_processed, labels)
    ch_score = calinski_harabasz_score(X_processed, labels)
    
    logging.info(f"Silhouette Score (Cohesion/Separation): {silhouette:.3f}")
    logging.info(f"Davies-Bouldin Index: {db_index:.3f}")
    logging.info(f"Calinski-Harabasz Variance Ratio: {ch_score:.3f}")
    
    # 2. Cluster Profiling
    df['Cluster'] = labels
    logging.info("\n--- Cluster Profiles ---")
    
    # Get average values for numerical features per cluster
    numeric_profile = df.groupby('Cluster').mean(numeric_only=True)
    logging.info(f"\nNumeric Averages per Cluster:\n{numeric_profile}")
    
    # Get mode (most frequent) for categorical features
    # (Just an example using 'property_type_en' and 'area_name_en')
    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster_id]
        most_freq_prop = cluster_data['property_type_en'].mode().values[0] if 'property_type_en' in df.columns else 'N/A'
        most_freq_area = cluster_data['area_name_en'].mode().values[0] if 'area_name_en' in df.columns else 'N/A'
        
        logging.info(f"Cluster {cluster_id}: Most frequent Property Type = {most_freq_prop}, Area = {most_freq_area}")
    
    return silhouette, db_index
