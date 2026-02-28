import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath('..'))
from src.data_ingestion import load_data
from src.data_validation import validate_data

# Set plot styling
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

def visualize_clusters():
    print("Loading data...")
    df_raw = load_data(sample_frac=0.05)
    df_clean = validate_data(df_raw)
    
    print("Loading trained pipeline...")
    with open('models/segmentation_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
        
    preprocessor = pipeline.named_steps['preprocessor']
    kmeans_model = pipeline.named_steps['kmeans']
    
    print("Transforming and predicting...")
    X_transformed = preprocessor.transform(df_clean)
    cluster_labels = kmeans_model.predict(X_transformed)
    
    print("Applying PCA...")
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(X_transformed)
    
    pca_df = pd.DataFrame(data=pca_features, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = cluster_labels
    
    cluster_names = {
        0: 'Regular Residential Buyers',
        1: 'Luxury High-Value Investors',
        2: 'Commercial/Corporate Renters',
        3: 'Off-Plan Flippers',
        4: 'Bulk/Portfolio Acquirers'
    }
    
    pca_df['Segment'] = pca_df['Cluster'].map(cluster_names).fillna(pca_df['Cluster'].astype(str))
    
    print("Plotting...")
    sns.scatterplot(x='PC1', y='PC2', hue='Segment', palette='tab10', data=pca_df, alpha=0.6, s=50)
    plt.title('2D PCA Projection of Transaction Clusters')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} Variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} Variance)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = 'cluster_visualization.png'
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

if __name__ == '__main__':
    visualize_clusters()
