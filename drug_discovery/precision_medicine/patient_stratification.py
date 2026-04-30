import pandas as pd
from sklearn.cluster import KMeans
from typing import List, Dict, Any

class PatientStratifier:
    """
    Cluster patients based on multi-omics or clinical data.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def stratify_patients(self, features: List[str], num_clusters: int = 3) -> pd.Series:
        """
        Group patients into clusters.
        """
        X = self.data[features]
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        return pd.Series(clusters, index=self.data.index)

    def get_cluster_characteristics(self, clusters: pd.Series) -> pd.DataFrame:
        """
        Analyze the mean values of features for each cluster.
        """
        df_with_clusters = self.data.copy()
        df_with_clusters['cluster'] = clusters
        return df_with_clusters.groupby('cluster').mean()
