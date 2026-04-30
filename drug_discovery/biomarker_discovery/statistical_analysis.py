import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple

class BiomarkerStatisticalAnalysis:
    """
    Statistical methods for biomarker identification.
    """
    
    def __init__(self, data: pd.DataFrame, group_col: str):
        """
        Args:
            data: DataFrame containing feature values and group labels.
            group_col: Name of the column containing group labels (e.g., 'case' vs 'control').
        """
        self.data = data
        self.group_col = group_col
        self.groups = data[group_col].unique()
        if len(self.groups) < 2:
            raise ValueError("Data must contain at least two groups for comparison.")

    def differential_expression(self, features: List[str]) -> pd.DataFrame:
        """
        Perform t-tests for each feature between two groups.
        
        Returns:
            DataFrame with t-statistic, p-value, and fold change.
        """
        results = []
        group1 = self.data[self.data[self.group_col] == self.groups[0]]
        group2 = self.data[self.data[self.group_col] == self.groups[1]]
        
        for feature in features:
            if feature == self.group_col:
                continue
            
            v1 = group1[feature].dropna()
            v2 = group2[feature].dropna()
            
            if len(v1) < 2 or len(v2) < 2:
                continue
                
            t_stat, p_val = stats.ttest_ind(v1, v2)
            fold_change = v1.mean() / (v2.mean() + 1e-9)
            
            results.append({
                "feature": feature,
                "t_statistic": t_stat,
                "p_value": p_val,
                "log2_fold_change": np.log2(fold_change + 1e-9)
            })
            
        return pd.DataFrame(results).sort_values("p_value")

    def correlation_analysis(self, target_feature: str, features: List[str]) -> pd.DataFrame:
        """
        Calculate correlation between features and a target feature.
        """
        correlations = []
        for feature in features:
            if feature == target_feature:
                continue
            
            corr, p_val = stats.pearsonr(self.data[feature], self.data[target_feature])
            correlations.append({
                "feature": feature,
                "correlation": corr,
                "p_value": p_val
            })
            
        return pd.DataFrame(correlations).sort_values("p_value")
