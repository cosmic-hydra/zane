import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve
from typing import List, Dict, Any, Tuple

class BiomarkerMLDiscovery:
    """
    Machine Learning methods for biomarker identification and ranking.
    """
    
    def __init__(self, data: pd.DataFrame, target_col: str):
        self.data = data
        self.target_col = target_col
        
    def rank_features_by_importance(self, features: List[str]) -> pd.DataFrame:
        """
        Use Random Forest to rank features by their importance in predicting the target.
        """
        X = self.data[features]
        y = self.data[self.target_col]
        
        # Simple binary classification or regression check
        if y.dtype == object or len(np.unique(y)) < 10:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            
        model.fit(X, y)
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        ranked_features = []
        for f in range(X.shape[1]):
            ranked_features.append({
                "feature": features[indices[f]],
                "importance": importances[indices[f]]
            })
            
        return pd.DataFrame(ranked_features)

    def evaluate_biomarker_panel(self, panel_features: List[str]) -> Dict[str, float]:
        """
        Evaluate the predictive power of a set of biomarkers.
        """
        X = self.data[panel_features]
        y = self.data[self.target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        
        return {
            "auc_roc": auc,
            "num_features": len(panel_features)
        }
