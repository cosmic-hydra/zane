from typing import List, Dict, Any, Optional, Callable
import numpy as np
import pandas as pd
from .optimizer import BayesianOptimizer
from drug_discovery.data import MolecularFeaturizer

class ActiveLearningOrchestrator:
    """
    Orchestrates the Active Learning loop:
    1. Featurize a large pool of unlabeled molecules.
    2. Use Bayesian Optimization to suggest which ones to label (experiment/simulate).
    3. Retrain the model on new labels.
    """
    
    def __init__(
        self,
        model_trainer: Any,
        featurizer: MolecularFeaturizer,
        optimizer: Optional[BayesianOptimizer] = None
    ):
        self.trainer = model_trainer
        self.featurizer = featurizer
        self.optimizer = optimizer or BayesianOptimizer()
        self.labeled_data = pd.DataFrame()
        
    def run_cycle(self, unlabeled_smiles: List[str], oracle: Callable[[List[str]], List[float]]) -> Dict[str, Any]:
        """
        Run one AL cycle.
        """
        # 1. Featurize unlabeled pool
        X_pool = []
        valid_smiles = []
        for smiles in unlabeled_smiles:
            fp = self.featurizer.smiles_to_fingerprint(smiles)
            if fp is not None:
                X_pool.append(fp)
                valid_smiles.append(smiles)
        
        X_pool = np.array(X_pool)
        
        # 2. Suggest candidates
        indices = self.optimizer.suggest(X_pool)
        suggested_smiles = [valid_smiles[i] for i in indices]
        
        # 3. Query oracle (simulation or real experiment)
        labels = oracle(suggested_smiles)
        
        # 4. Update dataset
        new_data = pd.DataFrame({'smiles': suggested_smiles, 'target': labels})
        self.labeled_data = pd.concat([self.labeled_data, new_data]).drop_duplicates('smiles')
        
        # 5. Tell optimizer
        X_new = X_pool[indices]
        self.optimizer.tell(X_new, np.array(labels))
        
        # 6. Retrain model (optional)
        # self.trainer.train_on_dataframe(self.labeled_data)
        
        return {
            "num_new_labels": len(labels),
            "suggested_smiles": suggested_smiles,
            "total_labeled": len(self.labeled_data)
        }
