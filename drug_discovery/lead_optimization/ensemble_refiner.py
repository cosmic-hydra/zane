from typing import List, Dict, Any, Optional
import torch
import numpy as np

class EnsembleRefiner:
    """
    Refines drug leads by combining multiple scoring functions:
    - ML-based property prediction
    - ADMET scoring
    - Physics-based binding affinity (optional)
    - Synthetic Accessibility (SA)
    """
    
    def __init__(
        self,
        property_predictor: Any,
        admet_predictor: Any,
        physics_simulator: Optional[Any] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        self.property_predictor = property_predictor
        self.admet_predictor = admet_predictor
        self.physics_simulator = physics_simulator
        self.weights = weights or {
            'property': 0.4,
            'admet': 0.3,
            'physics': 0.2,
            'sa': 0.1
        }

    def calculate_ensemble_score(self, smiles: str, target_protein_pdb: Optional[str] = None) -> float:
        """
        Calculate a weighted aggregate score for a molecule.
        """
        scores = {}
        
        # 1. Property Score
        try:
            scores['property'] = self.property_predictor.predict(smiles)
        except Exception:
            scores['property'] = 0.0
            
        # 2. ADMET Score (QED is a good proxy)
        try:
            scores['admet'] = self.admet_predictor.calculate_qed(smiles)
        except Exception:
            scores['admet'] = 0.0
            
        # 3. SA Score (Lower is better, so we invert it)
        try:
            sa = self.admet_predictor.calculate_synthetic_accessibility(smiles)
            scores['sa'] = 1.0 - (sa / 10.0)
        except Exception:
            scores['sa'] = 0.0
            
        # 4. Physics Score
        if self.physics_simulator and target_protein_pdb:
            try:
                # Assuming simulate_md returns a dict with binding_energy
                res = self.physics_simulator.simulate_md(smiles, target_protein_pdb)
                # Lower energy is better binding, so we negate and normalize
                scores['physics'] = -res.get('binding_energy', 0.0) / 100.0
            except Exception:
                scores['physics'] = 0.0
        else:
            scores['physics'] = 0.0
            
        # Weighted sum
        total_score = sum(self.weights.get(k, 0) * scores.get(k, 0) for k in self.weights)
        return float(total_score)

    def rank_candidates(self, smiles_list: List[str], target_protein_pdb: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []
        for smiles in smiles_list:
            score = self.calculate_ensemble_score(smiles, target_protein_pdb)
            results.append({
                'smiles': smiles,
                'ensemble_score': score
            })
        return sorted(results, key=lambda x: x['ensemble_score'], reverse=True)
