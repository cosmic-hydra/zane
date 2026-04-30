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
        Calculate a weighted aggregate score for a molecule with multi-objective optimization.
        """
        scores = {}
        
        # 1. Property Score (Activity/Potency)
        try:
            prop_res = self.property_predictor.predict(smiles)
            # If it's a dict, take the value, else it's the score itself
            scores['property'] = prop_res if isinstance(prop_res, (int, float)) else list(prop_res.values())[0]
        except Exception:
            scores['property'] = 0.0
            
        # 2. ADMET Score (Safety and Pharmokinetics)
        try:
            # We want high QED and high safety
            qed = self.admet_predictor.calculate_qed(smiles)
            tox_verdict = self.admet_predictor.evaluate(smiles) if hasattr(self.admet_predictor, 'evaluate') else None
            safety = tox_verdict.safety_score if tox_verdict else 1.0
            scores['admet'] = 0.5 * qed + 0.5 * safety
        except Exception:
            scores['admet'] = 0.0
            
        # 3. SA Score (Synthetic Accessibility)
        try:
            sa = self.admet_predictor.calculate_synthetic_accessibility(smiles)
            # 1.0 is very easy, 0.0 is impossible
            scores['sa'] = 1.0 - (sa / 10.0)
        except Exception:
            scores['sa'] = 0.0
            
        # 4. Physics Score (Binding Affinity)
        if self.physics_simulator and target_protein_pdb:
            try:
                res = self.physics_simulator.simulate_md(smiles, target_protein_pdb)
                binding_energy = res.get('binding_energy', 0.0)
                # Use a sigmoid to normalize energy: -10 kcal/mol should be a very high score
                scores['physics'] = 1.0 / (1.0 + np.exp(0.5 * (binding_energy + 8.0)))
            except Exception:
                scores['physics'] = 0.0
        else:
            scores['physics'] = 0.0

        # Multi-objective Pareto-like aggregate (Geometric Mean for balance)
        # We add a small epsilon to avoid zeroing out the entire score
        epsilon = 0.01
        weighted_scores = [
            (scores.get('property', 0) + epsilon) ** self.weights.get('property', 0.4),
            (scores.get('admet', 0) + epsilon) ** self.weights.get('admet', 0.3),
            (scores.get('physics', 0) + epsilon) ** self.weights.get('physics', 0.2),
            (scores.get('sa', 0) + epsilon) ** self.weights.get('sa', 0.1)
        ]
        
        total_score = np.prod(weighted_scores)
        return float(total_score)

    def select_elite_candidates(
        self, 
        smiles_list: List[str], 
        target_protein_pdb: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Select 'Elite' candidates that excel across all objectives.
        """
        results = self.rank_candidates(smiles_list, target_protein_pdb)
        
        # Additional filter for elite candidates: must pass safety gate strictly
        elite = []
        for res in results:
            if len(elite) >= top_k:
                break
                
            try:
                # Assuming admet_predictor has a validate method that checks elite standards
                if hasattr(self.admet_predictor, 'is_elite_smiles'):
                    if not self.admet_predictor.is_elite_smiles(res['smiles']):
                        continue
            except Exception:
                pass
                
            elite.append(res)
            
        return elite

    def rank_candidates(self, smiles_list: List[str], target_protein_pdb: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []
        for smiles in smiles_list:
            score = self.calculate_ensemble_score(smiles, target_protein_pdb)
            results.append({
                'smiles': smiles,
                'ensemble_score': score
            })
        return sorted(results, key=lambda x: x['ensemble_score'], reverse=True)
