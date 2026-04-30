"""
Drug Combination Testing - Synergy and Antagonism Prediction

Predicts drug-drug interactions including:
- Synergistic effects (combined effect > sum of individual effects)
- Antagonistic effects (combined effect < sum of individual effects)
- Additive effects (combined effect ≈ sum of individual effects)

Uses Bliss independence model, Loewe additivity, and ML models.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from drug_discovery.external_tooling import canonicalize_smiles, gt4sd_properties
from drug_discovery.utils.rdkit_fallback import heuristic_props, is_smiles_plausible

try:  # pragma: no cover - optional dependency
    from rdkit import Chem, DataStructs  # type: ignore
    from rdkit.Chem import AllChem, Descriptors  # type: ignore
except Exception:  # pragma: no cover - default path when RDKit unavailable
    Chem = None  # type: ignore
    DataStructs = None  # type: ignore
    AllChem = None  # type: ignore
    Descriptors = None  # type: ignore

logger = logging.getLogger(__name__)


class DrugCombinationTester:
    """Test drug combinations for synergy/antagonism."""

    def __init__(self, use_ml_models: bool = True):
        """
        Initialize drug combination tester.

        Args:
            use_ml_models: Whether to use ML models for prediction
        """
        self.use_ml_models = use_ml_models
        self.synergy_model = None
        if use_ml_models:
            self._init_models()

    def _init_models(self) -> None:
        """Initialize ML models for synergy prediction."""
        logger.info("Initializing drug combination synergy models")
        # Placeholder - would be trained on drug combination databases
        self.synergy_model = {
            "rf": RandomForestRegressor(n_estimators=100, random_state=42),
            "gb": GradientBoostingRegressor(n_estimators=100, random_state=42),
        }

    def _compute_combination_features(
        self,
        smiles1: str,
        smiles2: str,
    ) -> np.ndarray | None:
        """
        Compute features for drug combination.

        Args:
            smiles1: First drug SMILES
            smiles2: Second drug SMILES

        Returns:
            Feature vector or None if computation fails
        """
        try:
            smiles1 = canonicalize_smiles(smiles1) or smiles1
            smiles2 = canonicalize_smiles(smiles2) or smiles2

            if Chem is None:
                if not (is_smiles_plausible(smiles1) and is_smiles_plausible(smiles2)):
                    return None
                props1 = heuristic_props(smiles1)
                props2 = heuristic_props(smiles2)
                desc1 = np.array(
                    [props1.molecular_weight, props1.logp, props1.h_donors, props1.h_acceptors, props1.tpsa]
                )
                desc2 = np.array(
                    [props2.molecular_weight, props2.logp, props2.h_donors, props2.h_acceptors, props2.tpsa]
                )
                set1, set2 = set(smiles1), set(smiles2)
                denom = len(set1 | set2) or 1
                similarity = len(set1 & set2) / denom
            else:
                mol1 = Chem.MolFromSmiles(smiles1)
                mol2 = Chem.MolFromSmiles(smiles2)

                if mol1 is None or mol2 is None:
                    return None

                desc1 = np.array(
                    [
                        Descriptors.MolWt(mol1),
                        Descriptors.MolLogP(mol1),
                        Descriptors.NumHDonors(mol1),
                        Descriptors.NumHAcceptors(mol1),
                        Descriptors.TPSA(mol1),
                    ]
                )

                desc2 = np.array(
                    [
                        Descriptors.MolWt(mol2),
                        Descriptors.MolLogP(mol2),
                        Descriptors.NumHDonors(mol2),
                        Descriptors.NumHAcceptors(mol2),
                        Descriptors.TPSA(mol2),
                    ]
                )

                fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
                similarity = float(DataStructs.TanimotoSimilarity(fp1, fp2))

            gt4sd_1 = gt4sd_properties(smiles1)
            gt4sd_2 = gt4sd_properties(smiles2)
            gt4sd_features = np.array(
                [
                    gt4sd_1.get("qed", 0.0),
                    gt4sd_2.get("qed", 0.0),
                    gt4sd_2.get("qed", 0.0) - gt4sd_1.get("qed", 0.0),
                    gt4sd_1.get("logp", 0.0),
                    gt4sd_2.get("logp", 0.0),
                    abs(gt4sd_2.get("logp", 0.0) - gt4sd_1.get("logp", 0.0)),
                ]
            )

            features = np.concatenate(
                [
                    desc1,
                    desc2,
                    desc2 - desc1,
                    [similarity],
                    gt4sd_features,
                ]
            )

            return features

        except Exception as e:
            logger.warning(f"Feature computation failed: {e}")
            return None

    def predict_synergy_bliss(
        self,
        smiles1: str,
        smiles2: str,
        effect1: float,
        effect2: float,
    ) -> dict[str, float]:
        """
        Predict synergy using Bliss independence model.

        Bliss independence: E_combo = E1 + E2 - E1*E2

        Args:
            smiles1: First drug SMILES
            smiles2: Second drug SMILES
            effect1: Effect of drug 1 alone (0-1)
            effect2: Effect of drug 2 alone (0-1)

        Returns:
            Dictionary with expected effect and synergy score
        """
        # Bliss independence expected effect
        expected_effect = effect1 + effect2 - (effect1 * effect2)

        # Placeholder for observed effect - would come from experiments
        # Using simple heuristics based on molecular similarity
        features = self._compute_combination_features(smiles1, smiles2)

        if features is None:
            return {
                "expected_effect": expected_effect,
                "synergy_score": 0.0,
                "interaction_type": "unknown",
            }

        similarity = features[-1]

        # Simple heuristic: moderate similarity often leads to synergy
        observed_effect = expected_effect
        if 0.3 < similarity < 0.7:
            observed_effect += 0.1  # Potential synergy
        elif similarity > 0.9:
            observed_effect -= 0.05  # Potential antagonism (similar mechanism)

        synergy_score = observed_effect - expected_effect

        interaction_type = (
            "synergistic" if synergy_score > 0.05 else "antagonistic" if synergy_score < -0.05 else "additive"
        )

        return {
            "expected_effect": expected_effect,
            "observed_effect": observed_effect,
            "synergy_score": synergy_score,
            "interaction_type": interaction_type,
            "similarity": similarity,
        }

    def predict_synergy_loewe(
        self,
        smiles1: str,
        smiles2: str,
        dose1: float,
        dose2: float,
        ic50_1: float,
        ic50_2: float,
    ) -> dict[str, float]:
        """
        Predict synergy using Loewe additivity model.

        Loewe: dose1/IC50_1 + dose2/IC50_2 = 1 for additive effect

        Args:
            smiles1: First drug SMILES
            smiles2: Second drug SMILES
            dose1: Dose of drug 1
            dose2: Dose of drug 2
            ic50_1: IC50 of drug 1
            ic50_2: IC50 of drug 2

        Returns:
            Dictionary with combination index and interaction type
        """
        # Combination index (CI)
        ci = (dose1 / ic50_1) + (dose2 / ic50_2)

        # CI < 1: synergistic
        # CI = 1: additive
        # CI > 1: antagonistic

        interaction_type = "synergistic" if ci < 0.9 else "antagonistic" if ci > 1.1 else "additive"

        return {
            "combination_index": ci,
            "interaction_type": interaction_type,
            "synergy_strength": max(0, 1 - ci) if ci < 1 else 0,
            "antagonism_strength": max(0, ci - 1) if ci > 1 else 0,
        }

    def predict_synergy_ml(
        self,
        smiles1: str,
        smiles2: str,
    ) -> dict[str, float]:
        """
        Predict synergy using ML models.

        Args:
            smiles1: First drug SMILES
            smiles2: Second drug SMILES

        Returns:
            Dictionary with synergy prediction
        """
        features = self._compute_combination_features(smiles1, smiles2)

        if features is None:
            return {
                "synergy_score": 0.0,
                "confidence": 0.0,
                "interaction_type": "unknown",
            }

        # Placeholder - would use trained model
        # Using simple heuristics for demonstration
        similarity = features[-1]
        logp_diff = abs(features[6])  # Difference in logP
        gt4sd_qed_delta = abs(features[-4])

        synergy_score = 0.0
        if 0.3 < similarity < 0.7:
            synergy_score += 0.3
        if logp_diff > 2:
            synergy_score += 0.2  # Different physicochemical properties
        if gt4sd_qed_delta < 0.25:
            synergy_score += 0.1

        synergy_score = min(1.0, synergy_score)

        interaction_type = (
            "synergistic" if synergy_score > 0.5 else "antagonistic" if synergy_score < -0.3 else "additive"
        )

        return {
            "synergy_score": synergy_score,
            "confidence": 0.7,
            "interaction_type": interaction_type,
            "similarity": similarity,
        }

    def test_combination(
        self,
        smiles1: str,
        smiles2: str,
        method: str = "bliss",
        **kwargs,
    ) -> dict[str, float]:
        """
        Test drug combination using specified method.

        Args:
            smiles1: First drug SMILES
            smiles2: Second drug SMILES
            method: Prediction method ('bliss', 'loewe', 'ml')
            **kwargs: Additional parameters for specific methods

        Returns:
            Dictionary with synergy prediction
        """
        if method == "bliss":
            effect1 = kwargs.get("effect1", 0.5)
            effect2 = kwargs.get("effect2", 0.5)
            return self.predict_synergy_bliss(smiles1, smiles2, effect1, effect2)

        elif method == "loewe":
            dose1 = kwargs.get("dose1", 1.0)
            dose2 = kwargs.get("dose2", 1.0)
            ic50_1 = kwargs.get("ic50_1", 1.0)
            ic50_2 = kwargs.get("ic50_2", 1.0)
            return self.predict_synergy_loewe(smiles1, smiles2, dose1, dose2, ic50_1, ic50_2)

        elif method == "ml":
            return self.predict_synergy_ml(smiles1, smiles2)

        else:
            logger.error(f"Unknown method: {method}")
            return {}

    def batch_test_combinations(
        self,
        combinations: list[tuple[str, str]],
        method: str = "ml",
    ) -> pd.DataFrame:
        """
        Test multiple drug combinations.

        Args:
            combinations: List of (smiles1, smiles2) tuples
            method: Prediction method

        Returns:
            DataFrame with results
        """
        results = []

        for smiles1, smiles2 in combinations:
            prediction = self.test_combination(smiles1, smiles2, method=method)

            row = {
                "smiles1": smiles1,
                "smiles2": smiles2,
                "interaction_type": prediction.get("interaction_type", "unknown"),
                "synergy_score": prediction.get("synergy_score", 0.0),
            }

            if "similarity" in prediction:
                row["similarity"] = prediction["similarity"]
            if "confidence" in prediction:
                row["confidence"] = prediction["confidence"]

            results.append(row)

        df = pd.DataFrame(results)
        logger.info(f"Tested {len(combinations)} drug combinations")

        return df

    def find_toxicity_balancers(
        self,
        toxic_smiles: str,
        potential_balancers: list[str],
    ) -> pd.DataFrame:
        """
        Find molecules that balance the toxicity of a target compound.
        
        A balancer is a molecule that has an antagonistic effect on the toxicity
        of another molecule, effectively 'balancing' it.
        """
        results = []
        for balancer_smiles in potential_balancers:
            # We use ML synergy model but look for antagonistic interactions on toxicity
            # In this context, antagonism is good because it reduces the toxic effect.
            prediction = self.predict_synergy_ml(toxic_smiles, balancer_smiles)
            
            # If the interaction is antagonistic, it might be a good balancer
            if prediction["interaction_type"] == "antagonistic":
                results.append({
                    "toxic_smiles": toxic_smiles,
                    "balancer_smiles": balancer_smiles,
                    "balancing_efficiency": abs(prediction["synergy_score"]),
                    "confidence": prediction["confidence"]
                })
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("balancing_efficiency", ascending=False)
        return df

    def find_synergistic_pairs(
        self,
        smiles_list: list[str],
        threshold: float = 0.5,
        max_pairs: int | None = None,
    ) -> pd.DataFrame:
        """
        Find potentially synergistic drug pairs from a list.

        Args:
            smiles_list: List of drug SMILES
            threshold: Minimum synergy score threshold
            max_pairs: Maximum number of pairs to return

        Returns:
            DataFrame with top synergistic pairs
        """
        combinations = []
        for i, smiles1 in enumerate(smiles_list):
            for smiles2 in smiles_list[i + 1 :]:
                combinations.append((smiles1, smiles2))

        # Test all combinations
        results = self.batch_test_combinations(combinations, method="ml")

        # Filter synergistic pairs
        synergistic = results[(results["interaction_type"] == "synergistic") & (results["synergy_score"] >= threshold)]

        # Sort by synergy score
        synergistic = synergistic.sort_values("synergy_score", ascending=False)

        if max_pairs is not None:
            synergistic = synergistic.head(max_pairs)

        logger.info(f"Found {len(synergistic)} synergistic pairs above threshold {threshold}")

        return synergistic
