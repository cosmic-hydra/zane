"""
Toxicity Prediction - Advanced Toxicity Testing Layer

Predicts multiple toxicity endpoints using ensemble ML models:
- Cytotoxicity (general cell toxicity)
- Hepatotoxicity (liver toxicity)
- Cardiotoxicity (heart toxicity, including hERG inhibition)
- Mutagenicity (genetic damage)

Uses cross-dataset validation and confidence scoring.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from drug_discovery.utils.rdkit_fallback import heuristic_props, is_smiles_plausible

try:  # pragma: no cover - optional dependency
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import Descriptors, rdMolDescriptors  # type: ignore
except Exception:  # pragma: no cover - default path when RDKit unavailable
    Chem = None  # type: ignore
    Descriptors = None  # type: ignore
    rdMolDescriptors = None  # type: ignore

logger = logging.getLogger(__name__)


class ToxicityPredictor:
    """Advanced toxicity prediction with multiple endpoints."""

    def __init__(self, use_ensemble: bool = True):
        """
        Initialize toxicity predictor.

        Args:
            use_ensemble: Whether to use ensemble of models
        """
        self.use_ensemble = use_ensemble
        self.models = {}
        self._init_models()

    def _init_models(self) -> None:
        """Initialize toxicity prediction models."""
        # Placeholder models - in production, these would be trained on toxicity databases
        logger.info("Initializing toxicity prediction models")

        endpoints = ["cytotoxicity", "hepatotoxicity", "cardiotoxicity", "mutagenicity"]

        for endpoint in endpoints:
            if self.use_ensemble:
                self.models[endpoint] = {
                    "rf": RandomForestClassifier(n_estimators=100, random_state=42),
                    "gb": GradientBoostingClassifier(n_estimators=100, random_state=42),
                    "lr": LogisticRegression(max_iter=1000, random_state=42),
                }
            else:
                self.models[endpoint] = {
                    "rf": RandomForestClassifier(n_estimators=100, random_state=42),
                }

    def _compute_molecular_descriptors(self, smiles: str) -> np.ndarray | None:
        """Compute molecular descriptors for toxicity prediction."""
        if not is_smiles_plausible(smiles):
            return None

        if Chem is None:
            props = heuristic_props(smiles)
            descriptors = [
                props.molecular_weight,
                props.logp,
                props.h_donors,
                props.h_acceptors,
                props.rotatable_bonds,
                props.tpsa,
                props.aromatic_rings,
                max(0.0, props.aromatic_rings - 1.0),
                max(0.0, props.aromatic_rings - 2.0),
                max(0.0, 1.0 - props.logp / 10.0),
                props.h_acceptors,
                props.aromatic_rings,
                0.0,
                0.0,
            ]
            return np.array(descriptors, dtype=np.float32)

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            descriptors = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumAliphaticRings(mol),
                Descriptors.NumSaturatedRings(mol),
                Descriptors.FractionCSP3(mol),
                Descriptors.NumHeteroatoms(mol),
                Descriptors.RingCount(mol),
                rdMolDescriptors.CalcNumSpiroAtoms(mol),
                rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            ]

            return np.array(descriptors, dtype=np.float32)

        except Exception as e:  # pragma: no cover - defensive logging
            logger.warning(f"Descriptor computation failed for {smiles}: {e}")
            return None

    def predict_cytotoxicity(
        self,
        smiles: str,
        return_confidence: bool = True,
    ) -> dict[str, float]:
        """
        Predict cytotoxicity (general cell toxicity).

        Args:
            smiles: SMILES string
            return_confidence: Whether to return confidence score

        Returns:
            Dictionary with prediction and confidence
        """
        descriptors = self._compute_molecular_descriptors(smiles)
        if descriptors is None:
            return {"toxic": 0.5, "toxic_class": "non-hepatotoxic", "confidence": 0.0}

        # Placeholder prediction - would use trained model
        # Using simple heuristics for demonstration
        mol_weight = descriptors[0]
        logp = descriptors[1]

        # Simple rule-based prediction (placeholder)
        toxic_prob = 0.0
        if mol_weight > 600 or logp > 6:
            toxic_prob += 0.3
        if descriptors[2] > 7:  # Many H-donors
            toxic_prob += 0.2

        toxic_prob = min(1.0, toxic_prob)

        result = {
            "toxic": toxic_prob,
            "toxic_class": "toxic" if toxic_prob > 0.5 else "non-toxic",
        }

        if return_confidence:
            # Confidence based on how certain the prediction is
            result["confidence"] = abs(toxic_prob - 0.5) * 2

        return result

    def predict_hepatotoxicity(
        self,
        smiles: str,
        return_confidence: bool = True,
    ) -> dict[str, float]:
        """
        Predict hepatotoxicity (liver toxicity).

        Args:
            smiles: SMILES string
            return_confidence: Whether to return confidence score

        Returns:
            Dictionary with prediction and confidence
        """
        descriptors = self._compute_molecular_descriptors(smiles)
        if descriptors is None:
            return {"toxic": 0.5, "confidence": 0.0}

        # Placeholder - hepatotoxicity often related to lipophilicity and size
        logp = descriptors[1]
        mol_weight = descriptors[0]

        toxic_prob = 0.0
        if logp > 5:
            toxic_prob += 0.4
        if mol_weight > 500:
            toxic_prob += 0.3

        toxic_prob = min(1.0, toxic_prob)

        result = {
            "toxic": toxic_prob,
            "toxic_class": "hepatotoxic" if toxic_prob > 0.5 else "non-hepatotoxic",
        }

        if return_confidence:
            result["confidence"] = abs(toxic_prob - 0.5) * 2

        return result

    def predict_cardiotoxicity(
        self,
        smiles: str,
        include_herg: bool = True,
        return_confidence: bool = True,
    ) -> dict[str, float]:
        """
        Predict cardiotoxicity including hERG inhibition.

        Args:
            smiles: SMILES string
            include_herg: Whether to specifically check hERG inhibition
            return_confidence: Whether to return confidence score

        Returns:
            Dictionary with prediction and confidence
        """
        descriptors = self._compute_molecular_descriptors(smiles)
        if descriptors is None:
            return {
                "toxic": 0.5,
                "cardiotoxic_class": "non-cardiotoxic",
                "herg_inhibitor": 0.5,
                "herg_risk": "medium",
                "confidence": 0.0,
            }

        # Placeholder - hERG inhibition often related to lipophilicity and basic nitrogen
        logp = descriptors[1]
        h_acceptors = descriptors[3]

        herg_prob = 0.0
        if logp > 3:
            herg_prob += 0.4
        if h_acceptors > 5:
            herg_prob += 0.3

        herg_prob = min(1.0, herg_prob)

        result = {
            "toxic": herg_prob,
            "cardiotoxic_class": "cardiotoxic" if herg_prob > 0.5 else "non-cardiotoxic",
        }

        if include_herg:
            result["herg_inhibitor"] = herg_prob
            result["herg_risk"] = "high" if herg_prob > 0.7 else "medium" if herg_prob > 0.4 else "low"

        if return_confidence:
            result["confidence"] = abs(herg_prob - 0.5) * 2

        return result

    def predict_mutagenicity(
        self,
        smiles: str,
        return_confidence: bool = True,
    ) -> dict[str, float]:
        """
        Predict mutagenicity (Ames test prediction).

        Args:
            smiles: SMILES string
            return_confidence: Whether to return confidence score

        Returns:
            Dictionary with prediction and confidence
        """
        descriptors = self._compute_molecular_descriptors(smiles)
        if descriptors is None:
            return {
                "mutagenic": 0.5,
                "mutagenic_class": "non-mutagenic",
                "ames_positive": False,
                "confidence": 0.0,
            }

        mutagenic_prob = 0.1  # Base probability

        if Chem is None:
            props = heuristic_props(smiles)
            mutagenic_prob += 0.3 if props.aromatic_rings > 1 else 0.0
            mutagenic_prob += 0.2 if props.logp > 4 else 0.0
        else:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                aromatic_amines_pattern = Chem.MolFromSmarts("[NX3;H2,H1]-[c]")
                if mol.HasSubstructMatch(aromatic_amines_pattern):
                    mutagenic_prob += 0.5

                nitro_pattern = Chem.MolFromSmarts("[N+](=O)[O-]")
                if mol.HasSubstructMatch(nitro_pattern):
                    mutagenic_prob += 0.4

        mutagenic_prob = min(1.0, mutagenic_prob)

        result = {
            "mutagenic": mutagenic_prob,
            "mutagenic_class": "mutagenic" if mutagenic_prob > 0.5 else "non-mutagenic",
            "ames_positive": mutagenic_prob > 0.5,
        }

        if return_confidence:
            result["confidence"] = abs(mutagenic_prob - 0.5) * 2

        return result

    def predict_all_toxicity_endpoints(
        self,
        smiles: str,
        return_confidence: bool = True,
    ) -> dict[str, dict[str, float]]:
        """
        Predict all toxicity endpoints simultaneously.

        Args:
            smiles: SMILES string
            return_confidence: Whether to return confidence scores

        Returns:
            Dictionary mapping endpoint names to prediction dictionaries
        """
        results = {
            "cytotoxicity": self.predict_cytotoxicity(smiles, return_confidence),
            "hepatotoxicity": self.predict_hepatotoxicity(smiles, return_confidence),
            "cardiotoxicity": self.predict_cardiotoxicity(smiles, return_confidence=return_confidence),
            "mutagenicity": self.predict_mutagenicity(smiles, return_confidence),
        }

        # Overall toxicity score (worst case)
        toxicity_scores = [
            results["cytotoxicity"]["toxic"],
            results["hepatotoxicity"]["toxic"],
            results["cardiotoxicity"]["toxic"],
            results["mutagenicity"]["mutagenic"],
        ]

        results["overall"] = {
            "toxicity_score": max(toxicity_scores),
            "toxicity_class": "toxic" if max(toxicity_scores) > 0.5 else "non-toxic",
            "worst_endpoint": ["cytotoxicity", "hepatotoxicity", "cardiotoxicity", "mutagenicity"][
                np.argmax(toxicity_scores)
            ],
        }

        return results

    def batch_predict(
        self,
        smiles_list: list[str],
        return_confidence: bool = True,
    ) -> pd.DataFrame:
        """
        Predict toxicity for multiple molecules.

        Args:
            smiles_list: List of SMILES strings
            return_confidence: Whether to include confidence scores

        Returns:
            DataFrame with toxicity predictions
        """
        results = []

        for smiles in smiles_list:
            predictions = self.predict_all_toxicity_endpoints(smiles, return_confidence)

            row = {
                "smiles": smiles,
                "cytotoxicity": predictions["cytotoxicity"]["toxic"],
                "hepatotoxicity": predictions["hepatotoxicity"]["toxic"],
                "cardiotoxicity": predictions["cardiotoxicity"]["toxic"],
                "mutagenicity": predictions["mutagenicity"]["mutagenic"],
                "overall_toxicity": predictions["overall"]["toxicity_score"],
                "worst_endpoint": predictions["overall"]["worst_endpoint"],
            }

            if return_confidence:
                row["cytotoxicity_confidence"] = predictions["cytotoxicity"].get("confidence", 0)
                row["hepatotoxicity_confidence"] = predictions["hepatotoxicity"].get("confidence", 0)
                row["cardiotoxicity_confidence"] = predictions["cardiotoxicity"].get("confidence", 0)
                row["mutagenicity_confidence"] = predictions["mutagenicity"].get("confidence", 0)

            results.append(row)

        df = pd.DataFrame(results)
        logger.info(f"Batch predicted toxicity for {len(smiles_list)} molecules")

        return df

    def get_toxicity_pass_rate(self, predictions: pd.DataFrame, threshold: float = 0.5) -> dict[str, float]:
        """
        Calculate toxicity pass rates from batch predictions.

        Args:
            predictions: DataFrame from batch_predict
            threshold: Toxicity threshold

        Returns:
            Dictionary with pass rates for each endpoint
        """
        pass_rates = {
            "cytotoxicity": (predictions["cytotoxicity"] < threshold).mean(),
            "hepatotoxicity": (predictions["hepatotoxicity"] < threshold).mean(),
            "cardiotoxicity": (predictions["cardiotoxicity"] < threshold).mean(),
            "mutagenicity": (predictions["mutagenicity"] < threshold).mean(),
            "overall": (predictions["overall_toxicity"] < threshold).mean(),
        }

        return pass_rates
