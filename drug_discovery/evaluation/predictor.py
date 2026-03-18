"""
Molecular Property Prediction and Evaluation
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Any


class PropertyPredictor:
    """
    Predicts various molecular properties using trained models
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        """
        Args:
            model: Trained PyTorch model
            device: Device to run predictions
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def predict(self, features: torch.Tensor) -> np.ndarray:
        """
        Predict properties for given features

        Args:
            features: Input features

        Returns:
            Predictions
        """
        with torch.no_grad():
            features = features.to(self.device)
            predictions = self.model(features)
            return predictions.cpu().numpy()

    def predict_from_smiles(self, smiles: str, featurizer) -> float | None:
        """
        Predict property from SMILES string

        Args:
            smiles: SMILES string
            featurizer: Molecular featurizer

        Returns:
            Predicted property value
        """
        features = featurizer.smiles_to_fingerprint(smiles)
        if features is None:
            return None

        features = torch.FloatTensor(features).unsqueeze(0)
        prediction = self.predict(features)

        return float(prediction[0])


class ADMETPredictor:
    """
    Predicts ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties
    """

    def __init__(self):
        pass

    def calculate_lipinski_properties(self, smiles: str) -> dict[str, float] | None:
        """
        Calculate Lipinski's Rule of Five properties

        Args:
            smiles: SMILES string

        Returns:
            Dictionary of Lipinski properties
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol_wt = getattr(Descriptors, "MolWt")
        mol_logp = getattr(Crippen, "MolLogP")
        num_h_donors = getattr(Lipinski, "NumHDonors")
        num_h_acceptors = getattr(Lipinski, "NumHAcceptors")
        num_rotatable_bonds = getattr(Lipinski, "NumRotatableBonds")
        num_aromatic_rings = getattr(Lipinski, "NumAromaticRings")

        properties = {
            "molecular_weight": float(mol_wt(mol)),
            "logp": float(mol_logp(mol)),
            "h_bond_donors": float(num_h_donors(mol)),
            "h_bond_acceptors": float(num_h_acceptors(mol)),
            "rotatable_bonds": float(num_rotatable_bonds(mol)),
            "aromatic_rings": float(num_aromatic_rings(mol)),
        }

        return properties

    def check_lipinski_rule(self, smiles: str) -> dict[str, Any] | None:
        """
        Check if molecule passes Lipinski's Rule of Five

        Args:
            smiles: SMILES string

        Returns:
            Dictionary with pass/fail and violations
        """
        props = self.calculate_lipinski_properties(smiles)
        if props is None:
            return None

        violations = []

        if props["molecular_weight"] > 500:
            violations.append("molecular_weight > 500")
        if props["logp"] > 5:
            violations.append("logP > 5")
        if props["h_bond_donors"] > 5:
            violations.append("H-bond donors > 5")
        if props["h_bond_acceptors"] > 10:
            violations.append("H-bond acceptors > 10")

        return {
            "passes": len(violations) == 0,
            "violations": violations,
            "num_violations": len(violations),
            "properties": props,
        }

    def calculate_qed(self, smiles: str) -> float | None:
        """
        Calculate Quantitative Estimate of Drug-likeness

        Args:
            smiles: SMILES string

        Returns:
            QED score (0-1, higher is better)
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        return QED.qed(mol)

    def calculate_synthetic_accessibility(self, smiles: str) -> float | None:
        """
        Estimate synthetic accessibility score (1-10, lower is easier)

        Args:
            smiles: SMILES string

        Returns:
            SA score
        """
        try:
            from rdkit.Chem import Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Simple heuristic (in production, use proper SA score calculation)
            complexity = Descriptors.BertzCT(mol)
            sa_score = min(10, max(1, complexity / 100))

            return sa_score
        except Exception:
            return None

    def predict_toxicity_flags(self, smiles: str) -> dict[str, bool] | None:
        """
        Check for common toxicity flags using structural alerts

        Args:
            smiles: SMILES string

        Returns:
            Dictionary of toxicity flags
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Common PAINS (Pan Assay Interference Compounds) patterns
        pains_patterns = [
            "c1ccc2c(c1)ncs2",  # Benzothiazole
            "[N;D2]=[N;D2]",  # Azo compounds
            "[S;D2](=O)=O",  # Sulfonyl
        ]

        flags = {
            "contains_reactive_groups": False,
            "potential_pains": False,
        }

        # Check for PAINS
        for pattern in pains_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                flags["potential_pains"] = True
                break

        # Check for reactive groups (simplified)
        reactive_smarts = ["[N+](=O)[O-]", "C(=O)Cl", "[S;D2]S"]
        for pattern in reactive_smarts:
            try:
                if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    flags["contains_reactive_groups"] = True
                    break
            except Exception:
                pass

        return flags


class ModelEvaluator:
    """
    Evaluates model performance on various metrics
    """

    def __init__(self):
        self.metrics = {}

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """
        Evaluate regression model

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }

        # Pearson correlation
        correlation = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
        metrics["pearson_r"] = correlation

        self.metrics = metrics
        return metrics

    def evaluate_classification(
        self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
    ) -> dict[str, float]:
        """
        Evaluate classification model

        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            threshold: Classification threshold

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

        y_pred_binary = (y_pred > threshold).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred_binary),
            "precision": precision_score(y_true, y_pred_binary, zero_division=0),
            "recall": recall_score(y_true, y_pred_binary, zero_division=0),
            "f1": f1_score(y_true, y_pred_binary, zero_division=0),
        }

        # ROC-AUC for probability predictions
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
        except Exception:
            metrics["roc_auc"] = 0.0

        self.metrics = metrics
        return metrics

    def print_metrics(self):
        """Print evaluation metrics"""
        print("\n=== Model Evaluation Metrics ===")
        for key, value in self.metrics.items():
            print(f"{key.upper()}: {value:.4f}")
        print("=" * 35)
