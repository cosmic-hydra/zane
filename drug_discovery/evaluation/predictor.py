"""
Molecular Property Prediction and Evaluation
"""

from typing import Any

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class PropertyPredictor:
    """Predicts various molecular properties using trained models."""

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        """Initialize property predictor.

        Args:
            model: Trained PyTorch model.
            device: Device to run predictions on.
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def predict(self, features: torch.Tensor) -> np.ndarray:
        """Predict properties for given features.

        Args:
            features: Input feature tensor.

        Returns:
            Predictions as numpy array.
        """
        with torch.no_grad():
            features = features.to(self.device)
            predictions = self.model(features)
            return predictions.cpu().numpy()

    def predict_from_smiles(self, smiles: str, featurizer) -> float | None:
        """Predict property from SMILES string.

        Args:
            smiles: SMILES string.
            featurizer: Molecular featurizer object.

        Returns:
            Predicted property value or None if SMILES is invalid.
        """
        features = featurizer.smiles_to_fingerprint(smiles)
        if features is None:
            return None

        features = torch.FloatTensor(features).unsqueeze(0)
        prediction = self.predict(features)

        return float(prediction[0])


class ADMETPredictor:
    """Predicts ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties."""

    def __init__(self):
        """Initialize ADMET predictor."""
        pass

    def calculate_lipinski_properties(self, smiles: str) -> dict[str, float] | None:
        """Calculate Lipinski's Rule of Five properties.

        Args:
            smiles: SMILES string.

        Returns:
            Dictionary of Lipinski properties or None if invalid SMILES.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        properties = {
            "molecular_weight": float(Descriptors.MolWt(mol)),
            "logp": float(Crippen.MolLogP(mol)),
            "h_bond_donors": float(Lipinski.NumHDonors(mol)),
            "h_bond_acceptors": float(Lipinski.NumHAcceptors(mol)),
            "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
            "aromatic_rings": float(Lipinski.NumAromaticRings(mol)),
        }

        return properties

    def check_lipinski_rule(self, smiles: str) -> dict[str, Any] | None:
        """Check if molecule passes Lipinski's Rule of Five.

        Args:
            smiles: SMILES string.

        Returns:
            Dictionary with pass/fail status, violations list, and properties,
            or None if SMILES is invalid.
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
        """Calculate Quantitative Estimate of Drug-likeness.

        Args:
            smiles: SMILES string.

        Returns:
            QED score (0-1, higher is more drug-like) or None if invalid.
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

    def expected_calibration_error_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_uncertainty: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute regression ECE by comparing uncertainty and absolute error."""
        true_v = np.asarray(y_true).reshape(-1)
        pred_v = np.asarray(y_pred).reshape(-1)
        unc_v = np.asarray(y_uncertainty).reshape(-1)
        if len(true_v) == 0:
            return 0.0

        # Normalize uncertainty to [0, 1] for comparability across scales.
        unc_min = float(np.min(unc_v))
        unc_max = float(np.max(unc_v))
        if unc_max - unc_min > 1e-12:
            unc_norm = (unc_v - unc_min) / (unc_max - unc_min)
        else:
            unc_norm = np.zeros_like(unc_v)

        abs_err = np.abs(true_v - pred_v)
        err_max = float(np.max(abs_err))
        err_norm = abs_err / err_max if err_max > 1e-12 else np.zeros_like(abs_err)

        bins = np.linspace(0.0, 1.0, max(2, int(n_bins)) + 1)
        ece = 0.0
        for i in range(len(bins) - 1):
            left, right = bins[i], bins[i + 1]
            mask = (unc_norm >= left) & (unc_norm < right if i < len(bins) - 2 else unc_norm <= right)
            count = int(np.sum(mask))
            if count == 0:
                continue
            bin_unc = float(np.mean(unc_norm[mask]))
            bin_err = float(np.mean(err_norm[mask]))
            ece += (count / len(unc_norm)) * abs(bin_unc - bin_err)
        return float(ece)

    def prediction_interval_coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        z_score: float = 1.96,
    ) -> float:
        """Fraction of labels that fall inside Gaussian prediction intervals."""
        true_v = np.asarray(y_true).reshape(-1)
        pred_v = np.asarray(y_pred).reshape(-1)
        std_v = np.maximum(np.asarray(y_std).reshape(-1), 0.0)
        if len(true_v) == 0:
            return 0.0

        lower = pred_v - z_score * std_v
        upper = pred_v + z_score * std_v
        covered = (true_v >= lower) & (true_v <= upper)
        return float(np.mean(covered))

    def print_metrics(self):
        """Print evaluation metrics"""
        print("\n=== Model Evaluation Metrics ===")
        for key, value in self.metrics.items():
            print(f"{key.upper()}: {value:.4f}")
        print("=" * 35)
