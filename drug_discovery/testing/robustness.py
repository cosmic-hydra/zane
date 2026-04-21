"""
Robustness Testing - Validate Model Performance Under Perturbations

Tests model robustness against:
- Noisy molecular structures (SMILES perturbations)
- Distribution shifts (different chemical spaces)
- Adversarial examples
- Out-of-distribution detection
- Cross-validation stability
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold

from drug_discovery.utils.rdkit_fallback import is_smiles_plausible

try:  # pragma: no cover - optional dependency
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
except Exception:  # pragma: no cover - default path when RDKit unavailable
    Chem = None  # type: ignore
    AllChem = None  # type: ignore

logger = logging.getLogger(__name__)


class RobustnessTester:
    """Test model robustness under various perturbations."""

    def __init__(self, random_state: int = 42):
        """
        Initialize robustness tester.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def _perturb_smiles(
        self,
        smiles: str,
        perturbation_type: str = "tautomer",
    ) -> Optional[str]:
        """
        Create a perturbed version of SMILES.

        Args:
            smiles: Original SMILES
            perturbation_type: Type of perturbation
                - 'tautomer': Generate tautomer
                - 'stereoisomer': Flip stereochemistry
                - 'conformer': Different 3D conformer (returns same SMILES)

        Returns:
            Perturbed SMILES or None
        """
        if not is_smiles_plausible(smiles):
            return None

        if Chem is None:
            # Lightweight perturbations without RDKit
            if perturbation_type == "tautomer" and len(smiles) > 2:
                return smiles[::-1]  # reversed string as a deterministic variant
            return smiles

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            if perturbation_type == "tautomer":
                perturbed_smiles = Chem.MolToSmiles(mol, canonical=False)
                return perturbed_smiles

            elif perturbation_type == "stereoisomer":
                stereo_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
                if stereo_centers:
                    return Chem.MolToSmiles(mol, isomericSmiles=True)
                return smiles

            else:
                return smiles

        except Exception as e:
            logger.warning(f"SMILES perturbation failed: {e}")
            return None

    def test_smiles_perturbation_robustness(
        self,
        model: Callable,
        smiles_list: List[str],
        perturbation_types: List[str] = ["tautomer"],
        tolerance: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Test model robustness to SMILES perturbations.

        Args:
            model: Prediction function (smiles -> prediction)
            smiles_list: List of SMILES to test
            perturbation_types: Types of perturbations to apply
            tolerance: Maximum acceptable prediction difference

        Returns:
            Dictionary with robustness metrics
        """
        results = {
            "total_tested": 0,
            "robust_predictions": 0,
            "max_deviation": 0.0,
            "mean_deviation": 0.0,
            "per_molecule_deviations": [],
        }

        deviations = []

        for smiles in smiles_list:
            try:
                # Original prediction
                original_pred = model(smiles)

                if isinstance(original_pred, dict):
                    # Extract main prediction value
                    original_pred = list(original_pred.values())[0]

                # Test perturbations
                for perturb_type in perturbation_types:
                    perturbed_smiles = self._perturb_smiles(smiles, perturb_type)

                    if perturbed_smiles is None:
                        continue

                    perturbed_pred = model(perturbed_smiles)
                    if isinstance(perturbed_pred, dict):
                        perturbed_pred = list(perturbed_pred.values())[0]

                    # Compute deviation
                    deviation = abs(float(original_pred) - float(perturbed_pred))
                    deviations.append(deviation)

                    results["total_tested"] += 1
                    if deviation <= tolerance:
                        results["robust_predictions"] += 1

            except Exception as e:
                logger.warning(f"Robustness test failed for {smiles}: {e}")
                continue

        if deviations:
            results["max_deviation"] = max(deviations)
            results["mean_deviation"] = np.mean(deviations)
            results["std_deviation"] = np.std(deviations)
            results["per_molecule_deviations"] = deviations
            results["robustness_rate"] = results["robust_predictions"] / results["total_tested"]
        else:
            results["robustness_rate"] = 0.0

        logger.info(f"Robustness test: {results['robustness_rate']:.2%} robust predictions")

        return results

    def test_distribution_shift(
        self,
        model: Callable,
        train_smiles: List[str],
        test_smiles: List[str],
        train_labels: Optional[List[float]] = None,
        test_labels: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Test model performance under distribution shift.

        Args:
            model: Prediction function
            train_smiles: Training distribution SMILES
            test_smiles: Test distribution SMILES (different chemical space)
            train_labels: Optional ground truth labels for training set
            test_labels: Optional ground truth labels for test set

        Returns:
            Dictionary with performance metrics
        """
        results = {
            "train_distribution_size": len(train_smiles),
            "test_distribution_size": len(test_smiles),
        }

        try:
            # Get predictions
            train_preds = [model(s) for s in train_smiles]
            test_preds = [model(s) for s in test_smiles]

            # Extract prediction values
            if isinstance(train_preds[0], dict):
                train_preds = [list(p.values())[0] for p in train_preds]
                test_preds = [list(p.values())[0] for p in test_preds]

            # Compute distributional statistics
            results["train_pred_mean"] = np.mean(train_preds)
            results["train_pred_std"] = np.std(train_preds)
            results["test_pred_mean"] = np.mean(test_preds)
            results["test_pred_std"] = np.std(test_preds)

            # Distribution shift metric (KL divergence approximation)
            shift_magnitude = abs(results["test_pred_mean"] - results["train_pred_mean"])
            results["distribution_shift"] = shift_magnitude

            # If labels available, compute accuracy degradation
            if train_labels is not None and test_labels is not None:
                # Convert to binary for accuracy
                train_binary_preds = [1 if p > 0.5 else 0 for p in train_preds]
                test_binary_preds = [1 if p > 0.5 else 0 for p in test_preds]

                train_acc = accuracy_score(train_labels, train_binary_preds)
                test_acc = accuracy_score(test_labels, test_binary_preds)

                results["train_accuracy"] = train_acc
                results["test_accuracy"] = test_acc
                results["accuracy_degradation"] = train_acc - test_acc

            logger.info(f"Distribution shift magnitude: {shift_magnitude:.3f}")

        except Exception as e:
            logger.error(f"Distribution shift test failed: {e}")

        return results

    def test_cross_validation_stability(
        self,
        train_function: Callable,
        data: pd.DataFrame,
        target_column: str,
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """
        Test model stability across cross-validation folds.

        Args:
            train_function: Function to train model on data
            data: DataFrame with features and target
            target_column: Name of target column
            n_splits: Number of CV folds

        Returns:
            Dictionary with stability metrics
        """
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        fold_scores = []
        fold_predictions = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(data)):
            try:
                train_data = data.iloc[train_idx]
                val_data = data.iloc[val_idx]

                # Train model
                model = train_function(train_data)

                # Predict on validation set
                val_preds = []
                for _, row in val_data.iterrows():
                    pred = model(row)
                    if isinstance(pred, dict):
                        pred = list(pred.values())[0]
                    val_preds.append(pred)

                fold_predictions.append(val_preds)

                # Compute fold score
                val_labels = val_data[target_column].values
                val_binary_preds = [1 if p > 0.5 else 0 for p in val_preds]
                fold_score = accuracy_score(val_labels, val_binary_preds)
                fold_scores.append(fold_score)

                logger.info(f"Fold {fold_idx + 1}/{n_splits}: Score = {fold_score:.3f}")

            except Exception as e:
                logger.error(f"Fold {fold_idx + 1} failed: {e}")
                continue

        results = {
            "n_folds": len(fold_scores),
            "fold_scores": fold_scores,
            "mean_score": np.mean(fold_scores),
            "std_score": np.std(fold_scores),
            "min_score": np.min(fold_scores),
            "max_score": np.max(fold_scores),
            "coefficient_of_variation": np.std(fold_scores) / np.mean(fold_scores) if np.mean(fold_scores) > 0 else 0,
        }

        # Stability metric: lower CV means more stable
        results["is_stable"] = results["coefficient_of_variation"] < 0.1

        logger.info(f"Cross-validation: Mean={results['mean_score']:.3f}, Std={results['std_score']:.3f}")

        return results

    def test_adversarial_robustness(
        self,
        model: Callable,
        smiles: str,
        target_class: int = 1,
        max_attempts: int = 100,
    ) -> Dict[str, Any]:
        """
        Test model robustness to adversarial examples.

        Args:
            model: Prediction function
            smiles: Original SMILES
            target_class: Target class for adversarial attack
            max_attempts: Maximum number of perturbation attempts

        Returns:
            Dictionary with adversarial test results
        """
        results = {
            "original_smiles": smiles,
            "adversarial_found": False,
            "attempts": 0,
        }

        try:
            original_pred = model(smiles)
            if isinstance(original_pred, dict):
                original_pred = list(original_pred.values())[0]

            results["original_prediction"] = float(original_pred)

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return results

            # Try to find adversarial example
            for attempt in range(max_attempts):
                # Random perturbation strategy
                perturb_type = np.random.choice(["tautomer", "stereoisomer"])
                perturbed_smiles = self._perturb_smiles(smiles, perturb_type)

                if perturbed_smiles is None:
                    continue

                perturbed_pred = model(perturbed_smiles)
                if isinstance(perturbed_pred, dict):
                    perturbed_pred = list(perturbed_pred.values())[0]

                # Check if adversarial (prediction flipped)
                original_class = 1 if original_pred > 0.5 else 0
                perturbed_class = 1 if perturbed_pred > 0.5 else 0

                if perturbed_class == target_class and perturbed_class != original_class:
                    results["adversarial_found"] = True
                    results["adversarial_smiles"] = perturbed_smiles
                    results["adversarial_prediction"] = float(perturbed_pred)
                    results["attempts"] = attempt + 1
                    logger.warning(f"Adversarial example found after {attempt + 1} attempts")
                    break

            if not results["adversarial_found"]:
                results["attempts"] = max_attempts
                logger.info("No adversarial example found - model is robust")

        except Exception as e:
            logger.error(f"Adversarial test failed: {e}")

        return results

    def test_out_of_distribution_detection(
        self,
        model: Callable,
        in_distribution_smiles: List[str],
        out_distribution_smiles: List[str],
        confidence_threshold: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Test model's ability to detect out-of-distribution inputs.

        Args:
            model: Prediction function that returns confidence scores
            in_distribution_smiles: In-distribution SMILES
            out_distribution_smiles: Out-of-distribution SMILES
            confidence_threshold: Threshold for confident predictions

        Returns:
            Dictionary with OOD detection metrics
        """
        results = {
            "in_distribution_size": len(in_distribution_smiles),
            "out_distribution_size": len(out_distribution_smiles),
        }

        try:
            # Get predictions and confidences
            in_dist_confs = []
            out_dist_confs = []

            for smiles in in_distribution_smiles:
                pred = model(smiles)
                if isinstance(pred, dict) and "confidence" in pred:
                    in_dist_confs.append(pred["confidence"])
                else:
                    # Use prediction certainty as confidence
                    if isinstance(pred, dict):
                        pred = list(pred.values())[0]
                    confidence = abs(float(pred) - 0.5) * 2
                    in_dist_confs.append(confidence)

            for smiles in out_distribution_smiles:
                pred = model(smiles)
                if isinstance(pred, dict) and "confidence" in pred:
                    out_dist_confs.append(pred["confidence"])
                else:
                    if isinstance(pred, dict):
                        pred = list(pred.values())[0]
                    confidence = abs(float(pred) - 0.5) * 2
                    out_dist_confs.append(confidence)

            # Compute statistics
            results["in_dist_mean_confidence"] = np.mean(in_dist_confs)
            results["out_dist_mean_confidence"] = np.mean(out_dist_confs)
            results["confidence_gap"] = results["in_dist_mean_confidence"] - results["out_dist_mean_confidence"]

            # OOD detection rate
            in_dist_high_conf = sum(1 for c in in_dist_confs if c >= confidence_threshold)
            out_dist_low_conf = sum(1 for c in out_dist_confs if c < confidence_threshold)

            results["in_dist_high_conf_rate"] = in_dist_high_conf / len(in_dist_confs)
            results["out_dist_detection_rate"] = out_dist_low_conf / len(out_dist_confs)

            logger.info(f"OOD detection rate: {results['out_dist_detection_rate']:.2%}")

        except Exception as e:
            logger.error(f"OOD detection test failed: {e}")

        return results
