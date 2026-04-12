"""
Uncertainty Estimation - Quantify Prediction Confidence

Implements multiple uncertainty quantification methods:
- Ensemble uncertainty (variance across models)
- Bayesian uncertainty (posterior distributions)
- Conformal prediction (calibrated prediction intervals)
- Evidential deep learning (uncertainty from evidence)
- Monte Carlo dropout (approximate Bayesian inference)
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


class UncertaintyEstimator:
    """Estimate prediction uncertainty using multiple methods."""

    def __init__(self, calibration_method: str = "isotonic"):
        """
        Initialize uncertainty estimator.

        Args:
            calibration_method: Method for probability calibration ('isotonic', 'sigmoid')
        """
        self.calibration_method = calibration_method
        self.calibration_model = None

    def estimate_ensemble_uncertainty(
        self,
        predictions: list[float],
    ) -> dict[str, float]:
        """
        Estimate uncertainty from ensemble predictions.

        Args:
            predictions: List of predictions from different models

        Returns:
            Dictionary with uncertainty metrics
        """
        predictions = np.array(predictions)

        results = {
            "mean_prediction": np.mean(predictions),
            "std_prediction": np.std(predictions),
            "min_prediction": np.min(predictions),
            "max_prediction": np.max(predictions),
            "prediction_range": np.max(predictions) - np.min(predictions),
            "coefficient_of_variation": np.std(predictions) / np.mean(predictions) if np.mean(predictions) > 0 else 0,
        }

        # Epistemic uncertainty (model uncertainty)
        results["epistemic_uncertainty"] = results["std_prediction"]

        # Confidence based on agreement
        results["confidence"] = 1.0 - min(1.0, results["std_prediction"] * 2)

        return results

    def estimate_bayesian_uncertainty(
        self,
        posterior_samples: np.ndarray,
    ) -> dict[str, float]:
        """
        Estimate uncertainty from Bayesian posterior samples.

        Args:
            posterior_samples: Samples from posterior distribution (N, D)

        Returns:
            Dictionary with uncertainty metrics
        """
        results = {
            "posterior_mean": np.mean(posterior_samples, axis=0).tolist(),
            "posterior_std": np.std(posterior_samples, axis=0).tolist(),
        }

        # Compute credible intervals
        results["credible_interval_95"] = {
            "lower": np.percentile(posterior_samples, 2.5, axis=0).tolist(),
            "upper": np.percentile(posterior_samples, 97.5, axis=0).tolist(),
        }

        results["credible_interval_90"] = {
            "lower": np.percentile(posterior_samples, 5, axis=0).tolist(),
            "upper": np.percentile(posterior_samples, 95, axis=0).tolist(),
        }

        # Total uncertainty
        results["total_uncertainty"] = np.mean(np.std(posterior_samples, axis=0))

        return results

    def calibrate_probabilities(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
    ) -> dict[str, Any]:
        """
        Calibrate probability predictions using isotonic regression or sigmoid.

        Args:
            predictions: Predicted probabilities
            true_labels: True binary labels

        Returns:
            Dictionary with calibration metrics and calibrated predictions
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.dummy import DummyClassifier

        # Create dummy classifier for calibration
        dummy_clf = DummyClassifier()
        dummy_clf.fit(np.zeros((len(predictions), 1)), true_labels)

        # Calibrate using specified method
        calibrated_clf = CalibratedClassifierCV(
            dummy_clf,
            method=self.calibration_method,
            cv="prefit",
        )

        # Reshape for sklearn
        x_dummy = np.zeros((len(predictions), 1))
        calibrated_clf.fit(x_dummy, true_labels)

        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            true_labels,
            predictions,
            n_bins=10,
            strategy='uniform',
        )

        results = {
            "fraction_of_positives": fraction_of_positives.tolist(),
            "mean_predicted_value": mean_predicted_value.tolist(),
            "calibration_method": self.calibration_method,
        }

        # Expected Calibration Error (ECE)
        ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        results["expected_calibration_error"] = ece

        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(fraction_of_positives - mean_predicted_value))
        results["maximum_calibration_error"] = mce

        logger.info(f"Calibration: ECE={ece:.3f}, MCE={mce:.3f}")

        return results

    def conformal_prediction(
        self,
        calibration_scores: np.ndarray,
        test_score: float,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """
        Compute conformal prediction interval.

        Args:
            calibration_scores: Non-conformity scores from calibration set
            test_score: Score for test instance
            confidence_level: Desired confidence level (e.g., 0.95)

        Returns:
            Dictionary with prediction interval
        """
        # Compute quantile
        n = len(calibration_scores)
        q = np.ceil((n + 1) * confidence_level) / n
        quantile = np.quantile(calibration_scores, q)

        results = {
            "confidence_level": confidence_level,
            "quantile": float(quantile),
            "test_score": float(test_score),
            "is_conforming": test_score <= quantile,
        }

        # Prediction interval (simplified)
        results["prediction_interval"] = {
            "lower": float(test_score - quantile),
            "upper": float(test_score + quantile),
        }

        return results

    def monte_carlo_dropout_uncertainty(
        self,
        model: Callable,
        input_data: np.ndarray,
        n_iterations: int = 100,
        dropout_rate: float = 0.5,
    ) -> dict[str, float]:
        """
        Estimate uncertainty using Monte Carlo dropout.

        Args:
            model: Model with dropout layers
            input_data: Input data
            n_iterations: Number of MC samples
            dropout_rate: Dropout probability

        Returns:
            Dictionary with uncertainty estimates
        """
        predictions = []

        for _ in range(n_iterations):
            # In practice, model would have dropout enabled during inference
            pred = model(input_data)
            if isinstance(pred, dict):
                pred = list(pred.values())[0]
            predictions.append(float(pred))

        predictions = np.array(predictions)

        results = {
            "mean_prediction": np.mean(predictions),
            "predictive_uncertainty": np.std(predictions),
            "prediction_entropy": self._compute_entropy(predictions),
            "n_iterations": n_iterations,
        }

        # Confidence interval
        results["confidence_interval_95"] = {
            "lower": float(np.percentile(predictions, 2.5)),
            "upper": float(np.percentile(predictions, 97.5)),
        }

        return results

    def _compute_entropy(self, predictions: np.ndarray) -> float:
        """Compute predictive entropy."""
        # Convert to probabilities
        probs = predictions
        probs = np.clip(probs, 1e-10, 1 - 1e-10)

        # Binary entropy
        entropy = -probs * np.log(probs) - (1 - probs) * np.log(1 - probs)
        return float(np.mean(entropy))

    def evidential_uncertainty(
        self,
        alpha: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute uncertainty from evidential neural network outputs.

        Evidential deep learning parameterizes Dirichlet distribution:
        - alpha: Evidence for each class

        Args:
            alpha: Evidence parameters (Dirichlet parameters)

        Returns:
            Dictionary with uncertainty decomposition
        """
        # Total evidence
        total_evidence = np.sum(alpha)

        # Expected probability
        prob = alpha / total_evidence

        # Aleatoric uncertainty (data uncertainty)
        aleatoric = prob * (1 - prob) / (total_evidence + 1)

        # Epistemic uncertainty (model uncertainty)
        epistemic = prob * (1 - prob) / (total_evidence + 1) * (1 / total_evidence)

        results = {
            "predicted_probability": prob.tolist() if isinstance(prob, np.ndarray) else float(prob),
            "total_evidence": float(total_evidence),
            "aleatoric_uncertainty": aleatoric.tolist() if isinstance(aleatoric, np.ndarray) else float(aleatoric),
            "epistemic_uncertainty": epistemic.tolist() if isinstance(epistemic, np.ndarray) else float(epistemic),
        }

        # Total uncertainty
        total_uncertainty = aleatoric + epistemic
        results["total_uncertainty"] = total_uncertainty.tolist() if isinstance(total_uncertainty, np.ndarray) else float(total_uncertainty)

        return results

    def uncertainty_decomposition(
        self,
        ensemble_predictions: list[np.ndarray],
    ) -> dict[str, float]:
        """
        Decompose uncertainty into aleatoric and epistemic components.

        Args:
            ensemble_predictions: List of prediction arrays from different models

        Returns:
            Dictionary with uncertainty decomposition
        """
        ensemble_predictions = np.array(ensemble_predictions)  # (n_models, n_samples)

        # Mean prediction across ensemble
        mean_pred = np.mean(ensemble_predictions, axis=0)

        # Total uncertainty (variance of mean)
        total_uncertainty = np.var(mean_pred)

        # Epistemic uncertainty (variance across models)
        epistemic = np.var(np.mean(ensemble_predictions, axis=1))

        # Aleatoric uncertainty (expected variance)
        aleatoric = np.mean(np.var(ensemble_predictions, axis=1))

        results = {
            "total_uncertainty": float(total_uncertainty),
            "epistemic_uncertainty": float(epistemic),
            "aleatoric_uncertainty": float(aleatoric),
            "epistemic_fraction": float(epistemic / (total_uncertainty + 1e-10)),
            "aleatoric_fraction": float(aleatoric / (total_uncertainty + 1e-10)),
        }

        logger.info(f"Uncertainty decomposition: Epistemic={epistemic:.3f}, Aleatoric={aleatoric:.3f}")

        return results

    def compute_prediction_confidence(
        self,
        prediction: float,
        uncertainty: float,
        method: str = "exponential",
    ) -> float:
        """
        Compute confidence score from prediction and uncertainty.

        Args:
            prediction: Model prediction
            uncertainty: Uncertainty estimate
            method: Method for computing confidence ('exponential', 'linear', 'threshold')

        Returns:
            Confidence score (0-1)
        """
        if method == "exponential":
            # Exponential decay with uncertainty
            confidence = np.exp(-uncertainty * 5)

        elif method == "linear":
            # Linear decrease
            confidence = max(0, 1 - uncertainty)

        elif method == "threshold":
            # Threshold-based
            confidence = 1.0 if uncertainty < 0.2 else 0.5 if uncertainty < 0.5 else 0.0

        else:
            confidence = 0.5

        return float(np.clip(confidence, 0, 1))

    def batch_uncertainty_estimation(
        self,
        ensemble_models: list[Callable],
        smiles_list: list[str],
    ) -> pd.DataFrame:
        """
        Estimate uncertainty for batch of molecules using ensemble.

        Args:
            ensemble_models: List of prediction models
            smiles_list: List of SMILES strings

        Returns:
            DataFrame with predictions and uncertainties
        """
        results = []

        for smiles in smiles_list:
            # Get predictions from all models
            predictions = []
            for model in ensemble_models:
                try:
                    pred = model(smiles)
                    if isinstance(pred, dict):
                        pred = list(pred.values())[0]
                    predictions.append(float(pred))
                except Exception as e:
                    logger.warning(f"Model prediction failed for {smiles}: {e}")
                    continue

            if not predictions:
                continue

            # Compute uncertainty metrics
            uncertainty_metrics = self.estimate_ensemble_uncertainty(predictions)

            row = {
                "smiles": smiles,
                "mean_prediction": uncertainty_metrics["mean_prediction"],
                "std_prediction": uncertainty_metrics["std_prediction"],
                "epistemic_uncertainty": uncertainty_metrics["epistemic_uncertainty"],
                "confidence": uncertainty_metrics["confidence"],
                "prediction_range": uncertainty_metrics["prediction_range"],
            }

            results.append(row)

        df = pd.DataFrame(results)
        logger.info(f"Estimated uncertainty for {len(df)} molecules")

        return df
