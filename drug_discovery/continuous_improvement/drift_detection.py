"""
Continuous Improvement with Data Drift Detection

Implements continuous monitoring and auto-retraining:
- Data drift detection (distribution shift, concept drift)
- Performance degradation monitoring
- Automatic model retraining triggers
- A/B testing for model updates
- Feature importance tracking
- Data quality monitoring over time
"""

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Report on detected drift."""
    timestamp: str
    drift_detected: bool
    drift_type: str  # 'data', 'concept', 'feature'
    drift_score: float
    affected_features: list[str]
    recommendations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """Performance metric snapshot."""
    timestamp: str
    metric_name: str
    metric_value: float
    dataset_size: int
    metadata: dict[str, Any] = field(default_factory=dict)


class DataDriftDetector:
    """Detect distribution shifts in data."""

    def __init__(
        self,
        window_size: int = 1000,
        drift_threshold: float = 0.1,
        method: str = "ks",
    ):
        """
        Initialize drift detector.

        Args:
            window_size: Size of sliding window
            drift_threshold: Threshold for drift detection
            method: Detection method ('ks', 'wasserstein', 'psi')
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.method = method

        # Reference distribution (baseline)
        self.reference_distributions: dict[str, np.ndarray] = {}

        # Historical data windows
        self.data_windows: dict[str, deque] = {}

    def set_reference_distribution(
        self,
        feature_name: str,
        data: np.ndarray,
    ) -> None:
        """
        Set reference distribution for a feature.

        Args:
            feature_name: Feature name
            data: Reference data
        """
        self.reference_distributions[feature_name] = data
        self.data_windows[feature_name] = deque(maxlen=self.window_size)

        logger.info(f"Set reference distribution for {feature_name}: {len(data)} samples")

    def detect_drift(
        self,
        feature_name: str,
        new_data: np.ndarray,
    ) -> dict[str, Any]:
        """
        Detect drift in a feature.

        Args:
            feature_name: Feature name
            new_data: New data samples

        Returns:
            Dictionary with drift detection results
        """
        if feature_name not in self.reference_distributions:
            logger.warning(f"No reference distribution for {feature_name}")
            return {"drift_detected": False, "drift_score": 0.0}

        reference = self.reference_distributions[feature_name]

        # Compute drift score based on method
        if self.method == "ks":
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(reference, new_data)
            drift_score = statistic
            drift_detected = p_value < self.drift_threshold

        elif self.method == "wasserstein":
            # Wasserstein distance
            drift_score = stats.wasserstein_distance(reference, new_data)
            drift_detected = drift_score > self.drift_threshold

        elif self.method == "psi":
            # Population Stability Index
            drift_score = self._compute_psi(reference, new_data)
            drift_detected = drift_score > self.drift_threshold

        else:
            logger.error(f"Unknown method: {self.method}")
            return {"drift_detected": False, "drift_score": 0.0}

        # Update data window
        self.data_windows[feature_name].extend(new_data)

        result = {
            "drift_detected": drift_detected,
            "drift_score": float(drift_score),
            "method": self.method,
            "reference_size": len(reference),
            "new_data_size": len(new_data),
        }

        if drift_detected:
            logger.warning(f"Drift detected in {feature_name}: score={drift_score:.3f}")

        return result

    def _compute_psi(
        self,
        reference: np.ndarray,
        new_data: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Population Stability Index.

        Args:
            reference: Reference distribution
            new_data: New distribution
            n_bins: Number of bins

        Returns:
            PSI score
        """
        # Create bins from reference
        bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))

        # Compute distributions
        ref_hist, _ = np.histogram(reference, bins=bins)
        new_hist, _ = np.histogram(new_data, bins=bins)

        # Normalize
        ref_dist = ref_hist / len(reference)
        new_dist = new_hist / len(new_data)

        # Avoid division by zero
        ref_dist = np.maximum(ref_dist, 1e-10)
        new_dist = np.maximum(new_dist, 1e-10)

        # Compute PSI
        psi = np.sum((new_dist - ref_dist) * np.log(new_dist / ref_dist))

        return float(psi)

    def batch_detect_drift(
        self,
        feature_data: dict[str, np.ndarray],
    ) -> dict[str, dict[str, Any]]:
        """
        Detect drift across multiple features.

        Args:
            feature_data: Dictionary mapping feature names to data

        Returns:
            Dictionary of drift detection results per feature
        """
        results = {}

        for feature_name, data in feature_data.items():
            results[feature_name] = self.detect_drift(feature_name, data)

        # Summarize
        total_features = len(results)
        drifted_features = sum(1 for r in results.values() if r["drift_detected"])

        logger.info(f"Drift detection: {drifted_features}/{total_features} features drifted")

        return results


class ConceptDriftDetector:
    """Detect changes in target variable relationship."""

    def __init__(
        self,
        window_size: int = 1000,
        error_threshold: float = 0.15,
    ):
        """
        Initialize concept drift detector.

        Args:
            window_size: Size of sliding window
            error_threshold: Error rate threshold for drift
        """
        self.window_size = window_size
        self.error_threshold = error_threshold

        # Historical errors
        self.error_window = deque(maxlen=window_size)
        self.baseline_error_rate = None

    def set_baseline_error_rate(self, error_rate: float) -> None:
        """
        Set baseline error rate.

        Args:
            error_rate: Baseline error rate
        """
        self.baseline_error_rate = error_rate
        logger.info(f"Set baseline error rate: {error_rate:.3f}")

    def detect_concept_drift(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
    ) -> dict[str, Any]:
        """
        Detect concept drift based on prediction errors.

        Args:
            predictions: Model predictions
            true_labels: True labels

        Returns:
            Dictionary with drift detection results
        """
        # Compute errors
        errors = (predictions != true_labels).astype(int)
        current_error_rate = np.mean(errors)

        # Update error window
        self.error_window.extend(errors)

        # Detect drift
        if self.baseline_error_rate is None:
            self.baseline_error_rate = current_error_rate
            drift_detected = False
            drift_score = 0.0
        else:
            drift_score = abs(current_error_rate - self.baseline_error_rate)
            drift_detected = drift_score > self.error_threshold

        result = {
            "drift_detected": drift_detected,
            "drift_score": float(drift_score),
            "current_error_rate": float(current_error_rate),
            "baseline_error_rate": float(self.baseline_error_rate) if self.baseline_error_rate else None,
            "window_size": len(self.error_window),
        }

        if drift_detected:
            logger.warning(f"Concept drift detected: error rate increased by {drift_score:.3f}")

        return result


class PerformanceMonitor:
    """Monitor model performance over time."""

    def __init__(self, alert_threshold: float = 0.1):
        """
        Initialize performance monitor.

        Args:
            alert_threshold: Threshold for performance degradation alerts
        """
        self.alert_threshold = alert_threshold
        self.metrics_history: list[PerformanceMetric] = []

    def log_metric(
        self,
        metric_name: str,
        metric_value: float,
        dataset_size: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a performance metric.

        Args:
            metric_name: Name of metric
            metric_value: Value of metric
            dataset_size: Size of evaluation dataset
            metadata: Optional metadata
        """
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            metric_name=metric_name,
            metric_value=metric_value,
            dataset_size=dataset_size,
            metadata=metadata or {},
        )

        self.metrics_history.append(metric)

    def check_degradation(
        self,
        metric_name: str,
        lookback_period: int = 10,
    ) -> dict[str, Any]:
        """
        Check for performance degradation.

        Args:
            metric_name: Metric to check
            lookback_period: Number of recent measurements to consider

        Returns:
            Dictionary with degradation analysis
        """
        # Filter metrics
        recent_metrics = [
            m for m in self.metrics_history[-lookback_period:]
            if m.metric_name == metric_name
        ]

        if len(recent_metrics) < 2:
            return {"degradation_detected": False}

        # Compare recent to baseline (first half vs second half)
        mid = len(recent_metrics) // 2
        baseline_metrics = recent_metrics[:mid]
        recent_metrics = recent_metrics[mid:]

        baseline_value = np.mean([m.metric_value for m in baseline_metrics])
        recent_value = np.mean([m.metric_value for m in recent_metrics])

        degradation = baseline_value - recent_value
        degradation_detected = degradation > self.alert_threshold

        result = {
            "degradation_detected": degradation_detected,
            "degradation_amount": float(degradation),
            "baseline_value": float(baseline_value),
            "recent_value": float(recent_value),
            "metric_name": metric_name,
        }

        if degradation_detected:
            logger.warning(f"Performance degradation in {metric_name}: {degradation:.3f}")

        return result

    def get_metrics_summary(self) -> pd.DataFrame:
        """Get summary of all metrics."""
        if not self.metrics_history:
            return pd.DataFrame()

        data = []
        for metric in self.metrics_history:
            row = {
                "timestamp": metric.timestamp,
                "metric_name": metric.metric_name,
                "metric_value": metric.metric_value,
                "dataset_size": metric.dataset_size,
            }
            data.append(row)

        df = pd.DataFrame(data)
        return df


class ContinuousImprovementSystem:
    """Comprehensive continuous improvement system."""

    def __init__(
        self,
        model_retraining_trigger: Callable | None = None,
        retraining_window_days: int = 30,
    ):
        """
        Initialize continuous improvement system.

        Args:
            model_retraining_trigger: Function to trigger model retraining
            retraining_window_days: Days between automatic retraining
        """
        self.data_drift_detector = DataDriftDetector()
        self.concept_drift_detector = ConceptDriftDetector()
        self.performance_monitor = PerformanceMonitor()

        self.model_retraining_trigger = model_retraining_trigger
        self.retraining_window_days = retraining_window_days
        self.last_retraining_date: datetime | None = None

        # Drift reports
        self.drift_reports: list[DriftReport] = []

    def monitor_data_health(
        self,
        feature_data: dict[str, np.ndarray],
        predictions: np.ndarray | None = None,
        true_labels: np.ndarray | None = None,
    ) -> DriftReport:
        """
        Monitor data health and detect drifts.

        Args:
            feature_data: Dictionary of feature data
            predictions: Optional model predictions
            true_labels: Optional true labels

        Returns:
            Drift report
        """
        timestamp = datetime.now().isoformat()

        # Detect data drift
        data_drift_results = self.data_drift_detector.batch_detect_drift(feature_data)

        data_drift_detected = any(r["drift_detected"] for r in data_drift_results.values())
        affected_features = [
            f for f, r in data_drift_results.items()
            if r["drift_detected"]
        ]

        # Detect concept drift
        concept_drift_detected = False
        if predictions is not None and true_labels is not None:
            concept_drift_result = self.concept_drift_detector.detect_concept_drift(
                predictions,
                true_labels,
            )
            concept_drift_detected = concept_drift_result["drift_detected"]

        # Determine drift type
        if data_drift_detected and concept_drift_detected:
            drift_type = "data_and_concept"
        elif data_drift_detected:
            drift_type = "data"
        elif concept_drift_detected:
            drift_type = "concept"
        else:
            drift_type = "none"

        # Generate recommendations
        recommendations = []
        if data_drift_detected:
            recommendations.append("Retrain model with recent data to adapt to distribution shift")
        if concept_drift_detected:
            recommendations.append("Model performance degraded - immediate retraining recommended")
        if len(affected_features) > len(feature_data) / 2:
            recommendations.append("Major data shift detected - verify data collection pipeline")

        # Create drift report
        drift_report = DriftReport(
            timestamp=timestamp,
            drift_detected=data_drift_detected or concept_drift_detected,
            drift_type=drift_type,
            drift_score=max([r["drift_score"] for r in data_drift_results.values()]) if data_drift_results else 0.0,
            affected_features=affected_features,
            recommendations=recommendations,
            metadata={
                "data_drift_results": data_drift_results,
                "concept_drift_detected": concept_drift_detected,
            },
        )

        self.drift_reports.append(drift_report)

        # Check if retraining needed
        if drift_report.drift_detected:
            self._check_retraining_trigger()

        return drift_report

    def _check_retraining_trigger(self) -> None:
        """Check if model retraining should be triggered."""
        # Time-based trigger
        should_retrain = False

        if self.last_retraining_date is None:
            should_retrain = True
        else:
            days_since_retraining = (datetime.now() - self.last_retraining_date).days
            if days_since_retraining >= self.retraining_window_days:
                should_retrain = True

        # Drift-based trigger
        recent_drifts = [r for r in self.drift_reports[-10:] if r.drift_detected]
        if len(recent_drifts) >= 3:
            should_retrain = True

        if should_retrain:
            logger.warning("Retraining trigger activated")
            if self.model_retraining_trigger:
                self.model_retraining_trigger()
                self.last_retraining_date = datetime.now()

    def get_system_status(self) -> dict[str, Any]:
        """Get overall system status."""
        recent_reports = self.drift_reports[-10:] if len(self.drift_reports) >= 10 else self.drift_reports

        status = {
            "total_drift_reports": len(self.drift_reports),
            "recent_drift_rate": sum(1 for r in recent_reports if r.drift_detected) / len(recent_reports) if recent_reports else 0,
            "last_retraining_date": self.last_retraining_date.isoformat() if self.last_retraining_date else None,
            "days_since_retraining": (datetime.now() - self.last_retraining_date).days if self.last_retraining_date else None,
            "performance_metrics_count": len(self.performance_monitor.metrics_history),
        }

        return status
