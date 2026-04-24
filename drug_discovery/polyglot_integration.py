"""
Polyglot Integration Layer for ZANE.

Provides unified Python interface to all language-specific implementations:
- Julia: Scientific computing and numerical algorithms
- Go: High-performance CLI tools and batch processing
- Cython: Optimized fingerprint operations
- R: Statistical analysis and visualization

This module handles runtime selection of implementations based on availability.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class JuliaCompute:
    """Wrapper for Julia-based molecular computations."""

    def __init__(self):
        """Initialize Julia compute environment."""
        self._julia = None
        self._available = self._check_julia()

    def _check_julia(self) -> bool:
        """Check if Julia is available in the system."""
        try:
            import julia

            self._julia = julia.Julia(compiled_modules=False)
            return True
        except ImportError:
            logger.warning("Julia not available, falling back to Python implementations")
            return False

    @property
    def available(self) -> bool:
        """Check if Julia environment is ready."""
        return self._available

    def predict_admet_batch(self, properties_matrix: np.ndarray) -> np.ndarray:
        """Predict ADMET scores using Julia backend.

        Args:
            properties_matrix: Array of shape (n_molecules, n_properties)

        Returns:
            Array of ADMET scores
        """
        if not self.available:
            raise RuntimeError("Julia environment not available")

        # Execute Julia function
        result = self._julia.include("julia/molecular_properties.jl")
        return result.batch_admet_prediction(properties_matrix)

    def molecular_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Calculate Tanimoto similarity using Julia.

        Args:
            fp1: First fingerprint
            fp2: Second fingerprint

        Returns:
            Similarity score (0-1)
        """
        if not self.available:
            raise RuntimeError("Julia environment not available")

        result = self._julia.include("julia/molecular_properties.jl")
        return result.molecular_similarity(fp1, fp2)


class GoAccelerator:
    """Wrapper for Go-based high-performance operations."""

    def __init__(self, go_binary_path: Path | None = None):
        """Initialize Go accelerator.

        Args:
            go_binary_path: Path to compiled Go binary, or None to use default
        """
        self.binary_path = go_binary_path or Path("tools/go/admet/admet")
        self._available = self._check_go()

    def _check_go(self) -> bool:
        """Check if Go binary exists and is executable."""
        if self.binary_path.exists():
            return True
        logger.warning(f"Go binary not found at {self.binary_path}")
        return False

    @property
    def available(self) -> bool:
        """Check if Go is available."""
        return self._available

    def predict_admet_single(
        self, molecular_weight: float, logp: float, hbd: int, hba: int, rotatable_bonds: int
    ) -> dict[str, Any]:
        """Predict ADMET using Go backend.

        Args:
            molecular_weight: Molecular weight
            logp: LogP value
            hbd: Hydrogen bond donors
            hba: Hydrogen bond acceptors
            rotatable_bonds: Count of rotatable bonds

        Returns:
            ADMET prediction dictionary
        """
        if not self.available:
            raise RuntimeError("Go binary not available")

        cmd = [
            str(self.binary_path),
            "-mw",
            str(molecular_weight),
            "-logp",
            str(logp),
            "-hbd",
            str(hbd),
            "-hba",
            str(hba),
            "-rb",
            str(rotatable_bonds),
            "-output",
            "json",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f"Go execution failed: {result.stderr}")
            return json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Go execution timed out")

    def predict_admet_batch(self, properties_json: str) -> list[dict[str, float]]:
        """Batch predict ADMET using Go backend.

        Args:
            properties_json: JSON file path with batch properties

        Returns:
            List of ADMET predictions
        """
        if not self.available:
            raise RuntimeError("Go binary not available")

        cmd = [str(self.binary_path), "-batch", properties_json, "-output", "json"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"Go execution failed: {result.stderr}")
            return json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Go batch execution timed out")


class CythonOptimized:
    """Wrapper for Cython-optimized operations."""

    def __init__(self):
        """Initialize Cython optimized module."""
        self._module = None
        self._available = self._check_cython()

    def _check_cython(self) -> bool:
        """Check if Cython extensions are available."""
        try:
            from zane import fingerprints as fp_module

            self._module = fp_module
            return True
        except ImportError:
            logger.warning("Cython extensions not available, falling back to NumPy")
            return False

    @property
    def available(self) -> bool:
        """Check if Cython is available."""
        return self._available

    def tanimoto_batch(self, fingerprints1: np.ndarray, fingerprints2: np.ndarray) -> np.ndarray:
        """Compute pairwise Tanimoto similarity (Cython-optimized).

        Args:
            fingerprints1: Array of shape (n1, n_features)
            fingerprints2: Array of shape (n2, n_features)

        Returns:
            Similarity matrix of shape (n1, n2)
        """
        if not self.available:
            # Fallback to NumPy
            return self._tanimoto_numpy(fingerprints1, fingerprints2)

        return self._module.tanimoto_similarity_batch(fingerprints1, fingerprints2)

    @staticmethod
    def _tanimoto_numpy(fp1: np.ndarray, fp2: np.ndarray) -> np.ndarray:
        """NumPy fallback for Tanimoto similarity."""
        similarities = np.zeros((fp1.shape[0], fp2.shape[0]))
        for i in range(fp1.shape[0]):
            for j in range(fp2.shape[0]):
                intersection = np.minimum(fp1[i], fp2[j]).sum()
                union = np.maximum(fp1[i], fp2[j]).sum()
                similarities[i, j] = intersection / union if union > 0 else 0
        return similarities

    def euclidean_distance_batch(self, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances (Cython-optimized)."""
        if not self.available:
            return np.linalg.cdist(points1, points2)

        return self._module.euclidean_distance_batch(points1, points2)

    def sigmoid_transform(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid transformation (Cython-optimized)."""
        if not self.available:
            return 1.0 / (1.0 + np.exp(-x))

        return self._module.sigmoid_transform(x)


class RStatistics:
    """Wrapper for R-based statistical analysis."""

    def __init__(self):
        """Initialize R statistics environment."""
        self._r = None
        self._available = self._check_r()

    def _check_r(self) -> bool:
        """Check if R environment is available."""
        try:
            import rpy2.robjects as robjects

            self._r = robjects
            return True
        except ImportError:
            logger.warning("R/rpy2 not available, statistical functions disabled")
            return False

    @property
    def available(self) -> bool:
        """Check if R is available."""
        return self._available

    def analyze_training_trends(
        self, epochs: list[int], train_loss: list[float], val_loss: list[float]
    ) -> dict[str, Any]:
        """Analyze training trends using R statistics.

        Args:
            epochs: Epoch numbers
            train_loss: Training losses
            val_loss: Validation losses

        Returns:
            Statistical analysis results
        """
        if not self.available:
            raise RuntimeError("R environment not available")

        # Execute R analysis
        robjects = self._r
        robjects.r.source("R/analysis.R")

        result = robjects.r.analyze_training_trends(epochs, train_loss, val_loss)
        return dict(result)

    def rank_drug_candidates(self, candidates_df: Any, weights: dict[str, float] | None = None) -> Any:
        """Rank drug candidates using multi-objective scoring in R.

        Args:
            candidates_df: Data frame with candidate properties
            weights: Scoring weights

        Returns:
            Ranked data frame
        """
        if not self.available:
            raise RuntimeError("R environment not available")

        robjects = self._r
        robjects.r.source("R/analysis.R")

        if weights is None:
            result = robjects.r.rank_drug_candidates(candidates_df)
        else:
            result = robjects.r.rank_drug_candidates(candidates_df, weights)

        return result


class PolyglotPipeline:
    """Unified interface to all polyglot components."""

    def __init__(self):
        """Initialize all available backends."""
        self.julia = JuliaCompute()
        self.go = GoAccelerator()
        self.cython = CythonOptimized()
        self.r = RStatistics()

        logger.info(f"Julia available: {self.julia.available}")
        logger.info(f"Go available: {self.go.available}")
        logger.info(f"Cython available: {self.cython.available}")
        logger.info(f"R available: {self.r.available}")

    def predict_admet(
        self,
        molecular_weight: float,
        logp: float,
        hbd: int,
        hba: int,
        rotatable_bonds: int,
        prefer_backend: str = "auto",
    ) -> dict[str, Any]:
        """Predict ADMET with automatic backend selection.

        Args:
            molecular_weight: Molecular weight
            logp: LogP value
            hbd: Hydrogen bond donors
            hba: Hydrogen bond acceptors
            rotatable_bonds: Rotatable bonds count
            prefer_backend: Preferred backend ("go", "julia", "python")

        Returns:
            ADMET prediction
        """
        # Try preferred backend first
        if prefer_backend == "go" and self.go.available:
            return self.go.predict_admet_single(molecular_weight, logp, hbd, hba, rotatable_bonds)
        elif prefer_backend == "julia" and self.julia.available:
            # Julia prefers batch operations
            logger.info("Using Julia backend for single prediction")
            return {"note": "Julia batch-optimized"}

        # Fallback to pure Python
        logger.info("Using Python fallback for ADMET prediction")
        return self._python_admet(molecular_weight, logp, hbd, hba, rotatable_bonds)

    @staticmethod
    def _python_admet(mw: float, logp: float, hbd: int, hba: int, rb: int) -> dict[str, Any]:
        """Python ADMET fallback implementation."""
        violations = []
        if mw > 500:
            violations.append("mw > 500")
        if logp > 5:
            violations.append("logp > 5")
        if hbd > 5:
            violations.append("hbd > 5")
        if hba > 10:
            violations.append("hba > 10")

        score = 0.7 if not violations else 0.4
        return {"lipinski": {"passes": len(violations) == 0, "violations": violations}, "admet": {"overall": score}}

    def get_backend_info(self) -> dict[str, bool]:
        """Get information about available backends."""
        return {
            "julia": self.julia.available,
            "go": self.go.available,
            "cython": self.cython.available,
            "r": self.r.available,
        }
