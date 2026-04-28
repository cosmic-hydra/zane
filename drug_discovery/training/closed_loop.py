"""Closed-Loop Active Learning System with Bayesian Optimization.

Implements generate -> surrogate-filter -> oracle-evaluate -> retrain cycles.

The :class:`SurrogateModel` uses GPyTorch/BoTorch (when available) to build a
Gaussian Process that predicts binding free energy from molecular fingerprints.
Only the top 0.1 % of candidates by Expected Improvement are forwarded to the
expensive :class:`~drug_discovery.polyglot_integration.PhysicsOracle` for
physical FEP simulation.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    import gpytorch  # type: ignore[import-untyped]

    _GPYTORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    gpytorch = None  # type: ignore[assignment]
    _GPYTORCH_AVAILABLE = False

try:
    from botorch.acquisition import ExpectedImprovement  # type: ignore[import-untyped]
    from botorch.fit import fit_gpytorch_mll  # type: ignore[import-untyped]
    from botorch.models import SingleTaskGP  # type: ignore[import-untyped]

    _BOTORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BOTORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def smiles_to_fingerprint(smiles: str, nbits: int = 256) -> np.ndarray:
    """Convert a SMILES string into a fixed-length bit vector.

    Uses RDKit Morgan fingerprints when available; falls back to a
    deterministic hash-based encoding otherwise.
    """
    try:
        from rdkit import Chem  # type: ignore[import-untyped]
        from rdkit.Chem import AllChem  # type: ignore[import-untyped]

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
            return np.array(fp, dtype=np.float64)
    except Exception:
        pass

    # Deterministic hash fallback
    digest = hashlib.sha256(smiles.encode()).hexdigest()
    bits = np.zeros(nbits, dtype=np.float64)
    for i, ch in enumerate(digest):
        bits[i % nbits] += int(ch, 16) / 15.0
    return bits / max(bits.max(), 1e-8)


# ---------------------------------------------------------------------------
# Surrogate Model (BoTorch / GPyTorch)
# ---------------------------------------------------------------------------
class SurrogateModel:
    """Gaussian Process surrogate that predicts delta-G from fingerprints.

    When BoTorch/GPyTorch are installed, it fits a proper
    :class:`~botorch.models.SingleTaskGP`. Otherwise it uses a lightweight
    kernel-weighted nearest-neighbour estimator so the closed-loop pipeline
    always runs.
    """

    def __init__(self, fp_dim: int = 256, device: str = "cpu"):
        self.fp_dim = fp_dim
        self.device = torch.device(device)
        self._train_X: list[np.ndarray] = []
        self._train_Y: list[float] = []
        self._gp_model: Any = None
        self._gp_likelihood: Any = None
        self._best_y: float = float("inf")
        self._fit_count: int = 0

    # ------------------------------------------------------------------
    # Training data management
    # ------------------------------------------------------------------
    def observe(self, fingerprint: np.ndarray, delta_g: float) -> None:
        """Record one (fingerprint, delta_g) observation."""
        self._train_X.append(fingerprint)
        self._train_Y.append(delta_g)
        if delta_g < self._best_y:
            self._best_y = delta_g

    def observe_batch(self, fingerprints: Sequence[np.ndarray], delta_gs: Sequence[float]) -> None:
        for fp, dg in zip(fingerprints, delta_gs):
            self.observe(fp, dg)

    @property
    def n_observations(self) -> int:
        return len(self._train_Y)

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------
    def fit(self) -> None:
        """(Re-)fit the surrogate GP on current observations."""
        if self.n_observations < 2:
            logger.warning("SurrogateModel.fit: need >= 2 observations, have %d", self.n_observations)
            return

        train_X = torch.tensor(np.array(self._train_X), dtype=torch.float64, device=self.device)
        train_Y = torch.tensor(self._train_Y, dtype=torch.float64, device=self.device).unsqueeze(-1)

        if _BOTORCH_AVAILABLE:
            self._fit_botorch(train_X, train_Y)
        else:
            self._fit_fallback(train_X, train_Y)
        self._fit_count += 1

    def _fit_botorch(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Fit a BoTorch SingleTaskGP."""
        self._gp_model = SingleTaskGP(train_X, train_Y)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._gp_model.likelihood, self._gp_model)
        fit_gpytorch_mll(mll)
        logger.info("BoTorch SingleTaskGP fitted on %d observations", train_X.shape[0])

    def _fit_fallback(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Store tensors for the KNN fallback predictor."""
        self._gp_model = None
        self._fallback_X = train_X
        self._fallback_Y = train_Y
        logger.info("Fallback surrogate fitted on %d observations", train_X.shape[0])

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, fingerprints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and std for an array of fingerprints.

        Args:
            fingerprints: ``(N, fp_dim)`` array.

        Returns:
            Tuple of ``(means, stds)`` each of shape ``(N,)``.
        """
        X = torch.tensor(fingerprints, dtype=torch.float64, device=self.device)
        if X.ndim == 1:
            X = X.unsqueeze(0)

        if _BOTORCH_AVAILABLE and self._gp_model is not None:
            return self._predict_botorch(X)
        return self._predict_fallback(X)

    def _predict_botorch(self, X: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        self._gp_model.eval()
        self._gp_model.likelihood.eval()
        with torch.no_grad():
            posterior = self._gp_model.posterior(X)
            mean = posterior.mean.squeeze(-1).cpu().numpy()
            std = posterior.variance.sqrt().squeeze(-1).cpu().numpy()
        return mean, std

    def _predict_fallback(self, X: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Kernel-weighted KNN surrogate."""
        if not hasattr(self, "_fallback_X") or self._fallback_X is None:
            return np.zeros(X.shape[0]), np.ones(X.shape[0])

        # RBF kernel distances
        dists = torch.cdist(X.double(), self._fallback_X.double())
        weights = torch.exp(-0.5 * dists**2)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        means = (weights @ self._fallback_Y).squeeze(-1).cpu().numpy()
        # Variance as weighted squared deviation
        y_flat = self._fallback_Y.squeeze(-1)
        var = (weights * (y_flat.unsqueeze(0) - torch.tensor(means, device=self.device).unsqueeze(1)) ** 2).sum(1)
        stds = var.sqrt().cpu().numpy()
        return means, stds

    # ------------------------------------------------------------------
    # Acquisition: Expected Improvement
    # ------------------------------------------------------------------
    def expected_improvement(self, fingerprints: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Compute Expected Improvement for candidate fingerprints.

        Lower delta-G is better (more negative binding energy), so EI is
        computed with respect to the current *minimum* observed value.

        Args:
            fingerprints: ``(N, fp_dim)`` array.
            xi: Exploration parameter.

        Returns:
            ``(N,)`` array of EI values (higher = more promising).
        """
        means, stds = self.predict(fingerprints)
        best_y = self._best_y if self._best_y < float("inf") else 0.0

        # We want to *minimise* delta_g.  EI for minimisation:
        #   EI = (best_y - mu - xi) * Phi(Z) + sigma * phi(Z)
        improvement = best_y - means - xi
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.where(stds > 1e-8, improvement / stds, 0.0)
        ei = np.where(
            stds > 1e-8,
            improvement * _normal_cdf(Z) + stds * _normal_pdf(Z),
            np.maximum(improvement, 0.0),
        )
        return np.maximum(ei, 0.0)

    def select_top_candidates(
        self,
        smiles_list: Sequence[str],
        top_fraction: float = 0.001,
        min_candidates: int = 1,
        xi: float = 0.01,
    ) -> list[str]:
        """Return the top *top_fraction* SMILES ranked by Expected Improvement.

        Args:
            smiles_list: Full candidate pool.
            top_fraction: Fraction to keep (default 0.1 %).
            min_candidates: Minimum number of candidates returned.
            xi: EI exploration parameter.

        Returns:
            List of selected SMILES strings.
        """
        if not smiles_list:
            return []

        fps = np.array([smiles_to_fingerprint(s, nbits=self.fp_dim) for s in smiles_list])
        ei_values = self.expected_improvement(fps, xi=xi)

        k = max(min_candidates, int(math.ceil(len(smiles_list) * top_fraction)))
        k = min(k, len(smiles_list))
        top_indices = np.argsort(ei_values)[-k:][::-1]
        return [smiles_list[i] for i in top_indices]

    # ------------------------------------------------------------------
    # Serialization (warm-start)
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist training data and metadata to *path* for warm-starting.

        The GP model itself is re-fitted on load; only the observations
        and configuration are serialized.
        """
        state = {
            "fp_dim": self.fp_dim,
            "train_X": np.array(self._train_X) if self._train_X else np.empty((0, self.fp_dim)),
            "train_Y": np.array(self._train_Y),
            "best_y": self._best_y,
            "fit_count": self._fit_count,
        }
        np.savez_compressed(path, **state)
        logger.info("SurrogateModel saved to %s (%d observations)", path, self.n_observations)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> SurrogateModel:
        """Load a previously saved surrogate and re-fit.

        Returns a new :class:`SurrogateModel` with all prior observations
        restored.  Call :meth:`fit` to rebuild the GP.
        """
        data = np.load(path, allow_pickle=False)
        fp_dim = int(data["fp_dim"])
        model = cls(fp_dim=fp_dim, device=device)
        train_X = data["train_X"]
        train_Y = data["train_Y"]
        for i in range(len(train_Y)):
            model.observe(train_X[i], float(train_Y[i]))
        model._fit_count = int(data.get("fit_count", 0))
        if model.n_observations >= 2:
            model.fit()
        logger.info("SurrogateModel loaded from %s (%d observations)", path, model.n_observations)
        return model


# ---------------------------------------------------------------------------
# Normal distribution helpers (avoid scipy dependency)
# ---------------------------------------------------------------------------
def _normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf_approx(x / math.sqrt(2.0)))


def _erf_approx(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun approximation (max error ~1.5e-7)."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-(x**2))
    return sign * y


# ---------------------------------------------------------------------------
# Closed-Loop Learner (updated)
# ---------------------------------------------------------------------------
class ClosedLoopLearner:
    """Closed-loop active learning for drug discovery.

    Generates candidates, filters through a :class:`SurrogateModel` to find
    the top 0.1 % by Expected Improvement, evaluates those with the
    :class:`~drug_discovery.polyglot_integration.PhysicsOracle`, and retrains.
    """

    def __init__(
        self,
        pipeline: Any = None,
        docking_engine: Any = None,
        admet_predictor: Any = None,
        bayesian_optimizer: Any = None,
        physics_oracle: Any = None,
        surrogate: SurrogateModel | None = None,
    ):
        """
        Args:
            pipeline: Main drug discovery pipeline.
            docking_engine: Docking engine for evaluation.
            admet_predictor: ADMET predictor.
            bayesian_optimizer: Legacy Bayesian optimizer (kept for compat).
            physics_oracle: :class:`~drug_discovery.polyglot_integration.PhysicsOracle` instance.
            surrogate: :class:`SurrogateModel` instance (created automatically if *None*).
        """
        self.pipeline = pipeline
        self.docking_engine = docking_engine
        self.admet_predictor = admet_predictor
        self.bayesian_optimizer = bayesian_optimizer
        self.physics_oracle = physics_oracle
        self.surrogate = surrogate or SurrogateModel()

        self.iteration_history: list[dict[str, Any]] = []

    def run_closed_loop(
        self,
        target_protein: str,
        num_iterations: int = 10,
        candidates_per_iteration: int = 50,
        top_k_for_training: int = 10,
        surrogate_top_fraction: float = 0.001,
    ) -> list[dict]:
        """Run closed-loop learning cycles.

        Args:
            target_protein: Target protein for optimisation.
            num_iterations: Number of learning iterations.
            candidates_per_iteration: Candidates generated per iteration.
            top_k_for_training: Top candidates to use for retraining.
            surrogate_top_fraction: Fraction of candidates sent to the oracle.

        Returns:
            List of iteration results.
        """
        logger.info("=" * 80)
        logger.info("STARTING CLOSED-LOOP ACTIVE LEARNING")
        logger.info("=" * 80)

        results: list[dict[str, Any]] = []

        for iteration in range(num_iterations):
            logger.info("\n%s", "=" * 80)
            logger.info("ITERATION %d/%d", iteration + 1, num_iterations)
            logger.info("%s\n", "=" * 80)

            # Step 1: Generate candidates
            logger.info("Step 1: Generating candidates...")
            candidates = self._generate_candidates(
                target_protein=target_protein, num_candidates=candidates_per_iteration
            )

            # Step 2: Surrogate filter -- pick top 0.1 % by EI
            logger.info("Step 2: Surrogate filtering (top %.2f%%)...", surrogate_top_fraction * 100)
            smiles_pool = [c["smiles"] for c in candidates]
            if self.surrogate.n_observations >= 2:
                self.surrogate.fit()
                selected_smiles = self.surrogate.select_top_candidates(
                    smiles_pool, top_fraction=surrogate_top_fraction
                )
            else:
                # Not enough data yet -- send all candidates
                selected_smiles = smiles_pool
            logger.info("  %d / %d candidates selected for oracle", len(selected_smiles), len(smiles_pool))

            # Step 3: Physics Oracle evaluation
            logger.info("Step 3: Evaluating selected candidates...")
            evaluated = self._evaluate_candidates_with_oracle(selected_smiles, target_protein)

            # Feed results back into the surrogate
            for ev in evaluated:
                if ev.get("delta_g") is not None:
                    fp = smiles_to_fingerprint(ev["smiles"], nbits=self.surrogate.fp_dim)
                    self.surrogate.observe(fp, ev["delta_g"])

            # Step 4: Also run legacy evaluation for all candidates
            all_evaluated = self._evaluate_candidates(candidates, target_protein)

            # Step 5: Select top candidates
            logger.info("Step 4: Selecting top candidates...")
            top_candidates = self._select_top_candidates(all_evaluated, top_k=top_k_for_training)

            # Step 6: Retrain model
            logger.info("Step 5: Retraining model...")
            retrain_metrics = self._retrain_model(top_candidates)

            # Record iteration results
            iteration_result = {
                "iteration": iteration + 1,
                "num_generated": len(candidates),
                "num_surrogate_selected": len(selected_smiles),
                "num_oracle_evaluated": len(evaluated),
                "num_evaluated": len(all_evaluated),
                "top_candidates": top_candidates,
                "retrain_metrics": retrain_metrics,
                "best_score": top_candidates[0].get("overall_score", 0) if top_candidates else 0,
                "surrogate_observations": self.surrogate.n_observations,
            }

            results.append(iteration_result)
            self.iteration_history.append(iteration_result)

            logger.info("\nIteration %d Summary:", iteration + 1)
            logger.info("  Generated: %d", len(candidates))
            logger.info("  Surrogate-selected: %d", len(selected_smiles))
            logger.info("  Oracle-evaluated: %d", len(evaluated))
            logger.info("  Best score: %.4f", iteration_result["best_score"])

        logger.info("\n%s", "=" * 80)
        logger.info("CLOSED-LOOP LEARNING COMPLETE")
        logger.info("=" * 80)

        return results

    # ------------------------------------------------------------------
    # Oracle evaluation
    # ------------------------------------------------------------------
    def _evaluate_candidates_with_oracle(
        self, smiles_list: list[str], target_protein: str
    ) -> list[dict[str, Any]]:
        """Run the Physics Oracle on a short-list of SMILES."""
        if self.physics_oracle is None:
            # No oracle configured -- return mock evaluations
            return [
                {"smiles": s, "delta_g": -7.0 + hash(s) % 100 / 50.0, "success": True}
                for s in smiles_list
            ]

        results = self.physics_oracle.score_batch_sync(smiles_list)
        return [r.as_dict() for r in results]

    # ------------------------------------------------------------------
    # Legacy helpers (preserved from original)
    # ------------------------------------------------------------------
    # A small pool of drug-like SMILES used when no generative model is
    # attached.  Provides structural diversity for the surrogate to learn from.
    _SEED_SMILES = [
        "CCO",
        "c1ccccc1",
        "CC(=O)O",
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # testosterone
        "OC(=O)c1ccccc1O",  # salicylic acid
        "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",  # pyrene
        "CC(=O)NC1=CC=C(C=C1)O",  # acetaminophen
        "C1CCCCC1",  # cyclohexane
        "c1ccncc1",  # pyridine
        "C1=CC=C(C=C1)C(=O)O",  # benzoic acid
        "OC1=CC=CC=C1",  # phenol
        "c1ccc(cc1)N",  # aniline
        "CC(C)(C)c1ccc(cc1)O",  # 4-tert-butylphenol
    ]

    def _generate_candidates(self, target_protein: str, num_candidates: int) -> list[dict]:
        """Generate drug candidates.

        When no generative model is attached, samples from a pool of
        drug-like seed SMILES to provide structural diversity.
        """
        candidates = []
        pool = self._SEED_SMILES
        for i in range(num_candidates):
            smiles = pool[i % len(pool)]
            candidates.append(
                {"id": f"iter_candidate_{i}", "smiles": smiles, "generation_method": "active_learning"}
            )
        return candidates

    def _evaluate_candidates(self, candidates: list[dict], target_protein: str) -> list[dict]:
        """Evaluate candidates on multiple objectives."""
        evaluated = []
        for candidate in candidates:
            evaluation = {**candidate, "evaluations": {}}

            if self.docking_engine:
                binding_score = -7.5
                evaluation["evaluations"]["binding"] = binding_score

            if self.admet_predictor:
                qed_score = 0.75
                evaluation["evaluations"]["qed"] = qed_score

            evaluation["overall_score"] = self._calculate_overall_score(evaluation["evaluations"])
            evaluated.append(evaluation)

        return evaluated

    def _select_top_candidates(self, evaluated: list[dict], top_k: int) -> list[dict]:
        """Select top k candidates by overall score."""
        sorted_candidates = sorted(evaluated, key=lambda x: x.get("overall_score", 0), reverse=True)
        return sorted_candidates[:top_k]

    def _calculate_overall_score(self, evaluations: dict[str, float]) -> float:
        """Calculate weighted overall score."""
        weights = {"binding": 2.0, "qed": 1.5, "toxicity": -1.0}

        total_score = 0.0
        total_weight = 0.0

        for key, value in evaluations.items():
            weight = weights.get(key, 1.0)
            total_score += weight * value
            total_weight += abs(weight)

        if total_weight > 0:
            return total_score / total_weight
        return 0.0

    def _retrain_model(self, top_candidates: list[dict]) -> dict:
        """Retrain model using top candidates."""
        metrics = {
            "num_new_samples": len(top_candidates),
            "retrain_loss": 0.01,
            "validation_score": 0.85,
        }
        logger.info("Model retrained with %d new samples", len(top_candidates))
        return metrics

    def save_iteration_history(self, filepath: str) -> None:
        """Save iteration history to file."""
        df = pd.DataFrame(self.iteration_history)
        df.to_csv(filepath, index=False)
        logger.info("Iteration history saved to %s", filepath)
