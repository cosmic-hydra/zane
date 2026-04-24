"""Active Learning Pipeline for ZANE.
Uncertainty-driven molecule selection for wet-lab feedback loops.
Ref: Reker & Schneider "Active learning for computational chemogenomics" (2015)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ActiveLearningConfig:
    acquisition: str = "uncertainty"  # "uncertainty","thompson","expected_improvement","diversity"
    n_select: int = 10
    mc_samples: int = 30
    diversity_weight: float = 0.3
    exploration_factor: float = 1.0
    seed: int = 42


class ActiveLearner:
    """Active learning for drug discovery.
    Example:
        learner = ActiveLearner(ActiveLearningConfig(n_select=10))
        learner.fit(X_train, y_train, model_fn)
        indices = learner.select(X_pool)
        learner.update(X_new, y_new)
    """

    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.X_train = None
        self.y_train = None
        self.model = None
        self.predict_fn = None
        self.history = []
        self.cycle = 0

    def fit(self, X, y, model_fn: Callable):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.predict_fn = model_fn
        self.model = model_fn(self.X_train, self.y_train)

    def select(self, X_pool, fingerprints=None):
        acq = self.config.acquisition
        if acq == "uncertainty":
            scores = self._uncertainty(X_pool)
        elif acq == "thompson":
            scores = self._thompson(X_pool)
        elif acq == "expected_improvement":
            scores = self._ei(X_pool)
        else:
            scores = self.rng.rand(len(X_pool))
        if self.config.diversity_weight > 0 and fingerprints is not None:
            div = self._diversity(fingerprints)
            scores = (1 - self.config.diversity_weight) * scores + self.config.diversity_weight * div
        top_k = min(self.config.n_select, len(X_pool))
        indices = np.argsort(-scores)[:top_k]
        self.history.append({"cycle": self.cycle, "selected": indices.tolist()})
        self.cycle += 1
        return indices

    def update(self, X_new, y_new):
        self.X_train = np.vstack([self.X_train, X_new])
        self.y_train = np.concatenate([self.y_train, y_new])
        if self.predict_fn:
            self.model = self.predict_fn(self.X_train, self.y_train)
        logger.info(f"AL update: {len(self.X_train)} total samples")

    def _uncertainty(self, X):
        preds = [self._predict(X + self.rng.normal(0, 0.1, X.shape)) for _ in range(self.config.mc_samples)]
        return np.std(preds, axis=0)

    def _thompson(self, X):
        mean = self._predict(X)
        std = self._uncertainty(X)
        return mean + self.config.exploration_factor * std * self.rng.randn(len(X))

    def _ei(self, X):
        mean = self._predict(X)
        std = self._uncertainty(X)
        best = self.y_train.max() if self.y_train is not None else 0
        z = (mean - best) / (std + 1e-8)
        try:
            from scipy.stats import norm

            return (mean - best) * norm.cdf(z) + std * norm.pdf(z)
        except ImportError:
            return mean + std

    def _diversity(self, fps):
        scores = np.ones(len(fps))
        if self.X_train is not None and len(self.X_train) > 0:
            for i, fp in enumerate(fps):
                sims = [
                    float(np.sum(fp * t) / max(np.sum(fp) + np.sum(t) - np.sum(fp * t), 1e-12))
                    for t in self.X_train[:50]
                ]
                scores[i] = 1 - min(sims) if sims else 1.0
        return scores

    def _predict(self, X):
        if self.model and callable(self.model):
            return np.array(self.model(X)).flatten()
        return self.rng.rand(len(X))

    def summary(self):
        return {
            "cycles": self.cycle,
            "train_size": len(self.X_train) if self.X_train is not None else 0,
            "acquisition": self.config.acquisition,
        }
